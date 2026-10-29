"""Cold component resource binding onto existing native ExecutionPlans."""

import hashlib
import json
import threading
from types import MappingProxyType
import weakref

from .._dimensions import check_constraints, evaluate_extent, extent_dependencies, validate_constraints, validate_extent
from .._static_values import read_static_key, static_key


def _control_identity(contract):
    """Canonical compiler-fact projection; no layout or execution interpretation."""
    ignored = {
        "semantic_key", "compilation", "configuration", "name", "execution",
        "execution_entries", "execution_fingerprint", "target", "fixed_bindings", "field_io_bindings",
    }
    facts = {key: value for key, value in contract.items() if key not in ignored}
    facts["required_capabilities"] = [
        value for value in facts["required_capabilities"] if value != "component_static_choices_v1"
    ]
    return hashlib.sha256(json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dimensions(declarations, supplied):
    base = {name for name, schema in declarations.items() if len(schema) == 4}
    if set(supplied) != base:
        raise ValueError(
            f"Component dimension inputs differ: missing={sorted(base-set(supplied))}, extra={sorted(set(supplied)-base)}"
        )
    result = {}

    def resolve(name, stack=()):
        if name in stack or name not in declarations:
            raise ValueError(f"Unknown or cyclic component dimension {name!r}")
        if name in result:
            return result[name]
        schema = declarations[name]
        value = (
            evaluate_extent(schema[4], lambda dependency: resolve(dependency, (*stack, name)))
            if len(schema) == 5
            else supplied[name]
        )
        if type(value) is not int or not (1 if schema[2] == "extent" else 0) <= value <= 2**31 - 1:
            raise ValueError(f"Component dimension {name!r} violates its i32 {schema[2]} range")
        if schema[3] and value > resolve(schema[3], (*stack, name)):
            raise ValueError(f"Active dimension {name!r} exceeds capacity {schema[3]!r}")
        result[name] = value
        return value

    for name in declarations:
        resolve(name)
    return result


class Component:
    """Loaded component code and immutable compiler-produced resource facts.

    Native SNode construction/layout and existing plan preparation remain the
    authority. This object never interprets the component's computation.
    """

    def __init__(self, module, name):
        from urllib.parse import quote
        from zipfile import ZipFile
        from ._runtime import _name

        _name(name)
        with module._runtime._lock:
            module._check()
            relative = "components/" + quote(name, safe="-_.") + ".json"
            if module._source_path.is_dir():
                payload = (module._source_path / relative).read_bytes()
            else:
                with ZipFile(module._source_path) as archive:
                    payload = archive.read(relative)
            contract = json.loads(payload)
            required = {
                "version",
                "groups",
                "dimensions",
                "resources",
                "values",
                "bindings",
                "explicit",
                "entries",
                "required_capabilities",
                "semantic_key",
                "name",
                "execution",
                "execution_entries",
                "target",
            }
            if contract.get("version") in (2, 3, 4, 5, 6, 7):
                required.add("compilation")
            if contract.get("version") in (3, 4, 5, 6, 7):
                required.add("configuration")
            if contract.get("version") in (4, 5, 6, 7):
                required.update(("control", "internal_arguments", "execution_fingerprint"))
            if contract.get("version") in (5, 6, 7):
                required.add("constraints")
            if contract.get("version") in (6, 7):
                required.update(("fixed_groups", "fixed_bindings"))
            if contract.get("version") == 7:
                required.update(("field_io", "field_io_bindings"))
            if set(contract) != required or contract["version"] not in (1, 2, 3, 4, 5, 6, 7) or contract["name"] != name:
                raise ValueError("Unsupported component contract")
            capabilities = {"dense_snode_group_v1", "component_root_v1"}
            if contract["version"] == 7:
                capabilities.add("component_field_coordinates_v1")
            fixed_groups = contract.get("fixed_groups", {})
            if contract["version"] >= 6:
                if not fixed_groups or set(fixed_groups) & set(contract["groups"]):
                    raise ValueError("Fixed SNode component groups must have distinct declared identities")
                capabilities.add("component_fixed_snode_groups_v1")
            if any(resource.get("kind") == "ndarray" for resource in contract["resources"].values()):
                capabilities.add("component_array_storage_v1")
            if any(descriptor[0] == "ndarray" for descriptor in contract["explicit"].values()):
                capabilities.add("component_explicit_ndarray_v1")
            if contract["version"] >= 2:
                capabilities.add("component_compiler_facts_v1")
                compilation = contract["compilation"]
                if (
                    compilation.get("version") != 1
                    or len(compilation.get("entries", ())) != len(contract["entries"])
                    or compilation.get("task_scope") != "compiled_not_observed_replay"
                ):
                    raise ValueError("Invalid component compiler facts")
            self._configuration = {}
            if contract["version"] >= 4:
                capabilities.add("component_control_v1")
                if not isinstance(contract["configuration"], dict):
                    raise ValueError("Invalid component configuration domain")
                if _control_identity(contract) != contract["semantic_key"]:
                    raise ValueError("Component control facts differ from their semantic identity")
            if contract["version"] == 3 or (contract["version"] >= 4 and contract["configuration"]):
                capabilities.add("component_static_choices_v1")
                if not isinstance(contract["configuration"], dict) or not contract["configuration"]:
                    raise ValueError("Invalid component configuration domain")
                for key, spec in contract["configuration"].items():
                    if not key.isidentifier() or set(spec) != {"selected", "choices"} or not spec["choices"]:
                        raise ValueError("Invalid component static choice declaration")
                    selected = static_key(read_static_key(spec["selected"]))
                    choices = tuple(static_key(read_static_key(value)) for value in spec["choices"])
                    if len(set(choices)) != len(choices) or selected not in choices:
                        raise ValueError("Component selected configuration is outside its domain")
                    for entry in contract["entries"]:
                        for dependency, value in entry["static"]:
                            if dependency == key and static_key(read_static_key(value)) != selected:
                                raise ValueError("Component configuration differs from compiled static dependencies")
                    self._configuration[key] = (selected, choices)
            if contract["version"] == 5 or contract["version"] >= 6 and contract["constraints"]:
                capabilities.add("component_dimension_constraints_v1")
                if not contract["constraints"]:
                    raise ValueError("Constrained component requires nonempty dimension constraints")
                triggers = set(contract["dimensions"]) | set(contract["values"]) | set(contract["resources"])
                triggers.update(name for entry in contract["entries"] for name, _ in entry["static"])
                triggers.update(alias for resource in contract["resources"].values() for alias in resource["aliases"])
                validate_constraints(contract["constraints"], contract["dimensions"].keys(), triggers)
                sources = contract["compilation"].get("constraint_sources")
                if (
                    not isinstance(sources, dict) or set(sources) != set(contract["constraints"])
                    or any(
                        not isinstance(source, list) or len(source) != 2 or not isinstance(source[0], str)
                        or type(source[1]) is not int or source[1] <= 0 for source in sources.values()
                    )
                ):
                    raise ValueError("Invalid component constraint declaration sources")
            if set(contract["required_capabilities"]) != capabilities:
                raise ValueError("Unsupported required component capability")
            if contract["target"] not in ("x64", "cuda") or contract["target"] != module._runtime.arch:
                raise ValueError("Component target does not match this Runtime")
            declarations = contract["dimensions"]
            for key, schema in declarations.items():
                if (
                    not key.isidentifier()
                    or not isinstance(schema, list)
                    or len(schema) not in (4, 5)
                    or schema[:2] != ["dimension", "i32"]
                    or schema[2] not in ("extent", "active")
                    or not isinstance(schema[3], str)
                    or (schema[2] == "extent" and schema[3])
                ):
                    raise ValueError("Invalid component dimension declaration")
                if schema[3] and (schema[3] not in declarations or declarations[schema[3]][2] != "extent"):
                    raise ValueError("Component active count requires a declared extent")
                if len(schema) == 5:
                    validate_extent(schema[4])
            self._layout_dimensions = set()

            def depend(name, stack=()):
                if name not in declarations or name in stack:
                    raise ValueError("Invalid component layout dimension dependency")
                schema = declarations[name]
                if len(schema) == 4:
                    self._layout_dimensions.add(name)
                else:
                    for dependency in extent_dependencies(schema[4]):
                        depend(dependency, (*stack, name))

            for schema in contract["groups"].values():
                for node in json.loads(schema)["nodes"]:
                    for extent in node["extents"]:
                        if extent["parameter"]:
                            depend(extent["parameter"])
            aliases = {}
            self._array_layouts = {}
            for key, resource in contract["resources"].items():
                array = resource.get("kind") == "ndarray"
                fixed = resource.get("kind") == "fixed_snode"
                if fixed:
                    if (set(resource) != {"kind", "group", "paths", "dtype", "element_shape", "shape", "aliases",
                                          "access", "effects_precision", "initialization"}
                            or resource["group"] not in fixed_groups
                            or any(type(extent) is not int or extent <= 0 for extent in resource["shape"])):
                        raise ValueError("Invalid fixed SNode component resource")
                if array:
                    if set(resource) != {"kind", "shape", "dtype", "element_shape", "layout", "aliases",
                                         "access", "effects_precision", "initialization"}:
                        raise ValueError("Invalid component array contract")
                    for extent in resource["shape"]:
                        if isinstance(extent, str):
                            depend(extent)
                        elif type(extent) is not int or extent < 0:
                            raise ValueError("Invalid component array extent")
                    if resource["layout"]:
                        if resource["dtype"] != "struct" or resource["element_shape"]:
                            raise ValueError("Invalid component structure array contract")
                        self._array_layouts[key] = module.ndarray_layout(resource["layout"])
                    else:
                        from ._runtime import _dtype, _shape
                        _dtype(resource["dtype"])
                        _shape(resource["element_shape"])
                if (
                    (not array and not fixed and (resource.get("kind") is not None or resource.get("group") not in contract["groups"]))
                    or not (
                        resource.get("access") == "read_write"
                        and resource.get("effects_precision") == "conservative"
                        or contract["version"] >= 2
                        and resource.get("effects_precision") == "compiler_proven"
                        and resource.get("access") in ("none", "read", "write", "read_write", "atomic")
                    )
                    or resource.get("initialization") != ("empty_activation_and_zero_values" if fixed else "consumer_explicit")
                ):
                    raise ValueError("Invalid component resource contract")
                for alias in (key, *resource["aliases"]):
                    if alias in aliases and aliases[alias] != key:
                        raise ValueError("Component resource aliases conflict")
                    aliases[alias] = key
            plan = module.execution_plan(contract["execution"])
            self._fixed_layouts = {}
            if fixed_groups:
                native_layouts = module._fixed_snode_plan_layouts(plan)
                fixed_bindings = contract["fixed_bindings"]
                if (set(fixed_bindings) != set(fixed_groups)
                        or any(type(tree) is not int for tree in fixed_bindings.values())
                        or len(set(fixed_bindings.values())) != len(fixed_bindings)
                        or set(fixed_bindings.values()) != set(native_layouts)):
                    raise ValueError("Fixed SNode component bindings differ from the compiled Plan closure")
                for group, schema in fixed_groups.items():
                    if (set(schema) != {"kind", "version", "allocation", "access_fingerprint"}
                            or schema["kind"] != "fixed_snode" or schema["version"] != 1
                            or schema["allocation"] != native_layouts[fixed_bindings[group]]):
                        raise ValueError("Fixed SNode component layout differs from its native compiled layout")
                    self._fixed_layouts[group] = schema["allocation"]
                effect_groups = {
                    effect["group"] for entry in contract["compilation"]["entries"]
                    for effect in entry["effects"] if "group" in effect
                }
                if effect_groups != set(fixed_groups):
                    raise ValueError("Fixed SNode component effects do not cover its group closure")
            document = json.loads(plan.to_json())
            operations = document["operations"]
            if contract["version"] >= 4:
                fingerprint = hashlib.sha256(
                    json.dumps(
                        dict(plan=document, semantic_key=contract["semantic_key"]),
                        sort_keys=True, separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                if (
                    fingerprint != contract["execution_fingerprint"]
                    or list(plan.kernel_entries) != contract["execution_entries"]
                ):
                    raise ValueError("Component control contract differs from its actual execution plan")
                if plan.data_entries:
                    raise ValueError("Generic component Plan contains producer data")
            elif (
                document.get("initialization")
                or document.get("resources")
                or len(operations) != len(contract["entries"])
                or len(operations) != len(contract["execution_entries"])
                or any(
                    operation["kind"] != "kernel"
                    or operation["entry"] != entry_name
                    or operation["arguments"] != entry["arguments"]
                    for operation, entry, entry_name in zip(
                        operations, contract["entries"], contract["execution_entries"]
                    )
                )
            ):
                raise ValueError("Component contract differs from its actual execution plan")
            if set(plan.argument_names) != set(contract["bindings"]) | set(contract["explicit"]):
                raise ValueError("Component contract does not cover its execution plan bindings")
            used = {"resource": set(), "value": set(), "dimension": set()}
            for binding in contract["bindings"].values():
                kind, key = binding["kind"], binding["name"]
                table = {"resource": contract["resources"], "value": contract["values"], "dimension": declarations}.get(
                    kind
                )
                if table is None or key not in table:
                    raise ValueError("Invalid component symbolic binding")
                used[kind].add(key)
            bound_resources = {name for name, resource in contract["resources"].items() if resource.get("kind") != "fixed_snode"}
            if used["resource"] != bound_resources or used["value"] != set(contract["values"]):
                raise ValueError("Component contains unreferenced resource/value requirements")
            self._field_io = {}
            if contract["version"] == 7:
                from ._field_coordinates import load_transfers
                self._field_io = load_transfers(module, contract)
            self._module, self._plan = module, plan
            self._contract_json = json.dumps(contract)
            self._aliases = aliases

    @property
    def contract(self):
        self._module._check()
        return json.loads(self._contract_json)

    def instantiate(self, dimensions, *, configuration=None):
        supplied = dict(configuration or {})
        if set(supplied) != set(self._configuration):
            raise ValueError(
                "Component configuration inputs differ: "
                f"missing={sorted(set(self._configuration)-set(supplied))}, "
                f"extra={sorted(set(supplied)-set(self._configuration))}"
            )
        for name, value in supplied.items():
            selected, choices = self._configuration[name]
            actual = static_key(value)
            if actual not in choices:
                raise ValueError(f"Component configuration {name!r} is outside its declared domain")
            if actual != selected:
                raise ValueError(f"Component configuration {name!r} requires a different compiled variant")
        return ComponentInstance(self, dimensions)


class ComponentInstance:
    """Private resource groups with immutable per-prepare argument bindings.

    Each preparation owns its argument snapshot, including on an independent
    execution domain. Close revokes submission then retires accepted work;
    it is a blocking cold operation, never a module root-pointer swap.
    """

    def __init__(self, component, dimensions):
        self._component = component
        self._groups, self._resources = {}, {}
        self._fixed_state = None
        self._prepared = weakref.WeakSet()
        self._binding_domain = None
        self._transfer_active = False
        self._close_lock = threading.RLock()
        self._closed = self._retired = False
        self._initialized = set()
        contract = component.contract
        self._inputs = dict(dimensions)
        self._dimensions = _dimensions(contract["dimensions"], self._inputs)
        check_constraints(
            contract.get("constraints", {}), self._dimensions.__getitem__,
            contract.get("compilation", {}).get("constraint_sources"),
            entry=contract["name"],
        )
        runtime = component._module._runtime
        try:
            with runtime._lock:
                component._module._check()
                if contract.get("fixed_groups"):
                    self._fixed_state = component._module.snode_instance(plan=component._plan)
                for name, schema in contract["groups"].items():
                    required = {
                        extent["parameter"]
                        for node in json.loads(schema)["nodes"]
                        for extent in node["extents"]
                        if extent["parameter"]
                    }
                    self._groups[name] = runtime.dense_snode_group(
                        schema, {key: self._dimensions[key] for key in required}
                    )
                for name, resource in contract["resources"].items():
                    if resource.get("kind") == "fixed_snode":
                        if name in component._field_io:
                            from ._field_coordinates import FieldCoordinates
                            self._resources[name] = FieldCoordinates(self, resource, component._field_io[name])
                            self._initialized.add(name)
                        continue
                    shape = tuple(
                        self._dimensions[item] if isinstance(item, str) else item for item in resource["shape"]
                    )
                    if resource.get("kind") == "ndarray":
                        view = runtime.ndarray(component._array_layouts.get(name, resource["dtype"]), shape,
                                               element_shape=resource["element_shape"])
                    else:
                        view = self._groups[resource["group"]].member(
                            resource["paths"], resource["dtype"], element_shape=resource["element_shape"]
                        )
                    self._resources[name] = view
                    if view.shape != shape:
                        raise ValueError("Native SNode layout differs from the component extent declaration")
        except BaseException:
            self.close()
            raise

    def _check(self):
        self._component._module._check()
        if self._closed:
            raise RuntimeError("Component instance is closed")

    @property
    def dimensions(self):
        return MappingProxyType(dict(self._dimensions))

    @property
    def layout(self):
        """Actual native SNode layout facts, including complete root storage."""
        self._check()
        return {**{name: group.layout for name, group in self._groups.items()},
                **json.loads(json.dumps(self._component._fixed_layouts))}

    @property
    def nbytes(self):
        """Root and independent-array bytes; excludes code and driver overhead."""
        self._check()
        if self._fixed_state is not None:
            raise NotImplementedError("Sparse allocator usage is dynamic; fixed root bytes are not total storage usage")
        contract = self._component.contract
        return sum(group.nbytes for group in self._groups.values()) + sum(
            self._resources[name].nbytes for name, resource in contract["resources"].items()
            if resource.get("kind") == "ndarray")

    def resource(self, name):
        self._check()
        key = self._component._aliases[name]
        if key not in self._resources:
            raise NotImplementedError("Sparse resources retain SNode storage; use original component entries to initialize/read values")
        return self._resources[key]

    def initialize(self, data=None, *, zero=()):
        with self._component._module._runtime._lock:
            self._initialize(data, zero=zero)

    def _initialize(self, data, *, zero):
        self._check()
        if self._transfer_active or self._has_live_bindings():
            raise RuntimeError("Component initialization requires closing prepared calls at a cold boundary")
        selected = {}
        for name, value in dict(data or {}).items():
            key = self._component._aliases[name]
            if key not in self._resources or self._component.contract["resources"][key].get("kind") == "fixed_snode":
                raise ValueError("Sparse model data must use a declared numerical initializer, not a dense byte upload")
            if key in selected:
                raise ValueError("Initialization repeats an aliased component resource")
            selected[key] = value
        for name in zero:
            key = self._component._aliases[name]
            if key not in self._resources or self._component.contract["resources"][key].get("kind") == "fixed_snode":
                raise ValueError("Sparse reset requires the original component reset entry or a new empty instance")
            if key in selected:
                raise ValueError("Initialization repeats an aliased component resource")
            selected[key] = bytes(self._resources[key].nbytes)
        for name, value in selected.items():
            self._resources[name].write(value)
            self._initialized.add(name)

    def _has_live_bindings(self):
        runtime = self._component._module._runtime
        return any(call._handle is not None for call in self._prepared) or any(
            self in owners for bindings in runtime._domain_owners.values() for owners in bindings.values()
        )

    def prepare(self, *, values=None, dimensions=None, arguments=None, execution_domain=None):
        from ._runtime import Scalar

        runtime = self._component._module._runtime
        with runtime._lock:
            self._check()
            if self._transfer_active:
                raise RuntimeError("Component Field transfer is in progress")
            if self._has_live_bindings() and execution_domain is not self._binding_domain:
                raise ValueError("Component state requires one ordered execution domain until its bindings retire")
            if self._initialized != set(self._resources):
                raise ValueError("Component resources require explicit consumer initialization before preparation")
            updates = dict(dimensions or {})
            if set(updates) & self._component._layout_dimensions:
                raise ValueError("Changing layout dimensions requires a new component instance")
            contract = self._component.contract
            resolved = _dimensions(contract["dimensions"], {**self._inputs, **updates})
            check_constraints(
                contract.get("constraints", {}), resolved.__getitem__,
                contract.get("compilation", {}).get("constraint_sources"),
                entry=contract["name"],
            )
            values, bindings = dict(values or {}), dict(arguments or {})
            if set(values) != set(contract["values"]) or set(bindings) != set(contract["explicit"]):
                raise ValueError("Component value/explicit argument inputs differ from its contract")
            for symbol, value in tuple(bindings.items()):
                descriptor = contract["explicit"][symbol]
                if descriptor[0] == "ndarray":
                    # Resource type, runtime, dimensions and lifetime remain
                    # the existing native preparation contract's authority.
                    continue
                bindings[symbol] = (
                    Scalar(descriptor[1][1], value)
                    if descriptor[0] == "scalar"
                    else self._component._plan.value(self._component._module, symbol, value)
                )
            for symbol, binding in contract["bindings"].items():
                kind, name = binding["kind"], binding["name"]
                if kind == "resource":
                    bindings[symbol] = self._resources[name]
                elif kind == "dimension":
                    bindings[symbol] = resolved[name]
                else:
                    schema = contract["values"][name]
                    bindings[symbol] = (
                        self._component._plan.value(self._component._module, symbol, values[name])
                        if schema[2]
                        else Scalar(schema[1], values[name])
                    )
            call = self._component._module.prepare_execution(
                self._component._plan, bindings, execution_domain=execution_domain, snode_instance=self._fixed_state
            )
            try:
                call._retain_binding_owner(self)
            except BaseException:
                call.close()
                raise
            self._prepared.add(call)
            self._binding_domain = execution_domain
            return call

    def close(self):
        with self._close_lock:
            self._close()

    def _close(self):
        runtime = self._component._module._runtime
        with runtime._lock:
            if getattr(self, "_retired", True):
                return
            self._closed = True
            calls = tuple(self._prepared)
        # PreparedCall.close() releases the runtime submission lock while
        # retiring its accepted domain work. Do not enclose it in that lock.
        for call in calls:
            call.close()
        for resource in self._resources.values():
            resource.close()
        for group in self._groups.values():
            group.close()
        if self._fixed_state is not None:
            self._fixed_state.close()
        self._retired = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
