"""Cold component resource binding onto existing native ExecutionPlans."""

import json
from types import MappingProxyType
import weakref

from .._dimensions import evaluate_extent, extent_dependencies, validate_extent
from .._static_values import read_static_key, static_key


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
            if len(payload) > 4 * 1024 * 1024:
                raise ValueError("Component contract is too large")
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
            if contract.get("version") in (2, 3):
                required.add("compilation")
            if contract.get("version") == 3:
                required.add("configuration")
            if set(contract) != required or contract["version"] not in (1, 2, 3) or contract["name"] != name:
                raise ValueError("Unsupported component contract")
            capabilities = {"dense_snode_group_v1", "component_root_v1"}
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
            if contract["version"] == 3:
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
            for key, resource in contract["resources"].items():
                if (
                    resource.get("group") not in contract["groups"]
                    or not (
                        resource.get("access") == "read_write"
                        and resource.get("effects_precision") == "conservative"
                        or contract["version"] >= 2
                        and resource.get("effects_precision") == "compiler_proven"
                        and resource.get("access") in ("none", "read", "write", "read_write", "atomic")
                    )
                    or resource.get("initialization") != "consumer_explicit"
                ):
                    raise ValueError("Invalid component resource contract")
                for alias in (key, *resource["aliases"]):
                    if alias in aliases and aliases[alias] != key:
                        raise ValueError("Component resource aliases conflict")
                    aliases[alias] = key
            plan = module.execution_plan(contract["execution"])
            document = json.loads(plan.to_json())
            operations = document["operations"]
            if (
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
            if used["resource"] != set(contract["resources"]) or used["value"] != set(contract["values"]):
                raise ValueError("Component contains unreferenced resource/value requirements")
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

    Initial implementation provides cold close/reprepare. It does not swap any
    module-level root pointer and makes no nonblocking retirement claim.
    """

    def __init__(self, component, dimensions):
        self._component = component
        self._groups, self._resources = {}, {}
        self._prepared = weakref.WeakSet()
        self._closed = self._retired = False
        self._initialized = set()
        contract = component.contract
        self._inputs = dict(dimensions)
        self._dimensions = _dimensions(contract["dimensions"], self._inputs)
        runtime = component._module._runtime
        try:
            with runtime._lock:
                component._module._check()
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
                    view = self._groups[resource["group"]].member(
                        resource["paths"], resource["dtype"], element_shape=resource["element_shape"]
                    )
                    self._resources[name] = view
                    shape = tuple(
                        self._dimensions[item] if isinstance(item, str) else item for item in resource["shape"]
                    )
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
        return {name: group.layout for name, group in self._groups.items()}

    @property
    def nbytes(self):
        """Root allocation bytes; excludes executable code and driver overhead."""
        self._check()
        return sum(group.nbytes for group in self._groups.values())

    def resource(self, name):
        self._check()
        return self._resources[self._component._aliases[name]]

    def initialize(self, data=None, *, zero=()):
        with self._component._module._runtime._lock:
            self._initialize(data, zero=zero)

    def _initialize(self, data, *, zero):
        self._check()
        if any(call._handle is not None for call in self._prepared):
            raise RuntimeError("Component initialization requires closing prepared calls at a cold boundary")
        selected = {}
        for name, value in dict(data or {}).items():
            key = self._component._aliases[name]
            if key in selected:
                raise ValueError("Initialization repeats an aliased component resource")
            selected[key] = value
        for name in zero:
            key = self._component._aliases[name]
            if key in selected:
                raise ValueError("Initialization repeats an aliased component resource")
            selected[key] = bytes(self._resources[key].nbytes)
        for name, value in selected.items():
            self._resources[name].write(value)
            self._initialized.add(name)

    def prepare(self, *, values=None, dimensions=None, arguments=None, execution_domain=None):
        from ._runtime import Scalar

        if execution_domain is not None:
            raise ValueError("Component execution domains require separate lifecycle qualification")
        runtime = self._component._module._runtime
        with runtime._lock:
            self._check()
            if self._initialized != set(self._resources):
                raise ValueError("Component resources require explicit consumer initialization before preparation")
            updates = dict(dimensions or {})
            if set(updates) & self._component._layout_dimensions:
                raise ValueError("Changing layout dimensions requires a new component instance")
            contract = self._component.contract
            resolved = _dimensions(contract["dimensions"], {**self._inputs, **updates})
            values, bindings = dict(values or {}), dict(arguments or {})
            if set(values) != set(contract["values"]) or set(bindings) != set(contract["explicit"]):
                raise ValueError("Component value/explicit argument inputs differ from its contract")
            for symbol, value in tuple(bindings.items()):
                descriptor = contract["explicit"][symbol]
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
                self._component._plan, bindings, execution_domain=execution_domain
            )
            call._owners = (*call._owners, self)
            if call._replay_owners is not None:
                call._replay_owners = (*call._replay_owners, self)
            self._prepared.add(call)
            return call

    def close(self):
        with self._component._module._runtime._lock:
            self._close()

    def _close(self):
        if getattr(self, "_retired", True):
            return
        self._closed = True
        for call in tuple(self._prepared):
            call.close()
        for resource in self._resources.values():
            resource.close()
        for group in self._groups.values():
            group.close()
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
