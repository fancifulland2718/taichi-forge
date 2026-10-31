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
        "execution_entries", "execution_fingerprint", "target", "fixed_bindings", "field_io_bindings", "setup_bindings", "operation_bindings",
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


def _validate_input_sources(contract):
    sources = contract["compilation"]["input_sources"]
    expected = dict(
        dimensions={name for name, schema in contract["dimensions"].items() if len(schema) == 4},
        values=set(contract["values"]), resources=set(contract["resources"]),
    )
    if not isinstance(sources, dict) or set(sources) != set(expected):
        raise ValueError("Invalid component input source roles")
    for role, names in expected.items():
        if not isinstance(sources[role], dict) or set(sources[role]) != names:
            raise ValueError(f"Component input sources must cover {role}")

    def validate(source, *, resource=False):
        if not isinstance(source, dict):
            raise ValueError("Invalid component input source path")
        kind = source.get("kind")
        keys = {"kind", "path"}
        if resource and kind == "borrowed_group_member":
            keys.add("member")
            if not isinstance(source.get("member"), str) or not source["member"]:
                raise ValueError("Invalid component borrowed input member")
        elif kind != "attribute":
            raise ValueError("Invalid component input source kind")
        if (set(source) != keys or not isinstance(source.get("path"), list) or not source["path"]
                or any(not isinstance(part, str) or not part.isidentifier() for part in source["path"])):
            raise ValueError("Invalid component input source path")

    for role in ("dimensions", "values"):
        for source in sources[role].values():
            validate(source)
    for name, aliases in sources["resources"].items():
        if not isinstance(aliases, dict) or set(aliases) != set(contract["resources"][name]["aliases"]):
            raise ValueError("Component input sources must cover resource aliases")
        for source in aliases.values():
            validate(source, resource=True)


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
            if contract.get("version") in (2, 3, 4, 5, 6, 7, 8, 9, 10):
                required.add("compilation")
            if contract.get("version") in (3, 4, 5, 6, 7, 8, 9, 10):
                required.add("configuration")
            if contract.get("version") in (4, 5, 6, 7, 8, 9, 10):
                required.update(("control", "internal_arguments", "execution_fingerprint"))
            if contract.get("version") in (5, 6, 7, 8, 9, 10):
                required.add("constraints")
            if contract.get("version") in (6, 7, 8, 9, 10):
                required.update(("fixed_groups", "fixed_bindings"))
            if contract.get("version") in (7, 8, 9, 10):
                required.update(("field_io", "field_io_bindings"))
            if contract.get("version") in (8, 9, 10):
                required.add("lifecycle")
            if contract.get("version") in (9, 10):
                required.add("setup_bindings")
            if contract.get("version") == 10:
                required.add("operation_bindings")
            host_resource_capability = "component_host_resources_v1" in contract.get("required_capabilities", ())
            if host_resource_capability:
                required.add("host_resources")
            if set(contract) != required or contract["version"] not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) or contract["name"] != name:
                raise ValueError("Unsupported component contract")
            capabilities = {"dense_snode_group_v1", "component_root_v1"}
            pending = [contract.get("control", {})]
            native_sort = False
            while pending:
                region = pending.pop()
                pending.extend(region.get("children", ()))
                if region.get("kind") != "native_sort":
                    continue
                native_sort = True
                operations = region.get("operations")
                if set(region) != {"kind", "operations"} or not isinstance(operations, list) or not operations:
                    raise ValueError("Invalid component native sort definition")
                for operation in operations:
                    if (operation.get("kind") != "sort" or not {"kind", "keys"} <= set(operation)
                            or set(operation) - {"kind", "keys", "values"}):
                        raise ValueError("Invalid component native sort operation")
                    for port, symbol in operation.items():
                        if port == "kind":
                            continue
                        binding = contract["bindings"].get(symbol, {})
                        resource = contract["resources"].get(binding.get("name"), {})
                        if (binding.get("kind") != "resource" or resource.get("kind") != "ndarray"
                                or len(resource.get("shape", ())) != 1
                                or port == "keys" and resource.get("element_shape")):
                            raise ValueError("Component native sort requires declared one-dimensional arrays")
            if native_sort:
                capabilities.add("component_native_sort_v1")
            host_resources = contract.get("host_resources", [])
            if host_resource_capability:
                if (not isinstance(host_resources, list) or not host_resources
                        or any(not isinstance(name, str) for name in host_resources)
                        or len(host_resources) != len(set(host_resources))
                        or any(name not in contract["resources"] or contract["resources"][name].get("kind") is not None
                               for name in host_resources)):
                    raise ValueError("Invalid component Host-visible resources")
                capabilities.add("component_host_resources_v1")
            if contract["version"] >= 7 and contract["field_io"]:
                capabilities.add("component_field_coordinates_v1")
            fixed_groups = contract.get("fixed_groups", {})
            if contract["version"] in (6, 7) or fixed_groups:
                if not fixed_groups or set(fixed_groups) & set(contract["groups"]):
                    raise ValueError("Fixed SNode component groups must have distinct declared identities")
                capabilities.add("component_fixed_snode_groups_v1")
            if contract["version"] in (8, 9, 10):
                lifecycle = contract["lifecycle"]
                lifecycle_keys = {"initialization_nodes", "initialization", "reprepare"}
                if contract["version"] in (9, 10):
                    lifecycle_keys.add("setup_nodes")
                    nodes, start, names = lifecycle.get("setup_nodes"), 0, set()
                    if not isinstance(nodes, list) or (not nodes and contract["version"] == 9):
                        raise ValueError("Invalid component setup nodes")
                    for item in nodes:
                        if not isinstance(item, list) or len(item) != 2:
                            raise ValueError("Invalid component setup nodes")
                        label, end = item
                        if (not isinstance(label, str) or not label.isidentifier() or label in names
                                or type(end) is not int or not start < end < lifecycle.get("initialization_nodes", 0)):
                            raise ValueError("Invalid component setup nodes")
                        names.add(label)
                        start = end
                    if nodes:
                        capabilities.add("component_cold_setup_v1")
                if contract["version"] == 10:
                    lifecycle_keys.add("operation_nodes")
                    nodes = lifecycle.get("operation_nodes")
                    start, names = lifecycle.get("initialization_nodes", 0), set()
                    if not isinstance(nodes, list) or not nodes:
                        raise ValueError("Invalid component operation nodes")
                    for item in nodes:
                        if not isinstance(item, list) or len(item) != 2:
                            raise ValueError("Invalid component operation nodes")
                        label, end = item
                        if (not isinstance(label, str) or not label.isidentifier() or label in names
                                or type(end) is not int or not start < end):
                            raise ValueError("Invalid component operation nodes")
                        names.add(label)
                        start = end
                    capabilities.add("component_named_operations_v1")
                if (set(lifecycle) != lifecycle_keys
                        or type(lifecycle["initialization_nodes"]) is not int or lifecycle["initialization_nodes"] <= 0
                        or lifecycle["initialization"] != "explicit_instance" or lifecycle["reprepare"] != "preserve_state"):
                    raise ValueError("Invalid component instance lifecycle")
                if not fixed_groups and contract["fixed_bindings"]:
                    raise ValueError("Component without fixed groups cannot contain fixed bindings")
                capabilities.add("component_instance_initializer_v1")
            if any(resource.get("kind") == "ndarray" for resource in contract["resources"].values()):
                capabilities.add("component_array_storage_v1")
            if any(resource.get("initialization") == "zero" for resource in contract["resources"].values()):
                capabilities.add("component_zero_initial_state_v1")
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
            configuration_sources = contract.get("compilation", {}).get("configuration_sources")
            if "configuration_sources" in contract.get("compilation", {}):
                if not isinstance(configuration_sources, dict) or set(configuration_sources) != set(self._configuration):
                    raise ValueError("Component configuration sources must cover the admitted inputs")
                for source in configuration_sources.values():
                    if (
                        not isinstance(source, dict) or set(source) != {"kind", "path"}
                        or source["kind"] not in ("attribute", "recording_argument")
                        or not isinstance(source["path"], list) or not source["path"]
                        or any(not isinstance(part, str) or not part.isidentifier() for part in source["path"])
                        or source["kind"] == "recording_argument" and len(source["path"]) != 1
                    ):
                        raise ValueError("Invalid component configuration source path")
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
            if contract["target"] not in ("x64", "cuda", "vulkan") or contract["target"] != module._runtime.arch:
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
                    or resource.get("initialization") not in (
                        ("empty_activation_and_zero_values",) if fixed else ("consumer_explicit", "zero"))
                ):
                    raise ValueError("Invalid component resource contract")
                for alias in (key, *resource["aliases"]):
                    if alias in aliases and aliases[alias] != key:
                        raise ValueError("Component resource aliases conflict")
                    aliases[alias] = key
            if "input_sources" in contract.get("compilation", {}):
                _validate_input_sources(contract)
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
                execution_facts = dict(plan=document, semantic_key=contract["semantic_key"])
                if contract["version"] in (9, 10):
                    execution_facts["setup_bindings"] = contract["setup_bindings"]
                if contract["version"] == 10:
                    execution_facts["operation_bindings"] = contract["operation_bindings"]
                fingerprint = hashlib.sha256(
                    json.dumps(
                        execution_facts,
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
            if used["resource"] | set(host_resources) != bound_resources or used["value"] != set(contract["values"]):
                raise ValueError("Component contains unreferenced resource/value requirements")
            self._field_io = {}
            if contract["version"] >= 7:
                from ._field_coordinates import load_transfers
                self._field_io = load_transfers(module, contract)
            self._instance_plan = plan
            if self._field_io:
                from ._plan import ExecutionPlan

                # Allocation/binding closure only; this projection is never
                # executed. Cold Field I/O needs its own private Kernel views,
                # in addition to the numerical and initializer entries. Every
                # transfer above already passed native tree/contract checks.
                closure = json.loads(plan.to_json())
                closure.pop("version")
                selected = set(plan.kernel_entries)
                for _, transfers in self._field_io.values():
                    for transfer in transfers.values():
                        for operation in json.loads(transfer.to_json())["operations"]:
                            if operation["entry"] not in selected:
                                closure["operations"].append(operation)
                                selected.add(operation["entry"])
                self._instance_plan = ExecutionPlan(**closure)
            self._operation_plans = {}
            self._setup_plans = {}
            self._initializer_plan = None
            self._replay_plan = plan
            if contract["version"] in (8, 9, 10):
                from ._component_lifecycle import split_component_lifecycle

                # Both phases are projections of the validated native Plan.
                # Keep the full Plan for native tree-closure instantiation.
                # A fresh replay preparation never reruns the model initializer.
                self._initializer_plan, self._replay_plan = split_component_lifecycle(plan)
                if contract["version"] in (9, 10):
                    from ._component_lifecycle import split_component_setup
                    self._setup_plans, self._initializer_plan = split_component_setup(
                        self._initializer_plan, contract["setup_bindings"])
                    if list(self._setup_plans) != [name for name, _ in contract["lifecycle"]["setup_nodes"]]:
                        raise ValueError("Component setup bindings differ from the declared entries")
            if contract["version"] == 10:
                from ._component_lifecycle import split_component_operations
                self._operation_plans, self._replay_plan = split_component_operations(
                    self._replay_plan, contract["operation_bindings"])
                if list(self._operation_plans) != [name for name, _ in contract["lifecycle"]["operation_nodes"]]:
                    raise ValueError("Component operation bindings differ from the declared entries")
            self._module, self._plan = module, plan
            self._contract_json = json.dumps(contract)
            self._aliases = aliases

    @property
    def contract(self):
        self._module._check()
        return json.loads(self._contract_json)

    @property
    def setup_entries(self):
        """Names of original cold preparation recordings exported with this root."""
        self._module._check()
        return tuple(self._setup_plans)

    @property
    def operation_entries(self):
        """Named normal recordings sharing this component instance's state."""
        self._module._check()
        return tuple(self._operation_plans)

    def _operation_plan(self, name):
        self._module._check()
        if name is None:
            return self._replay_plan
        if not isinstance(name, str) or name not in self._operation_plans:
            raise ValueError(f"Unknown component operation {name!r}")
        return self._operation_plans[name]

    def inputs(self, *, initialization=False, setup=None, operation=None):
        """Query external values/arguments needed by one lifecycle phase."""
        if sum((setup is not None, bool(initialization), operation is not None)) > 1:
            raise ValueError("Choose setup, final initialization or operation, not both")
        plan = (self._setup_plan(setup) if setup is not None
                else self._initializer_plan if initialization else self._operation_plan(operation))
        if plan is None:
            raise ValueError("Component has no registered instance initializer")
        contract = self.contract
        names = set(plan.argument_names)
        return dict(
            values={binding["name"]: contract["values"][binding["name"]]
                    for symbol, binding in contract["bindings"].items()
                    if symbol in names and binding["kind"] == "value"},
            arguments={name: spec for name, spec in contract["explicit"].items() if name in names},
        )

    def _setup_plan(self, name):
        self._module._check()
        if not isinstance(name, str) or name not in self._setup_plans:
            raise ValueError(f"Unknown component setup entry {name!r}")
        return self._setup_plans[name]

    def configuration_sources(self):
        """Query declared Host attribute paths and normal recording arguments.

        The consumer supplies its own model values under the returned input IDs.
        Paths describe provenance; this query does not traverse Python objects,
        execute getters, select variants or bypass instantiate's static checks.
        Older artifacts remain loadable but must be rebuilt for this query.
        """
        sources = self.contract.get("compilation", {}).get("configuration_sources")
        if sources is None:
            raise ValueError("Artifact has no configuration source paths; rebuild it to query model provenance")
        return sources

    def input_sources(self):
        """Query normal sources of base dimensions, dynamic values and resources.

        Resource entries retain every declared alias. Borrowed members identify
        the anchor path and the original member separately; the anchor is not
        necessarily the requested member. Derived dimensions remain expressions
        in ``contract``. This returns a copy and never evaluates Python objects.
        """
        sources = self.contract.get("compilation", {}).get("input_sources")
        if sources is None:
            raise ValueError("Artifact has no input source paths; rebuild it to query model provenance")
        return sources

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
        self._model_initialized = component._initializer_plan is None
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
                    self._fixed_state = component._module.snode_instance(plan=component._instance_plan)
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
                # Only the allocation site's explicit model policy supplies
                # these initial values. Never infer zero from producer bytes,
                # read/write effects or a resource's name. Reprepare does not
                # revisit this construction boundary or reset persistent state.
                self._initialize(None, zero=[name for name, resource in contract["resources"].items()
                                             if resource["initialization"] == "zero"])
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

    def prepare(self, *, values=None, dimensions=None, arguments=None, execution_domain=None, operation=None):
        runtime = self._component._module._runtime
        with runtime._lock:
            self._check()
            if self._transfer_active:
                raise RuntimeError("Component Field transfer is in progress")
            if not self._model_initialized:
                raise ValueError("Component requires initialize_state() before preparation")
            if self._has_live_bindings() and execution_domain is not self._binding_domain:
                raise ValueError("Component state requires one ordered execution domain until its bindings retire")
            if self._initialized != set(self._resources):
                raise ValueError("Component resources require explicit consumer initialization before preparation")
            plan = self._component._operation_plan(operation)
            bindings = self._bindings(plan, values, dimensions, arguments)
            call = self._component._module.prepare_execution(
                plan, bindings, execution_domain=execution_domain, snode_instance=self._fixed_state
            )
            try:
                call._retain_binding_owner(self)
            except BaseException:
                call.close()
                raise
            self._prepared.add(call)
            self._binding_domain = execution_domain
            return call

    def _bindings(self, plan, values, dimensions, arguments):
        from ._runtime import Scalar

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
        names = set(plan.argument_names)
        required_values = {binding["name"] for symbol, binding in contract["bindings"].items()
                           if symbol in names and binding["kind"] == "value"}
        if set(values) != required_values or set(bindings) != set(contract["explicit"]) & names:
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
                else plan.value(self._component._module, symbol, value)
            )
        for symbol, binding in contract["bindings"].items():
            if symbol not in names:
                continue
            kind, name = binding["kind"], binding["name"]
            if kind == "resource":
                bindings[symbol] = self._resources[name]
            elif kind == "dimension":
                bindings[symbol] = resolved[name]
            else:
                schema = contract["values"][name]
                bindings[symbol] = (
                    plan.value(self._component._module, symbol, values[name])
                    if schema[2]
                    else Scalar(schema[1], values[name])
                )
        return bindings

    def initialize_state(self, *, values=None, dimensions=None, arguments=None, reset=False):
        """Run the original model initializer at a cold instance boundary.

        Repeated initialization requires explicit reset=True. A failed accepted
        initializer may have written state; replay stays disabled until a
        successful retry. No numerical writes are transactionally rolled back.
        """
        return self._run_cold_plan(self._component._initializer_plan, values=values,
                                   dimensions=dimensions, arguments=arguments, reset=reset, finish=True)

    def run_setup(self, name, *, values=None, dimensions=None, arguments=None, reset=False):
        """Complete one original cold preparation entry on this instance.

        Host model preparation may read/adjust resources between calls. This
        never makes the model ready; initialize_state() remains the final gate.
        No interpreter, host callback or module-global pointer swap is involved.
        """
        return self._run_cold_plan(self._component._setup_plan(name), values=values,
                                   dimensions=dimensions, arguments=arguments, reset=reset, finish=False)

    def _run_cold_plan(self, plan, *, values, dimensions, arguments, reset, finish):
        from contextlib import ExitStack, closing

        component = self._component
        runtime = component._module._runtime
        with self._close_lock:
            with runtime._lock:
                self._check()
                if plan is None:
                    raise ValueError("Component has no registered instance initializer")
                if type(reset) is not bool:
                    raise TypeError("reset must be a bool")
                if self._transfer_active or self._has_live_bindings():
                    raise RuntimeError("State initialization requires closing prepared calls at a cold boundary")
                if self._model_initialized and not reset:
                    raise ValueError("Component state is already initialized; use reset=True explicitly")
                required = (set(self._resources) if finish else {
                    binding["name"] for symbol, binding in component.contract["bindings"].items()
                    if symbol in plan.argument_names and binding["kind"] == "resource"})
                if not required <= self._initialized:
                    raise ValueError("Component resources require explicit consumer initialization before state initialization")
                bindings = self._bindings(plan, values, dimensions, arguments)
                self._transfer_active = True
            try:
                with ExitStack() as owners:
                    domain = (owners.enter_context(runtime.execution_domain(capacity=1, instance_capacity=1))
                              if runtime.arch == "cuda" else None)
                    call = owners.enter_context(component._module.prepare_execution(
                        plan, bindings, execution_domain=domain, snode_instance=self._fixed_state))
                    with runtime._lock:
                        self._model_initialized = False
                    call.initialize()
                    if domain is None:
                        call.run()
                    else:
                        with closing(call.submit()) as ticket:
                            ticket.wait()
                with runtime._lock:
                    self._model_initialized = finish
            finally:
                with runtime._lock:
                    self._transfer_active = False

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
