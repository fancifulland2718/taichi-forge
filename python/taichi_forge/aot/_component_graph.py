"""Component capture glue for the existing Graph builder and AOT lowering.

This module collects contracts. Graph owns operation order and executable
lowering; there is no component execution interpreter.
"""

import inspect
import json

from taichi_forge.aot._component import (
    Array,
    DenseField,
    Dimension,
    RuntimeValue,
    RootOption,
    StaticValue,
    _digest,
    _static_value,
    _dimension_value,
    _capture_id,
    _dense_group,
)
from taichi_forge.aot.utils import produce_injected_args_for_graph
from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge.lang.matrix import MatrixType
from taichi_forge.types._argument_descriptor import describe_symbolic_arg, describe_element_type


class _GraphCapture:
    def __init__(self, component, record, args=(), kwargs=None, build=None):
        self.component = component
        self.owner = getattr(record, "__self__", None)
        self.build = build
        self.codes = []
        self.dispatches = {}
        self.natives = {}
        self.groups = {}
        self.fixed_groups = {}
        self.fixed_dependencies = {}
        self.dimensions = {}
        self.resources = {}
        self.host_resources = set()
        self.bindings = {}
        self.values = {}
        self.explicit = {}
        self.entries = []
        self.symbols = {}
        self.array_types = {}
        self._array_values = {}
        self._shape_values = {}
        self._members = {}
        self._constants = tuple(
            name for name, declaration in component.declarations.items()
            if isinstance(declaration, StaticValue) and declaration.choices is None
            and self.owner is not None and component.available(self.owner, name)
        )
        self._constant_snapshot = self._constant_values()
        self._choices = {
            name: declaration
            for name, declaration in component.declarations.items()
            if isinstance(declaration, StaticValue)
            and declaration.choices is not None
            and (self.owner is None or component.available(self.owner, name))
        }
        self._options = {}
        if any(isinstance(declaration, RootOption) for declaration in self._choices.values()):
            signature = inspect.signature(record)
            parameters = signature.bind(object(), *args, **(kwargs or {}))
            parameters.apply_defaults()
            builder_name = next(iter(signature.parameters))
            for name, declaration in self._choices.items():
                if isinstance(declaration, RootOption):
                    if name == builder_name or name not in parameters.arguments:
                        raise TaichiCompilationError(f"Component recording option {name!r} has no normal argument")
                    self._options[name] = parameters.arguments[name]
        self._root_name = getattr(record, "__qualname__", "component root")
        self.configuration = {}
        if self._choices:
            self.owner = getattr(record, "__self__", None)
            if self.owner is None:
                raise TaichiCompilationError(
                    "A finite component domain requires a bound normal recording method; "
                    "its configuration must be frozen before Python graph construction"
                )
            self.configuration = self._configuration()

    def _constant_values(self):
        return {name: _static_value(self.component.read(self.owner, name)) for name in self._constants}

    def _configuration(self):
        return {
            name: dict(
                selected=declaration.validate(
                    (
                        self._options[name]
                        if isinstance(declaration, RootOption)
                        else self.component.read(self.owner, name)
                    ),
                    name,
                    self._root_name,
                ),
                choices=[_static_value(value) for value in declaration.choices],
            )
            for name, declaration in self._choices.items()
        }

    def dispatch(self, builder, entry, args, template_args, label):
        from taichi_forge.graph._graph import Arg, ArgKind, flatten_args
        from taichi_forge.lang._ndarray import Ndarray

        owner = getattr(entry, "_kernel_owner", None)
        kernel = getattr(entry, "_primal", None)
        if kernel is None:
            raise TaichiCompilationError("A component Graph requires original decorated kernels")
        expected_templates = {} if template_args is None else dict(template_args)
        if owner is not None:
            name = kernel.arguments[0].name
            if name in expected_templates and expected_templates[name] is not owner:
                raise TaichiCompilationError("Component Graph templates must name the same declared owner")
            expected_templates[name] = owner
        if self.owner is None:
            self.owner = owner
        if self.owner is None:
            raise TaichiCompilationError("A component Graph root requires a declared resource owner")
        if owner is not None:
            self.component._entry_owner_path(self.owner, owner)
        owner = self.owner
        managed_symbols = set()
        normalized = []
        for value in args:
            if isinstance(value, Ndarray):
                matches = [name for name, declaration in self.component.declarations.items()
                           if isinstance(declaration, Array) and self.component.available(owner, name)
                           and self.component.read(owner, name) is value]
                if not matches:
                    raise TaichiCompilationError("Component Graph managed arrays require a normal ResourceScope declaration")
                declaration = self.component.declarations[matches[0]]
                value = self._array_argument(declaration, matches)
                managed_symbols.add(value.name)
            normalized.append(value)
        args = tuple(normalized)
        injected = produce_injected_args_for_graph(kernel, args, expected_templates)
        code = self.component.compile(
            entry,
            *(value for i, value in enumerate(injected) if i not in kernel.template_slot_locations),
            owner=owner,
            template_args=expected_templates,
        )
        if self.build is not None:
            self.build.admit_kernel(code)
        for arg in flatten_args(args):
            if arg.name in managed_symbols:
                continue
            if arg.name.startswith("__component_"):
                raise TaichiCompilationError("Explicit Graph arguments cannot use the reserved __component_ prefix")
            descriptor = describe_symbolic_arg(arg).structural_key()
            if arg.name in self.explicit and self.explicit[arg.name] != descriptor:
                raise TaichiCompilationError(f"Component Graph argument {arg.name!r} has incompatible uses")
            self.explicit[arg.name] = descriptor
        contract = code.contract
        for group, schema in code.fixed_groups.items():
            if self.fixed_groups.setdefault(group, schema) != schema or (
                self.fixed_dependencies.setdefault(group, code.fixed_dependencies[group]) != code.fixed_dependencies[group]
            ):
                raise TaichiCompilationError("Fixed component group changed while recording")
        for name, (group, paths) in code.fixed_members.items():
            declaration = self.component.declarations[name]
            member_key = json.dumps((group, paths), sort_keys=True)
            identity = self._members.get(member_key)
            if identity is None:
                identity = name
                if identity in self.resources:
                    raise TaichiCompilationError("Component fixed resource identity collision")
                self._members[member_key] = identity
                self.resources[identity] = dict(
                    kind="fixed_snode", group=group, paths=paths, dtype=declaration.schema[1][1],
                    element_shape=declaration.schema[1][2], shape=declaration.shape, aliases=[],
                    access="read_write", effects_precision="conservative",
                    initialization="empty_activation_and_zero_values")
            if name not in self.resources[identity]["aliases"]:
                self.resources[identity]["aliases"].append(name)
        for target, additions in ((self.groups, contract["groups"]), (self.dimensions, contract["dimensions"])):
            for name, schema in additions.items():
                if name in target and target[name] != schema:
                    raise TaichiCompilationError(f"Component Graph declaration {name!r} changed while recording")
                target[name] = schema
        hidden = []
        for name in code.dynamic:
            declaration = self.component.declarations[name]
            schema = declaration.schema
            if isinstance(declaration, DenseField):
                group, paths = contract["members"][name]
                member_key = json.dumps((group, paths), sort_keys=True)
                identity = self._members.get(member_key)
                if identity is None:
                    field = self.component.read(owner, name)
                    normal = getattr(field, "_logical_resource_declaration", None)
                    path = self.component._paths[name]
                    identity = _capture_id((*path[:-1], normal["identity"])) if normal else name
                    if identity in self.resources:
                        raise TaichiCompilationError(f"Component resource identity collision: {identity!r}")
                    self._members[member_key] = identity
                    self.resources[identity] = dict(
                        group=group,
                        paths=paths,
                        dtype=schema[1][1],
                        element_shape=schema[1][2],
                        shape=schema[2],
                        aliases=[],
                        access="read_write",
                        effects_precision="conservative",
                        initialization=(normal or {}).get("initialization", "consumer_explicit"),
                    )
                for alias, location in contract["members"].items():
                    if (
                        json.dumps(location, sort_keys=True) == member_key
                        and alias not in self.resources[identity]["aliases"]
                    ):
                        self.resources[identity]["aliases"].append(alias)
                symbol_name = "__component_resource_" + _digest((group, paths))
                binding = dict(kind="resource", name=identity)
                symbol = Arg(ArgKind.NDARRAY, symbol_name, declaration.dtype, ndim=len(declaration.shape))
            elif isinstance(declaration, (RuntimeValue, Dimension)):
                kind = "dimension" if isinstance(declaration, Dimension) else "value"
                symbol_name = "__component_" + kind + "_" + name
                binding = dict(kind=kind, name=name)
                dtype = declaration.annotation
                symbol = Arg(ArgKind.MATRIX if isinstance(dtype, MatrixType) else ArgKind.SCALAR, symbol_name, dtype)
                if kind == "value":
                    self.values[name] = schema
            elif isinstance(declaration, Array):
                symbol = self._array_argument(declaration, [alias for alias in code.used
                    if isinstance(self.component.declarations[alias], Array)
                    and self.component.declarations[alias].resource_id == declaration.resource_id])
                symbol_name = symbol.name
                binding = self.bindings[symbol_name]
            else:
                raise TaichiCompilationError(f"Unsupported component Graph capture {name!r}")
            if symbol_name in self.bindings and self.bindings[symbol_name] != binding:
                raise TaichiCompilationError("Component Graph binding identity changed while recording")
            self.bindings[symbol_name] = binding
            hidden.append(self.symbols.setdefault(symbol_name, symbol))
        compiled = kernel.compiled_kernels[code.key]
        arguments = flatten_args((*args, *hidden))
        entry_fact = dict(
            semantic_key=code.semantic_key, static=contract["static"], arguments=[arg.name for arg in arguments]
        )
        if contract.get("constants"):
            entry_fact["constants"] = contract["constants"]
        # This label identifies a captured source call, independent of producer
        # objects and native kernel numbering. Control construction order need
        # not be execution order; query the actual Graph IR after compilation.
        capture_label = "__component_call_" + _digest((entry_fact, label))
        self.dispatches[capture_label] = (code, entry_fact)
        builder._record_dispatch(compiled, arguments, capture_label)
        self.entries.append(entry_fact)
        self.codes.append(code)
        return builder

    def _array_argument(self, declaration, aliases):
        from taichi_forge.graph._graph import Arg, ArgKind
        from taichi_forge.lang._resource_declaration import extent_dependencies

        if declaration.shape is None or not declaration.resource_id:
            raise TaichiCompilationError("Component Graph arrays require normal ResourceScope ndarray declarations")
        identity = declaration.resource_id
        for alias in aliases:
            value = self.component.read(self.owner, alias)
            declaration.validate(value, self.component, self.owner)
            if self._array_values.setdefault(identity, value) is not value:
                raise TaichiCompilationError("Component array aliases must refer to the same allocation")

        def dimension(name):
            spec = self.component.declarations[name]
            _dimension_value(self.component, self.owner, name)
            self.dimensions[name] = json.loads(json.dumps(spec.schema))
            for dependency in extent_dependencies(spec.expression) if spec.expression else ():
                dimension(dependency)
            if spec.capacity:
                dimension(spec.capacity)

        for extent in declaration.shape:
            if isinstance(extent, str):
                dimension(extent)
        element = describe_element_type(declaration.dtype)
        layout = "__component_array_" + _digest(element.structural_key()) if element.category == "struct" else ""
        if layout:
            self.array_types[layout] = declaration.dtype
        resource = dict(kind="ndarray", shape=declaration.shape,
            dtype=str(element.scalar_type) if not layout else "struct", element_shape=element.shape, layout=layout,
            aliases=sorted(aliases), access="read_write", effects_precision="conservative",
            initialization=getattr(value, "_logical_resource_declaration", {}).get("initialization", "consumer_explicit"))
        previous = self.resources.get(identity)
        if previous is not None:
            if {k: v for k, v in previous.items() if k != "aliases"} != {k: v for k, v in resource.items() if k != "aliases"}:
                raise TaichiCompilationError(f"Incompatible component array identity {identity!r}")
            resource["aliases"] = sorted(set(previous["aliases"]) | set(resource["aliases"]))
        self.resources[identity] = resource
        symbol_name = "__component_resource_" + _digest(("ndarray", identity))
        self.bindings[symbol_name] = dict(kind="resource", name=identity)
        return self.symbols.setdefault(symbol_name, Arg(ArgKind.NDARRAY, symbol_name, declaration.dtype, ndim=declaration.ndim))

    def native_array(self, value):
        aliases = [name for name, declaration in self.component.declarations.items()
                   if isinstance(declaration, Array) and self.component.available(self.owner, name)
                   and self.component.read(self.owner, name) is value]
        if not aliases:
            raise TaichiCompilationError("Component native arrays require normal ResourceScope ndarray declarations")
        return self._array_argument(self.component.declarations[aliases[0]], aliases).name

    def array_extent(self, value):
        """Project a normal declared array for shape-only private allocation."""
        from taichi_forge.lang._ndarray import Ndarray

        aliases = [name for name, declaration in self.component.declarations.items()
                   if isinstance(declaration, (Array, DenseField))
                   and self.component.available(self.owner, name)
                   and self.component.read(self.owner, name) is value]
        if not aliases:
            raise TaichiCompilationError("ArrayExtent requires a declared component Dense Field or Ndarray")
        for alias in aliases:
            self._shape_values[alias] = (value, tuple(value.shape))
        if isinstance(value, Ndarray):
            return self.native_array(value)
        return self._retain_dense_resources((value,), host_visible=False)[0]

    def native(self, executable):
        # Concrete providers opt in; inherited hooks cannot hide extra behavior.
        convert = type(executable).__dict__.get("_capture_component")
        if convert is None:
            raise TaichiCompilationError(f"Component native capture is not qualified for {type(executable).__name__}")
        definition = convert(executable, self)
        self.natives[definition.graph_ir_node.name] = definition
        return definition

    def retain_host_resources(self, resources):
        """Keep explicitly published normal state without adding kernel inputs.

        These are normal declared Dense Fields, not producer snapshots. Layout
        and dimensions still come from the same native SNode schema projection.
        """
        self._retain_dense_resources(resources, host_visible=True)

    def _retain_dense_resources(self, resources, *, host_visible):
        from taichi_forge.graph._graph import Arg, ArgKind

        groups, symbols = {}, []
        for field in resources:
            aliases = [name for name, declaration in self.component.declarations.items()
                       if isinstance(declaration, DenseField)
                       and self.component.available(self.owner, name)
                       and self.component.read(self.owner, name) is field]
            if not aliases:
                raise TaichiCompilationError("Host-visible resources require declared Dense Fields on this component")
            declaration = self.component.declarations[aliases[0]]
            group = declaration.group
            if group not in groups:
                groups[group] = _dense_group(self.component, self.owner, group)
            schema, members, dimensions, _ = groups[group]
            if self.groups.setdefault(group, schema) != schema:
                raise TaichiCompilationError("Host-visible resource layout changed during recording")
            def dimension(name):
                from taichi_forge.lang._resource_declaration import extent_dependencies
                spec = self.component.declarations[name]
                _dimension_value(self.component, self.owner, name)
                self.dimensions[name] = json.loads(json.dumps(spec.schema))
                for dependency in extent_dependencies(spec.expression) if spec.expression else ():
                    dimension(dependency)
                if spec.capacity:
                    dimension(spec.capacity)
            for name in dimensions:
                dimension(name)
            paths = members[aliases[0]]
            key = json.dumps((group, paths), sort_keys=True)
            identity = self._members.get(key)
            if identity is None:
                name = aliases[0]
                normal = getattr(field, "_logical_resource_declaration", None)
                path = self.component._paths[name]
                identity = _capture_id((*path[:-1], normal["identity"])) if normal else name
                if identity in self.resources:
                    raise TaichiCompilationError("Host-visible resource identity collision")
                self._members[key] = identity
                self.resources[identity] = dict(group=group, paths=paths,
                    dtype=declaration.schema[1][1], element_shape=declaration.schema[1][2],
                    shape=declaration.shape, aliases=[], access="read_write",
                    effects_precision="conservative",
                    initialization=(normal or {}).get("initialization", "consumer_explicit"))
            resource = self.resources[identity]
            resource["aliases"] = sorted(set(resource["aliases"]) | set(aliases))
            if host_visible:
                self.host_resources.add(identity)
            else:
                symbol_name = "__component_resource_" + _digest((group, paths))
                self.bindings[symbol_name] = dict(kind="resource", name=identity)
                self.symbols.setdefault(symbol_name, Arg(ArgKind.NDARRAY, symbol_name,
                                                         declaration.dtype, ndim=len(declaration.shape)))
                symbols.append(symbol_name)
        return symbols

    def finish(self, builder):
        if not self.codes:
            raise TaichiCompilationError("Component roots require a nonempty captured dispatch closure")
        if self.configuration != self._configuration():
            raise TaichiCompilationError("Component configuration changed while recording its Graph; retry")
        if self._constant_snapshot != self._constant_values():
            raise TaichiCompilationError("Component definition constants changed while recording its Graph; retry")
        # Detect declaration/static mutation over the whole normal recording.
        for code in self.codes:
            if not code.matches(self.component, self.owner):
                raise TaichiCompilationError("Component dependencies changed while recording its Graph")
        for alias, (value, shape) in self._shape_values.items():
            if self.component.read(self.owner, alias) is not value or tuple(value.shape) != shape:
                raise TaichiCompilationError("Component array extent source changed while recording its Graph")
        for identity, value in self._array_values.items():
            for alias in self.resources[identity]["aliases"]:
                if self.component.read(self.owner, alias) is not value:
                    raise TaichiCompilationError("Component array binding changed while recording its Graph")
                self.component.declarations[alias].validate(value, self.component, self.owner)
        facts = dict(
            version=1,
            groups=self.groups,
            dimensions=self.dimensions,
            resources=self.resources,
            values=self.values,
            bindings=self.bindings,
            explicit={
                name: schema for name, schema in self.explicit.items() if name not in builder._declared_private_bindings
            },
            entries=self.entries,
            required_capabilities=["dense_snode_group_v1", "component_root_v1"],
        )
        internal = {
            name: schema for name, schema in self.explicit.items() if name in builder._declared_private_bindings
        }
        if internal:
            facts["internal_arguments"] = internal
        if any(descriptor[0] == "ndarray" for descriptor in facts["explicit"].values()):
            facts["required_capabilities"].append("component_explicit_ndarray_v1")
        if self.array_types or any(value.get("kind") == "ndarray" for value in self.resources.values()):
            facts["required_capabilities"].append("component_array_storage_v1")
        if self.fixed_groups:
            facts["fixed_groups"] = self.fixed_groups
        if self.host_resources:
            facts["host_resources"] = sorted(self.host_resources)
            facts["required_capabilities"].append("component_host_resources_v1")
        facts["semantic_key"] = _digest(facts)
        return facts


class _LifecycleExport:
    """Graph-node boundaries and their authoritative lowering projection."""

    def __init__(self, initialization_nodes, setup_nodes=(), operation_nodes=()):
        self.initialization_nodes = initialization_nodes
        self.setup_nodes = tuple(setup_nodes)
        self.setup_operations = ()
        self.operation_nodes = tuple(operation_nodes)
        self.operation_bindings = ()


class ComponentGraph:
    """Frozen normal Graph plus its compiler-produced resource contract."""

    def __init__(self, graph, contract, array_types=None, fixed_dependencies=None, field_io=None,
                 initialization_nodes=0, setup_nodes=(), operation_nodes=()):
        self._graph = graph
        self._contract = json.dumps(contract, sort_keys=True)
        self._array_types = dict(array_types or {})
        self._fixed_dependencies = dict(fixed_dependencies or {})
        self._field_io = dict(field_io or {})
        self._initialization_nodes = initialization_nodes
        self._setup_nodes = tuple(setup_nodes)
        self._operation_nodes = tuple(operation_nodes)

    @property
    def contract(self):
        return json.loads(self._contract)

    def export(self, module, name):
        if name in module._component_roots:
            raise ValueError(f"Duplicate component root {name!r}")
        lifecycle = _LifecycleExport(self._initialization_nodes, self._setup_nodes, self._operation_nodes)
        plan = module.add_execution_plan(name, self._graph, snapshot_state=False,
                                         _fixed_snode_dependencies=tuple(self._fixed_dependencies.values()),
                                         _component_lifecycle=lifecycle)
        contract = self.contract
        if contract["version"] >= 8:
            contract["fixed_bindings"] = {}
            contract["field_io_bindings"] = {}
        if contract["version"] >= 9:
            contract["setup_bindings"] = list(lifecycle.setup_operations)
        if contract["version"] >= 10:
            contract["operation_bindings"] = list(lifecycle.operation_bindings)
        try:
            if self._field_io:
                contract["field_io_bindings"] = {}
                for resource, operations in self._field_io.items():
                    bindings = {}
                    for operation, code in operations.items():
                        entry = "__component_field_" + _digest((name, resource, operation))
                        code.export(module, entry)
                        bindings[operation] = entry
                    contract["field_io_bindings"][resource] = bindings
            for layout in {resource.get("layout") for resource in contract["resources"].values()} - {None, ""}:
                module.add_ndarray_layout(layout, self._array_types[layout])
        except BaseException:
            module._graph_export_failed = True
            raise
        contract.update(
            name=name,
            execution=name,
            target=module._arch.name,
            execution_entries=(
                list(plan.kernel_entries)
                if contract["version"] >= 4
                else [operation["entry"] for operation in json.loads(plan.to_json())["operations"]]
            ),
        )
        if contract["version"] >= 4:
            if plan.data_entries:
                # Export already added native definitions. Preserve the
                # module's existing fail-closed publication contract.
                module._graph_export_failed = True
                raise TaichiCompilationError("Generic control components cannot capture producer state")
            facts = dict(plan=json.loads(plan.to_json()), semantic_key=contract["semantic_key"])
            if contract["version"] >= 9:
                facts["setup_bindings"] = contract["setup_bindings"]
            if contract["version"] >= 10:
                facts["operation_bindings"] = contract["operation_bindings"]
            contract["execution_fingerprint"] = _digest(facts)
        module._component_roots[name] = contract
        if self._fixed_dependencies:
            module._component_fixed_sources[name] = self._fixed_dependencies
        module._content.append("component:" + name)
        return plan


def _complete_compiler_contract(graph, contract, capture, *, lifecycle=False):
    """Project existing compiled Graph effects and final native task facts.

    This is a cold query, not another effects analyzer or launch evaluator.
    Task manifests describe compiled choices; actual replay needs observation.
    """
    from taichi_forge.graph._ir import (
        DispatchNode,
        NativeCallNode,
        GraphAccess,
        SequentialRegion,
        WhileRegion,
        _merge_parallel_access,
        graph_ir_to_dict,
    )
    from taichi_forge.lang import impl

    # Root and Sequential private declarations converge in Graph's actual
    # fixed binding contract. A nested factory need not use root declarations.
    for name in set(contract["explicit"]) & set(graph._spec.fixed_runtime_args):
        contract.setdefault("internal_arguments", {})[name] = contract["explicit"].pop(name)
    nodes = []
    native_nodes = []
    entries_in_order, codes = [], []
    has_control = False
    skipped = 0

    def dispatch_count(node):
        return 1 if isinstance(node, DispatchNode) else sum(dispatch_count(child) for child in node.children)

    def visit(node):
        nonlocal has_control, skipped
        if isinstance(node, DispatchNode):
            if node.dispatch_label not in capture.dispatches:
                raise TaichiCompilationError("Component Graph contains a dispatch outside its capture context")
            code, entry = capture.dispatches[node.dispatch_label]
            if [binding.name for binding in node.bindings] != entry["arguments"]:
                raise TaichiCompilationError("Component Graph changed captured argument connections")
            nodes.append(node)
            codes.append(code)
            entries_in_order.append(entry)
            return dict(kind="kernel", semantic_key=code.semantic_key, arguments=entry["arguments"])
        elif isinstance(node, SequentialRegion):
            return dict(kind="sequence", children=[visit(child) for child in node.children])
        elif isinstance(node, NativeCallNode):
            definition = capture.natives.get(node.name)
            if definition is None or node != definition.graph_ir_node:
                raise TaichiCompilationError("Component Graph contains an unqualified native definition")
            has_control = True
            native_nodes.append(node)
            return dict(kind="native_sort", operations=definition.operations)
        elif isinstance(node, WhileRegion):
            has_control = True
            facts = graph_ir_to_dict(node)
            children = [visit(node.condition)]
            if node.max_iterations:
                children.append(visit(node.body))
            else:
                # The ordinary zero-budget while still evaluates its condition,
                # but exports no body code. Do not demand dead captures/values.
                skipped += dispatch_count(node.body)
                children.append(dict(kind="sequence", children=[]))
            # The Graph IR owns these control policies. Display names, compiler
            # counters and derived analysis facts are not semantic identities.
            return {
                **{
                    key: facts[key]
                    for key in (
                        "kind",
                        "predicate",
                        "counter",
                        "status",
                        "max_iterations",
                        "control_inputs",
                        "carried_state",
                        "chunk_size",
                        "compound_chunk_size",
                        "vulkan_first_chunk_strategy",
                        "masked_execution",
                        "lowering_mode",
                    )
                },
                "children": children,
            }
        else:
            raise TaichiCompilationError("Component compiler facts require qualified sequential/while regions")

    control = visit(graph._spec.ir_root)
    if len(nodes) + skipped != len(capture.codes):
        raise TaichiCompilationError("Component compiled Graph changed its entry closure")
    contract["entries"] = entries_in_order
    fixed_trees = set()
    code_fixed_trees = {}
    for code in dict.fromkeys(codes):
        if code.fixed_groups:
            manifest = impl.get_runtime().prog._debug_snode_relocation_manifest(code.kernel.compiled_kernels[code.key])
            trees = {item["tree_id"] for item in manifest["tree_dependencies"]}
            code_fixed_trees[code] = trees
            fixed_trees.update(trees)
    fixed_groups = {group: schema for group, schema in capture.fixed_groups.items()
                    if capture.fixed_dependencies[group][0] in fixed_trees}
    if fixed_groups:
        contract["fixed_groups"] = fixed_groups
    else:
        contract.pop("fixed_groups", None)
    used_symbols = {symbol for entry in entries_in_order for symbol in entry["arguments"]}
    used_symbols.update(binding.name for node in native_nodes for binding in node.bindings)
    private_shapes = {
        name: requirement.deployment_shape()
        for name, requirement in graph._spec.fixed_runtime_args.items()
        if name in used_symbols and getattr(requirement, "shape_bindings", ())
    }
    if private_shapes:
        contract["private_shapes"] = private_shapes
        contract["required_capabilities"].append("component_private_array_extents_v1")
        used_symbols.update(value["array"] for shape in private_shapes.values()
                            for value in shape if isinstance(value, dict))
    for key in ("bindings", "explicit", "internal_arguments"):
        if key in contract:
            contract[key] = {name: value for name, value in contract[key].items() if name in used_symbols}
    for key, kind in (("resources", "resource"), ("values", "value")):
        required = {value["name"] for value in contract["bindings"].values() if value["kind"] == kind}
        if kind == "resource":
            required.update(capture.host_resources)
            required.update(name for name, resource in contract["resources"].items()
                            if resource.get("kind") == "fixed_snode" and resource["group"] in fixed_groups)
        contract[key] = {name: value for name, value in contract[key].items() if name in required}
    required_groups = {value["group"] for value in contract["resources"].values() if "group" in value}
    contract["groups"] = {name: value for name, value in contract["groups"].items() if name in required_groups}
    required_dimensions = {name for code in codes for name in code.contract["dimensions"]}
    def require_dimension(name):
        from taichi_forge.lang._resource_declaration import extent_dependencies
        if name in required_dimensions:
            return
        required_dimensions.add(name)
        spec = contract["dimensions"][name]
        for dependency in extent_dependencies(spec[4]) if len(spec) == 5 else ():
            require_dimension(dependency)
        if spec[3]:
            require_dimension(spec[3])
    for resource in contract["resources"].values():
        if resource.get("kind") == "ndarray":
            for extent in resource["shape"]:
                if isinstance(extent, str):
                    require_dimension(extent)
    # Layout-only siblings and Host-only groups can have independent extents.
    for schema in contract["groups"].values():
        for node in json.loads(schema)["nodes"]:
            for extent in node["extents"]:
                if extent["parameter"]:
                    require_dimension(extent["parameter"])
    contract["dimensions"] = {
        name: value for name, value in contract["dimensions"].items() if name in required_dimensions
    }
    used_declarations = {name for code in codes for name in code.used}
    used_declarations.update(alias for resource in contract["resources"].values() for alias in resource["aliases"])
    constraints, constraint_dimensions, constraint_sources = capture.component.constraint_facts(
        capture.owner, used_declarations, entry=capture._root_name
    )
    contract["dimensions"].update(constraint_dimensions)
    if not any(descriptor[0] == "ndarray" for descriptor in contract["explicit"].values()):
        contract["required_capabilities"] = [
            name for name in contract["required_capabilities"] if name != "component_explicit_ndarray_v1"
        ]
    if not any(value.get("kind") == "ndarray" for value in contract["resources"].values()):
        contract["required_capabilities"] = [name for name in contract["required_capabilities"]
                                             if name != "component_array_storage_v1"]
    if any(value["initialization"] == "zero" for value in contract["resources"].values()):
        contract["required_capabilities"].append("component_zero_initial_state_v1")
    accesses, uncertain, entries, task_cache = {}, set(), [], {}
    for node, code, entry in zip(nodes, codes, contract["entries"]):
        effects = []
        for effect in node.effects:
            if not effect.runtime_bound and effect.resource.startswith("snode:"):
                tree = int(effect.resource.split(":")[1])
                group = next((group for group in fixed_groups if capture.fixed_dependencies[group][0] == tree), None)
                if group is not None:
                    effects.append(dict(group=group, access=effect.access.value))
                    # Activation/listgen can affect every member of a group.
                    # Preserve the compiler access fact at group granularity.
                    for identity, resource in contract["resources"].items():
                        if resource.get("kind") == "fixed_snode" and resource["group"] == group:
                            previous = accesses.get(identity)
                            accesses[identity] = effect.access if previous is None else _merge_parallel_access(previous, effect.access)
                            if node.opaque:
                                uncertain.add(identity)
                    continue
            binding = contract["bindings"].get(effect.resource)
            explicit = contract["explicit"].get(effect.resource) or contract.get("internal_arguments", {}).get(
                effect.resource
            )
            if explicit is not None and effect.runtime_bound:
                if explicit[0] == "ndarray":
                    effects.append(dict(argument=effect.resource, access=effect.access.value))
                continue
            # The opaque Graph fallback includes scalar/aggregate bindings too.
            if node.opaque and (
                effect.resource in contract["explicit"] or binding is not None and binding["kind"] != "resource"
            ):
                continue
            if binding is None or binding["kind"] != "resource" or not effect.runtime_bound:
                raise TaichiCompilationError(f"Component Graph has an undeclared resource effect: {effect}")
            identity = binding["name"]
            previous = accesses.get(identity)
            accesses[identity] = effect.access if previous is None else _merge_parallel_access(previous, effect.access)
            effects.append(dict(resource=identity, access=effect.access.value))
        # Sparse lowering can conservatively classify allocator/list state
        # without publishing per-member Graph effects. The compiler's actual
        # tree closure still requires an opaque group effect, never 'none'.
        for group in fixed_groups:
            if capture.fixed_dependencies[group][0] not in code_fixed_trees.get(code, ()):
                continue
            if not any(effect.get("group") == group for effect in effects):
                effects.append(dict(group=group, access="opaque"))
            uncertain.update(identity for identity, resource in contract["resources"].items()
                             if resource.get("kind") == "fixed_snode" and resource["group"] == group)
        if node.opaque:
            uncertain.update(
                contract["bindings"][symbol]["name"]
                for symbol in entry["arguments"]
                if symbol in contract["bindings"] and contract["bindings"][symbol]["kind"] == "resource"
            )
        if code.semantic_key not in task_cache:
            task_cache[code.semantic_key] = impl.get_runtime().prog._kernel_task_manifest(
                code.kernel.compiled_kernels[code.key]
            )
        entries.append(
            dict(
                effects=effects,
                opaque=node.opaque,
                side_effects=list(node.side_effects),
                iteration_domain=node.iteration_domain,
                fusion_blocker=node.fusion_blocker,
                tasks=task_cache[code.semantic_key],
            )
        )
        if code.objects:
            entries[-1]["record_presence"] = dict(code.contract["objects"])
            entries[-1]["capture_paths"] = code.contract["access_paths"]
    for node in native_nodes:
        for effect in node.effects:
            binding = contract["bindings"].get(effect.resource)
            if binding is None or binding["kind"] != "resource":
                raise TaichiCompilationError("Component native effect has no declared resource")
            name = binding["name"]
            previous = accesses.get(name)
            accesses[name] = effect.access if previous is None else _merge_parallel_access(previous, effect.access)
    for name, resource in contract["resources"].items():
        if name not in uncertain and accesses.get(name) != GraphAccess.OPAQUE:
            access = accesses.get(name)
            resource["access"] = access.value if access is not None else "none"
            resource["effects_precision"] = "compiler_proven"
    contract["version"] = 2
    contract["required_capabilities"].append("component_compiler_facts_v1")
    if native_nodes:
        contract["required_capabilities"].append("component_native_sort_v1")
    if has_control or "internal_arguments" in contract or constraints or fixed_groups or lifecycle:
        contract["version"] = 5 if constraints else 4
        contract["control"] = control
        contract.setdefault("internal_arguments", {})
        contract["configuration"] = {}
        contract["required_capabilities"].append("component_control_v1")
    if constraints:
        contract["constraints"] = constraints
        contract["required_capabilities"].append("component_dimension_constraints_v1")
    if fixed_groups:
        contract["version"] = 6
        contract.setdefault("constraints", {})
        contract["required_capabilities"].append("component_fixed_snode_groups_v1")
    contract["semantic_key"] = _digest({name: value for name, value in contract.items() if name != "semantic_key"})
    contract["compilation"] = dict(version=1, entries=entries, task_scope="compiled_not_observed_replay")
    if constraints:
        contract["compilation"]["constraint_sources"] = constraint_sources
    return contract


def compile_component_graph(component, record, *args, initializer=None, setup=None, operations=None,
                            host_resources=(), _build=None, **kwargs):
    from taichi_forge.graph import GraphBuilder

    capture = _GraphCapture(component, record, args, kwargs, _build)
    builder = GraphBuilder(_capture_recipe_sources=False, _component_capture=capture)
    initialization_nodes = ()
    setup_nodes = []
    operation_nodes = []
    if setup is not None:
        from collections.abc import Mapping

        if initializer is None or not isinstance(setup, Mapping) or not setup:
            raise ValueError("Component setup requires named normal recordings and a final initializer")
        for name, entry in setup.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("Component setup names must be identifiers")
            if not callable(entry) or getattr(entry, "__self__", None) is not capture.owner:
                raise TaichiCompilationError("Component setup requires normal recordings with the same owner")
            entry(builder)
            builder._flush_graph_builder()
            end = len(builder._nodes)
            if end <= (setup_nodes[-1][1] if setup_nodes else 0):
                raise TaichiCompilationError("Component setup must record a nonempty normal Graph")
            setup_nodes.append((name, end))
    if initializer is not None:
        if not callable(initializer) or getattr(initializer, "__self__", None) is not capture.owner:
            raise TaichiCompilationError("Component initializer requires a normal recording with the same owner")
        initializer(builder)
        # Flush through the ordinary Graph compiler. This keeps the lifecycle
        # boundary outside any CGraph fusion/flattening; operation counts come
        # from the existing lowering, not the number of captured kernels.
        builder._flush_graph_builder()
        initialization_nodes = tuple(builder._nodes)
        if len(initialization_nodes) <= (setup_nodes[-1][1] if setup_nodes else 0):
            raise TaichiCompilationError("Component initializer must record a nonempty normal Graph")
    if operations is not None:
        from collections.abc import Mapping

        if initializer is None or not isinstance(operations, Mapping) or not operations:
            raise ValueError("Component operations require named normal recordings and an initializer")
        for name, entry in operations.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("Component operation names must be identifiers")
            if not callable(entry) or getattr(entry, "__self__", None) is not capture.owner:
                raise TaichiCompilationError("Component operations require normal recordings with the same owner")
            entry(builder)
            builder._flush_graph_builder()
            end = len(builder._nodes)
            if end <= (operation_nodes[-1][1] if operation_nodes else len(initialization_nodes)):
                raise TaichiCompilationError("Component operations must record nonempty normal Graphs")
            operation_nodes.append((name, end))
    operation_prefix = tuple(builder._nodes) if operation_nodes else ()
    record(builder, *args, **kwargs)
    capture.retain_host_resources(host_resources)
    contract = capture.finish(builder)
    if _build is not None:
        _build.check_time("before backend compilation")
    graph = builder.compile()
    try:
        if initialization_nodes and (
            tuple(graph._spec.nodes[:len(initialization_nodes)]) != initialization_nodes
            or len(graph._spec.nodes) == len(initialization_nodes)
        ):
            raise TaichiCompilationError("Component lifecycle requires a preserved initialization/replay boundary")
        if operation_nodes and (len(graph._spec.nodes) <= operation_nodes[-1][1]
                or tuple(graph._spec.nodes[:len(operation_prefix)]) != operation_prefix):
            raise TaichiCompilationError("Component operations require preserved boundaries and nonempty default replay")
        contract = _complete_compiler_contract(graph, contract, capture, lifecycle=bool(initialization_nodes))
        # Host choices remain part of admission even if no numerical entry uses
        # them. Explicit kernel-only policy follows the final compiler closure,
        # including dead-region pruning, rather than the whole owner registry.
        used_static = {name for entry in contract["entries"] for name, _ in entry["static"]}
        configuration = {
            name: spec
            for name, spec in capture.configuration.items()
            if not capture._choices[name].kernel_only or name in used_static
        }
        if configuration:
            # Admission describes the whole normal root, including Python choices
            # which may alter its dispatch sequence. It never widens a kernel ABI
            # or its actual static dependency key. Same code may admit several
            # configurations; this fixed root exports the selected one explicitly.
            contract["version"] = max(contract["version"], 3)
            contract["required_capabilities"].append("component_static_choices_v1")
            contract["configuration"] = configuration
        # Public provenance for normal Host model preparation. Attribute paths
        # come from declarations, never from decoding a generated capture ID.
        # Recording options belong to the root call, not to the model owner.
        contract["compilation"]["configuration_sources"] = {
            name: dict(
                kind="recording_argument" if isinstance(capture._choices[name], RootOption) else "attribute",
                path=list(component._paths[name]),
            )
            for name in configuration
        }
        if _build is not None:
            _build.check_time("after backend compilation")
        field_io = _compile_fixed_field_io(component, capture.owner, contract, _build)
        contract["compilation"]["input_sources"] = _input_sources(component, contract)
        if initialization_nodes:
            # Version 8 adds lifecycle to the same compiler projection. Empty
            # optional resource families stay empty, rather than acquiring a
            # synthetic fixed tree just to use an initializer.
            contract["version"] = 8
            for key in ("configuration", "constraints", "fixed_groups", "field_io", "internal_arguments"):
                contract.setdefault(key, {})
            if "component_control_v1" not in contract["required_capabilities"]:
                contract["required_capabilities"].append("component_control_v1")
            contract["lifecycle"] = dict(initialization_nodes=len(initialization_nodes),
                                         initialization="explicit_instance", reprepare="preserve_state")
            contract["required_capabilities"].append("component_instance_initializer_v1")
            if setup_nodes:
                contract["version"] = 9
                contract["lifecycle"]["setup_nodes"] = setup_nodes
                contract["required_capabilities"].append("component_cold_setup_v1")
            if operation_nodes:
                contract["version"] = 10
                contract["lifecycle"].setdefault("setup_nodes", [])
                contract["lifecycle"]["operation_nodes"] = operation_nodes
                contract["required_capabilities"].append("component_named_operations_v1")
        if contract["version"] >= 4:
            from taichi_forge_runtime.aot._component import _control_identity

            contract["semantic_key"] = _control_identity(contract)
        fixed = {group: capture.fixed_dependencies[group] for group in contract.get("fixed_groups", {})}
        return ComponentGraph(graph, contract, capture.array_types, fixed, field_io,
                              len(initialization_nodes), setup_nodes, operation_nodes)
    except BaseException:
        graph.close()
        raise


def _input_sources(component, contract):
    """Project normal declaration paths after pruning the compiled closure."""
    # Fixed Field I/O additionally exposes readable dotted aliases. They were
    # created from these paths, not by decoding compiler-generated identities.
    readable = {".".join(path): path for path in component._paths.values()}

    def source(name, *, fixed=False):
        allocation = component._source_scope_paths.get(name)
        if allocation is not None:
            return dict(kind="attribute", path=list(allocation))
        borrowed = component._borrowed_reads.get(name)
        if borrowed is not None:
            anchor, anchor_member, member, _ = borrowed
            if member == anchor_member:
                return dict(kind="attribute", path=list(anchor))
            return dict(kind="borrowed_group_member", path=list(anchor), member=member)
        path = component._paths.get(name)
        if path is None and fixed:
            path = readable.get(name)
        if path is None:
            raise TaichiCompilationError(f"Missing component declaration source for {name!r}")
        return dict(kind="attribute", path=list(path))

    return dict(
        # Derived extents are evaluated from their authoritative expressions.
        # A synthetic extent identity is not an attribute on a model object.
        dimensions={name: source(name) for name, schema in contract["dimensions"].items() if len(schema) == 4},
        values={name: source(name) for name in contract["values"]},
        resources={
            name: {alias: source(alias, fixed=resource.get("kind") == "fixed_snode")
                   for alias in resource["aliases"]}
            for name, resource in contract["resources"].items()
        },
    )


def _compile_fixed_field_io(component, owner, contract, build=None):
    """Compile ordinary Field coordinate transfers for the actual closure.

    These are library data transfers, shared with normal Field.read_at/write_at;
    application numerical functions and signatures are not wrapped or rewritten.
    """
    from taichi_forge._kernels import field_read_coordinates, field_write_coordinates
    from taichi_forge.lang import impl
    from taichi_forge.lang._field_coordinates import element_type

    result = {}
    for name, resource in contract["resources"].items():
        if resource.get("kind") != "fixed_snode":
            continue
        field = component.read(owner, resource["aliases"][0])
        # Keep declaration IDs for cache/ABI identity and expose their original
        # member paths for public consumption, without decoding internal names.
        resource["aliases"] = list(dict.fromkeys([
            *resource["aliases"],
            *(".".join(component._paths[alias]) for alias in resource["aliases"] if alias in component._paths),
        ]))
        indices = impl.ndarray(impl.i32, shape=(1, max(len(field.shape), 1)))
        values = impl.ndarray(element_type(field), shape=1)
        operations = {}
        for operation, entry in (("read", field_read_coordinates), ("write", field_write_coordinates)):
            code = component.compile(entry, indices, values, owner=owner, template_args={"field": field})
            if code.dynamic or set(code.fixed_groups) != {resource["group"]}:
                raise TaichiCompilationError("Field transfer captured unexpected resources or dynamic state")
            if build is not None:
                build.admit_kernel(code)
            operations[operation] = code
        result[name] = operations
    if result:
        contract["version"] = 7
        contract["field_io"] = {
            name: dict(entries={operation: code.semantic_key for operation, code in operations.items()},
                       offsets=list(component.read(owner, contract["resources"][name]["aliases"][0]).snode.ptr.offset)
                       or [0] * len(contract["resources"][name]["shape"]))
            for name, operations in result.items()
        }
        contract["required_capabilities"].append("component_field_coordinates_v1")
    return result
