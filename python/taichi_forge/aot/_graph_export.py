"""AOT publication and snapshot transaction over shared frozen Graph lowering."""

import hashlib

from taichi_forge._lib import core
from taichi_forge.graph._lowering import GraphLoweringContext, _snapshot, lower_frozen_graph
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge_runtime.aot import ExecutionPlan


class _ExportContext(GraphLoweringContext):
    """Deployment adapter; actual compiled closure selects snapshots at finalization."""

    def __init__(
        self,
        module,
        reserved_names,
        *,
        snapshot_state=False,
        source_arrays=None,
        source_resources=None,
        resource_inputs=None,
        fixed_snode_dependencies=(),
    ):
        super().__init__(
            module._arch,
            set(reserved_names) | set(module._exported_kernels) | set(module._fields),
            source_arrays=source_arrays,
            source_resources=source_resources,
            resource_inputs=resource_inputs,
        )
        self.snapshot_state = snapshot_state
        self.fixed_snode_dependencies = {item[0]: tuple(item) for item in fixed_snode_dependencies}
        if self.fixed_snode_dependencies and snapshot_state:
            raise TaichiRuntimeError("Generic fixed SNode groups cannot snapshot producer state")
        self.kernels = {}
        self.kernel_launch_abis = {}
        self._kernel_names = {
            (id(kernel), module._kernel_launch_abis.get(name, "ordinary")): name
            for name, kernel in module._exported_kernels.items()
        }
        self.data_payloads = {}
        self._data_keys = {}
        self._readonly_data_owners = {}
        self._provider_data = {}
        self.root_dependencies = ()
        self.root_payloads = ()
        from ._graph_storage_bindings import RetainedArrayBindings

        self.retained_arrays = RetainedArrayBindings(module, self)

    def retained_kernel(self, kernel, arguments, symbols, label, selected):
        specialization = kernel.ensure_compiled(*arguments, _for_aot=True)
        self.kernel(kernel.compiled_kernels[specialization], symbols, label)

    def require_frozen_host_state(self):
        if not self.snapshot_state:
            raise TaichiRuntimeError("AOT raster camera/light state requires explicit snapshot_state=True at export")

    def _capture_private_array(self, owner):
        from taichi_forge.lang._ndarray import StructNdarray
        from ._graph_storage_bindings import _SOURCE_NDARRAY_TYPES

        return self.snapshot_state and type(owner) in (*_SOURCE_NDARRAY_TYPES, StructNdarray)

    def data_resource(
        self, owner, data, *, shape, dtype=None, element_shape=(), layout=None, name=None, readonly=False, usage=None
    ):
        """Stage already frozen data; never sample a live owner during export."""
        from taichi_forge.graph._lowering_array_data import array_resource_fields

        fields = array_resource_fields(shape=shape, dtype=dtype, element_shape=element_shape, layout=layout)
        if type(data) is not bytes:
            raise TypeError("AOT resource data must be an immutable byte snapshot")
        previous = self._data_keys.get(id(data))
        if previous is None:
            key = hashlib.sha256(data).hexdigest()
            if key in self.data_payloads and self.data_payloads[key] != data:
                raise ValueError("AOT data digest collision")
            self.data_payloads[key] = data
            self._data_keys[id(data)] = (data, key)
        else:
            key = previous[1]
        if readonly:
            # Qualified exporters must never pass shared data to a writable
            # operation or expose it as mutable state.
            if name is not None:
                raise ValueError("Shared read-only AOT data cannot reserve a private binding name")
            identity = key, dtype, layout, tuple(shape), tuple(element_shape), tuple(sorted(usage or ()))
            owner = self._readonly_data_owners.setdefault(identity, owner)
        return self._resource(
            owner,
            dict(
                kind="ndarray",
                **fields,
                initial_data=key,
                **({"usage": sorted(usage)} if usage is not None else {}),
            ),
            name=name,
        )

    def host_blob_resource(self, data):
        """Retain qualified immutable provider code/data without device storage."""
        if type(data) is not bytes:
            raise TypeError("AOT host blobs must be immutable bytes")
        key = hashlib.sha256(data).hexdigest()
        previous = self.data_payloads.setdefault(key, data)
        if previous != data:
            raise ValueError("AOT data digest collision")
        owner = self._readonly_data_owners.setdefault(("host_blob", key), object())
        return self._resource(owner, dict(kind="host_blob", data=key, byte_size=len(data)))

    def provider_data_binding(self, name, owner, *, mutable=False):
        """Declare qualified provider storage; sample the complete closure once."""
        from taichi_forge.graph._native import ProviderOwnedNdarrayBinding
        from taichi_forge.lang import impl
        from taichi_forge.graph._lowering_array_data import ArrayDataLayout, ArraySnapshot

        if mutable and not self.snapshot_state:
            raise TaichiRuntimeError("AOT provider workspace requires explicit snapshot_state=True at export")
        if type(owner) is not ProviderOwnedNdarrayBinding or owner._runtime_prog is not impl.get_runtime().prog:
            raise TaichiRuntimeError("AOT provider data requires a live qualified generation binding")
        identity = owner._runtime_allocation_identity
        cached = self._provider_data.get(identity)
        array = owner.arr
        dtype, shape = str(array.element_data_type()), tuple(array.shape)
        if tuple(array.total_shape()) != shape:
            raise TaichiRuntimeError("AOT provider data currently requires scalar arrays")
        layout = ArrayDataLayout(shape, core.data_type_size(array.element_data_type()), dtype=dtype)
        if cached is None:
            resource = self._name("resource")
            self._provider_data[identity] = ArraySnapshot(owner, resource, layout)
        else:
            resource = cached.name
            if layout != cached.layout:
                raise TaichiRuntimeError("AOT provider allocation changed its data description")
        if name in self.private_bindings and self.private_bindings[name] is not owner:
            raise TaichiRuntimeError("AOT provider data binding changed within one export")
        self.private_bindings[name] = owner
        self._private_names[name] = resource

    def field_state(self, dependencies):
        """Retain source lifetime proofs; the compiled AOT closure selects bytes."""
        for dependency in dependencies:
            if not self.snapshot_state and self.fixed_snode_dependencies.get(dependency[0]) != tuple(dependency):
                raise TaichiRuntimeError("AOT live Field state requires explicit snapshot_state=True at export")
            dependency = tuple(dependency)
            previous = self._state_dependencies.setdefault(dependency[0], dependency)
            if previous != dependency:
                raise TaichiRuntimeError("AOT Graph has conflicting Field lifetime dependencies")

    def materialize_provider_data(self, module):
        """Sample compiled roots, provider arrays and immutable recipe data once."""
        import numpy as np

        from taichi_forge.lang import impl

        pending = (*self._provider_data.values(), *self.retained_arrays.snapshots.values())
        arrays = [item.owner.arr for item in pending]
        program = impl.get_runtime().prog
        ids = sorted(
            set(module._aot_builder._used_snode_tree_ids(sorted(self._used_kernel_entries)))
            | self.retained_arrays.tree_ids
        )
        dependencies = tuple(program._snapshot_snode_tree_dependencies(ids))
        if self.fixed_snode_dependencies:
            if self.retained_arrays.tree_ids & self.fixed_snode_dependencies.keys():
                raise TaichiRuntimeError("Fixed instance trees cannot be borrowed as module-root arrays")
            if any(self.fixed_snode_dependencies.get(item[0]) != tuple(item) for item in dependencies):
                raise TaichiRuntimeError("Compiled fixed SNode closure differs from its declared component lifetime")
            # The actual compiled closure is validated, but no producer bytes
            # or active sets are sampled. Runtime instances start empty.
            dependencies = ()
        if dependencies:
            if not self.snapshot_state:
                raise TaichiRuntimeError("AOT compiled Field state requires explicit snapshot_state=True at export")
            for dependency in dependencies:
                expected = self._state_dependencies.get(dependency[0])
                if expected is not None and expected != tuple(dependency):
                    raise TaichiRuntimeError("AOT compiled Field lifetime changed during export")
            snapshot = program._snapshot_dense_state(dependencies, arrays)
            self.root_dependencies, self.root_payloads = dependencies, snapshot["roots"]
            for item, data in zip(pending, snapshot["arrays"], strict=True):
                self.data_resource(item.owner, data, **item.layout.resource_fields(), name=item.name)
            return
        if not pending:
            return
        # Snapshot whole storage records, including structure padding. The
        # existing Program transfer checks exact byte counts and one completion
        # boundary for the complete batch; no per-member packing is needed.
        hosts = [np.empty(item.layout.byte_size, dtype=np.uint8) for item in pending]
        program.copy_ndarrays_to_host(arrays, hosts, snapshot=True)
        for item, host in zip(pending, hosts, strict=True):
            self.data_resource(item.owner, host.tobytes(), **item.layout.resource_fields(), name=item.name)

    def _kernel_entry(self, kernel, launch_abi, jit_source):
        identity = (id(kernel), launch_abi)
        name = self._kernel_names.get(identity)
        if name is None:
            name = self._name("kernel" if launch_abi == "ordinary" else "gated_kernel")
            self._kernel_names[identity] = name
            self.kernels[name] = kernel
            if launch_abi != "ordinary":
                self.kernel_launch_abis[name] = launch_abi
        return name


def export_graph(
    module,
    graph,
    *,
    gradients=None,
    snapshot_state=False,
    source_bindings=None,
    resource_bindings=None,
    argument_resources=None,
    fixed_snode_dependencies=(),
    component_initialization_nodes=0,
):
    """Preflight the entire region before publishing any module definitions."""
    with graph._lifecycle_lock:
        graph._check_runtime_valid()
        if core.arch_name(module._arch) != graph._execution_arch:
            raise TaichiRuntimeError("AOT execution Graph and module backends must match")
        if graph._workspace_lane_capacity != 1 or graph._qualified_fusion_selector is not None:
            raise TaichiRuntimeError("AOT execution Graph needs a fixed single-frame execution recipe")
        spec = graph._spec
        binding_factory = getattr(spec, "_binding_executor_factory", None)
        binding_strategy = None
        if binding_factory is not None:
            # Only a concrete executor can qualify equivalence with the fixed
            # AOT submission path. Inherited hooks could lose queue/segment work.
            if getattr(binding_factory, "__dict__", {}).get("_validate_aot_export") is None:
                raise TaichiRuntimeError("AOT execution Graph requires the selected storage/binding recipe")
            binding_strategy = binding_factory._validate_aot_export(spec, graph._execution_arch)
        gradient_names = set((gradients or {}).values())
        from ._graph_source_bindings import SourceArrayBindings
        from taichi_forge.graph._lowering_resource_sources import SourceResourceBindings
        from taichi_forge.graph._lowering_resource_bindings import ResourceInputBindings

        source_resources = SourceResourceBindings(source_bindings, spec.fixed_runtime_args)
        source_arrays = SourceArrayBindings(source_resources.arrays, spec.fixed_runtime_args, module)
        resource_inputs = ResourceInputBindings(resource_bindings, spec.fixed_runtime_args)
        non_resource_names = (
            (set(spec.runtime_arg_names) - source_resources.owners.keys())
            | source_arrays.arrays.keys()
            | gradient_names
        )
        if resource_inputs.public_names & non_resource_names:
            raise ValueError("Public resource names conflict with Graph array/scalar/gradient arguments")
        context = _ExportContext(
            module,
            set(spec.runtime_arg_names)
            | set(spec.fixed_runtime_args)
            | gradient_names
            | source_arrays.arrays.keys()
            | source_resources.owners.keys()
            | resource_inputs.names,
            snapshot_state=snapshot_state,
            source_arrays=source_arrays,
            source_resources=source_resources,
            resource_inputs=resource_inputs,
            fixed_snode_dependencies=fixed_snode_dependencies,
        )
        operation_ends = lower_frozen_graph(context, graph)
        if component_initialization_nodes:
            if (type(component_initialization_nodes) is not int
                    or not 0 < component_initialization_nodes < len(operation_ends)):
                raise TaichiRuntimeError("Component initialization must precede a nonempty replay region")
            if (context.initialization or binding_strategy is not None
                    or context.storage_recipe is not None or gradients or argument_resources):
                raise TaichiRuntimeError(
                    "Component instance initializers with Plan-private resources/recipes require shared-lifetime qualification")
            boundary = operation_ends[component_initialization_nodes - 1]
            context.initialization = context.operations[:boundary]
            context.operations = context.operations[boundary:]
        submission_recipe = None
        if getattr(binding_factory, "__dict__", {}).get("_export_aot_submission") is not None:
            submission_recipe = binding_factory._export_aot_submission(spec, operation_ends)
        mapped_gradients = {}
        for primal, gradient in (gradients or {}).items():
            if primal in context._scalar_values or gradient in context._scalar_values:
                raise TaichiRuntimeError("AOT gradient bindings require ndarrays")
            primal, gradient = context.binding(primal), context.binding(gradient)
            if primal in mapped_gradients and mapped_gradients[primal] != gradient:
                raise TaichiRuntimeError("AOT aliases of one primal must share the same gradient resource")
            mapped_gradients[primal] = gradient
        from taichi_forge.graph._lowering_constraints import export_constraints

        constraints = export_constraints(spec, context.binding, additional=context.binding_constraints)
        if component_initialization_nodes:
            from taichi_forge_runtime.aot._component_lifecycle import split_component_lifecycle

            try:
                split_component_lifecycle(ExecutionPlan(
                    context.operations, initialization=context.initialization,
                    resources=context.resources, constraints=constraints or None))
            except ValueError as error:
                raise TaichiRuntimeError(str(error)) from error
        if context.storage_recipe is not None:
            if not context._buffer_only_kernels:
                raise TaichiRuntimeError("AOT storage rings require ordinary buffer dispatches")
            # Validate aliases, AD and the entire selected schedule before any
            # code is published; the arena proof alone cannot cover these.
            ExecutionPlan(
                context.operations,
                initialization=context.initialization or None,
                resources=context.resources,
                gradients=mapped_gradients or None,
                constraints=constraints or None,
                binding_strategy=binding_strategy,
                submission_recipe=submission_recipe,
                storage_recipe=context.storage_recipe,
            )
        if argument_resources is not None:
            argument_resources = _snapshot(list(argument_resources))
            # Validate the full binding policy before publishing kernels, roots
            # or views. Pending provider data already has its resource identity;
            # its bytes are sampled at the existing completion boundary below.
            policy_plan = ExecutionPlan(
                context.operations,
                initialization=context.initialization or None,
                resources=context.resources,
                gradients=mapped_gradients or None,
                constraints=constraints or None,
                binding_strategy=binding_strategy,
                submission_recipe=submission_recipe,
                storage_recipe=context.storage_recipe,
            ).with_argument_resources(argument_resources)
            missing_data = set(policy_plan.data_entries) - context.data_payloads.keys() - module._data_payloads.keys()
            if missing_data:
                raise ValueError("Argument resources reference unexported data: " + ", ".join(sorted(missing_data)))
        try:
            source_arrays.publish()
            context.retained_arrays.publish()
            for name, kernel in context.kernels.items():
                if name in context.kernel_launch_abis:
                    module._aot_builder._add_with_launch_abi(name, kernel, context.kernel_launch_abis[name])
                else:
                    module._aot_builder.add(name, kernel)
            # A different AOT compile request can retain a different Field
            # closure than the selected JIT graph. Inspect the actual exports,
            # then sample roots/provider arrays at one completion boundary.
            context.materialize_provider_data(module)
            plan = ExecutionPlan(
                context.operations,
                initialization=context.initialization or None,
                resources=context.resources,
                gradients=mapped_gradients or None,
                constraints=constraints or None,
                binding_strategy=binding_strategy,
                submission_recipe=submission_recipe,
                storage_recipe=context.storage_recipe,
            )
            # Provider workspace must not accidentally become a public input.
            expected_names = (
                (set(spec.runtime_arg_names) - source_resources.used)
                | source_arrays.used
                | resource_inputs.used_names
                | (gradient_names - context.private_bindings.keys())
            )
            if set(plan.argument_names) != expected_names:
                raise TaichiRuntimeError("AOT Graph reconstruction changed the public binding contract")
            if argument_resources is not None:
                plan = plan.with_argument_resources(argument_resources)
            for key in plan.data_entries:
                if key not in context.data_payloads:
                    if key not in module._data_payloads:
                        raise TaichiRuntimeError("AOT Graph reconstruction omitted an initial data payload")
                    continue
                if key in module._data_payloads and module._data_payloads[key] != context.data_payloads[key]:
                    raise ValueError("AOT data digest collision")
            module._add_initial_root_snapshots(context.root_dependencies, context.root_payloads)
        except Exception:
            # Builders cannot roll back compiled entries. Finalization failures
            # also forbid publication, even if earlier kernels compiled.
            module._graph_export_failed = True
            raise
        module._exported_kernels.update(context.kernels)
        module._kernel_launch_abis.update(context.kernel_launch_abis)
        module._content.extend("kernel:" + name for name in context.kernels)
        module._data_payloads.update(
            {key: context.data_payloads[key] for key in plan.data_entries if key in context.data_payloads}
        )
        return plan
