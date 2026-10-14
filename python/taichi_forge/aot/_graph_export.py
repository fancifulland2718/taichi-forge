"""Cold Graph lowering into the shared, pointer-free execution description."""

import ctypes
import hashlib
import json
import struct

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge_runtime.aot import ExecutionPlan, Scalar


def _snapshot(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


class _ExportContext:
    """One export transaction; object identities never become artifact data."""

    def __init__(self, module, reserved_names, *, snapshot_state=False):
        self.snapshot_state = snapshot_state
        self.operations = []
        self.initialization = []
        self._initialization_resources = {}
        self.data_payloads = {}
        self._data_keys = {}
        self._readonly_data_owners = {}
        self.resources = []
        self.kernels = {}
        self._resource_owners = {}
        self._indices = {}
        self._names = set(reserved_names) | set(module._exported_kernels)
        self._kernel_names = {id(kernel): name for name, kernel in module._exported_kernels.items()}
        self.reconstructed_binding_validators = set()
        self.reconstructed_argument_binders = set()
        self.reconstructed_derived_bindings = set()
        self.binding_constraints = []
        self._provider_data = {}
        self.private_bindings = {}
        self._private_names = {}
        self._scalar_values = {}
        self._scalar_symbols = {}
        self._scalar_resources = {}

    def _name(self, kind):
        index = self._indices.get(kind, 0)
        while (name := f"__forge_aot_{kind}_{index}") in self._names:
            index += 1
        self._names.add(name)
        self._indices[kind] = index + 1
        return name

    def resource(self, owner, provider, operation, *, parameters, bindings=None, version=1):
        return self._resource(
            owner,
            dict(
                kind="native",
                provider=provider,
                operation=operation,
                version=version,
                parameters=parameters,
                bindings={port: self.binding(slot) for port, slot in (bindings or {}).items()},
            ),
        )

    def resource_once(self, owner, build):
        """Sample a qualified live resource once within this export transaction."""
        previous = self._resource_owners.get(id(owner))
        if previous is not None:
            return previous[1]
        name = build()
        created = self._resource_owners.get(id(owner))
        if created is None or created[1] != name:
            raise ValueError("AOT resource factory did not register its owner")
        return name

    def _resource(self, owner, description, *, name=None):
        """Share only an identical producer resource, with an identical description."""
        if owner is None:
            raise ValueError("AOT reconstruction requires a resource owner identity")
        description = _snapshot(description)
        previous = self._resource_owners.get(id(owner))
        if previous is not None:
            _, name, frozen = previous
            if frozen != description:
                raise ValueError("AOT resource reconstruction changed within one export")
            return name
        name = name or self._name("resource")
        self._resource_owners[id(owner)] = (owner, name, description)
        self.resources.append(dict(description, name=name))
        return name

    def declare_private_bindings(self, bindings):
        from taichi_forge.graph._ir import InternalNdarrayRequirement
        from taichi_forge.lang.util import cook_dtype

        for name, owner in bindings.items():
            if type(owner) in (bool, int, float):
                # The final spec value is the shared binding frame. Source
                # nodes may retain equal but nonidentical earlier values.
                self.private_bindings[name] = self._scalar_values[name] = owner
                continue
            if not isinstance(owner, InternalNdarrayRequirement):
                continue
            dtype = cook_dtype(owner.dtype)
            element_type = str(dtype.element_type())
            if element_type not in {"f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"}:
                raise TaichiRuntimeError("AOT private ndarray requires a supported scalar element type")
            if core.data_type_size(dtype) != owner.element_bytes:
                raise TaichiRuntimeError("AOT private ndarray element size differs from its declaration")
            self._private_names[name] = self._resource(
                owner,
                dict(
                    kind="ndarray",
                    dtype=element_type,
                    shape=owner.shape,
                    element_shape=dtype.shape(),
                    zero_initialize=True,
                ),
            )
            self.private_bindings[name] = owner

    def data_resource(self, owner, data, *, dtype, shape, element_shape=(), name=None, readonly=False):
        """Stage already frozen data; never sample a live owner during export."""
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
            identity = key, dtype, tuple(shape), tuple(element_shape)
            owner = self._readonly_data_owners.setdefault(identity, owner)
        return self._resource(
            owner,
            dict(kind="ndarray", dtype=dtype, shape=shape, element_shape=element_shape, initial_data=key),
            name=name,
        )

    def binding(self, name):
        if name in self._scalar_values:
            raise TaichiRuntimeError("AOT fixed scalar requires a typed kernel argument")
        return self._private_names.get(name, name)

    def view_binding(self, name, parent, offset, shape, *, owner):
        if name in self.private_bindings and self.private_bindings[name] is not owner:
            raise TaichiRuntimeError("AOT derived view conflicts with an existing private binding")
        self._private_names[name] = self._resource(
            owner, dict(kind="ndarray_view", array=self.binding(parent), offset=offset, shape=shape)
        )
        self.private_bindings[name] = owner
        self.reconstructed_derived_bindings.add(name)

    def provider_data_binding(self, name, owner, *, mutable=False):
        """Declare qualified provider storage; sample the complete closure once."""
        from taichi_forge.graph._native import ProviderOwnedNdarrayBinding
        from taichi_forge.lang import impl

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
        if cached is None:
            resource = self._name("resource")
            self._provider_data[identity] = (owner, resource, dtype, shape)
        else:
            _, resource, previous_dtype, previous_shape = cached
            if (dtype, shape) != (previous_dtype, previous_shape):
                raise TaichiRuntimeError("AOT provider allocation changed its data description")
        if name in self.private_bindings and self.private_bindings[name] is not owner:
            raise TaichiRuntimeError("AOT provider data binding changed within one export")
        self.private_bindings[name] = owner
        self._private_names[name] = resource

    def materialize_provider_data(self):
        """Complete and read all selected allocations under one submission lock."""
        import numpy as np

        from taichi_forge.lang import impl
        from taichi_forge.lang.util import to_numpy_type

        pending = tuple(self._provider_data.values())
        if not pending:
            return
        arrays = [owner.arr for owner, _, _, _ in pending]
        hosts = [
            np.empty(shape, dtype=to_numpy_type(array.element_data_type()))
            for array, (_, _, _, shape) in zip(arrays, pending)
        ]
        impl.get_runtime().prog.copy_ndarrays_to_host(arrays, hosts, snapshot=True)
        for (owner, name, dtype, shape), host in zip(pending, hosts):
            self.data_resource(owner, host.tobytes(), dtype=dtype, shape=shape, name=name)

    def kernel_binding(self, arg):
        if arg.name not in self._scalar_values:
            return self.binding(arg.name)
        if arg.tag != core.ArgKind.SCALAR:
            raise TaichiRuntimeError("AOT fixed scalar is used as a non-scalar argument")
        dtype = str(arg.dtype())
        symbol = (arg.name, dtype)
        if symbol in self._scalar_symbols:
            return self._scalar_symbols[symbol]
        value = self._scalar_values[arg.name]
        try:
            if dtype in ("f16", "f32"):
                # F32 matches Graph's C++ cast, including overflow to infinity;
                # F16 also uses a float32 payload in the AOT scalar ABI.
                value = ctypes.c_float(float(value)).value
            resource = Scalar(dtype, value).resource(arg.name)
        except (ValueError, TypeError, OverflowError, struct.error) as error:
            raise TaichiRuntimeError(f"AOT fixed scalar {arg.name!r} cannot be encoded as {dtype}") from error
        identity = (dtype, resource["bits"])
        name = self._scalar_resources.get(identity)
        if name is None:
            name = self._name("resource")
            self._scalar_resources[identity] = name
            resource["name"] = name
            self.resources.append(resource)
        self._scalar_symbols[symbol] = name
        return name

    def declared_bindings(self, bindings):
        return {
            name: owner
            for name, owner in bindings.items()
            if name in self.private_bindings
            and (
                self.private_bindings[name] is owner
                or (
                    name in self._scalar_values
                    and type(owner) in (bool, int, float)
                    and owner == self._scalar_values[name]
                )
            )
        }

    def require_private_bindings(self, bindings):
        if len(self.declared_bindings(bindings)) != len(bindings):
            raise TaichiRuntimeError(
                "AOT fixed bindings require a reconstruction description; live arrays and unsupported values are not declarations"
            )

    def native(self, provider, operation, *, bindings, parameters=None, version=1):
        self.operations.append(self._native_description(provider, operation, bindings, parameters, version))

    def _native_description(self, provider, operation, bindings, parameters, version):
        return _snapshot(
            dict(
                kind="native",
                provider=provider,
                operation=operation,
                version=version,
                parameters=parameters or {},
                bindings={port: self.binding(slot) for port, slot in bindings.items()},
            )
        )

    def initialize_native_once(self, resource, provider, operation, *, bindings, parameters=None, version=1):
        """One consistent setup per shared resource, outside replay."""
        description = self._native_description(provider, operation, bindings, parameters, version)
        previous = self._initialization_resources.get(resource)
        if previous is not None:
            if previous != description:
                raise ValueError("AOT resource initialization changed within one export")
            return
        self._initialization_resources[resource] = description
        self.initialization.append(description)

    def kernel(self, kernel, args, label=""):
        if any(arg.tag not in (core.ArgKind.SCALAR, core.ArgKind.NDARRAY) for arg in args):
            raise TaichiRuntimeError("AOT execution Graph currently requires scalar or ndarray kernel bindings")
        name = self._kernel_names.get(id(kernel))
        if name is None:
            name = self._name("kernel")
            self._kernel_names[id(kernel)] = name
            self.kernels[name] = kernel
        operation = dict(kind="kernel", entry=name, arguments=[self.kernel_binding(arg) for arg in args])
        if label:
            operation["label"] = label
        self.operations.append(operation)


def _native(context, executable):
    action = executable.recordable_action
    if action is None or action.backend_command_recording is None:
        raise TaichiRuntimeError(f"AOT native Graph needs a reconstruction description: {type(executable).__name__}")
    if action.temporary_bindings:
        raise TaichiRuntimeError("AOT native Graph temporary bindings require a resource recipe")
    if any(name in context._scalar_values for name in action.fixed_bindings):
        raise TaichiRuntimeError("AOT native fixed scalars require a typed reconstruction recipe")
    recording = action.backend_command_recording
    # Subclasses must explicitly qualify their own semantics. Inheriting the
    # base export of a callback/epilogue variant would silently lose work.
    exporter = type(recording).__dict__.get("_export_aot")
    if exporter is None:
        raise TaichiRuntimeError(f"AOT reconstruction is unavailable for recording {type(recording).__name__}")
    before = len(context.operations)
    recreated = set(exporter(recording, context))
    if len(context.operations) == before:
        raise TaichiRuntimeError("AOT recording export must contain an operation")
    if recreated | context.declared_bindings(action.fixed_bindings).keys() != set(action.fixed_bindings):
        raise TaichiRuntimeError("AOT recording did not reconstruct all private bindings")
    from taichi_forge.hardware._native_adapter import HardwareRecordingExecutable

    validate = getattr(executable, "validate_graph_bindings", None)
    if (
        getattr(validate, "__func__", None) is HardwareRecordingExecutable.validate_graph_bindings
        and executable._recording is recording
    ):
        # The standard wrapper only forwards to the exact recording whose
        # concrete export contract just succeeded. Additional validation from
        # a custom executable or an unrelated lease still needs its own recipe.
        context.reconstructed_binding_validators.add(id(executable))
    return dict(action.fixed_bindings)


def _sequence(context, sequence):
    if (
        sequence._has_indirect_dispatch
        or sequence._structured_depth
        or sequence._temporary_actions
        or set(sequence._derived_runtime_arg_names) - context.reconstructed_derived_bindings
    ):
        raise TaichiRuntimeError("AOT sequence requires explicit control/private resource reconstruction")
    context.require_private_bindings(sequence._fixed_runtime_args)
    for item in sequence._items:
        if item[0] != "dispatch":
            raise TaichiRuntimeError(f"AOT sequence cannot reconstruct {item[0]!r} yet")
        context.kernel(*item[1:4])


def _node(context, node):
    from taichi_forge.graph._graph import (
        _CompiledCGraphNode,
        _CompiledNativeGraphNode,
        _CompiledSequentialRegionNode,
    )
    from taichi_forge.graph._ir import DispatchNode, SequentialRegion

    if isinstance(node, _CompiledSequentialRegionNode):
        for child in node.nodes:
            _node(context, child)
        return
    if isinstance(node, _CompiledNativeGraphNode):
        if set(node.derived_runtime_arg_names) - context.reconstructed_derived_bindings:
            raise TaichiRuntimeError("AOT native derived bindings require a reconstruction recipe")
        _native(context, node.executable)
        return
    if not isinstance(node, _CompiledCGraphNode):
        raise TaichiRuntimeError(f"AOT execution Graph cannot reconstruct {type(node).__name__} yet")
    if (
        node.parallel_dispatch_groups
        or node.temporary_actions
        or set(node.derived_runtime_arg_names) - context.reconstructed_derived_bindings
    ):
        raise TaichiRuntimeError("AOT execution Graph requires the selected composition/resource recipe")
    ir_nodes = node.ir_node.children if isinstance(node.ir_node, SequentialRegion) else (node.ir_node,)
    if any(isinstance(item, DispatchNode) and item.bounded_domain is not None for item in ir_nodes):
        raise TaichiRuntimeError("AOT execution Graph does not yet reconstruct bounded dispatch")
    if any(dispatch.dispatch_packet is not None for dispatch in node.recording_dispatches):
        raise TaichiRuntimeError("AOT execution Graph does not yet reconstruct indirect dispatch")
    if any(
        manifest.execution_kind not in ("kernel_dispatch", "backend_command")
        for manifest in node.native_action_manifests
    ):
        raise TaichiRuntimeError("AOT execution Graph contains an unsupported native execution kind")
    command_count = sum(manifest.execution_kind == "backend_command" for manifest in node.native_action_manifests)
    if not command_count:
        context.require_private_bindings(node.fixed_runtime_args)
        if len(node.recording_dispatches) != node.dispatch_count:
            raise TaichiRuntimeError("AOT execution Graph lacks complete symbolic dispatch metadata")
        dispatches = node.compiled_graph._physical_jit_dispatch_sources
        if len(dispatches) != node.physical_dispatch_count:
            raise TaichiRuntimeError("AOT execution Graph lacks complete physical dispatch metadata")
        for dispatch in dispatches:
            context.kernel(*dispatch)
        return
    if node.composer_applied_groups or node.physical_dispatch_count != node.dispatch_count:
        raise TaichiRuntimeError("AOT mixed Graph requires the selected composition recipe")
    # Capture-native commands coexist with kernels inside one CGraph. Only the
    # frozen operation sequence retains their ordering; dispatches alone omit them.
    recreated = context.declared_bindings(node.fixed_runtime_args)
    native_count, kernel_count = 0, 0
    for item in node.recipe_operations:
        if item[0] == "native":
            for name, owner in _native(context, item[1]).items():
                if name in recreated and recreated[name] is not owner:
                    raise TaichiRuntimeError("AOT recording private binding owners conflict")
                recreated[name] = owner
            native_count += 1
        elif item[0] == "dispatch":
            context.kernel(*item[1:4])
            kernel_count += 1
        elif item[0] == "sequential":
            before = len(context.operations)
            _sequence(context, item[1])
            kernel_count += len(context.operations) - before
        else:
            raise TaichiRuntimeError(f"AOT execution Graph cannot reconstruct recipe {item[0]!r}")
    if (
        native_count != command_count
        or kernel_count != len(node.recording_dispatches)
        or native_count + kernel_count != node.dispatch_count
    ):
        raise TaichiRuntimeError("AOT recording sequence does not cover the materialized Graph")
    if recreated.keys() != node.fixed_runtime_args.keys() or any(
        recreated[name] is not value for name, value in node.fixed_runtime_args.items()
    ):
        raise TaichiRuntimeError("AOT execution Graph has unreconstructed fixed bindings")


def export_graph(module, graph, *, gradients=None, snapshot_state=False):
    """Preflight the entire region before publishing any module definitions."""
    with graph._lifecycle_lock:
        graph._check_runtime_valid()
        if core.arch_name(module._arch) != graph._execution_arch:
            raise TaichiRuntimeError("AOT execution Graph and module backends must match")
        if graph._workspace_lane_capacity != 1 or graph._qualified_fusion_selector is not None:
            raise TaichiRuntimeError("AOT execution Graph needs a fixed single-frame execution recipe")
        spec = graph._spec
        if getattr(spec, "_binding_executor_factory", None) is not None or getattr(spec, "_storage_plans", ()):
            raise TaichiRuntimeError("AOT execution Graph requires the selected storage/binding recipe")
        gradient_names = set((gradients or {}).values())
        context = _ExportContext(
            module,
            set(spec.runtime_arg_names) | set(spec.fixed_runtime_args) | gradient_names,
            snapshot_state=snapshot_state,
        )
        context.declare_private_bindings(spec.fixed_runtime_args)
        # Dispatch actions may already be flattened into ordinary physical
        # kernels. Reconstruct their concrete binding contracts before lowering
        # those kernels, without regenerating the selected dispatch recipe.
        for lease in spec.lifetime_leases:
            exporter = type(lease).__dict__.get("_export_aot_bindings")
            if exporter is not None:
                exporter(lease, context)
        for node in graph._spec.nodes:
            _node(context, node)
        if any(
            (
                callable(getattr(lease, "bind_graph_arguments", None))
                and id(lease) not in context.reconstructed_argument_binders
            )
            or (
                callable(getattr(lease, "validate_graph_bindings", None))
                and id(lease) not in context.reconstructed_binding_validators
            )
            for lease in spec.lifetime_leases
        ):
            raise TaichiRuntimeError("AOT execution Graph requires the selected binding validation recipe")
        context.materialize_provider_data()
        mapped_gradients = {}
        for primal, gradient in (gradients or {}).items():
            if primal in context._scalar_values or gradient in context._scalar_values:
                raise TaichiRuntimeError("AOT gradient bindings require ndarrays")
            primal, gradient = context.binding(primal), context.binding(gradient)
            if primal in mapped_gradients and mapped_gradients[primal] != gradient:
                raise TaichiRuntimeError("AOT aliases of one primal must share the same gradient resource")
            mapped_gradients[primal] = gradient
        from ._graph_constraints import export_constraints

        plan = ExecutionPlan(
            context.operations,
            initialization=context.initialization or None,
            resources=context.resources,
            gradients=mapped_gradients or None,
            constraints=export_constraints(spec, context.binding, additional=context.binding_constraints) or None,
        )
        # An opaque provider workspace must not accidentally become a public
        # input to some other kernel. Resource recipes must cover the full use.
        if set(plan.argument_names) != set(spec.runtime_arg_names) | (gradient_names - context.private_bindings.keys()):
            raise TaichiRuntimeError("AOT Graph reconstruction changed the public binding contract")
        for key in plan.data_entries:
            if key not in context.data_payloads:
                raise TaichiRuntimeError("AOT Graph reconstruction omitted an initial data payload")
            if key in module._data_payloads and module._data_payloads[key] != context.data_payloads[key]:
                raise ValueError("AOT data digest collision")
        try:
            for name, kernel in context.kernels.items():
                module._aot_builder.add(name, kernel)
        except Exception:
            # Backend builders do not support rollback of successful preceding
            # kernel compilations. Never publish a partially compiled archive.
            module._graph_export_failed = True
            raise
        module._exported_kernels.update(context.kernels)
        module._content.extend("kernel:" + name for name in context.kernels)
        module._data_payloads.update({key: context.data_payloads[key] for key in plan.data_entries})
        return plan
