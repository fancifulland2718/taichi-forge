"""Cold Graph lowering into the shared, pointer-free execution description."""

import json

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge_runtime.aot import ExecutionPlan


def _snapshot(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


class _ExportContext:
    """One export transaction; object identities never become artifact data."""

    def __init__(self, module, reserved_names):
        self.operations = []
        self.resources = []
        self.kernels = {}
        self._resource_owners = {}
        self._indices = {}
        self._names = set(reserved_names) | set(module._exported_kernels)
        self._kernel_names = {id(kernel): name for name, kernel in module._exported_kernels.items()}
        self.reconstructed_binding_validators = set()

    def _name(self, kind):
        index = self._indices.get(kind, 0)
        while (name := f"__forge_aot_{kind}_{index}") in self._names:
            index += 1
        self._names.add(name)
        self._indices[kind] = index + 1
        return name

    def resource(self, owner, provider, operation, *, parameters, bindings=None, version=1):
        """Share only an identical producer resource, with an identical description."""
        if owner is None:
            raise ValueError("AOT reconstruction requires a resource owner identity")
        description = _snapshot(
            dict(
                kind="native",
                provider=provider,
                operation=operation,
                version=version,
                parameters=parameters,
                bindings=bindings or {},
            )
        )
        previous = self._resource_owners.get(id(owner))
        if previous is not None:
            _, name, frozen = previous
            if frozen != description:
                raise ValueError("AOT resource reconstruction changed within one export")
            return name
        name = self._name("resource")
        self._resource_owners[id(owner)] = (owner, name, description)
        self.resources.append(dict(description, name=name))
        return name

    def native(self, provider, operation, *, bindings, parameters=None, version=1):
        self.operations.append(
            _snapshot(
                dict(
                    kind="native",
                    provider=provider,
                    operation=operation,
                    version=version,
                    parameters=parameters or {},
                    bindings=bindings,
                )
            )
        )

    def kernel(self, kernel, args, label=""):
        if label:
            raise TaichiRuntimeError("AOT execution Graph does not yet preserve dispatch labels")
        if any(arg.tag not in (core.ArgKind.SCALAR, core.ArgKind.NDARRAY) for arg in args):
            raise TaichiRuntimeError("AOT execution Graph currently requires scalar or ndarray kernel bindings")
        name = self._kernel_names.get(id(kernel))
        if name is None:
            name = self._name("kernel")
            self._kernel_names[id(kernel)] = name
            self.kernels[name] = kernel
        self.operations.append(dict(kind="kernel", entry=name, arguments=[arg.name for arg in args]))


def _native(context, executable):
    action = executable.recordable_action
    if action is None or action.backend_command_recording is None:
        raise TaichiRuntimeError(f"AOT native Graph needs a reconstruction description: {type(executable).__name__}")
    if action.temporary_bindings:
        raise TaichiRuntimeError("AOT native Graph temporary bindings require a resource recipe")
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
    if recreated != set(action.fixed_bindings):
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
        or sequence._fixed_runtime_args
        or sequence._temporary_actions
        or sequence._derived_runtime_arg_names
    ):
        raise TaichiRuntimeError("AOT sequence requires explicit control/private resource reconstruction")
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
        if node.derived_runtime_arg_names:
            raise TaichiRuntimeError("AOT native derived bindings require a reconstruction recipe")
        _native(context, node.executable)
        return
    if not isinstance(node, _CompiledCGraphNode):
        raise TaichiRuntimeError(f"AOT execution Graph cannot reconstruct {type(node).__name__} yet")
    if node.parallel_dispatch_groups or node.temporary_actions or node.derived_runtime_arg_names:
        raise TaichiRuntimeError("AOT execution Graph requires the selected composition/resource recipe")
    ir_nodes = node.ir_node.children if isinstance(node.ir_node, SequentialRegion) else (node.ir_node,)
    if any(isinstance(item, DispatchNode) and item.bounded_domain is not None for item in ir_nodes):
        raise TaichiRuntimeError("AOT execution Graph does not yet reconstruct bounded dispatch")
    if any(dispatch.dispatch_packet is not None for dispatch in node.recording_dispatches):
        raise TaichiRuntimeError("AOT execution Graph does not yet reconstruct indirect dispatch")
    if not node.source_native_count:
        if node.fixed_runtime_args or len(node.recording_dispatches) != node.dispatch_count:
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
    recreated = {}
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
        native_count != node.source_native_count
        or kernel_count != len(node.recording_dispatches)
        or native_count + kernel_count != node.dispatch_count
    ):
        raise TaichiRuntimeError("AOT recording sequence does not cover the materialized Graph")
    if recreated.keys() != node.fixed_runtime_args.keys() or any(
        recreated[name] is not value for name, value in node.fixed_runtime_args.items()
    ):
        raise TaichiRuntimeError("AOT execution Graph has unreconstructed fixed bindings")


def export_graph(module, graph, *, gradients=None):
    """Preflight the entire region before publishing any module definitions."""
    with graph._lifecycle_lock:
        graph._check_runtime_valid()
        if core.arch_name(module._arch) != graph._execution_arch:
            raise TaichiRuntimeError("AOT execution Graph and module backends must match")
        if graph._workspace_lane_capacity != 1 or graph._qualified_fusion_selector is not None:
            raise TaichiRuntimeError("AOT execution Graph needs a fixed single-frame execution recipe")
        spec = graph._spec
        if (
            getattr(spec, "_binding_executor_factory", None) is not None
            or getattr(spec, "_storage_plans", ())
            or spec.memory_recipe_binding_names
        ):
            raise TaichiRuntimeError("AOT execution Graph requires the selected storage/binding recipe")
        gradient_names = set((gradients or {}).values())
        context = _ExportContext(module, set(graph._spec.runtime_arg_names) | gradient_names)
        for node in graph._spec.nodes:
            _node(context, node)
        if any(
            callable(getattr(lease, "bind_graph_arguments", None))
            or (
                callable(getattr(lease, "validate_graph_bindings", None))
                and id(lease) not in context.reconstructed_binding_validators
            )
            for lease in spec.lifetime_leases
        ):
            raise TaichiRuntimeError("AOT execution Graph requires the selected binding validation recipe")
        plan = ExecutionPlan(context.operations, resources=context.resources, gradients=gradients)
        # An opaque provider workspace must not accidentally become a public
        # input to some other kernel. Resource recipes must cover the full use.
        if set(plan.argument_names) != set(graph._spec.runtime_arg_names) | gradient_names:
            raise TaichiRuntimeError("AOT Graph reconstruction changed the public binding contract")
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
        return plan
