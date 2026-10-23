"""Explicit immutable native recordings using the existing operation factories."""

from taichi_forge._lib import core
from taichi_forge.algorithms import _autodiff
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording, _CudaGraphCaptureRecipe
from taichi_forge.hardware._native_adapter import native_recording_node
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang._storage_view import DenseNdarrayView, describe_storage
from taichi_forge.lang.exception import TaichiRuntimeError


class _CaptureRecipe(_CudaGraphCaptureRecipe):
    kind = "builtin_fixed_primitive"

    def __init__(self, owner, arguments):
        self.owner, self.arguments = owner, arguments

    def append_to_graph(self, builder, program):
        builder._dispatch_cuda_capture_command(self.owner.cuda_command(), self.arguments)


class _Recording(BackendCommandRecording):
    def __init__(self, plan, owner, fixed, arguments):
        super().__init__(
            backend=plan._backend,
            binding_names=tuple(fixed),
            command_count=1,
            barrier_policy="internal",
            workspace_ownership="provider_generation",
            replay_mode="stream_capture" if plan._backend == "cuda" else "native_replay",
        )
        # Freeze the definition and source bindings, not the source plan's
        # mutable close state or its growable Program arena.
        for name, value in dict(
            _owner=owner, _fixed=fixed, _arguments=arguments,
            _program=plan._program, _owners=plan._owners,
            _binding=plan._native_binding, _ad_kind=plan._ad_kind,
            _effects=plan._native_binding.resource_effects
            + (ResourceEffect(owner, GraphAccess.READ_WRITE, runtime_bound=False),),
            _graph_physical_plan_id=plan._physical_id + ":persistent-v1",
        ).items():
            object.__setattr__(self, name, value)
        if plan._backend == "cuda":
            object.__setattr__(self, "_cuda_capture_recipe", _CaptureRecipe(owner, arguments))

    @property
    def resource_effects(self):
        return self._effects

    @property
    def workspace_bytes(self):
        """Private native scratch retained by this recording generation."""
        return self._owner.workspace_bytes()

    def execute(self, bindings):
        if impl.get_runtime().prog is not self._program:
            raise TaichiRuntimeError("Persistent primitive belongs to another runtime")
        _autodiff.reject_unsupported_automatic_ad(self._ad_kind)
        raise TaichiRuntimeError("Persistent CUDA primitives require a captured Graph execution route")

    def prepare_graph_execute(self, bindings):
        return lambda: self.execute(bindings)

    def _export_aot(self, context):
        self._binding.lower(context)
        return ()

    def _as_graph_native_node(self):
        return native_recording_node(
            self, runtime_bindings=(), fixed_bindings=self._fixed,
            lifetime_leases=(*self._owners, self._owner),
            debug_info=dict(
                kind="persistent_native_primitive", operation=self._binding.definition.operation,
                workspace_bytes=self._owner.workspace_bytes(),
                workspace_owner="recording_generation",
            ),
        )


class _VulkanRecording(_Recording):
    def prepare_graph_execute(self, bindings):
        # A baseline Graph also needs an actual prepared replay packet. The
        # whole-Graph binding-frame recipe can instead co-record our command
        # with neighboring kernels. Both retain the same private workspace.
        _autodiff.reject_unsupported_automatic_ad(self._ad_kind)
        if impl.get_runtime().prog is not self._program:
            raise TaichiRuntimeError("Persistent primitive belongs to another runtime")
        frame = getattr(self, "_frame", None)
        if frame is None:
            from taichi_forge.graph._graph import _GraphRunContext

            context = _GraphRunContext()
            context.begin(self._fixed)
            try:
                frame = core._prepare_vulkan_graph_recording(
                    self._program, [self._vulkan_graph_command()], context.flattened_args()
                )
            finally:
                context.end()
            object.__setattr__(self, "_frame", frame)
        return self._run_frame

    def _run_frame(self):
        if impl.get_runtime().prog is not self._program:
            raise TaichiRuntimeError("Persistent primitive belongs to another runtime")
        _autodiff.reject_unsupported_automatic_ad(self._ad_kind)
        self._frame.run()

    def execute(self, bindings):
        self.prepare_graph_execute(bindings)()

    def _vulkan_graph_command(self):
        return self._owner.vulkan_command()

    # Concrete exporter admission must not silently inherit changed behavior.
    def _export_aot(self, context):
        return super()._export_aot(context)


def persistent_recording(plan):
    from ._prepared_compact import PreparedCompactPlan
    from ._prepared_primitive import PreparedSortPlan
    from taichi_forge.graph._graph import ArgKind

    if type(plan) not in (PreparedSortPlan, PreparedCompactPlan):
        raise TypeError("Persistent primitive requires a concrete sort or compact plan")
    if plan._backend not in ("cuda", "vulkan"):
        raise TaichiRuntimeError("Persistent native recording requires CUDA or Vulkan")
    if isinstance(plan, PreparedSortPlan) and plan._definition.parameters["nan_policy"] != "last":
        raise TaichiRuntimeError("Persistent native sort requires nan_policy='last'")
    operation = plan._definition.operation
    factory = getattr(plan._program, "_record_primitive_" + operation, None)
    if factory is None:
        raise TaichiRuntimeError("Persistent primitives require matching native recording support")
    fixed, arguments, indices = {}, [], []
    prefix = f"__native_primitive_{id(plan):x}_{id(arguments):x}_"
    for index, binding in enumerate(plan._native_binding.arrays):
        value = binding.value
        source = value
        while isinstance(source, DenseNdarrayView):
            source = source._source
        if not isinstance(source, Ndarray):
            # Dense Field roots use the native command's tree dependencies.
            # They are not invented ndarray owners or copied to new storage.
            continue
        # Graph leases retain the owning allocation. The prepared storage
        # descriptor already freezes the exact view offset/layout; do not
        # invent a second ndarray owner for a subrange.
        descriptor = describe_storage(source).descriptor
        name = prefix + binding.port
        arguments.append(core.Arg(ArgKind.NDARRAY, name, descriptor.scalar_type,
                                  len(descriptor.index_shape), list(descriptor.element_shape)))
        indices.append(index)
        fixed[name] = source
    owner = factory(plan._command, arguments, indices)
    cls = _VulkanRecording if plan._backend == "vulkan" else _Recording
    return cls(plan, owner, fixed, arguments)
