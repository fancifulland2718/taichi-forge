"""Private integer scan recording with Graph-retained, immutable scratch."""

from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording, _CudaGraphCaptureRecipe
from taichi_forge.hardware._native_adapter import native_recording_node
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types.primitive_types import i32, u32


def scan_workspace_bytes(num_items, dtype):
    # Even a single-level scan binds one harmless word, keeping the existing
    # Graph ndarray lifetime contract instead of inventing a null allocation.
    query = getattr(_ti_core, "_cuda_scan_capture_workspace_bytes", None)
    if query is None:
        raise TaichiRuntimeError(
            "Graph segmented scan requires a native runtime with retained scan recording support"
        )
    return max(4, query(num_items, 0 if dtype == i32 else 2))


class _ScanCaptureRecipe(_CudaGraphCaptureRecipe):
    kind = "driver_scan_retained_workspace"

    def __init__(self, dtype, num_items):
        self.dtype = dtype
        self.num_items = num_items

    def append_to_graph(self, builder, program):
        from taichi_forge.graph._graph import Arg, ArgKind

        builder._dispatch_cuda_scan_capture_recipe(
            program,
            Arg(ArgKind.NDARRAY, "scanned", self.dtype, ndim=1),
            Arg(ArgKind.NDARRAY, "scan_scratch", u32, ndim=1),
            self.num_items,
            0 if self.dtype == i32 else 2,
        )


class RetainedScanRecording(BackendCommandRecording):
    def __init__(self, dtype, num_items):
        super().__init__(
            backend="cuda",
            binding_names=("scanned", "scan_scratch"),
            # Scratch is an explicit ndarray binding owned by the outer Graph
            # action, not a hidden provider plan allocation.
            command_count=1,
            workspace_ownership="none",
            replay_mode="stream_capture",
        )
        object.__setattr__(self, "_cuda_capture_recipe", _ScanCaptureRecipe(dtype, num_items))

    @property
    def resource_effects(self):
        return (
            ResourceEffect("scanned", GraphAccess.READ_WRITE),
            ResourceEffect("scan_scratch", GraphAccess.READ_WRITE),
        )

    def execute(self, bindings):
        raise TaichiRuntimeError("Retained scan is a CUDA Graph-only recording")

    def _as_graph_native_node(self):
        return native_recording_node(
            self,
            runtime_bindings=(("scanned", "ndarray"), ("scan_scratch", "ndarray")),
            lifetime_leases=(),
            debug_info={"kind": "retained_driver_scan"},
        )
