from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl


def sync():
    """Blocks the calling thread until all the previously
    launched Taichi kernels have completed.
    """
    impl.get_runtime().sync()


def get_last_spv_stats():
    """Returns the latest Vulkan SPIR-V per-task codegen stats."""
    return list(_ti_core.get_last_vulkan_spv_stats())


__all__ = ["sync", "get_last_spv_stats"]
