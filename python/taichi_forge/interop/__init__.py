"""Zero-copy storage interoperability.

The public adapters in this module are strict: unsupported storage raises
instead of silently materializing a copy. Existing NumPy, PyTorch, and Paddle
kernel-argument APIs retain their historical copy fallbacks.
"""

from ._dlpack import (
    ExternalDenseView,
    capabilities as _dlpack_capabilities,
    from_dlpack,
    from_external,
)
from ._vulkan_cuda import (
    VulkanCudaExternalAllocation,
    capabilities as _vulkan_cuda_capabilities,
    current_cuda_device_uuid,
    import_vulkan_cuda_allocation as _import_vulkan_cuda_allocation,
)


def import_external_allocation(provider, *args, **kwargs):
    """Import a raw external allocation through a qualified provider.

    ``"vulkan_cuda"`` is the only raw-allocation provider currently exposed.
    Provider-specific entry points remain available for source compatibility.
    """

    if provider == "vulkan_cuda":
        return _import_vulkan_cuda_allocation(*args, **kwargs)
    raise ValueError(
        "unsupported external allocation provider; expected 'vulkan_cuda'"
    )


def import_vulkan_cuda_allocation(*args, **kwargs):
    """Compatibility spelling for the Vulkan-CUDA allocation provider."""

    return import_external_allocation("vulkan_cuda", *args, **kwargs)


def capabilities():
    """Return backward-compatible and provider-scoped interop capabilities."""

    result = dict(_dlpack_capabilities())
    result["interop_schema_version"] = 2
    result["providers"] = {
        "dlpack": dict(_dlpack_capabilities()),
        "vulkan_cuda": _vulkan_cuda_capabilities(),
    }
    return result


__all__ = [
    "ExternalDenseView",
    "VulkanCudaExternalAllocation",
    "capabilities",
    "current_cuda_device_uuid",
    "from_dlpack",
    "from_external",
    "import_external_allocation",
    "import_vulkan_cuda_allocation",
]
