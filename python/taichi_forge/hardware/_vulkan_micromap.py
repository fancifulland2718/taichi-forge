"""Baked host input for the Vulkan BLAS opacity micromap adapter."""

from taichi_forge.hardware._micromap import BakedOpacityMicromap
from dataclasses import dataclass


@dataclass(frozen=True)
class VulkanOpacityMicromap(BakedOpacityMicromap):
    """Native Vulkan two/four-state micromap data, not a borrowed GPU handle.

    Pass to ``TriangleBLAS(..., opacity_micromap=asset)``. Descriptors are
    (byte offset, subdivision level, native format 1 or 2) triples or packed
    little-endian u32/u16/u16 bytes. Optional indices are signed int32; -1..-4
    denote native predefined opacity states. The caller owns classification
    and its agreement with geometry, UVs, alpha filtering and cutoff.
    """

    _identity_tag = b"forge-vulkan-baked-omm1"
