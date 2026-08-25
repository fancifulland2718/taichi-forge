"""Public Vulkan hardware ray-query provider API."""

from taichi_forge.hardware._ray import (
    InstanceTLAS,
    RayInstance,
    TriangleBLAS,
    TriangleScene,
    VulkanBLASBuildRecording,
    VulkanBLASRefitRecording,
    VulkanRayQueryRecording,
    VulkanRayRefitRecording,
    VulkanTLASBuildRecording,
    VulkanTLASRefitRecording,
    is_available,
)
from taichi_forge.hardware._optix import (
    OptixProvider,
    OptixRayQueryRecording,
    OptixRayRefitRecording,
    OptixTriangleScene,
    is_loaded as is_optix_loaded,
    load_provider as load_optix_provider,
)

__all__ = [
    "InstanceTLAS",
    "OptixProvider",
    "OptixRayQueryRecording",
    "OptixRayRefitRecording",
    "OptixTriangleScene",
    "RayInstance",
    "TriangleBLAS",
    "TriangleScene",
    "VulkanBLASBuildRecording",
    "VulkanBLASRefitRecording",
    "VulkanRayQueryRecording",
    "VulkanRayRefitRecording",
    "VulkanTLASBuildRecording",
    "VulkanTLASRefitRecording",
    "is_available",
    "is_optix_loaded",
    "load_optix_provider",
]
