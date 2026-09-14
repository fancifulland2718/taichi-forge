"""Native batch scenes plus explicit Vulkan and OptiX resource/program APIs."""

from taichi_forge.hardware._ray_factory import NativeTriangleScene, triangle_scene

from taichi_forge.hardware._vulkan_micromap import VulkanOpacityMicromap
from taichi_forge.hardware._shader_artifact import PtxModule, ShaderBuildInfo
from taichi_forge.hardware._shader_artifact import SpirvShader
from taichi_forge.hardware._vulkan_ray_program import (
    VulkanHitGroup, VulkanPreparedLaunch, VulkanProgramRecording, VulkanRayBinding,
    VulkanRayTracingPipeline, VulkanSbtRecord, is_program_available,
)
from taichi_forge.hardware._optix_parameters import OptixParameterField, OptixParameterLayout
from taichi_forge.hardware._optix_program import (
    OptixHitGroup, OptixPreparedLaunch, OptixProgram, OptixProgramRecording,
    OptixSbtRecord, OptixShaderEntry,
)

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
    VulkanTLASTransformRecording,
    is_available,
    is_opacity_micromap_available,
)
from taichi_forge.hardware._optix import (
    OptixAlphaMask,
    OptixOpacityMicromap,
    OptixGASRefitRecording,
    OptixInstanceRefitRecording,
    OptixInstanceScene,
    OptixProvider,
    OptixRayInstance,
    OptixRayQueryRecording,
    OptixRayRefitRecording,
    OptixTriangleScene,
    OptixTriangleGAS,
    is_loaded as is_optix_loaded,
    load_provider as load_optix_provider,
)

__all__ = [
    "NativeTriangleScene",
    "triangle_scene",
    "SpirvShader",
    "VulkanHitGroup",
    "VulkanPreparedLaunch",
    "VulkanProgramRecording",
    "VulkanRayBinding",
    "VulkanRayTracingPipeline",
    "VulkanSbtRecord",
    "is_program_available",
    "PtxModule",
    "ShaderBuildInfo",
    "OptixHitGroup",
    "OptixParameterField",
    "OptixParameterLayout",
    "OptixPreparedLaunch",
    "OptixProgram",
    "OptixProgramRecording",
    "OptixSbtRecord",
    "OptixShaderEntry",
    "is_opacity_micromap_available",
    "VulkanOpacityMicromap",
    "InstanceTLAS",
    "OptixAlphaMask",
    "OptixOpacityMicromap",
    "OptixGASRefitRecording",
    "OptixInstanceRefitRecording",
    "OptixInstanceScene",
    "OptixProvider",
    "OptixRayInstance",
    "OptixRayQueryRecording",
    "OptixRayRefitRecording",
    "OptixTriangleScene",
    "OptixTriangleGAS",
    "RayInstance",
    "TriangleBLAS",
    "TriangleScene",
    "VulkanBLASBuildRecording",
    "VulkanBLASRefitRecording",
    "VulkanRayQueryRecording",
    "VulkanRayRefitRecording",
    "VulkanTLASBuildRecording",
    "VulkanTLASRefitRecording",
    "VulkanTLASTransformRecording",
    "is_available",
    "is_optix_loaded",
    "load_optix_provider",
]
