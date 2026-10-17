#pragma once

#include "taichi/runtime/vulkan_ray_program_types.h"

#ifdef TI_WITH_VULKAN
#include <array>
#include <functional>
#include <map>
#include "taichi/rhi/vulkan/vulkan_ray_pipeline.h"

namespace taichi::lang {

using VulkanRayProgramResourceSets =
    std::map<std::uint32_t, std::unique_ptr<vulkan::VulkanResourceSet>>;

// Shared cold construction and recording. The caller owns resource leases,
// queue ordering and retirement; these functions never submit or use Program.
std::unique_ptr<vulkan::VulkanPipeline> prepare_vulkan_ray_program_pipeline(
    vulkan::VulkanDevice &device,
    const std::vector<VulkanRayProgramShader> &shaders,
    const std::vector<VulkanRayProgramGroup> &groups,
    std::uint32_t recursion_depth,
    bool opacity_micromap);

void validate_vulkan_ray_program_launch(
    const vulkan::VulkanDevice &device,
    const vulkan::VulkanPipeline &pipeline,
    const std::array<std::uint32_t, 3> &dimensions,
    const std::vector<std::uint8_t> &push_constants);

void prepare_vulkan_ray_program_bindings(
    vulkan::VulkanPipeline &pipeline,
    const VulkanRayProgramResourceSets &sets);

std::unique_ptr<vulkan::VulkanShaderBindingTable>
prepare_vulkan_ray_program_sbt(vulkan::VulkanDevice &device,
                               const vulkan::VulkanPipeline &pipeline,
                               const VulkanRayProgramRecord &raygen,
                               const std::vector<VulkanRayProgramRecord> &miss,
                               const std::vector<VulkanRayProgramRecord> &hit);

void record_vulkan_ray_program(
    CommandList *commands,
    vulkan::VulkanPipeline &pipeline,
    const vulkan::VulkanShaderBindingTable &sbt,
    const VulkanRayProgramResourceSets &sets,
    const std::vector<std::function<void(CommandList *)>> &scene_reads,
    const std::array<std::uint32_t, 3> &dimensions,
    const std::vector<std::uint8_t> &push_constants);

}  // namespace taichi::lang
#endif
