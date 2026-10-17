#include "taichi/runtime/vulkan_graphics_pipeline.h"

#ifdef TI_WITH_VULKAN
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
namespace {
std::size_t vertex_format_bytes(BufferFormat format) {
  switch (format) {
    case BufferFormat::r8:
    case BufferFormat::r8u:
    case BufferFormat::r8i:
      return 1;
    case BufferFormat::rg8:
    case BufferFormat::rg8u:
    case BufferFormat::rg8i:
    case BufferFormat::r16:
    case BufferFormat::r16u:
    case BufferFormat::r16i:
    case BufferFormat::r16f:
      return 2;
    case BufferFormat::rgba8:
    case BufferFormat::rgba8u:
    case BufferFormat::rgba8i:
    case BufferFormat::rg16:
    case BufferFormat::rg16u:
    case BufferFormat::rg16i:
    case BufferFormat::rg16f:
    case BufferFormat::r32u:
    case BufferFormat::r32i:
    case BufferFormat::r32f:
      return 4;
    case BufferFormat::rgb16:
    case BufferFormat::rgb16u:
    case BufferFormat::rgb16i:
    case BufferFormat::rgb16f:
      return 6;
    case BufferFormat::rgba16:
    case BufferFormat::rgba16u:
    case BufferFormat::rgba16i:
    case BufferFormat::rgba16f:
    case BufferFormat::rg32u:
    case BufferFormat::rg32i:
    case BufferFormat::rg32f:
      return 8;
    case BufferFormat::rgb32u:
    case BufferFormat::rgb32i:
    case BufferFormat::rgb32f:
      return 12;
    case BufferFormat::rgba32u:
    case BufferFormat::rgba32i:
    case BufferFormat::rgba32f:
      return 16;
    default:
      TI_ERROR("Unsupported Vulkan graphics vertex format {}.",
               static_cast<std::uint32_t>(format));
  }
}

}  // namespace

std::unique_ptr<Pipeline> prepare_vulkan_graphics_pipeline(
    vulkan::VulkanDevice &device,
    const std::vector<PipelineSourceDesc> &sources,
    const RasterParams &raster,
    const std::vector<VertexInputBinding> &bindings,
    const std::vector<VertexInputAttribute> &attributes,
    const std::string &name) {
  std::unordered_set<PipelineStageType> stages;
  for (const auto &source : sources) {
    TI_ERROR_IF(source.type != PipelineSourceType::spirv_binary ||
                    !source.data || source.size < 5 * sizeof(uint32_t) ||
                    source.size % sizeof(uint32_t),
                "Vulkan graphics shaders require complete SPIR-V words.");
    uint32_t magic;
    std::memcpy(&magic, source.data, sizeof(magic));
    TI_ERROR_IF(magic != 0x07230203u,
                "Vulkan graphics shaders require SPIR-V magic.");
    TI_ERROR_IF(!stages.insert(source.stage).second,
                "Vulkan graphics shader stages must be unique.");
    TI_ERROR_IF(source.stage != PipelineStageType::vertex &&
                    source.stage != PipelineStageType::fragment &&
                    source.stage != PipelineStageType::mesh &&
                    source.stage != PipelineStageType::task,
                "Unsupported Vulkan graphics shader stage.");
  }
  const bool mesh = stages.count(PipelineStageType::mesh);
  const bool task = stages.count(PipelineStageType::task);
  TI_ERROR_IF(
      !stages.count(PipelineStageType::fragment) ||
          mesh == bool(stages.count(PipelineStageType::vertex)) ||
          (task && !mesh),
      "Vulkan graphics requires fragment and either vertex or mesh shaders.");
  TI_ERROR_IF(
      mesh && (!device.vk_caps().mesh_shader ||
               (task && !device.vk_caps().task_shader)),
      "Vulkan mesh/task shader features are unavailable on the active device.");
  TI_ERROR_IF(mesh && (!bindings.empty() || !attributes.empty()),
              "Vulkan mesh pipelines cannot declare vertex inputs.");

  const auto &limits = device.get_vk_physical_device_props().limits;
  TI_ERROR_IF(bindings.size() > limits.maxVertexInputBindings ||
                  attributes.size() > limits.maxVertexInputAttributes,
              "Vulkan vertex input count exceeds the physical device limits.");
  std::unordered_map<uint32_t, size_t> strides;
  for (const auto &binding : bindings) {
    TI_ERROR_IF(
        binding.binding >= limits.maxVertexInputBindings || !binding.stride ||
            binding.stride > limits.maxVertexInputBindingStride,
        "Vulkan vertex binding or stride exceeds the physical device limits.");
    TI_ERROR_IF(!strides.emplace(binding.binding, binding.stride).second,
                "Vulkan vertex binding is duplicated.");
  }
  std::unordered_set<uint32_t> locations;
  for (const auto &attribute : attributes) {
    TI_ERROR_IF(
        attribute.location >= limits.maxVertexInputAttributes ||
            attribute.offset > limits.maxVertexInputAttributeOffset,
        "Vulkan vertex location or offset exceeds the physical device limits.");
    TI_ERROR_IF(!locations.insert(attribute.location).second,
                "Vulkan vertex attribute location is duplicated.");
    const auto found = strides.find(attribute.binding);
    TI_ERROR_IF(found == strides.end(),
                "Vulkan vertex attribute references an undeclared binding.");
    const auto bytes = vertex_format_bytes(attribute.format);
    TI_ERROR_IF(attribute.offset > found->second ||
                    bytes > found->second - attribute.offset,
                "Vulkan vertex attribute exceeds its binding stride.");
  }
  auto pipeline = device.create_raster_pipeline(sources, raster, bindings,
                                                attributes, name);
  TI_ERROR_IF(!pipeline, "Vulkan graphics pipeline creation failed.");
  return pipeline;
}
}  // namespace taichi::lang
#endif
