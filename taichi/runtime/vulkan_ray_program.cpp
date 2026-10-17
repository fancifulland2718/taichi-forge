#include "taichi/runtime/vulkan_ray_program.h"

#ifdef TI_WITH_VULKAN
#include <stdexcept>

namespace taichi::lang {
using vulkan::VulkanPipeline;
using vulkan::VulkanResourceSet;
namespace {
VkShaderStageFlagBits stage_bits(VulkanRayProgramShader::Stage stage) {
  switch (stage) {
    case VulkanRayProgramShader::Stage::raygen:
      return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    case VulkanRayProgramShader::Stage::miss:
      return VK_SHADER_STAGE_MISS_BIT_KHR;
    case VulkanRayProgramShader::Stage::closest_hit:
      return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    case VulkanRayProgramShader::Stage::any_hit:
      return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  }
  throw std::invalid_argument("Unsupported Vulkan ray shader stage");
}

vulkan::VulkanShaderBindingRecord native_record(
    const VulkanRayProgramRecord &record) {
  return {record.group, record.data};
}

}  // namespace

std::unique_ptr<VulkanPipeline> prepare_vulkan_ray_program_pipeline(
    vulkan::VulkanDevice &device,
    const std::vector<VulkanRayProgramShader> &shaders,
    const std::vector<VulkanRayProgramGroup> &groups,
    std::uint32_t recursion_depth,
    bool opacity_micromap) {
  VulkanPipeline::Params params;
  params.device = &device;
  params.name = "forge_ray_program";
  for (const auto &shader : shaders) {
    TI_ERROR_IF(shader.words.size() < 5 || shader.words.front() != 0x07230203u,
                "Ray shaders require a complete SPIR-V module");
    vulkan::SpirvCodeView view;
    view.data = shader.words.data();
    view.size = shader.words.size() * sizeof(std::uint32_t);
    view.stage = stage_bits(shader.stage);
    view.entry_point = shader.entry;
    params.code.push_back(std::move(view));
  }
  vulkan::VulkanRayTracingPipelineParams ray;
  ray.max_recursion_depth = recursion_depth;
  ray.opacity_micromap = opacity_micromap;
  for (const auto &group : groups) {
    vulkan::VulkanRayTracingGroup::Kind kind;
    switch (group.kind) {
      case VulkanRayProgramGroup::Kind::raygen:
        kind = vulkan::VulkanRayTracingGroup::Kind::raygen;
        break;
      case VulkanRayProgramGroup::Kind::miss:
        kind = vulkan::VulkanRayTracingGroup::Kind::miss;
        break;
      case VulkanRayProgramGroup::Kind::triangles:
        kind = vulkan::VulkanRayTracingGroup::Kind::triangles;
        break;
      default:
        throw std::invalid_argument("Invalid Vulkan ray group kind");
    }
    ray.groups.push_back(
        {kind, group.general, group.closest_hit, group.any_hit});
  }
  return std::make_unique<VulkanPipeline>(params, ray);
}

void validate_vulkan_ray_program_launch(
    const vulkan::VulkanDevice &device,
    const VulkanPipeline &pipeline,
    const std::array<std::uint32_t, 3> &dimensions,
    const std::vector<std::uint8_t> &push_constants) {
  vulkan::validate_ray_dispatch(device, dimensions[0], dimensions[1],
                                dimensions[2]);
  TI_ERROR_IF(
      push_constants.size() > 128 || push_constants.size() % 4 ||
          push_constants.size() < pipeline.required_ray_push_constant_bytes(),
      "Ray push constants must cover reflected blocks and fit the "
      "aligned 128-byte layout");
}

void prepare_vulkan_ray_program_bindings(
    VulkanPipeline &pipeline,
    const VulkanRayProgramResourceSets &sets) {
  const auto &expected = pipeline.get_resource_set_templates();
  TI_ERROR_IF(sets.size() != expected.size(),
              "Ray descriptor sets do not match shader reflection");
  for (const auto &[index, expected_set] : expected) {
    const auto found = sets.find(index);
    TI_ERROR_IF(found == sets.end(), "Missing ray descriptor set {}", index);
    const auto &actual = found->second->get_bindings();
    TI_ERROR_IF(actual.size() != expected_set.get_bindings().size(),
                "Ray descriptor binding coverage differs in set {}", index);
    for (const auto &[binding, declaration] : expected_set.get_bindings()) {
      const auto value = actual.find(binding);
      TI_ERROR_IF(value == actual.end() ||
                      value->second.type != declaration.type ||
                      VulkanResourceSet::descriptor_count(value->second) !=
                          VulkanResourceSet::descriptor_count(declaration),
                  "Ray descriptor type/count mismatch at set {} binding {}",
                  index, binding);
    }
  }
  for (const auto &[index, set] : sets) {
    TI_ERROR_IF(set->prepare_for_replay(false) != RhiResult::success,
                "Ray descriptor set {} preparation failed", index);
  }
}

std::unique_ptr<vulkan::VulkanShaderBindingTable>
prepare_vulkan_ray_program_sbt(vulkan::VulkanDevice &device,
                               const VulkanPipeline &pipeline,
                               const VulkanRayProgramRecord &raygen,
                               const std::vector<VulkanRayProgramRecord> &miss,
                               const std::vector<VulkanRayProgramRecord> &hit) {
  std::vector<vulkan::VulkanShaderBindingRecord> native_miss, native_hit;
  native_miss.reserve(miss.size());
  native_hit.reserve(hit.size());
  for (const auto &record : miss)
    native_miss.push_back(native_record(record));
  for (const auto &record : hit)
    native_hit.push_back(native_record(record));
  return std::make_unique<vulkan::VulkanShaderBindingTable>(
      device, pipeline, native_record(raygen), native_miss, native_hit);
}

void record_vulkan_ray_program(
    CommandList *commands,
    VulkanPipeline &pipeline,
    const vulkan::VulkanShaderBindingTable &sbt,
    const VulkanRayProgramResourceSets &sets,
    const std::vector<std::function<void(CommandList *)>> &scene_reads,
    const std::array<std::uint32_t, 3> &dimensions,
    const std::vector<std::uint8_t> &push_constants) {
  auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(commands);
  vulkan::record_ray_program_begin(*vk_commands);
  for (const auto &read : scene_reads)
    read(commands);
  vk_commands->bind_pipeline(&pipeline);
  for (const auto &[index, set] : sets) {
    TI_ERROR_IF(vk_commands->bind_shader_resources(set.get(), index) !=
                    RhiResult::success,
                "Failed to bind prepared Vulkan ray resources");
  }
  if (!push_constants.empty())
    vk_commands->push_constants(push_constants.data(), push_constants.size());
  vk_commands->trace_rays(sbt, dimensions[0], dimensions[1], dimensions[2]);
  vulkan::record_ray_program_end(*vk_commands);
}
}  // namespace taichi::lang
#endif
