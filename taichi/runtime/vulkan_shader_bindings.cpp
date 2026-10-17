#include "taichi/runtime/vulkan_shader_bindings.h"

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
PreparedVulkanShaderBindings prepare_vulkan_shader_bindings(
    vulkan::VulkanPipeline &pipeline,
    VulkanShaderResourceSets resources) {
  const auto &expected = pipeline.get_resource_set_templates();
  TI_ERROR_IF(resources.size() != expected.size(),
              "Vulkan shader descriptor sets do not match shader reflection.");
  for (const auto &[index, expected_set] : expected) {
    const auto found = resources.find(index);
    TI_ERROR_IF(
        found == resources.end() || !found->second ||
            !vulkan::VulkanResourceSet::SetLayoutCmp{}(
                expected_set,
                *static_cast<vulkan::VulkanResourceSet *>(found->second.get())),
        "Vulkan shader descriptor type/count/coverage mismatch in set {}.",
        index);
  }
  PreparedVulkanShaderBindings prepared;
  prepared.reserve(resources.size());
  for (auto &[index, resource] : resources) {
    const auto result = resource->prepare_for_replay(false);
    TI_ERROR_IF(
        result != RhiResult::success,
        "Vulkan shader resource set {} preparation failed: RhiResult({}).",
        index, result);
    prepared.emplace_back(index, std::move(resource));
  }
  return prepared;
}
}  // namespace taichi::lang
#endif
