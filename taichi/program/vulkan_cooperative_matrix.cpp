#include "taichi/program/program.h"

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {

bool Program::vulkan_cooperative_matrix_available() const {
  if (compile_config().arch != Arch::vulkan || !program_impl_) {
    return false;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_compute_device());
  return device != nullptr && device->vk_caps().cooperative_matrix &&
         !device->vk_caps().cooperative_matrix_properties.empty();
}

std::vector<std::unordered_map<std::string, std::uint64_t>>
Program::vulkan_cooperative_matrix_properties() const {
  std::vector<std::unordered_map<std::string, std::uint64_t>> result;
  if (compile_config().arch != Arch::vulkan || !program_impl_) {
    return result;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_compute_device());
  if (device == nullptr || !device->vk_caps().cooperative_matrix) {
    return result;
  }
  const auto &caps = device->vk_caps();
  result.reserve(caps.cooperative_matrix_properties.size());
  for (const auto &property : caps.cooperative_matrix_properties) {
    result.push_back({
        {"m", property.m},
        {"n", property.n},
        {"k", property.k},
        {"a_type", static_cast<std::uint64_t>(property.a_type)},
        {"b_type", static_cast<std::uint64_t>(property.b_type)},
        {"c_type", static_cast<std::uint64_t>(property.c_type)},
        {"result_type", static_cast<std::uint64_t>(property.result_type)},
        {"scope", static_cast<std::uint64_t>(property.scope)},
        {"saturating_accumulation", property.saturating_accumulation},
        {"subgroup_size", caps.subgroup_size},
        {"supported_stages", caps.cooperative_matrix_supported_stages},
    });
  }
  return result;
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

bool Program::vulkan_cooperative_matrix_available() const {
  return false;
}

std::vector<std::unordered_map<std::string, std::uint64_t>>
Program::vulkan_cooperative_matrix_properties() const {
  return {};
}

}  // namespace taichi::lang

#endif
