#include "taichi/runtime/gfx/cooperative_matrix.h"

#include <stdexcept>

#include "spirv/unified1/spirv.hpp"
#include "taichi/rhi/common/spirv_capability.h"
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi::lang::gfx {

void validate_cooperative_matrix_metadata(
    const spirv::TaskAttributes &task,
    const std::vector<std::uint32_t> &code,
    std::uint32_t metadata_version) {
  const bool uses_matrix = spirv_declares_capability(
      code.data(), code.size(), spv::CapabilityCooperativeMatrixKHR);
  const auto &requirements = task.cooperative_matrix_requirements;
  if (!uses_matrix && requirements.empty())
    return;
  if (metadata_version < 8 || uses_matrix == requirements.empty())
    throw std::invalid_argument(
        "Cooperative matrix shader requires matching version-8 deployment "
        "requirements; re-export or recompile the old artifact");
  // The existing lowering uses the device's fixed subgroup width to compute
  // tile addresses. SPIR-V 1.6 permits varying widths even without stage flags.
  if (code[1] >= 0x10600)
    throw std::invalid_argument(
        "Cooperative matrix fixed-subgroup lowering requires SPIR-V below 1.6");
  for (const auto &requirement : requirements) {
    if (!requirement.m || !requirement.n || !requirement.k ||
        !requirement.subgroup_size ||
        task.task_type != OffloadedTaskType::range_for ||
        !task.range_for_attribs || !task.range_for_attribs->const_range() ||
        task.range_for_attribs->begin != 0 ||
        task.range_for_attribs->end % requirement.subgroup_size ||
        task.advisory_num_threads_per_group <= 0 ||
        task.advisory_num_threads_per_group % requirement.subgroup_size)
      throw std::invalid_argument(
          "Invalid cooperative matrix dimensions, subgroup or launch contract");
  }
}

void validate_cooperative_matrix_device(
    Device *device,
    const spirv::TaskAttributes &task,
    const std::vector<std::uint32_t> &code) {
  validate_cooperative_matrix_metadata(task, code, 8);
  if (task.cooperative_matrix_requirements.empty())
    return;
#ifdef TI_WITH_VULKAN
  auto *vk = dynamic_cast<vulkan::VulkanDevice *>(device);
  if (vk != nullptr) {
    for (const auto &requirement : task.cooperative_matrix_requirements) {
      if (!vk->vk_caps().cooperative_matrix_f16_f32_valid(
              requirement.m, requirement.n, requirement.k,
              requirement.subgroup_size))
        throw std::invalid_argument(
            "Vulkan device does not support the cooperative matrix dimensions "
            "or fixed subgroup required by this kernel");
    }
    return;
  }
#endif
  throw std::invalid_argument(
      "Cooperative matrix deployment requires a compatible Vulkan device");
}

}  // namespace taichi::lang::gfx
