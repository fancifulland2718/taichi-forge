#pragma once

#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang::vulkan {

// Cold, device-specific admission for the selected mesh/task entry point.
// This supplements reflection, not SPIR-V validation. No specialization
// overrides are supplied by the current graphics interface. Unresolved constant
// expressions remain the driver's responsibility; never guess their value.
TI_DLL_EXPORT void validate_mesh_shader_limits(const void *code,
                                               size_t size,
                                               uint32_t entry_id,
                                               VkShaderStageFlagBits stage,
                                               const VulkanCapabilities &caps);

}  // namespace taichi::lang::vulkan
