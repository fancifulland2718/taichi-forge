#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "taichi/common/core.h"

namespace taichi::lang::vulkan {

// Explicit producer-side transformation. Does not create a device or change
// pipeline/runtime state. Unspecified scalar constants freeze to their
// defaults.
TI_DLL_EXPORT std::vector<uint32_t> specialize_spirv_shader(
    const std::vector<uint32_t> &code,
    const std::unordered_map<uint32_t, std::string> &values);

}  // namespace taichi::lang::vulkan
