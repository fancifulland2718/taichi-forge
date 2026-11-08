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

struct ShaderArgumentConstant {
  uint32_t offset;
  unsigned bits;
  uint64_t value;
};

// Consumer-side Vulkan 1.3 transform. Only constant-index integer loads from
// the declared uniform argument binding qualify. Other resources stay dynamic.
TI_DLL_EXPORT std::vector<uint32_t> specialize_spirv_arguments(
    const std::vector<uint32_t> &code, uint32_t argument_binding,
    const std::vector<ShaderArgumentConstant> &values, size_t *folded = nullptr);

}  // namespace taichi::lang::vulkan
