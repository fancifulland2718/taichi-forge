#pragma once

#include <cstddef>
#include <cstdint>

namespace taichi::lang {

// Capabilities are the first section in a logical SPIR-V module. Inspect only
// this small prefix, not every instruction in a potentially large shader.
inline bool spirv_declares_capability(const std::uint32_t *words,
                                      std::size_t count,
                                      std::uint32_t capability) {
  if (count < 5 || words[0] != 0x07230203)
    return false;
  for (std::size_t i = 5; i < count;) {
    const auto size = words[i] >> 16;
    if (size != 2 || (words[i] & 0xffff) != 17 || size > count - i)
      break;  // OpCapability has exactly two words.
    if (words[i + 1] == capability)
      return true;
    i += size;
  }
  return false;
}

}  // namespace taichi::lang
