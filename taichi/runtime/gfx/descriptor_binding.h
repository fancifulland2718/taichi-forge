#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "taichi/rhi/device.h"

namespace taichi::lang::gfx {

struct DescriptorBufferSlice {
  uint64_t offset{0};
  uint64_t size{0};
  uint32_t byte_bias{0};
};

inline DescriptorBufferSlice describe_descriptor_buffer_slice(
    Device::StorageBufferBindingLimits limits,
    uint64_t offset,
    uint64_t size,
    bool supports_byte_bias) {
  if (!limits.offset_alignment || size > UINT64_MAX - offset)
    throw std::invalid_argument("Invalid descriptor buffer range or alignment");
  // Empty arrays retain the existing harmless whole-allocation binding.
  if (!size)
    return {};
  const auto bias = offset % limits.offset_alignment;
  if (bias && !supports_byte_bias)
    throw std::invalid_argument(
        "Unaligned array view requires descriptor byte offsets; re-export "
        "the kernel with a current Forge compiler");
  if (bias > std::numeric_limits<uint32_t>::max() || size > UINT64_MAX - bias ||
      size + bias > limits.max_range)
    throw std::invalid_argument(
        "Array descriptor exceeds the device range or shader offset ABI");
  return {offset - bias, size + bias, uint32_t(bias)};
}

}  // namespace taichi::lang::gfx
