#pragma once

#include "taichi/rhi/device.h"

namespace taichi::lang {

bool is_cuda_to_vulkan_copy(Device *dst_device, Device *src_device);

void memcpy_cuda_to_vulkan_fast(DevicePtr dst, DevicePtr src, uint64_t size);

void memcpy_cuda_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size);

void memcpy_vulkan_to_cuda(DevicePtr dst, DevicePtr src, uint64_t size);

}  // namespace taichi::lang
