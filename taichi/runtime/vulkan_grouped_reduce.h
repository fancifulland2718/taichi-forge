#pragma once

#include <memory>
#include "taichi/rhi/device.h"

namespace taichi::lang {
class NativeOperation;
namespace gfx {

struct VulkanGroupedReduceView {
  DevicePtr pointer{kDeviceNullPtr};
  size_t stride{0};
};

// Cold fixed sum preparation. Keys are signed i32; invalid keys are ignored.
// Empty input still overwrites every output group with zero. The enclosing
// executor retains allocations; the operation owns pipelines and descriptors.
std::shared_ptr<NativeOperation> prepare_vulkan_grouped_reduce_operation(
    Device &device,
    VulkanGroupedReduceView keys,
    VulkanGroupedReduceView values,
    VulkanGroupedReduceView output,
    size_t count,
    size_t groups,
    int value_type);

}  // namespace gfx
}  // namespace taichi::lang
