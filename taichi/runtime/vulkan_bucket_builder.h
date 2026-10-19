#pragma once

#include <cstdint>
#include <memory>
#include "taichi/rhi/device.h"

namespace taichi::lang {
class NativeOperation;
namespace gfx {

struct VulkanBucketBuilderView {
  DevicePtr pointer{kDeviceNullPtr};
  size_t stride{0};
};

// Algorithm selection preferences, not input-capacity limits. Larger inputs
// or bin counts can use direct counting without per-chunk partial storage.
struct VulkanBucketBuilderPolicy {
  bool private_enabled{true};
  uint64_t private_min_items{65536};
  uint64_t private_max_bins{1024};
  uint64_t private_max_bytes{4ull << 20};
};

// The operation owns its cursor/partial buffers, pipelines and descriptors;
// its executor retains user allocations through completion. Keys/counts use
// i32; payloads are copied without arithmetic. Bucket order is unspecified.
std::shared_ptr<NativeOperation> prepare_vulkan_bucket_builder_operation(
    Device &device,
    VulkanBucketBuilderView keys,
    VulkanBucketBuilderView values,
    VulkanBucketBuilderView offsets,
    VulkanBucketBuilderView output,
    size_t count,
    size_t bins,
    size_t item_bytes,
    int value_type,
    const VulkanBucketBuilderPolicy &policy = {});

}  // namespace gfx
}  // namespace taichi::lang
