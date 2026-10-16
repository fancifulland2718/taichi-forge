#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "taichi/rhi/device.h"

namespace taichi::lang {
namespace vulkan {
class VulkanDevice;
}

// The existing shader ABI uses a nonempty uint32 key count. This validation
// performs no device access, allocation or compilation.
std::uint64_t vulkan_parallel_sort_storage_bytes(std::uint64_t key_count);

// Device-local execution shared by JIT and AOT; no Program/Ndarray handles.
// Creation records but never executes sorting. The caller serializes device
// use and retires all command buffers before device destruction. Commands
// retain input leases, shader pipelines and scratch after this handle closes.
class VulkanParallelSortPlanCore {
 public:
  virtual ~VulkanParallelSortPlanCore() = default;
  virtual void record(CommandList *commands, bool inline_recording) = 0;
  virtual std::unordered_map<std::string, std::uint64_t> statistics() const = 0;
};

std::shared_ptr<VulkanParallelSortPlanCore> make_vulkan_parallel_sort_plan(
    vulkan::VulkanDevice &device,
    DevicePtr keys,
    std::optional<DevicePtr> values,
    std::shared_ptr<void> storage_lease,
    std::uint64_t key_count,
    const std::vector<std::vector<std::uint32_t>> &shaders,
    // JIT retains a secondary sequence for cheap direct submissions; AOT
    // records into its own command list and does not need a second copy.
    bool retain_secondary);

}  // namespace taichi::lang
