#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "taichi/rhi/device.h"

namespace taichi::lang {
namespace vulkan {
class VulkanDevice;
}

// Compact row-major, in-place complex f32 storage. Zero batch_tile retains
// the adapter's ordinary plan; a selected nonzero tile is a physical choice.
struct VulkanFftPlanConfig {
  std::vector<std::uint64_t> dimensions;
  std::uint64_t batches{1};
  int direction{-1};
  bool normalize_inverse{false};
  std::uint64_t batch_tile{0};
};

// Shared execution core, independent of Program handles, Ndarray and Python.
// Creation may initialize LUTs on the ordered queue; it never transforms the
// input. Recording retains the plan, adapter and storage in the command list.
// The caller owns the device, serializes recording and supplies surrounding
// barriers, and retires all command buffers before destroying the device.
class VulkanFftPlanCore {
 public:
  virtual ~VulkanFftPlanCore() = default;
  virtual bool supports_inline_recording() const = 0;
  // prefer_inline=false appends the adapter's retained secondary sequence.
  // Inline recording is required inside an enclosing secondary command list.
  virtual void record(CommandList *commands, bool prefer_inline = true) = 0;
  virtual std::unordered_map<std::string, std::uint64_t> statistics() const = 0;
};

std::shared_ptr<VulkanFftPlanCore> make_vulkan_fft_plan(
    vulkan::VulkanDevice &device,
    DeviceAllocation storage,
    std::size_t storage_bytes,
    std::shared_ptr<void> storage_lease,
    const VulkanFftPlanConfig &config,
    const std::string &adapter_path);

}  // namespace taichi::lang
