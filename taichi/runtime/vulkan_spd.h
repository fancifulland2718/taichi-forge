#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include "taichi/rhi/device.h"

#ifdef TI_WITH_VULKAN
namespace taichi::lang {
namespace vulkan {
class VulkanDevice;
}

// Shared cold SPD preparation. The caller owns image leases, queue ordering,
// counter initialization submission and retirement; no Program is required.
struct VulkanSpdCommands {
  std::function<void(CommandList *)> record_inline;
  std::function<void(CommandList *)> replay;
  std::function<void(CommandList *)> initialize;
  std::unordered_map<std::string, uint64_t> statistics;
};

VulkanSpdCommands prepare_vulkan_spd(vulkan::VulkanDevice &device,
                                     const ImageParams &source_parameters,
                                     DeviceAllocation source,
                                     const ImageParams &output_parameters,
                                     DeviceAllocation output,
                                     const std::vector<uint32_t> &shader,
                                     std::shared_ptr<void> image_owners,
                                     bool secondary_replay = true);
}  // namespace taichi::lang
#endif
