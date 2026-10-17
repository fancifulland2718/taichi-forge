#pragma once

#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang::vulkan {

// Opt-in diagnostic state lives on the construction stack, not in the pipeline
// ABI. Events are flushed before driver calls so an interrupted process leaves
// an honest last-known phase, without inventing a VkResult.
class VulkanPipelineDiagnostics {
 public:
  explicit VulkanPipelineDiagnostics(const VulkanPipeline::Params &params);
  void event(const std::string &phase,
             const std::string &message = "",
             std::optional<int64_t> result = std::nullopt);

 private:
  std::string directory_;
};

std::string vulkan_device_identity_json(const VulkanDevice &device);

}  // namespace taichi::lang::vulkan
