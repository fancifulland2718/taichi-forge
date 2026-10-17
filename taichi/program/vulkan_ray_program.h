#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "taichi/rhi/device.h"
#include "taichi/runtime/vulkan_ray_program_types.h"

namespace taichi::lang {
class Texture;
namespace storage {
class DenseStorageDescriptor;
}

// JIT-only owner-bearing argument adapters.
struct VulkanRayProgramBuffer {
  std::uint32_t set{0};
  std::uint32_t binding{0};
  const storage::DenseStorageDescriptor *storage{nullptr};
  bool uniform{false};
  bool writable{false};
};
struct VulkanRayProgramImage {
  std::uint32_t set{0};
  std::uint32_t binding{0};
  Texture *texture{nullptr};
  bool storage{false};
  std::uint32_t mip_level{0};
};
struct VulkanRayProgramAS {
  std::uint32_t set{0};
  std::uint32_t binding{0};
  std::uint64_t handle{0};
};
struct VulkanRayProgramLaunch {
  std::array<std::uint32_t, 3> dimensions{1, 1, 1};
  VulkanRayProgramRecord raygen;
  std::vector<VulkanRayProgramRecord> miss;
  std::vector<VulkanRayProgramRecord> hit;
  std::vector<VulkanRayProgramBuffer> buffers;
  std::vector<VulkanRayProgramImage> images;
  std::vector<VulkanRayProgramAS> scenes;
  std::vector<std::uint8_t> push_constants;
};
struct PreparedVulkanRayLaunch {
  struct State;
  std::shared_ptr<State> state;
};
class VulkanRayProgramResource;
}  // namespace taichi::lang
