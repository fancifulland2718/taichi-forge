#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "taichi/rhi/device.h"

namespace taichi::lang {
class Texture;
namespace storage {
class DenseStorageDescriptor;
}

// Internal marshaling structures. Public artifacts and physical identities are
// Python/Forge-owned; Vulkan reflection and resource lifetime remain native.
struct VulkanRayProgramShader {
  enum class Stage { raygen, miss, closest_hit, any_hit };
  Stage stage{Stage::raygen};
  std::vector<std::uint32_t> words;
  std::string entry{"main"};
};
struct VulkanRayProgramGroup {
  enum class Kind { raygen, miss, triangles };
  Kind kind{Kind::raygen};
  std::uint32_t general{UINT32_MAX};
  std::uint32_t closest_hit{UINT32_MAX};
  std::uint32_t any_hit{UINT32_MAX};
};
struct VulkanRayProgramRecord {
  std::uint32_t group{0};
  std::vector<std::uint8_t> data;
};
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
