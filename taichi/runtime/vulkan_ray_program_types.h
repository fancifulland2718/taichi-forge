#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace taichi::lang {

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
}  // namespace taichi::lang
