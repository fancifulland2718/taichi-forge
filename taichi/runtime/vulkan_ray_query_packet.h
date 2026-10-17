#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include "taichi/rhi/device.h"

namespace taichi::lang {
// Cold-qualified device bindings. Lifetime/generation checks belong to the
// enclosing JIT or AOT owner rather than the shader packet.
struct VulkanRayQueryPacket {
  struct Binding {
    DevicePtr pointer;
    std::size_t bytes;
  };
  std::size_t ray_count{0};
  unsigned variant{0};  // bit 0: typed hits, bit 1: scalar subranges
  std::array<Binding, 3> shader_bindings{};
  std::array<std::uint32_t, 4> parameters{};
};
}  // namespace taichi::lang
