#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace taichi::lang {

enum class CudaContextMode : std::uint32_t {
  disabled = 0,
  required = 1,
  automatic = 2,
};

// Runtime initialization facts, never compiler or artifact metadata. The owner
// is a native lease of BOTH the external queue and its Vulkan device.
struct CudaContextConfig {
  CudaContextMode mode{CudaContextMode::disabled};
  std::array<std::uint8_t, 16> device_uuid{};
  std::vector<std::uint8_t> queue_data;
  std::shared_ptr<void> owner;
};

struct CudaContextInfo {
  bool initialized{false};
  bool vulkan_cig{false};
  bool driver_entry{false};
  bool device_capability{false};
  std::uint64_t generation{0};
  std::array<std::uint8_t, 16> device_uuid{};
  // Successful cold native checks: device, managed, async, owned pool, capture.
  // These facts do not assert application qualification or a speedup.
  std::uint32_t allocation_checks{0};
  std::string fallback_reason;
};

}  // namespace taichi::lang
