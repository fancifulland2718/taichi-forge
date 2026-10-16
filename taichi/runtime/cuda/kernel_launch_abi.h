#pragma once

#include <cstdint>

namespace taichi::lang::cuda {

// The argument prefix belongs to the registered code variant, not to an
// individual caller's preference. Ordinary and gated code cannot share frames.
enum class KernelLaunchABI : std::uint8_t {
  ordinary,
  graph_gate_v1,
};

inline constexpr const char *kGraphGateLaunchABI = "cuda_graph_gate_v1";

}  // namespace taichi::lang::cuda
