#pragma once

#include <cstdint>
#include <string_view>

namespace taichi::lang::cuda {

// The argument prefix belongs to the registered code variant, not to an
// individual caller's preference. Ordinary and gated code cannot share frames.
enum class KernelLaunchABI : std::uint8_t {
  ordinary,
  graph_gate_v1,
};

inline constexpr const char *kGraphGateLaunchABI = "cuda_graph_gate_v1";
inline constexpr const char *kContextLaunchABI = "runtime_context_v2";
inline constexpr const char *kGraphGateContextLaunchABI =
    "cuda_graph_gate_context_v2";

// AOT declares the full call contract; the in-memory gate enum continues to
// describe only the argument prefix. Context layout must not change a recipe's
// gate or ordinary execution semantics.
inline bool is_graph_gate_launch_abi(std::string_view abi) {
  return abi == kGraphGateLaunchABI || abi == kGraphGateContextLaunchABI;
}

inline unsigned launch_runtime_context_abi(std::string_view abi) {
  if (abi == kContextLaunchABI || abi == kGraphGateContextLaunchABI)
    return 2;
  if (abi == "ordinary" || abi == kGraphGateLaunchABI)
    return 1;
  return 0;
}

inline bool is_declared_launch_abi(std::string_view abi) {
  return abi != "ordinary" && launch_runtime_context_abi(abi) != 0;
}

}  // namespace taichi::lang::cuda
