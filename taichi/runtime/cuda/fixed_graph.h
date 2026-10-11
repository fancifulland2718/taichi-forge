#pragma once

#include <memory>
#include <vector>
#include "taichi/program/kernel_launcher.h"

namespace taichi::lang::cuda {
class KernelLauncher;

// Backend-owned fixed execution, independent of the Python/JIT Program. Source
// contexts are consumed only during construction. Registered code and bound
// device allocations must outlive this object and every submitted replay.
class FixedGraph {
 public:
  struct Dispatch {
    KernelLauncher *launcher;
    KernelLaunchHandle handle;
    LaunchContextBuilder *context;
  };
  explicit FixedGraph(const std::vector<Dispatch> &dispatches);
  ~FixedGraph();
  FixedGraph(const FixedGraph &) = delete;
  FixedGraph &operator=(const FixedGraph &) = delete;
  static bool available();
  void run();
  std::size_t argument_bytes() const;
  std::size_t node_count() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};
}  // namespace taichi::lang::cuda
