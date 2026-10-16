#pragma once

#include <memory>
#include <functional>
#include <variant>
#include <vector>
#include "taichi/program/kernel_launcher.h"
#include "taichi/runtime/cuda/capture_resources.h"

namespace taichi::lang {
class RuntimeFaultDomain;
}
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
    // Cold capture callback only; never retained or invoked during replay.
    std::function<void(void *)> external;
    std::function<std::shared_ptr<CudaGraphCaptureResources>()> take_resources;
    void *bounded_extent{nullptr};
    std::uint32_t bounded_capacity{0};
    std::uint32_t adaptive_block_dim{0};
    bool grouped_update{false};
    const void *bounded_binding_identity{nullptr};
  };
  struct While;
  using Operation = std::variant<Dispatch, std::shared_ptr<const While>>;
  struct While {
    void *predicate{nullptr};
    std::uint64_t max_iterations{0};
    bool continue_while_nonzero{true};
    std::vector<Operation> condition;
    std::vector<Operation> body;
  };
  explicit FixedGraph(const std::vector<Dispatch> &dispatches,
                      std::shared_ptr<RuntimeFaultDomain> fault_domain = {});
  explicit FixedGraph(const std::vector<Operation> &operations,
                      std::shared_ptr<RuntimeFaultDomain> fault_domain = {});
  ~FixedGraph();
  FixedGraph(const FixedGraph &) = delete;
  FixedGraph &operator=(const FixedGraph &) = delete;
  static bool available();
  void run();
  std::size_t argument_bytes() const;
  std::size_t node_count() const;
  std::size_t control_bytes() const;
  std::size_t capture_bytes() const;

 private:
  void initialize(const std::vector<Operation> &operations);
  struct State;
  std::unique_ptr<State> state_;
};
}  // namespace taichi::lang::cuda
