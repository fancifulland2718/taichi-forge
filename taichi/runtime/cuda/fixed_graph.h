#pragma once

#include <memory>
#include <functional>
#include <optional>
#include <variant>
#include <vector>
#include "taichi/program/kernel_launcher.h"
#include "taichi/program/runtime_completion.h"
#include "taichi/runtime/cuda/capture_resources.h"
#include "taichi/runtime/branch_selection.h"
#include "taichi/runtime/cuda_control_recipe.h"
#include "taichi/runtime/submission_parameters.h"

namespace taichi::lang {
class RuntimeFaultDomain;
class RuntimeClient;
class FixedExecution;
}
namespace taichi::lang::cuda {
class KernelLauncher;
class ExecutionDomain;
class ExecutionInstance;

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
    std::vector<SubmissionParameterPatch> parameters;
    // Cold, exact indices in the compiled task sequence. Flat ordinary graphs
    // use their runtime workspace; independent instances use private storage.
    std::vector<std::size_t> device_range_tasks;
  };
  struct While;
  struct Branch;
  struct Parallel;
  using Operation = std::variant<Dispatch,
                                 std::shared_ptr<const Branch>,
                                 std::shared_ptr<const While>,
                                 std::shared_ptr<const Parallel>>;
  struct Parallel {
    std::vector<std::vector<Operation>> branches;
  };
  struct Branch {
    void *selector{nullptr};
    BranchSelection selection;
    std::vector<Operation> condition;
    std::vector<std::vector<Operation>> bodies;
    BranchExecutionMode mode{BranchExecutionMode::native};
  };
  struct While {
    void *predicate{nullptr};
    std::uint64_t max_iterations{0};
    bool continue_while_nonzero{true};
    std::vector<Operation> condition;
    std::vector<Operation> body;
    // Explicit physical selection; never replaced by conditional nodes.
    std::optional<CudaControlRecipe<void *>> recipe;
  };
  explicit FixedGraph(const std::vector<Dispatch> &dispatches,
                      std::shared_ptr<RuntimeFaultDomain> fault_domain = {});
  explicit FixedGraph(const std::vector<Operation> &operations,
                      std::shared_ptr<RuntimeFaultDomain> fault_domain = {});
  ~FixedGraph();
  FixedGraph(const FixedGraph &) = delete;
  FixedGraph &operator=(const FixedGraph &) = delete;
  static bool available();
  // Cold preparation of an independent fixed execution. Registered code and
  // every bound allocation/provider are retained by operands. The caller
  // supplies the selected recipe and its shared-storage hazard contract;
  // this object isolates arguments, compiler temporaries and executable
  // control storage without changing that recipe or recompiling its code.
  static std::unique_ptr<FixedGraph> prepare_execution_instance(
      const std::vector<Operation> &operations,
      std::shared_ptr<RuntimeFaultDomain> owner,
      std::shared_ptr<ExecutionDomain> domain,
      std::shared_ptr<RuntimeCompletionResources> operands);
  RuntimeCompletion submit(
      const std::vector<RuntimeCompletion> &dependencies = {});
  // Cold, explicit retirement. Waits this instance's last replay, then releases
  // its native resources and domain budget. Driver/provider frees can impose
  // additional synchronization; wrapper destruction keeps its existing policy.
  void retire_execution_instance();
  struct InstanceSnapshot {
    std::uint64_t identity;
    std::uintptr_t temporary_storage;
    std::size_t temporary_bytes;
    bool retired;
  };
  InstanceSnapshot instance_snapshot() const;
  // Capture a flat immutable source without instantiating an executable.
  // The replay family retains this object and its operands through completion.
  static std::shared_ptr<FixedGraph> capture_binding_frame(
      const std::vector<Operation> &operations,
      std::shared_ptr<void> operands,
      std::shared_ptr<RuntimeFaultDomain> fault_domain = {});
  void *binding_source() const;
  // Transfer a fresh native-only retained recording to an enclosing domain.
  // No Taichi argument/temporary packet may enter through this adapter. A
  // recording already replayed on the legacy stream must be prepared anew.
  void validate_retained_provider() const;
  void adopt_retained_provider(
      const std::shared_ptr<RuntimeFaultDomain> &owner,
      std::vector<std::shared_ptr<RuntimeClient>> &parts);
  void enqueue_retained_provider(void *stream);
  void run();
  std::size_t argument_bytes() const;
  std::size_t node_count() const;
  std::size_t control_bytes() const;
  std::size_t capture_bytes() const;

 private:
  friend class taichi::lang::FixedExecution;
  // Parts are registered by one enclosing recipe before any backend work.
  // They have no domain owner of their own, preventing ownership cycles and
  // recursive admission. Only the enclosing domain may enqueue these parts.
  static std::unique_ptr<FixedGraph> prepare_execution_part(
      const std::vector<Operation> &operations,
      std::shared_ptr<RuntimeFaultDomain> owner,
      std::shared_ptr<RuntimeCompletionResources> operands,
      std::vector<std::shared_ptr<RuntimeClient>> &parts,
      ExecutionDomain &allocation_domain);
  void enqueue_execution_part(void *stream);
  void snapshot_parameters(const std::vector<SubmissionParameterValue> &values);
  void enqueue_parameter_upload(void *stream);
  FixedGraph(const std::vector<Operation> &operations,
             std::shared_ptr<RuntimeFaultDomain> fault_domain,
             std::shared_ptr<void> operands,
             bool binding_frame,
             std::shared_ptr<ExecutionDomain> domain = {},
             std::shared_ptr<RuntimeCompletionResources> instance_operands = {},
             std::vector<std::shared_ptr<RuntimeClient>> *parts = nullptr,
             ExecutionDomain *allocation_domain = nullptr);
  void initialize(const std::vector<Operation> &operations,
                  ExecutionDomain *allocation_domain = nullptr);
  struct State;
  std::shared_ptr<State> state_;
  std::shared_ptr<ExecutionDomain> domain_;
  std::shared_ptr<ExecutionInstance> instance_;
};
}  // namespace taichi::lang::cuda
