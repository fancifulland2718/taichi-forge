#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "taichi/util/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/offloaded_task_type.h"
#include "taichi/rhi/arch.h"
#include "taichi/program/callable.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/aot/graph_data.h"
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang {

class Program;

class TI_DLL_EXPORT Kernel : public Callable {
 public:
  enum class SNodeTreeDependencyState : std::uint8_t {
    unknown = 0,
    none = 1,
    present = 2,
  };

  enum class TaskLaunchPolicyMode {
    hint,
    require,
  };

  enum class TaskLaunchThreadLocalMode {
    automatic,
    enabled,
    disabled,
  };

  struct TaskLaunchPolicy {
    TaskLaunchPolicyMode mode{TaskLaunchPolicyMode::hint};
    TaskLaunchThreadLocalMode thread_local_mode{
        TaskLaunchThreadLocalMode::automatic};
    int block_dim{0};
    int cuda_min_blocks_per_sm{2};
    // -1 inherits CompileConfig::gpu_max_reg. Zero requests the CUDA
    // compiler/driver default, and a positive value is an explicit cap.
    int cuda_max_registers{-1};
    bool injected_block_dim{false};
    std::string optimization_spec_identity;
  };

  struct KernelOptimizationSpec {
    TaskLaunchThreadLocalMode thread_local_mode{
        TaskLaunchThreadLocalMode::automatic};
    int cuda_min_blocks_per_sm{2};
    // -1 inherits CompileConfig::gpu_max_reg. Zero requests the CUDA
    // compiler/driver default, and a positive value is an explicit cap.
    int cuda_max_registers{-1};
    std::string identity;
  };

  // Private Forge execution-plan metadata.  Unlike KernelOptimizationSpec,
  // this is complete and indexed by the physical offload ordinal.  The
  // compilation identity deliberately excludes launch-only controls so CUDA
  // code can be shared by multiple immutable launch plans.
  struct OffloadTaskOptimizationSpec {
    std::uint32_t task_index{0};
    std::string task_kind;
    int workgroup_size{0};  // zero inherits the compiler-selected value
    TaskLaunchThreadLocalMode thread_local_mode{
        TaskLaunchThreadLocalMode::automatic};
    int cuda_min_blocks_per_sm{2};
    // -1 inherits CompileConfig::gpu_max_reg. Zero requests the CUDA
    // compiler/driver default, and a positive value is an entry-specific cap.
    int cuda_max_registers{-1};
    int grid_residency_waves{0};  // zero is automatic
    int range_work_per_thread_target{1};
    std::string sparse_list_policy{"saturating"};
    std::string memory_strategy{"direct"};
    // Empty selects every eligible read-only stencil input. Otherwise this
    // is the canonical top-level external-argument subset staged by the
    // complete GraphMemory recipe.
    std::vector<int> memory_source_arg_indices;
    // Complete compiler-owned physical policy for a canonical flattened 2-D
    // range. Empty for direct and one-dimensional shared staging.
    std::vector<int> memory_domain_shape;
    std::vector<int> memory_domain_origin;
    std::vector<int> memory_tile_shape;
  };

  struct OffloadExecutionPlan {
    std::string compilation_identity;
    std::string execution_identity;
    // The source topology is checked immediately after offloading. ``tasks``
    // describes the physical topology after the optional compiler-owned
    // pointwise fusion transform and is the only topology seen by codegen and
    // launch-context construction.
    std::vector<OffloadTaskOptimizationSpec> source_tasks;
    std::vector<OffloadTaskOptimizationSpec> tasks;
    std::vector<std::vector<int>> fusion_groups;
    // Launch-only projections are materialized once with the immutable plan.
    // LaunchContextBuilder borrows these vectors instead of rebuilding and
    // validating them for every kernel invocation.
    std::vector<std::string> task_kinds;
    std::vector<int> grid_residency_waves;
    std::vector<int> range_work_per_thread_targets;
    // 0 keeps the hardware-saturating listgen grid; 1 applies the exact
    // compiler-proven parent-list capacity bound.
    std::vector<int> sparse_list_policies;
    LaunchContextBuilder::CudaTaskExecutionPlanDigest launch_content_digest{};
  };

  std::vector<SNode *> no_activate;

  bool is_accessor{false};

  Kernel(Program &program,
         const std::function<void()> &func,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  Kernel(Program &program,
         const std::function<void(Kernel *)> &func,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  Kernel(Program &program,
         std::unique_ptr<IRNode> &&ir,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  bool ir_is_ast() const {
    return ir_is_ast_;
  }

  LaunchContextBuilder make_launch_context(bool cpu_bounded_range = false);

  template <typename T>
  T fetch_ret(DataType dt, int i);

  [[nodiscard]] std::string get_name() const override;

  void set_kernel_key_for_cache(const std::string &kernel_key) const;

  const std::string &get_cached_kernel_key() const;

  void invalidate_kernel_key_for_cache() const;

  bool has_cached_offline_cache_body() const;

  const std::string &get_cached_offline_cache_body() const;

  void set_offline_cache_body(std::string body) const;

  // P-Compile-6: per-kernel compile_tier override.
  // When set, takes precedence over CompileConfig::compile_tier for this
  // kernel only. Empty optional = use program-level compile_tier (default).
  // Valid values: "fast", "balanced", "full". Invalid values are rejected at
  // the Python boundary; C++ side stores the string verbatim.
  void set_compile_tier_override(const std::string &tier);

  void clear_compile_tier_override();

  const std::optional<std::string> &get_compile_tier_override() const;

  // N1: immutable per-Kernel launch-policy metadata. Python materializes a
  // distinct Kernel for every policy specialization, so this is configured
  // before the first compilation and is never mutated on a warm launch.
  void set_task_launch_policy(
      const std::string &mode,
      int block_dim,
      bool injected_block_dim,
      const std::string &optimization_spec_identity,
      const std::string &thread_local_mode,
      int cuda_min_blocks_per_sm,
      int cuda_max_registers);

  const std::optional<TaskLaunchPolicy> &get_task_launch_policy() const;

  std::string task_launch_policy_cache_key() const;

  // Kernel-wide immutable optimization metadata. Unlike TaskLaunchPolicy,
  // this does not assume that the frontend contains exactly one range-for.
  void set_kernel_optimization_spec(
      const std::string &identity,
      const std::string &thread_local_mode,
      int cuda_min_blocks_per_sm,
      int cuda_max_registers);

  const std::optional<KernelOptimizationSpec> &get_kernel_optimization_spec()
      const;

  std::string optimization_spec_cache_key() const;

  const std::string &optimization_spec_identity() const;

  void set_offload_execution_plan(
      const std::string &compilation_identity,
      const std::string &execution_identity,
      const std::vector<int> &task_indices,
      const std::vector<std::string> &task_kinds,
      const std::vector<int> &workgroup_sizes,
      const std::vector<std::string> &thread_local_modes,
      const std::vector<int> &cuda_min_blocks_per_sm,
      const std::vector<int> &cuda_max_registers,
      const std::vector<int> &grid_residency_waves,
      const std::vector<int> &range_work_per_thread_targets,
      const std::vector<std::string> &sparse_list_policies,
      const std::vector<std::string> &memory_strategies,
      const std::vector<std::vector<int>> &memory_source_arg_indices,
      const std::vector<std::vector<int>> &memory_domain_shapes,
      const std::vector<std::vector<int>> &memory_domain_origins,
      const std::vector<std::vector<int>> &memory_tile_shapes,
      const std::vector<std::vector<int>> &fusion_groups);

  const std::optional<OffloadExecutionPlan> &get_offload_execution_plan()
      const;

  const OffloadTaskOptimizationSpec &offload_task_optimization_spec(
      std::size_t task_index,
      OffloadedTaskType task_type) const;

  const std::vector<int> &snode_tree_dependencies() const {
    return snode_tree_dependencies_;
  }

  void set_snode_tree_dependencies(
      const std::vector<int> &dependencies) const;

  SNodeTreeDependencyState snode_tree_dependency_state() const noexcept {
    return snode_tree_dependency_state_.load(std::memory_order_acquire);
  }

  bool snode_tree_dependencies_known() const noexcept {
    return snode_tree_dependency_state() != SNodeTreeDependencyState::unknown;
  }

  bool has_snode_tree_dependencies() const noexcept {
    return snode_tree_dependency_state() ==
           SNodeTreeDependencyState::present;
  }

  bool definition_retired() const {
    return ir == nullptr;
  }

  // Preserve the Kernel object's address for stale Graph/AOT plans while
  // releasing the frontend/lowered IR and specialization metadata that can no
  // longer be compiled after an explicitly bound SNodeTree is destroyed.
  void retire_definition(bool preserve_relocatable_abi = false);

 private:
  void init(Program &program,
            const std::function<void()> &func,
            const std::string &name = "",
            AutodiffMode autodiff_mode = AutodiffMode::kNone);

  // True if |ir| is a frontend AST. False if it's already offloaded to CHI IR.
  bool ir_is_ast_{false};
  mutable std::string kernel_key_;
  mutable bool kernel_key_valid_{false};
  mutable std::optional<std::string> offline_cache_body_;
  std::optional<std::string> compile_tier_override_;
  std::optional<TaskLaunchPolicy> task_launch_policy_;
  std::optional<KernelOptimizationSpec> kernel_optimization_spec_;
  std::optional<OffloadExecutionPlan> offload_execution_plan_;
  std::mutex offload_execution_plan_mutex_;
  std::atomic<bool> offload_execution_plan_frozen_{false};
  mutable std::mutex snode_tree_dependencies_mutex_;
  mutable std::atomic<SNodeTreeDependencyState> snode_tree_dependency_state_{
      SNodeTreeDependencyState::unknown};
  mutable std::vector<int> snode_tree_dependencies_;
};

}  // namespace taichi::lang
