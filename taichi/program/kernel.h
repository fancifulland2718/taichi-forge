#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include "taichi/util/lang_util.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/ir.h"
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

  struct TaskLaunchPolicy {
    TaskLaunchPolicyMode mode{TaskLaunchPolicyMode::hint};
    int block_dim{0};
    bool injected_block_dim{false};
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

  LaunchContextBuilder make_launch_context();

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
  void set_task_launch_policy(const std::string &mode,
                              int block_dim,
                              bool injected_block_dim);

  const std::optional<TaskLaunchPolicy> &get_task_launch_policy() const;

  std::string task_launch_policy_cache_key() const;

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
  void retire_definition();

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
  mutable std::mutex snode_tree_dependencies_mutex_;
  mutable std::atomic<SNodeTreeDependencyState> snode_tree_dependency_state_{
      SNodeTreeDependencyState::unknown};
  mutable std::vector<int> snode_tree_dependencies_;
};

}  // namespace taichi::lang
