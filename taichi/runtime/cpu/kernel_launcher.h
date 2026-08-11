#pragma once

#include <atomic>
#include <memory>
#include <unordered_map>

#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/kernel_launcher.h"

namespace taichi::lang {
namespace cpu {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct Context {
    using TaskFunc = int32 (*)(void *);
    struct SparseTaskMetadata {
      int sparse_list_op{OffloadedTask::kSparseListOpNone};
      int snode_id{-1};
      int parent_snode_id{-1};
    };
    JITModule *jit_module{nullptr};
    std::vector<int> snode_tree_ids;
    std::vector<TaskFunc> task_funcs;
    std::vector<OffloadedTaskType> task_types;
    std::vector<std::pair<std::string, std::string>> task_trace_metadata;
    std::vector<SparseTaskMetadata> sparse_task_metadata;
    std::vector<std::pair<std::vector<int>, Callable::Parameter>> parameters;
  };

 public:
  explicit KernelLauncher(Config config);

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;
  void retire_snode_tree(int tree_id) override;
  std::size_t debug_registered_kernel_count() override;
  void debug_reset_sparse_listgen_statistics() override;
  SparseSNodeTreeListgenStatistics debug_sparse_listgen_statistics(
      const std::vector<int> &snode_ids) override;
  void debug_reset_launch_attribution() override;
  std::unordered_map<std::string, std::uint64_t> debug_launch_attribution()
      const override;

 private:
  // A CPU kernel can contain multiple offloaded tasks that share LLVM runtime
  // scratch/list state. The ThreadPool queues individual parallel-for jobs,
  // not whole kernels, so two host callers could otherwise interleave those
  // tasks. Serialize at the kernel boundary while retaining full parallelism
  // inside each kernel and asynchronous producer/render host threads.
  std::mutex execution_mutex_;
  std::unordered_map<int, std::shared_ptr<const Context>> contexts_;
  bool sparse_listgen_telemetry_enabled_{false};
  std::unordered_map<int, SparseListgenNodeStatistics>
      sparse_listgen_telemetry_;
  struct LaunchAttributionCounters {
    bool enabled{false};
    std::atomic<std::uint64_t> launches{0};
    std::atomic<std::uint64_t> launch_wall_ns{0};
    std::atomic<std::uint64_t> launch_cpu_ns{0};
    std::atomic<std::uint64_t> execution_lock_wait_ns{0};
    std::atomic<std::uint64_t> execution_lock_hold_ns{0};
    std::atomic<std::uint64_t> registration_lock_wait_ns{0};
    std::atomic<std::uint64_t> registration_lock_hold_ns{0};
    std::atomic<std::uint64_t> context_lookup_ns{0};
    std::atomic<std::uint64_t> argument_binding_ns{0};
    std::atomic<std::uint64_t> task_execution_ns{0};
    std::atomic<std::uint64_t> task_invocations{0};
    std::atomic<std::uint64_t> serial_task_invocations{0};
    std::atomic<std::uint64_t> serial_task_execution_ns{0};
    std::atomic<std::uint64_t> range_task_invocations{0};
    std::atomic<std::uint64_t> range_task_execution_ns{0};
    std::atomic<std::uint64_t> other_task_invocations{0};
    std::atomic<std::uint64_t> other_task_execution_ns{0};
    std::atomic<std::uint64_t> labeled_launches{0};
    std::atomic<std::uint64_t> sparse_telemetry_launches{0};
    std::atomic<std::uint64_t> compile_profiler_enabled_launches{0};
  } launch_attribution_;
};

}  // namespace cpu
}  // namespace taichi::lang
