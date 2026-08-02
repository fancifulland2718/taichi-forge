#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/kernel_launcher.h"
#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

namespace taichi::lang {
namespace cuda {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct SparseListState {
    int64 dirty_epoch{0};
    int64 clean_epoch{-1};
    int64 clean_parent_version{-1};
    int64 version{0};
    std::uint64_t adaptive_window_bits{0};
    int adaptive_window_size{0};
    int adaptive_hit_count{0};
    bool adaptive_disabled{false};
  };

  struct Context {
    JITModule *jit_module{nullptr};
    std::vector<int> snode_tree_ids;
    std::vector<std::pair<std::vector<int>, Callable::Parameter>> parameters;
    std::vector<OffloadedTask> offloaded_tasks;
  };

 public:
  using Base::Base;

  struct GraphLaunchPacket {
    Handle handle;
    RuntimeContext context;
    void *device_arg_buffer{nullptr};
    std::size_t arg_buffer_size{0};
    std::size_t device_arg_buffer_size{0};
    std::size_t arg_buffer_prefix_size{0};
    std::string dispatch_label;
  };

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;
  // Register a private variant whose offloaded task entries uniformly test a
  // device gate before executing. The source kernel and ordinary registration
  // remain untouched, so only internal masked CUDA Graph replay pays the
  // branch cost.
  Handle register_llvm_kernel_graph_gated(
      const LLVM::CompiledKernelData &compiled);
  void retire_snode_tree(int tree_id) override;
  std::size_t debug_registered_kernel_count() override;
  void debug_reset_sparse_listgen_statistics() override;
  SparseSNodeTreeListgenStatistics debug_sparse_listgen_statistics(
      const std::vector<int> &snode_ids) override;
  bool prepare_cuda_graph_launch(Handle handle,
                                 LaunchContextBuilder &ctx,
                                 GraphLaunchPacket &packet,
                                 void *stream);
  bool prepare_cuda_graph_gated_launch(Handle handle,
                                       LaunchContextBuilder &ctx,
                                       GraphLaunchPacket &packet,
                                       void *gate,
                                       std::uint32_t expected,
                                       void *stream);
  bool update_cuda_graph_launch(const GraphLaunchPacket &packet,
                                LaunchContextBuilder &ctx,
                                std::vector<uint8_t> &host_arg_buffer,
                                void *stream);
  void capture_cuda_graph_launch(const GraphLaunchPacket &packet,
                                 void *stream);

 private:
  bool prepare_cuda_graph_context(Handle handle,
                                  LaunchContextBuilder &ctx,
                                  RuntimeContext &context);
  bool on_cuda_device(void *ptr);
  int64 get_sparse_list_version(int snode_id) const;
  void record_sparse_list_reuse_sample(SparseListState &state,
                                       bool would_skip) const;
  bool sparse_list_task_is_current(const OffloadedTask &task);
  void mark_sparse_list_task_launched(const OffloadedTask &task);
  void invalidate_sparse_list_cache(int sparse_mutation_snode_id);

  bool listgen_reuse_adaptive_{false};
  // Sparse-list reuse metadata describes one CUDA runtime, not one launch.
  // Dense kernels never take this lock. Sparse list generation/mutation holds
  // it only while checking state and enqueueing the corresponding task so
  // async host callers preserve submission order without a global launch lock.
  std::mutex sparse_list_mutex_;
  std::unordered_map<int, SparseListState> sparse_list_states_;
  bool sparse_listgen_telemetry_enabled_{false};
  std::unordered_map<int, SparseListgenNodeStatistics>
      sparse_listgen_telemetry_;
  std::unordered_map<int, std::shared_ptr<const Context>> contexts_;
};

}  // namespace cuda
}  // namespace taichi::lang
