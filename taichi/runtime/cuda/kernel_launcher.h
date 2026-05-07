#pragma once

#include <unordered_map>

#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/kernel_launcher.h"

namespace taichi::lang {
namespace cuda {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct SparseListState {
    int64 dirty_epoch{0};
    int64 clean_epoch{-1};
    int64 clean_parent_version{-1};
    int64 version{0};
  };

  struct Context {
    JITModule *jit_module{nullptr};
    std::vector<std::pair<std::vector<int>, Callable::Parameter>> parameters;
    std::vector<OffloadedTask> offloaded_tasks;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;

 private:
  bool on_cuda_device(void *ptr);
  int64 get_sparse_list_version(int snode_id) const;
  bool sparse_list_task_is_current(const OffloadedTask &task) const;
  void mark_sparse_list_task_launched(const OffloadedTask &task);
  void invalidate_sparse_list_cache(int sparse_mutation_snode_id);

  std::unordered_map<int, SparseListState> sparse_list_states_;
  std::vector<Context> contexts_;
};

}  // namespace cuda
}  // namespace taichi::lang
