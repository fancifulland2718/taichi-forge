#pragma once

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
    JITModule *jit_module{nullptr};
    std::vector<int> snode_tree_ids;
    std::vector<TaskFunc> task_funcs;
    std::vector<std::pair<std::vector<int>, Callable::Parameter>> parameters;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;
  void retire_snode_tree(int tree_id) override;
  std::size_t debug_registered_kernel_count() override;

 private:
  // A CPU kernel can contain multiple offloaded tasks that share LLVM runtime
  // scratch/list state. The ThreadPool queues individual parallel-for jobs,
  // not whole kernels, so two host callers could otherwise interleave those
  // tasks. Serialize at the kernel boundary while retaining full parallelism
  // inside each kernel and asynchronous producer/render host threads.
  std::mutex execution_mutex_;
  std::unordered_map<int, std::shared_ptr<const Context>> contexts_;
};

}  // namespace cpu
}  // namespace taichi::lang
