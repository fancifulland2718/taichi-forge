#include "taichi/runtime/llvm/kernel_launcher.h"
#include "taichi/system/profiler.h"

namespace taichi::lang {
namespace LLVM {

KernelLauncher::KernelLauncher(Config config) : config_(std::move(config)) {
}

void KernelLauncher::launch_kernel(
    const lang::CompiledKernelData &compiled_kernel_data,
    LaunchContextBuilder &ctx) {
  TI_AUTO_PROF;
  TI_ASSERT(arch_uses_llvm(compiled_kernel_data.arch()));
  const auto &llvm_ckd =
      dynamic_cast<const LLVM::CompiledKernelData &>(compiled_kernel_data);
  KernelLauncher::Handle handle;
  {
    TI_PROFILER("register_llvm_kernel");
    handle = register_llvm_kernel(llvm_ckd);
  }
  {
    TI_PROFILER("launch_llvm_kernel");
    launch_llvm_kernel(handle, ctx);
  }
}

}  // namespace LLVM
}  // namespace taichi::lang
