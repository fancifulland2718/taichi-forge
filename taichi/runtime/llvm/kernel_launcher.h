#pragma once

#include <mutex>
#include <shared_mutex>

#include "taichi/program/kernel_launcher.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"

namespace taichi::lang {
namespace LLVM {

class KernelLauncher : public lang::KernelLauncher {
 public:
  struct Config {
    LlvmRuntimeExecutor *executor{nullptr};
  };

  explicit KernelLauncher(Config config);

  void launch_kernel(const lang::CompiledKernelData &compiled_kernel_data,
                     LaunchContextBuilder &ctx) override;

  virtual void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) = 0;
  virtual Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) = 0;

 protected:
  Handle make_handle() {
    Handle handle;
    handle.set_launch_id(launch_id_counter_++);
    return handle;
  }

  LlvmRuntimeExecutor *get_runtime_executor() {
    return config_.executor;
  }

  std::shared_mutex &registration_mutex() {
    return registration_mutex_;
  }

 private:
  Config config_;
  int launch_id_counter_{0};
  // Registration/JIT construction mutates shared executor state and therefore
  // takes this mutex exclusively. Already-registered launches take it shared,
  // preserving concurrent simulation/render submission without allowing a
  // module or context table to change underneath an active launch.
  std::shared_mutex registration_mutex_;
};

}  // namespace LLVM
}  // namespace taichi::lang
