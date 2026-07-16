#pragma once

#include <mutex>

#include "taichi/program/kernel_launcher.h"
#include "taichi/runtime/gfx/runtime.h"

namespace taichi::lang {
namespace gfx {

class KernelLauncher : public lang::KernelLauncher {
 public:
  struct Config {
    GfxRuntime *gfx_runtime_{nullptr};
  };

  explicit KernelLauncher(Config config);

  void launch_kernel(const lang::CompiledKernelData &compiled_kernel_data,
                     LaunchContextBuilder &ctx) override;
  void retire_snode_tree(int tree_id) override;
  std::size_t debug_registered_kernel_count() override;

  Handle get_or_register_kernel(
      const lang::CompiledKernelData &compiled_kernel_data);

  GfxRuntime *runtime() const {
    return config_.gfx_runtime_;
  }

 private:
  Handle register_kernel(const lang::CompiledKernelData &compiled_kernel_data);

  Config config_;
  // CompiledKernelData caches a mutable launch handle. Its check/register/set
  // sequence and the runtime kernel table must be one atomic host operation.
  std::mutex registration_mutex_;
};

}  // namespace gfx
}  // namespace taichi::lang
