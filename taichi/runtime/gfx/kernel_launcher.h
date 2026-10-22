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
                     const KernelRegistrationState &registration,
                     LaunchContextBuilder &ctx) override;
  void launch_registered_kernel(const lang::CompiledKernelData &,
                                Handle handle,
                                LaunchContextBuilder &ctx) override {
    config_.gfx_runtime_->launch_kernel(handle, ctx);
  }

  void retire_snode_tree(int tree_id) override;
  std::size_t debug_registered_kernel_count() override;
  void debug_reset_sparse_listgen_statistics() override;
  SparseSNodeTreeListgenStatistics debug_sparse_listgen_statistics(
      const std::vector<int> &snode_ids, int tree_id = -1) override;

  Handle get_or_register_kernel(
      const lang::CompiledKernelData &compiled_kernel_data,
      const KernelRegistrationState &registration);

  GfxRuntime *runtime() const {
    return config_.gfx_runtime_;
  }

 private:
  Handle register_kernel(const lang::CompiledKernelData &compiled_kernel_data,
                         const KernelRegistrationState &registration);

  Config config_;
};

}  // namespace gfx
}  // namespace taichi::lang
