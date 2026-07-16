#pragma once

#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang {

class KernelLauncher {
 public:
  using Handle = KernelLaunchHandle;

  virtual void launch_kernel(const CompiledKernelData &compiled_kernel_data,
                             LaunchContextBuilder &ctx) = 0;

  // Cached Graph paths may already own a backend registration handle. The
  // default preserves non-LLVM backends; LLVM overrides this to avoid taking
  // the exclusive registration mutex on every steady-state dispatch.
  virtual void launch_registered_kernel(
      const CompiledKernelData &compiled_kernel_data,
      Handle handle,
      LaunchContextBuilder &ctx) {
    (void)handle;
    launch_kernel(compiled_kernel_data, ctx);
  }

  // Explicit SNodeTree destruction is a cold lifecycle transaction. Backends
  // may release registered executable state whose lowered kernels reference
  // |tree_id| after Program has drained host submissions and synchronized the
  // device. The default keeps backends without unload support source-compatible
  // and, importantly, never makes ordinary launch pay for lifecycle tracking.
  virtual void retire_snode_tree(int tree_id) {
  }

  virtual std::size_t debug_registered_kernel_count() {
    return 0;
  }

  virtual ~KernelLauncher() = default;
};

}  // namespace taichi::lang
