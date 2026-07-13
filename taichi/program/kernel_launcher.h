#pragma once

#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang {

class KernelLauncher {
 public:
  using Handle = KernelLaunchHandle;

  virtual void launch_kernel(const CompiledKernelData &compiled_kernel_data,
                             LaunchContextBuilder &ctx) = 0;

  // Explicit SNodeTree destruction is a cold lifecycle transaction. Backends
  // may release registered executable state whose lowered kernels reference
  // |tree_id| after Program has drained host submissions and synchronized the
  // device. The default keeps backends without unload support source-compatible
  // and, importantly, never makes ordinary launch pay for lifecycle tracking.
  virtual void retire_snode_tree(int tree_id) {
  }

  virtual ~KernelLauncher() = default;
};

}  // namespace taichi::lang
