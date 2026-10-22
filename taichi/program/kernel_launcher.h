#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>

#include "taichi/runtime/kernel_execution_handle.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/program/sparse_runtime_statistics.h"

namespace taichi::lang {

class KernelLauncher {
 public:
  using Handle = KernelLaunchHandle;

  KernelLauncher()
      : registration_owner_(
            next_registration_owner_.fetch_add(1, std::memory_order_relaxed)) {
    TI_ERROR_IF(
        registration_owner_ == 0 ||
            registration_owner_ == std::numeric_limits<std::uint64_t>::max(),
        "Kernel registration owner identity space exhausted");
  }

  virtual void launch_kernel(const CompiledKernelData &compiled_kernel_data,
                             const KernelRegistrationState &registration,
                             LaunchContextBuilder &ctx) = 0;

  // Prepared Graph and ordinary paths already own a backend registration.
  // Replay uses that exact handle without consulting payload registration
  // state or taking the exclusive registration mutex again.
  virtual void launch_registered_kernel(
      const CompiledKernelData &compiled_kernel_data,
      Handle handle,
      LaunchContextBuilder &ctx) = 0;

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

  virtual void debug_reset_sparse_listgen_statistics() {
  }

  virtual SparseSNodeTreeListgenStatistics
  debug_sparse_listgen_statistics(const std::vector<int> &snode_ids, int tree_id = -1) {
    (void)snode_ids;
    (void)tree_id;
    return {};
  }

  virtual void debug_reset_launch_attribution() {
  }

  virtual std::unordered_map<std::string, std::uint64_t>
  debug_launch_attribution() const {
    return {};
  }

  virtual ~KernelLauncher() = default;

 protected:
  void bind_registration(const KernelRegistrationState &registration) const {
    registration.bind_runtime(registration_owner_, registration_generation_);
  }

  void advance_registration_generation() {
    TI_ERROR_IF(
        registration_generation_ == std::numeric_limits<std::uint64_t>::max(),
        "Kernel registration generation space exhausted");
    ++registration_generation_;
  }

 private:
  inline static std::atomic<std::uint64_t> next_registration_owner_{1};
  const std::uint64_t registration_owner_;
  std::uint64_t registration_generation_{1};
};

}  // namespace taichi::lang
