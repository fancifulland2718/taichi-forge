#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>

#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/runtime/kernel_launch_handle.h"

namespace taichi::lang {

// Runtime registration belongs to an execution owner, never to code bytes.
// Callers use the launcher's existing registration/lifecycle gate. No global
// payload-address lookup or extra synchronization is required on replay.
class KernelRegistrationState {
 public:
  KernelRegistrationState() = default;
  KernelRegistrationState(const KernelRegistrationState &) = delete;
  KernelRegistrationState &operator=(const KernelRegistrationState &) = delete;

  void bind_runtime(std::uint64_t owner, std::uint64_t generation) const {
    TI_ERROR_IF(retired_, "Kernel execution registration has retired");
    TI_ERROR_IF(owner_ != 0 && (owner_ != owner || generation_ != generation),
                "Kernel execution registration belongs to another runtime "
                "or a retired registration generation");
    owner_ = owner;
    generation_ = generation;
  }

  void set_handle(KernelLaunchHandle handle) const {
    ordinary_ = handle;
  }
  const std::optional<KernelLaunchHandle> &get_handle() const {
    return ordinary_;
  }
  void set_graph_masked_handle(KernelLaunchHandle handle) const {
    masked_ = handle;
  }
  const std::optional<KernelLaunchHandle> &get_graph_masked_handle() const {
    return masked_;
  }
  void retire() noexcept {
    retired_ = true;
    ordinary_.reset();
    masked_.reset();
  }

 private:
  mutable std::uint64_t owner_{0};
  mutable std::uint64_t generation_{0};
  mutable std::optional<KernelLaunchHandle> ordinary_;
  mutable std::optional<KernelLaunchHandle> masked_;
  bool retired_{false};
};

// Stable ownership boundary between a frontend Kernel definition, compiled
// Graphs, and backend executable registration. The payload can outlive its
// compilation-cache entry, while retirement prevents a new launch from using
// registration IDs removed by an SNodeTree lifecycle transaction.
class KernelExecutionHandle {
 public:
  enum class State : std::uint8_t {
    active = 0,
    retired = 1,
  };

  KernelExecutionHandle(std::uint64_t identity,
                        std::shared_ptr<const CompiledKernelData> payload)
      : identity_(identity), payload_(std::move(payload)) {
  }

  std::uint64_t identity() const noexcept {
    return identity_;
  }

  bool active() const noexcept {
    return state_.load(std::memory_order_acquire) == State::active;
  }

  const CompiledKernelData &compiled() const {
    return *payload_;
  }

  std::shared_ptr<const CompiledKernelData> payload_lease() const {
    return payload_;
  }

  const KernelRegistrationState &registration() const noexcept {
    return registration_;
  }

  void retire() noexcept {
    if (state_.exchange(State::retired, std::memory_order_acq_rel) ==
        State::active) {
      registration_.retire();
    }
  }

 private:
  std::uint64_t identity_{0};
  std::shared_ptr<const CompiledKernelData> payload_;
  KernelRegistrationState registration_;
  std::atomic<State> state_{State::active};
};

}  // namespace taichi::lang
