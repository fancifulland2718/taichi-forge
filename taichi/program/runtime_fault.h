#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "taichi/common/core.h"
#include "taichi/rhi/arch.h"
#include "taichi/rhi/backend_error.h"
#include "taichi/rhi/public_device.h"

namespace taichi::lang {

// Backend return codes are not interchangeable with runtime lifecycle state.
// In particular, invalid arguments and allocation failures must not poison a
// healthy device, while context/device loss must reject later submissions.
enum class BackendErrorClassification : std::uint8_t {
  kSuccess,
  kRecoverable,
  kOperation,
  kFatal,
};

TI_DLL_EXPORT BackendErrorClassification classify_cuda_driver_error(
    std::uint32_t error) noexcept;
TI_DLL_EXPORT BackendErrorClassification classify_vulkan_result(
    std::int32_t result) noexcept;
TI_DLL_EXPORT BackendErrorClassification classify_rhi_result(
    RhiResult result) noexcept;

enum class RuntimeLifecycleState : std::uint8_t {
  kHealthy,
  kFaulted,
  kFinalizing,
  kFinalized,
};

TI_DLL_EXPORT const char *runtime_lifecycle_state_name(
    RuntimeLifecycleState state) noexcept;

struct RuntimeFaultRecord {
  Arch backend{Arch::x64};
  std::int64_t backend_code{0};
  std::uint64_t submission_sequence{0};
  std::string operation;
  std::string message;
};

struct RuntimeFaultSnapshot {
  RuntimeLifecycleState state{RuntimeLifecycleState::kHealthy};
  std::uint64_t program_domain{0};
  std::uint64_t rejected_submissions{0};
  std::optional<RuntimeFaultRecord> first_fault;
};

// Program-owned, backend-neutral first-fault state. Completion tickets may
// retain this object after Program teardown, so the record never stores a
// Program, kernel, Graph, Python-object, or backend-handle pointer.
class TI_DLL_EXPORT RuntimeFaultDomain final : public BackendFaultReporter {
 public:
  RuntimeFaultDomain(Arch backend, std::uint64_t program_domain) noexcept;

  RuntimeLifecycleState state() const noexcept {
    return state_.load(std::memory_order_acquire);
  }

  bool submission_allowed() const noexcept {
    return state_.load(std::memory_order_acquire) ==
           RuntimeLifecycleState::kHealthy;
  }

  bool backend_calls_safe() const noexcept override {
    return !fatal_observed_.load(std::memory_order_acquire);
  }

  bool has_fatal_fault() const noexcept {
    return fatal_observed_.load(std::memory_order_acquire);
  }

  // Returns true only for the call that records the immutable first fault.
  // Faults observed during finalization are retained for diagnostics but do
  // not move the lifecycle back from finalizing/finalized.
  bool report_fatal(RuntimeFaultRecord fault);

  void report_backend_error(const BackendRuntimeError &error,
                            std::uint64_t submission_sequence) noexcept
      override;

  void begin_finalizing() noexcept;
  void mark_finalized() noexcept;

  // Healthy is the only state that accepts new work. This throws directly as
  // TaichiRuntimeError instead of logging repeatedly after the first fault.
  void throw_if_submission_disallowed(
      const char *operation) const override;

  RuntimeFaultSnapshot snapshot() const;

 private:
  std::string rejection_message(const char *operation) const;

  const Arch backend_;
  const std::uint64_t program_domain_;
  std::atomic<RuntimeLifecycleState> state_{RuntimeLifecycleState::kHealthy};
  std::atomic<bool> fatal_observed_{false};
  mutable std::atomic<std::uint64_t> rejected_submissions_{0};
  mutable std::mutex mutex_;
  std::thread::id finalizer_thread_;
  std::optional<RuntimeFaultRecord> first_fault_;
};

}  // namespace taichi::lang
