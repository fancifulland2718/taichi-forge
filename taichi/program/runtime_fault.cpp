#include "taichi/program/runtime_fault.h"

#include <spdlog/fmt/fmt.h>

#include "taichi/common/exceptions.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {
namespace {

// Vulkan core/KHR numeric values are stable API constants. Keeping the small
// classification table independent from Vulkan headers lets CPU/CUDA-only
// builds compile the same runtime fault model.
constexpr std::int32_t kVkSuccess = 0;
constexpr std::int32_t kVkNotReady = 1;
constexpr std::int32_t kVkTimeout = 2;
constexpr std::int32_t kVkErrorDeviceLost = -4;
constexpr std::int32_t kVkSuboptimalKhr = 1000001003;
constexpr std::int32_t kVkErrorOutOfDateKhr = -1000001004;

}  // namespace

BackendErrorClassification classify_cuda_driver_error(
    std::uint32_t error) noexcept {
  if (error == CUDA_SUCCESS) {
    return BackendErrorClassification::kSuccess;
  }
  if (error == CUDA_ERROR_NOT_READY) {
    return BackendErrorClassification::kRecoverable;
  }
  switch (error) {
    case CUDA_ERROR_ILLEGAL_ADDRESS:
    case CUDA_ERROR_LAUNCH_TIMEOUT:
    case CUDA_ERROR_ASSERT:
    case CUDA_ERROR_HARDWARE_STACK_ERROR:
    case CUDA_ERROR_ILLEGAL_INSTRUCTION:
    case CUDA_ERROR_MISALIGNED_ADDRESS:
    case CUDA_ERROR_INVALID_ADDRESS_SPACE:
    case CUDA_ERROR_INVALID_PC:
    case CUDA_ERROR_LAUNCH_FAILED:
      return BackendErrorClassification::kFatal;
    default:
      return BackendErrorClassification::kOperation;
  }
}

BackendErrorClassification classify_vulkan_result(
    std::int32_t result) noexcept {
  if (result == kVkSuccess) {
    return BackendErrorClassification::kSuccess;
  }
  if (result == kVkNotReady || result == kVkTimeout ||
      result == kVkSuboptimalKhr || result == kVkErrorOutOfDateKhr) {
    return BackendErrorClassification::kRecoverable;
  }
  if (result == kVkErrorDeviceLost) {
    return BackendErrorClassification::kFatal;
  }
  return BackendErrorClassification::kOperation;
}

BackendErrorClassification classify_rhi_result(RhiResult result) noexcept {
  if (result == RhiResult::success) {
    return BackendErrorClassification::kSuccess;
  }
  // RhiResult intentionally loses backend-specific context. Never infer a
  // fatal device state from this coarse result alone.
  return BackendErrorClassification::kOperation;
}

const char *runtime_lifecycle_state_name(RuntimeLifecycleState state) noexcept {
  switch (state) {
    case RuntimeLifecycleState::kHealthy:
      return "healthy";
    case RuntimeLifecycleState::kFaulted:
      return "faulted";
    case RuntimeLifecycleState::kFinalizing:
      return "finalizing";
    case RuntimeLifecycleState::kFinalized:
      return "finalized";
  }
  return "unknown";
}

RuntimeFaultDomain::RuntimeFaultDomain(
    Arch backend,
    std::uint64_t program_domain) noexcept
    : backend_(backend), program_domain_(program_domain) {
}

bool RuntimeFaultDomain::report_fatal(RuntimeFaultRecord fault) {
  std::lock_guard<std::mutex> lock(mutex_);
  const RuntimeLifecycleState current =
      state_.load(std::memory_order_relaxed);
  if (current == RuntimeLifecycleState::kFinalized || first_fault_.has_value()) {
    return false;
  }
  // One Program owns one backend. Do not accept a caller-supplied backend that
  // could make the immutable first-fault record contradict its domain.
  fault.backend = backend_;
  first_fault_ = std::move(fault);
  if (current == RuntimeLifecycleState::kHealthy) {
    state_.store(RuntimeLifecycleState::kFaulted, std::memory_order_release);
  }
  return true;
}

void RuntimeFaultDomain::begin_finalizing() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  const RuntimeLifecycleState current =
      state_.load(std::memory_order_relaxed);
  if (current == RuntimeLifecycleState::kFinalizing ||
      current == RuntimeLifecycleState::kFinalized) {
    return;
  }
  state_.store(RuntimeLifecycleState::kFinalizing, std::memory_order_release);
}

void RuntimeFaultDomain::mark_finalized() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  state_.store(RuntimeLifecycleState::kFinalized, std::memory_order_release);
}

void RuntimeFaultDomain::throw_if_submission_disallowed(
    const char *operation) const {
  if (submission_allowed()) {
    return;
  }
  rejected_submissions_.fetch_add(1, std::memory_order_relaxed);
  throw TaichiRuntimeError(rejection_message(operation));
}

RuntimeFaultSnapshot RuntimeFaultDomain::snapshot() const {
  RuntimeFaultSnapshot result;
  std::lock_guard<std::mutex> lock(mutex_);
  result.state = state_.load(std::memory_order_relaxed);
  result.program_domain = program_domain_;
  result.rejected_submissions =
      rejected_submissions_.load(std::memory_order_relaxed);
  result.first_fault = first_fault_;
  return result;
}

std::string RuntimeFaultDomain::rejection_message(const char *operation) const {
  const RuntimeFaultSnapshot current = snapshot();
  std::string message = fmt::format(
      "Runtime is {}; refusing {}", runtime_lifecycle_state_name(current.state),
      operation ? operation : "a new submission");
  if (current.first_fault) {
    const RuntimeFaultRecord &fault = *current.first_fault;
    message += fmt::format(
        ". First fatal backend error: backend={}, code={}, operation={}, "
        "sequence={}: {}",
        arch_name(fault.backend), fault.backend_code,
        fault.operation.empty() ? "unknown" : fault.operation,
        fault.submission_sequence,
        fault.message.empty() ? "no backend message" : fault.message);
  }
  if (current.state == RuntimeLifecycleState::kFaulted) {
    message += ". Call ti.reset() to create a new Program/device";
  }
  return message;
}

}  // namespace taichi::lang
