#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "taichi/common/core.h"
#include "taichi/common/exceptions.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {

// A backend API error with machine-readable identity. RHI code reports the
// raw backend/code pair; the Program-owned fault domain remains the single
// authority that classifies it as recoverable, operation-local, or fatal.
class TI_DLL_EXPORT BackendRuntimeError final : public TaichiRuntimeError {
 public:
  BackendRuntimeError(Arch backend,
                      std::int64_t backend_code,
                      std::string operation,
                      std::string message)
      : TaichiRuntimeError(std::move(message)),
        backend_(backend),
        backend_code_(backend_code),
        operation_(std::move(operation)) {
  }

  Arch backend() const noexcept {
    return backend_;
  }
  std::int64_t backend_code() const noexcept {
    return backend_code_;
  }
  const std::string &operation() const noexcept {
    return operation_;
  }

 private:
  Arch backend_;
  std::int64_t backend_code_;
  std::string operation_;
};

// Implemented by a Program-owned object and retained by Device/completion
// owners. It contains no Device pointer, so attaching it cannot create an
// ownership cycle.
class TI_DLL_EXPORT BackendFaultReporter {
 public:
  virtual ~BackendFaultReporter() = default;

  virtual void report_backend_error(const BackendRuntimeError &error,
                                    std::uint64_t submission_sequence) noexcept =
      0;
  virtual bool backend_calls_safe() const noexcept = 0;
  virtual void throw_if_submission_disallowed(const char *operation) const = 0;
};

}  // namespace taichi::lang
