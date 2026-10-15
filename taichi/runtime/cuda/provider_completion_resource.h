#pragma once

#include <cstdint>

namespace taichi::lang {

// Mutable provider state may observe exactly when its submission lease retires.
// Both JIT and loaded execution own these resources; no Program pointer or
// provider-specific completion queue is part of the contract.
class CudaProviderCompletionResource {
 public:
  virtual ~CudaProviderCompletionResource() = default;
  virtual std::uint64_t submission_retirement_token() const {
    return 0;
  }
  virtual void on_submission_retired(std::uint64_t) noexcept {
  }
};

}  // namespace taichi::lang
