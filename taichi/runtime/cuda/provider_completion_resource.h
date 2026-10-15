#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace taichi::lang {
class RuntimeCompletion;

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
  // Ordered AOT commands share retirement across executions using this owner.
  // Weak tokens avoid a completion -> resource -> completion ownership cycle.
  void track_ordered_completion(const std::shared_ptr<RuntimeCompletion> &completion);
  void retire_completed_ordered_submissions();

 private:
  std::mutex completion_mutex_;
  std::vector<std::weak_ptr<RuntimeCompletion>> ordered_completions_;
};

}  // namespace taichi::lang
