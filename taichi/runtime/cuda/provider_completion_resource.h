#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include "taichi/runtime/runtime_client.h"

namespace taichi::lang {
class RuntimeCompletion;

// Mutable provider state may observe exactly when its submission lease retires.
// Both JIT and loaded execution own these resources; no Program pointer or
// provider-specific completion queue is part of the contract.
class CudaProviderCompletionResource : public RuntimeClient {
 public:
  virtual ~CudaProviderCompletionResource() = default;
  virtual std::uint64_t submission_retirement_token() const {
    return 0;
  }
  virtual void on_submission_retired(std::uint64_t) noexcept {
  }
  // Cold affinity lease. Zero denotes the legacy stream; nonzero identities
  // are unique ExecutionDomains. Prepared executions retain this lease until
  // their recordings have retired. Failed preparation drops its claim, and a
  // different domain may bind only after all previous claims have retired.
  std::shared_ptr<void> bind_execution_domain(std::uint64_t identity);
  // Providers with private streams prove their retirement here. The enclosing
  // domain already drained its own stream; failure retains the entire recipe.
  void retire() override {
  }
  // Ordered AOT commands share retirement across executions using this owner.
  // Weak tokens avoid a completion -> resource -> completion ownership cycle.
  void track_ordered_completion(const std::shared_ptr<RuntimeCompletion> &completion);
  void retire_completed_ordered_submissions();

 private:
  std::mutex completion_mutex_;
  std::vector<std::weak_ptr<RuntimeCompletion>> ordered_completions_;
  std::weak_ptr<std::uint64_t> execution_domain_binding_;
};

}  // namespace taichi::lang
