#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include "taichi/program/runtime_completion.h"

namespace taichi::lang {
class NativeOperation;
class RuntimeFaultDomain;
class BackendRuntimeError;
namespace cuda {

// One bounded domain admission owns these reservations. No private event queue:
// the domain's single completion covers every graph and provider in the recipe.
// reserve() is host-only; run() uses the admitted stream. On failed admission,
// destruction cancels only this batch's successful reservations. Once invoked,
// a provider token is retired only after the domain proves its prefix complete.
class OrderedSubmissionResources final : public RuntimeCompletionResources {
 public:
  OrderedSubmissionResources(
      std::size_t operation_count,
      std::shared_ptr<RuntimeCompletionResources> operands);
  ~OrderedSubmissionResources() override;
  void reserve(const std::vector<std::shared_ptr<NativeOperation>> &operations);
  void run(std::size_t index, void *stream);
  std::size_t retained_resource_count(
      std::uint32_t kind) const noexcept override;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

// Completion ownership for host-invoked, stream-ordered native operations.
// Caller holds the CUDA submission/context guards. No provider or Program API.
class OrderedSubmissions {
 public:
  OrderedSubmissions(std::size_t operations,
                     std::shared_ptr<RuntimeFaultDomain> fault_domain);
  ~OrderedSubmissions();
  void begin();
  void run(NativeOperation &operation);
  void finish();
  void finish_after_failure(
      const BackendRuntimeError *error = nullptr) noexcept;

 private:
  struct State;
  std::unique_ptr<State> state_;
};
}  // namespace cuda
}  // namespace taichi::lang
