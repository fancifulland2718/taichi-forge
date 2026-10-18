#pragma once

#include "taichi_core_impl.h"
#include "taichi/program/runtime_completion.h"

#ifdef TI_WITH_CUDA
namespace taichi::lang::cuda {
class ExecutionDomain;
}
namespace capi {

std::shared_ptr<taichi::lang::cuda::ExecutionDomain> get_execution_domain(
    TiCudaExecutionDomain handle);
taichi::lang::RuntimeCompletion get_completion(TiCompletion handle);

// Reserve all public ticket bookkeeping before a submit can affect the device.
// Failed admission destroys the unpublished reservation. commit only assigns
// an existing node, so publication cannot fail with a post-enqueue allocation.
class CompletionPublication {
 public:
  CompletionPublication();
  ~CompletionPublication();
  CompletionPublication(const CompletionPublication &) = delete;
  CompletionPublication &operator=(const CompletionPublication &) = delete;
  TiCompletion commit(taichi::lang::RuntimeCompletion completion);

 private:
  TiCompletion handle_{nullptr};
};

}  // namespace capi
#endif
