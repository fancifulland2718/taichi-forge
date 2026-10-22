#pragma once

#include "taichi_core_impl.h"
#include "taichi/program/runtime_completion.h"

namespace capi {
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
