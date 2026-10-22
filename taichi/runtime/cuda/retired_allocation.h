#pragma once

#include "taichi/rhi/cuda/cuda_context.h"

namespace taichi::lang::cuda {

// Only for cuMemAlloc storage whose last use has already completed. An
// explicit nonblocking cleanup stream avoids cuMemFree's context-wide wait.
// The owner drains that stream before destroying it or its device context.
// This is not the async-allocation wrapper: CIG retained-pool allocations have
// a separate owner and must not pass through this legacy-allocation contract.
inline void free_retired_allocation(void *pointer, void *cleanup_stream) {
  if (!pointer)
    return;
  auto &context = CUDAContext::get_instance();
  auto &driver = CUDADriver::get_instance();
  if (cleanup_stream && context.supports_mem_pool())
    driver.mem_free_async_impl(pointer, cleanup_stream);
  else
    driver.mem_free(pointer);
}

}  // namespace taichi::lang::cuda
