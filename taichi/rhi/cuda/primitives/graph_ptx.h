#pragma once

#include <cstddef>
#include <cstdint>

namespace taichi::lang::cuda {

struct CudaGraphConditionalControl {
  std::uintptr_t predicate{0};
  std::uint32_t iteration{0};
  std::uint32_t max_iterations{0};
  std::uint32_t continue_while_nonzero{1};
  std::uint32_t reserved{0};
};

static_assert(sizeof(CudaGraphConditionalControl) == 24);
static_assert(offsetof(CudaGraphConditionalControl, predicate) == 0);
static_assert(offsetof(CudaGraphConditionalControl, iteration) == 8);
static_assert(offsetof(CudaGraphConditionalControl, max_iterations) == 12);
static_assert(
    offsetof(CudaGraphConditionalControl, continue_while_nonzero) == 16);

bool driver_graph_conditional_setter_compiled();
void driver_graph_prepare_conditional_setter();
void driver_graph_set_conditional(
    CudaGraphConditionalControl *control,
    std::uint64_t conditional_handle,
    void *stream = nullptr);
void driver_graph_set_branch_conditional(
    CudaGraphConditionalControl *control,
    std::uint64_t conditional_handle,
    void *stream = nullptr);

// Toolkit-independent control kernels used by the pre-CUDA-12.8 bounded
// masking route. These are ordinary PTX kernels and do not reference
// cudaGraphSetConditional or any conditional-node driver symbol.
bool driver_graph_mask_latch_compiled();
void driver_graph_prepare_mask_latch();
void driver_graph_latch_while(void *predicate,
                              void *gate,
                              bool continue_while_nonzero,
                              void *stream = nullptr);
void driver_graph_latch_branch(void *selector,
                               void *gate,
                               std::uint32_t conditional_type,
                               std::uint32_t branch_count,
                               std::uint32_t default_branch,
                               void *stream = nullptr);

}  // namespace taichi::lang::cuda
