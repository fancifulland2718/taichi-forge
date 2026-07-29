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

}  // namespace taichi::lang::cuda
