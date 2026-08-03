#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

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
static_assert(offsetof(CudaGraphConditionalControl, continue_while_nonzero) ==
              16);

struct CudaGraphBoundedControl {
  std::uintptr_t device_node{0};
  std::uint32_t grid_x{0};
  std::uint32_t enabled{1};
  std::uint32_t status{0};
  std::uint32_t reserved{0};
};

static_assert(sizeof(CudaGraphBoundedControl) == 24);
static_assert(offsetof(CudaGraphBoundedControl, device_node) == 0);
static_assert(offsetof(CudaGraphBoundedControl, grid_x) == 8);
static_assert(offsetof(CudaGraphBoundedControl, enabled) == 12);
static_assert(offsetof(CudaGraphBoundedControl, status) == 16);

struct CudaGraphBoundedProbeResult {
  bool attempted{false};
  bool passed{false};
  bool ptx_linked{false};
  bool graph_uploaded{false};
  bool zero_count_skipped{false};
  std::uint32_t driver_error{0};
  std::uint32_t sparse_visited{0};
  std::uint32_t zero_visited{0};
  std::uint32_t rebound_visited{0};
  std::uint32_t baseline_visited{0};
  std::string reason{"not_attempted"};
};

bool driver_graph_conditional_setter_compiled();
void driver_graph_prepare_conditional_setter();
void driver_graph_set_conditional(CudaGraphConditionalControl *control,
                                  std::uint64_t conditional_handle,
                                  void *stream = nullptr);
void driver_graph_set_branch_conditional(CudaGraphConditionalControl *control,
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

// CUDA 12.4 device-updatable kernel-node lowering. The probe is explicit and
// cached; capability queries can inspect it without initializing a context or
// allocating device memory.
bool driver_graph_bounded_update_compiled();
bool driver_graph_prepare_bounded_update(std::uint32_t *driver_error = nullptr);
void driver_graph_update_bounded(CudaGraphBoundedControl *control,
                                 void *stream = nullptr);
void *driver_graph_bounded_probe_payload_function();
CudaGraphBoundedProbeResult driver_graph_bounded_probe(bool run);

}  // namespace taichi::lang::cuda
