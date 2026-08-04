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

struct CudaGraphBoundedExtentControl {
  std::uintptr_t device_node{0};
  std::uintptr_t extent{0};
  std::uint32_t capacity{0};
  std::uint32_t block_dim{0};
  std::uint32_t driver_status{0};
  std::uint32_t max_grid_dim{0};
};

static_assert(sizeof(CudaGraphBoundedExtentControl) == 32);
static_assert(offsetof(CudaGraphBoundedExtentControl, device_node) == 0);
static_assert(offsetof(CudaGraphBoundedExtentControl, extent) == 8);
static_assert(offsetof(CudaGraphBoundedExtentControl, capacity) == 16);
static_assert(offsetof(CudaGraphBoundedExtentControl, block_dim) == 20);
static_assert(offsetof(CudaGraphBoundedExtentControl, driver_status) == 24);
static_assert(offsetof(CudaGraphBoundedExtentControl, max_grid_dim) == 28);

// One stateful control record updates every consecutive bounded payload that
// consumes the same extent/capacity/block contract. The updater caches the
// last applied state because device-updated Graph node properties persist
// across replays.
struct CudaGraphBoundedGroupControl {
  std::uintptr_t extent{0};
  std::uintptr_t device_nodes{0};
  std::uint32_t node_count{0};
  std::uint32_t capacity{0};
  std::uint32_t block_dim{0};
  std::uint32_t max_grid_dim{0};
  std::uint32_t last_grid_x{0};
  std::uint32_t last_enabled{0};
  std::uint32_t initialized{0};
  std::uint32_t driver_status{0};
  std::uint32_t telemetry_enabled{0};
  std::uint32_t reserved{0};
  std::uint64_t replay_count{0};
  std::uint64_t state_change_count{0};
  std::uint64_t node_api_call_count{0};
};

static_assert(sizeof(CudaGraphBoundedGroupControl) == 80);
static_assert(offsetof(CudaGraphBoundedGroupControl, extent) == 0);
static_assert(offsetof(CudaGraphBoundedGroupControl, device_nodes) == 8);
static_assert(offsetof(CudaGraphBoundedGroupControl, node_count) == 16);
static_assert(offsetof(CudaGraphBoundedGroupControl, capacity) == 20);
static_assert(offsetof(CudaGraphBoundedGroupControl, block_dim) == 24);
static_assert(offsetof(CudaGraphBoundedGroupControl, max_grid_dim) == 28);
static_assert(offsetof(CudaGraphBoundedGroupControl, last_grid_x) == 32);
static_assert(offsetof(CudaGraphBoundedGroupControl, last_enabled) == 36);
static_assert(offsetof(CudaGraphBoundedGroupControl, initialized) == 40);
static_assert(offsetof(CudaGraphBoundedGroupControl, driver_status) == 44);
static_assert(offsetof(CudaGraphBoundedGroupControl, telemetry_enabled) == 48);
static_assert(offsetof(CudaGraphBoundedGroupControl, replay_count) == 56);
static_assert(offsetof(CudaGraphBoundedGroupControl, state_change_count) == 64);
static_assert(offsetof(CudaGraphBoundedGroupControl, node_api_call_count) == 72);

struct CudaGraphBoundedProbeResult {
  bool attempted{false};
  bool passed{false};
  bool ptx_linked{false};
  bool graph_uploaded{false};
  bool zero_count_skipped{false};
  bool launch_update_persists{false};
  bool external_update_persists{false};
  bool partial_failure_capacity_safe{false};
  std::uint32_t driver_error{0};
  std::uint32_t sparse_visited{0};
  std::uint32_t zero_visited{0};
  std::uint32_t rebound_visited{0};
  std::uint32_t baseline_visited{0};
  std::uint32_t persistent_sparse_visited{0};
  std::uint32_t persistent_disabled_visited{0};
  std::uint32_t external_update_visited{0};
  std::uint32_t external_reset_visited{0};
  std::uint32_t partial_failure_visited{0};
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
void driver_graph_update_bounded_extent(CudaGraphBoundedExtentControl *control,
                                        void *stream = nullptr);
void driver_graph_update_bounded_group(CudaGraphBoundedGroupControl *control,
                                       void *stream = nullptr);
void *driver_graph_bounded_probe_payload_function();
CudaGraphBoundedProbeResult driver_graph_bounded_probe(bool run);

}  // namespace taichi::lang::cuda
