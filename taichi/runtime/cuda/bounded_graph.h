#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "taichi/rhi/cuda/primitives/graph_ptx.h"
#include "taichi/runtime/cuda/kernel_launcher.h"

namespace taichi::lang::cuda {

// One entry per captured dispatch, including ordinary kernels and providers.
// Joining requires the same logical extent binding, not just equal addresses:
// two independently rebindable symbols can initially alias the same allocation.
struct BoundedGraphDispatch {
  std::uintptr_t extent{0};
  std::uint32_t capacity{0};
  std::uint32_t block_dim{0};
  bool adaptive{false};
  bool join_previous{false};
};

// Program-independent cold construction for device-updatable bounded kernels.
// The enclosing graph owns completion, context/locking, and allocation leases.
// It must release these controls after retiring its executable, or abandon them
// with release(false) when the backend is unsafe. No backend calls in
// destructor.
class BoundedGraphControls {
 public:
  struct Control {
    CudaGraphBoundedExtentControl *device_control{nullptr};
    CudaGraphBoundedExtentControl host_control;
  };
  struct Group {
    CudaGraphBoundedGroupControl *device_control{nullptr};
    std::uintptr_t *device_nodes{nullptr};
    CudaGraphBoundedGroupControl host_control;
    std::vector<std::uintptr_t> host_nodes;
    std::vector<std::size_t> dispatch_indices;
  };

  BoundedGraphControls() = default;
  BoundedGraphControls(const BoundedGraphControls &) = delete;
  BoundedGraphControls &operator=(const BoundedGraphControls &) = delete;
  bool initialize(const std::vector<BoundedGraphDispatch> &dispatches,
                  std::uint32_t saturation_grid_dim,
                  bool diagnostics_enabled,
                  std::uint32_t *driver_error);
  std::uint32_t capture(std::size_t index,
                        KernelLauncher &launcher,
                        const KernelLauncher::GraphLaunchPacket &packet,
                        void *stream);
  // Call after graph instantiation/upload; device node handles are not
  // portable.
  std::uint32_t upload() const;
  void release(bool backend_safe);
  std::uint64_t device_bytes() const;

  // Available to the graph owner for its binding patches and opt-in telemetry.
  std::vector<Control> controls;
  std::vector<Group> groups;
  std::vector<std::int32_t> group_indices;
  std::vector<std::int32_t> group_member_indices;
};

}  // namespace taichi::lang::cuda
