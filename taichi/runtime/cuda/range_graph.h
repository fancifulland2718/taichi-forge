#pragma once

#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/rhi/cuda/primitives/graph_ptx.h"

namespace taichi::lang::cuda {

// Cold lowering of selected ordinary tasks. The enclosing FixedGraph owns the
// temporary binding, executable and completion. Release only after
// executable retirement; release(false) abandons unsafe backend allocations.
class RangeGraphControls {
 public:
  RangeGraphControls() = default;
  RangeGraphControls(const RangeGraphControls &) = delete;
  RangeGraphControls &operator=(const RangeGraphControls &) = delete;
  bool initialize(
      const std::vector<KernelLauncher::GraphLaunchPacket> &packets,
      const std::vector<std::vector<std::size_t>> &selections,
      std::size_t temporary_bytes,
      std::uint32_t *driver_error);
  std::uint32_t capture(std::size_t dispatch,
                        KernelLauncher &launcher,
                        const KernelLauncher::GraphLaunchPacket &packet,
                        void *stream);
  std::uint32_t upload() const;
  void release(bool backend_safe, void *cleanup_stream = nullptr);
  std::size_t device_bytes() const;

 private:
  struct TaskControl {
    std::size_t task;
    std::size_t control;
  };
  std::vector<std::vector<TaskControl>> dispatches_;
  std::vector<CudaGraphRangeControl> host_;
  CudaGraphRangeControl *device_{nullptr};
};

}  // namespace taichi::lang::cuda
