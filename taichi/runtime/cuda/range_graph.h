#pragma once

#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/rhi/cuda/primitives/graph_ptx.h"

namespace taichi::lang::cuda {

// Cold lowering of selected ordinary tasks. The enclosing recording owns the
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
      std::uint32_t *driver_error,
      const std::vector<std::size_t> &recording_counts = {});
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
  // A condition may be captured in both the root and the loop body. Each
  // physical occurrence owns its node addresses, independently of replay count.
  struct DispatchControl {
    std::vector<TaskControl> tasks;
    std::size_t count{1};
    std::size_t next{0};
  };
  std::vector<DispatchControl> dispatches_;
  std::vector<CudaGraphRangeControl> host_;
  CudaGraphRangeControl *device_{nullptr};
};

}  // namespace taichi::lang::cuda
