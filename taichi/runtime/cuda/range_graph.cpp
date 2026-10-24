#include "taichi/runtime/cuda/range_graph.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/retired_allocation.h"

namespace taichi::lang::cuda {

bool RangeGraphControls::initialize(
    const std::vector<KernelLauncher::GraphLaunchPacket> &packets,
    const std::vector<std::vector<std::size_t>> &selections,
    std::size_t temporary_bytes,
    std::uint32_t *driver_error) {
  TI_ASSERT(driver_error);
  *driver_error = CUDA_SUCCESS;
  if (!dispatches_.empty() || device_ || packets.size() != selections.size())
    throw std::invalid_argument("Invalid CUDA range-control construction");
  // Validate the complete selection before allocating or capturing any work.
  std::vector<std::vector<TaskControl>> dispatches(packets.size());
  std::vector<CudaGraphRangeControl> host;
  auto &driver = CUDADriver::get_instance();
  int hardware_grid = 0;
  driver.device_get_attribute(&hardware_grid, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                              CUDAContext::get_instance().get_device());
  for (std::size_t i = 0; i < packets.size(); ++i) {
    if (selections[i].empty())
      continue;
    const auto &packet = packets[i];
    const auto base =
        reinterpret_cast<std::uintptr_t>(packet.context.global_tmp_buffer);
    if (packet.launch_abi != KernelLaunchABI::ordinary || packet.bounded_range ||
        !base || base % alignof(std::int32_t) || hardware_grid <= 0 ||
        temporary_bytes > std::numeric_limits<std::uintptr_t>::max() - base)
      throw std::invalid_argument("CUDA range task requires its private workspace");
    auto selected = selections[i];
    std::sort(selected.begin(), selected.end());
    if (std::adjacent_find(selected.begin(), selected.end()) != selected.end())
      throw std::invalid_argument("Duplicate CUDA range task selection");
    for (auto index : selected) {
      if (index >= packet.offloaded_tasks.size())
        throw std::invalid_argument("CUDA range task index is outside its kernel");
      const auto &task = packet.offloaded_tasks[index];
      if (task.device_range_dispatch_version != 1 ||
          !task.valid_device_range_dispatch_contract() || !task.native_function)
        throw std::invalid_argument("CUDA range task has no final IR qualification");
      CudaGraphRangeControl control;
      const auto bind = [&](const RangeBound &bound, std::uintptr_t &pointer,
                            std::int32_t &constant) {
        if (bound.kind == RangeBoundKind::constant_i32 && bound.valid()) {
          constant = static_cast<std::int32_t>(bound.value);
        } else if (bound.fits_global_tmps(temporary_bytes)) {
          pointer = base + static_cast<std::uintptr_t>(bound.value);
        } else {
          throw std::invalid_argument("CUDA range bound exceeds its workspace");
        }
      };
      bind(task.range_bounds->begin, control.begin_pointer, control.begin_constant);
      bind(task.range_bounds->end, control.end_pointer, control.end_constant);
      control.block_dim = task.block_dim;
      control.max_grid_dim = std::min(task.grid_dim, hardware_grid);
      dispatches[i].push_back({index, host.size()});
      host.push_back(control);
    }
  }
  if (host.empty())
    throw std::invalid_argument("CUDA range controls require a selected task");
  if (host.size() > std::numeric_limits<std::size_t>::max() / sizeof(host[0]))
    throw std::invalid_argument("CUDA range control storage overflow");
  if (!driver.launch_kernel_ex.available() || !driver.graph_upload.available()) {
    *driver_error = CUDA_ERROR_NOT_SUPPORTED;
    return false;
  }
  if (!driver_graph_prepare_bounded_update(driver_error))
    return false;
  dispatches_ = std::move(dispatches);
  host_ = std::move(host);
  *driver_error = driver.malloc.call(reinterpret_cast<void **>(&device_),
                                     host_.size() * sizeof(host_[0]));
  return *driver_error == CUDA_SUCCESS;
}

std::uint32_t RangeGraphControls::capture(
    std::size_t dispatch,
    KernelLauncher &launcher,
    const KernelLauncher::GraphLaunchPacket &packet,
    void *stream) {
  if (!device_ || dispatch >= dispatches_.size())
    return CUDA_ERROR_NOT_SUPPORTED;
  const auto &selected = dispatches_[dispatch];
  if (selected.empty()) {
    launcher.capture_cuda_graph_launch(packet, stream);
    return CUDA_SUCCESS;
  }
  // Packet slicing is cold only. Every serial producer/finalizer remains in
  // order; a controller reads its snapshot immediately before that payload.
  auto slice = packet;
  slice.offloaded_tasks.resize(1);
  std::size_t cursor = 0;
  for (std::size_t task = 0; task < packet.offloaded_tasks.size(); ++task) {
    slice.offloaded_tasks[0] = packet.offloaded_tasks[task];
    if (cursor == selected.size() || selected[cursor].task != task) {
      launcher.capture_cuda_graph_launch(slice, stream);
      continue;
    }
    const auto index = selected[cursor++].control;
    driver_graph_update_range(device_ + index, stream);
    std::vector<void *> nodes;
    std::uint32_t error = CUDA_SUCCESS;
    if (!launcher.capture_cuda_graph_updatable_launch(slice, stream, &nodes,
                                                       &error))
      return error == CUDA_SUCCESS ? CUDA_ERROR_NOT_SUPPORTED : error;
    if (nodes.size() != 1 || !nodes[0])
      return CUDA_ERROR_NOT_SUPPORTED;
    host_[index].device_node = reinterpret_cast<std::uintptr_t>(nodes[0]);
  }
  return CUDA_SUCCESS;
}

std::uint32_t RangeGraphControls::upload() const {
  if (!device_ || std::any_of(host_.begin(), host_.end(),
                             [](const auto &c) { return !c.device_node; }))
    return CUDA_ERROR_NOT_SUPPORTED;
  return CUDADriver::get_instance().memcpy_host_to_device.call(
      device_, const_cast<CudaGraphRangeControl *>(host_.data()), device_bytes());
}

void RangeGraphControls::release(bool backend_safe, void *cleanup_stream) {
  if (backend_safe && device_)
    free_retired_allocation(device_, cleanup_stream);
  device_ = nullptr;
  host_.clear();
  dispatches_.clear();
}

std::size_t RangeGraphControls::device_bytes() const {
  return device_ ? host_.size() * sizeof(host_[0]) : 0;
}

}  // namespace taichi::lang::cuda
