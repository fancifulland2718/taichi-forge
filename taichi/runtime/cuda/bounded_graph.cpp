#include "taichi/runtime/cuda/bounded_graph.h"
#include "taichi/runtime/cuda/retired_allocation.h"

#include <algorithm>
#include <limits>

#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang::cuda {

bool BoundedGraphControls::initialize(
    const std::vector<BoundedGraphDispatch> &dispatches,
    std::uint32_t saturation_grid_dim,
    bool diagnostics_enabled,
    std::uint32_t *driver_error) {
  TI_ASSERT(driver_error != nullptr);
  *driver_error = CUDA_SUCCESS;
  // Do not discard an earlier executable's controls or partially built state.
  if (!controls.empty() || !groups.empty() ||
      dispatches.size() >
          static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    *driver_error = CUDA_ERROR_NOT_SUPPORTED;
    return false;
  }
  controls.resize(dispatches.size());
  group_indices.assign(dispatches.size(), -1);
  group_member_indices.assign(dispatches.size(), -1);
  if (std::none_of(dispatches.begin(), dispatches.end(),
                   [](const auto &entry) { return entry.adaptive; }))
    return true;
  auto &driver = CUDADriver::get_instance();
  if (saturation_grid_dim == 0 || !driver.launch_kernel_ex.available() ||
      !driver.graph_upload.available()) {
    *driver_error = CUDA_ERROR_NOT_SUPPORTED;
    return false;
  }
  // Validate the complete grouping contract before allocating any controls.
  for (std::size_t i = 0; i < dispatches.size(); ++i) {
    const auto &entry = dispatches[i];
    if (!entry.adaptive)
      continue;
    if (!entry.extent || !entry.capacity || !entry.block_dim ||
        (entry.join_previous &&
         (i == 0 || !dispatches[i - 1].adaptive ||
          entry.extent != dispatches[i - 1].extent ||
          entry.capacity != dispatches[i - 1].capacity ||
          entry.block_dim != dispatches[i - 1].block_dim))) {
      *driver_error = CUDA_ERROR_NOT_SUPPORTED;
      return false;
    }
  }
  if (!driver_graph_prepare_bounded_update(driver_error))
    return false;
  for (std::size_t i = 0; i < dispatches.size();) {
    const auto &entry = dispatches[i];
    if (!entry.adaptive) {
      ++i;
      continue;
    }
    std::size_t end = i + 1;
    while (end < dispatches.size() && dispatches[end].adaptive &&
           dispatches[end].join_previous)
      ++end;
    if (end - i > 1) {
      // Publish ownership before allocation so every failure is retired by the
      // enclosing owner, including host allocation failure and driver errors.
      groups.emplace_back();
      auto &group = groups.back();
      group.host_control.extent = entry.extent;
      group.host_control.node_count = static_cast<std::uint32_t>(end - i);
      group.host_control.capacity = entry.capacity;
      group.host_control.block_dim = entry.block_dim;
      group.host_control.max_grid_dim = saturation_grid_dim;
      group.host_control.telemetry_enabled = diagnostics_enabled ? 1u : 0u;
      group.host_nodes.resize(end - i);
      group.dispatch_indices.reserve(end - i);
      for (auto index = i; index < end; ++index)
        group.dispatch_indices.push_back(index);
      *driver_error =
          driver.malloc.call(reinterpret_cast<void **>(&group.device_control),
                             sizeof(CudaGraphBoundedGroupControl));
      if (*driver_error != CUDA_SUCCESS)
        return false;
      *driver_error =
          driver.malloc.call(reinterpret_cast<void **>(&group.device_nodes),
                             group.host_nodes.size() * sizeof(std::uintptr_t));
      if (*driver_error != CUDA_SUCCESS)
        return false;
      group.host_control.device_nodes =
          reinterpret_cast<std::uintptr_t>(group.device_nodes);
      const auto group_index = static_cast<std::int32_t>(groups.size() - 1);
      for (std::size_t member = 0; member < group.dispatch_indices.size();
           ++member) {
        const auto index = group.dispatch_indices[member];
        group_indices[index] = group_index;
        group_member_indices[index] = static_cast<std::int32_t>(member);
      }
    } else {
      auto &control = controls[i];
      control.host_control.extent = entry.extent;
      control.host_control.capacity = entry.capacity;
      control.host_control.block_dim = entry.block_dim;
      control.host_control.max_grid_dim = saturation_grid_dim;
      *driver_error =
          driver.malloc.call(reinterpret_cast<void **>(&control.device_control),
                             sizeof(CudaGraphBoundedExtentControl));
      if (*driver_error != CUDA_SUCCESS)
        return false;
    }
    i = end;
  }
  return true;
}

std::uint32_t BoundedGraphControls::capture(
    std::size_t index,
    KernelLauncher &launcher,
    const KernelLauncher::GraphLaunchPacket &packet,
    void *stream) {
  if (index >= controls.size() || index >= group_indices.size() ||
      index >= group_member_indices.size())
    return CUDA_ERROR_NOT_SUPPORTED;
  const auto group_index = group_indices[index];
  if (group_index >= 0) {
    const auto member_index = group_member_indices[index];
    if (member_index < 0 ||
        static_cast<std::size_t>(group_index) >= groups.size())
      return CUDA_ERROR_NOT_SUPPORTED;
    auto &group = groups[group_index];
    if (static_cast<std::size_t>(member_index) >= group.host_nodes.size() ||
        !group.device_control || !group.device_nodes)
      return CUDA_ERROR_NOT_SUPPORTED;
    if (member_index == 0)
      driver_graph_update_bounded_group(group.device_control, stream);
    void *device_node = nullptr;
    std::uint32_t error = CUDA_SUCCESS;
    if (!launcher.capture_cuda_graph_bounded_launch(packet, stream,
                                                    &device_node, &error))
      return error == CUDA_SUCCESS ? CUDA_ERROR_NOT_SUPPORTED : error;
    group.host_nodes[member_index] =
        reinterpret_cast<std::uintptr_t>(device_node);
  } else {
    auto &control = controls[index];
    if (!control.device_control) {
      launcher.capture_cuda_graph_launch(packet, stream);
      return CUDA_SUCCESS;
    }
    driver_graph_update_bounded_extent(control.device_control, stream);
    void *device_node = nullptr;
    std::uint32_t error = CUDA_SUCCESS;
    if (!launcher.capture_cuda_graph_bounded_launch(packet, stream,
                                                    &device_node, &error))
      return error == CUDA_SUCCESS ? CUDA_ERROR_NOT_SUPPORTED : error;
    control.host_control.device_node =
        reinterpret_cast<std::uintptr_t>(device_node);
  }
  return CUDA_SUCCESS;
}

std::uint32_t BoundedGraphControls::upload() const {
  auto &driver = CUDADriver::get_instance();
  for (const auto &control : controls) {
    if (!control.device_control)
      continue;
    const auto error = driver.memcpy_host_to_device.call(
        control.device_control,
        const_cast<CudaGraphBoundedExtentControl *>(&control.host_control),
        sizeof(control.host_control));
    if (error != CUDA_SUCCESS)
      return error;
  }
  for (const auto &group : groups) {
    if (!group.device_control || !group.device_nodes ||
        group.host_nodes.empty())
      return CUDA_ERROR_NOT_SUPPORTED;
    auto error = driver.memcpy_host_to_device.call(
        group.device_nodes,
        const_cast<std::uintptr_t *>(group.host_nodes.data()),
        group.host_nodes.size() * sizeof(std::uintptr_t));
    if (error != CUDA_SUCCESS)
      return error;
    error = driver.memcpy_host_to_device.call(
        group.device_control,
        const_cast<CudaGraphBoundedGroupControl *>(&group.host_control),
        sizeof(group.host_control));
    if (error != CUDA_SUCCESS)
      return error;
  }
  return CUDA_SUCCESS;
}

void BoundedGraphControls::release(bool backend_safe, void *cleanup_stream) {
  for (auto &control : controls) {
    if (backend_safe && control.device_control)
      free_retired_allocation(control.device_control, cleanup_stream);
    control.device_control = nullptr;
  }
  controls.clear();
  for (auto &group : groups) {
    if (backend_safe && group.device_control)
      free_retired_allocation(group.device_control, cleanup_stream);
    group.device_control = nullptr;
    if (backend_safe && group.device_nodes)
      free_retired_allocation(group.device_nodes, cleanup_stream);
    group.device_nodes = nullptr;
  }
  groups.clear();
  group_indices.clear();
  group_member_indices.clear();
}

std::uint64_t BoundedGraphControls::device_bytes() const {
  std::uint64_t bytes =
      std::count_if(controls.begin(), controls.end(),
                    [](const auto &control) {
                      return control.device_control != nullptr;
                    }) *
      sizeof(CudaGraphBoundedExtentControl);
  for (const auto &group : groups) {
    if (group.device_control)
      bytes += sizeof(CudaGraphBoundedGroupControl);
    if (group.device_nodes)
      bytes += group.host_nodes.size() * sizeof(std::uintptr_t);
  }
  return bytes;
}

}  // namespace taichi::lang::cuda
