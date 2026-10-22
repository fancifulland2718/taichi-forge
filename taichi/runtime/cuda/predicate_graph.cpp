#include "taichi/runtime/cuda/predicate_graph.h"
#include "taichi/runtime/cuda/retired_allocation.h"

#include <limits>
#include <stdexcept>
#include <utility>

#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/runtime/control_encoding_cost.h"

namespace taichi::lang::cuda {
namespace {

std::optional<NestedControlEncodingCost> predicate_layout_cost(
    const NestedPredicateGraphLayout &layout) {
  if (!layout.predicate || !layout.gate ||
      layout.initial_dispatches >= layout.dispatches ||
      (!layout.inners.empty() && !layout.initial_dispatches))
    return std::nullopt;
  NestedControlCostAccumulator estimate(
      layout.dispatches, layout.initial_dispatches, layout.iterations);
  auto cursor = layout.initial_dispatches;
  for (const auto &inner : layout.inners) {
    if (!inner.predicate || !inner.gate || cursor > inner.condition_begin ||
        inner.condition_begin >= inner.body_begin ||
        inner.body_begin >= inner.end || inner.end >= layout.dispatches ||
        !estimate.add_inner(inner.end - inner.body_begin, inner.iterations))
      return std::nullopt;
    cursor = inner.end;
  }
  return estimate.finish();
}

}  // namespace

void capture_masked_while_graph(const NestedPredicateGraphLayout &layout,
                                bool continue_while_nonzero,
                                void *stream,
                                const MaskedGraphCapture &payload) {
  const auto cost = predicate_layout_cost(layout);
  if (!payload || !cost || !cost->total_actions(false))
    throw std::invalid_argument("Invalid CUDA masked while graph layout");
  for (std::size_t dispatch = 0; dispatch < layout.initial_dispatches; ++dispatch)
    payload(dispatch);
  for (std::uint64_t outer = 0; outer < layout.iterations; ++outer) {
    driver_graph_latch_while(reinterpret_cast<void *>(layout.predicate),
                             reinterpret_cast<void *>(layout.gate),
                             continue_while_nonzero, stream);
    auto cursor = layout.initial_dispatches;
    for (const auto &inner : layout.inners) {
      for (; cursor < inner.body_begin; ++cursor)
        payload(cursor);
      for (std::uint64_t iteration = 0; iteration < inner.iterations; ++iteration) {
        driver_graph_latch_nested_while(
            reinterpret_cast<void *>(layout.gate),
            reinterpret_cast<void *>(inner.predicate),
            reinterpret_cast<void *>(inner.gate), true, stream);
        for (auto dispatch = inner.body_begin; dispatch < inner.end; ++dispatch)
          payload(dispatch);
      }
      cursor = inner.end;
    }
    for (; cursor < layout.dispatches; ++cursor)
      payload(cursor);
  }
}

void capture_masked_branch_graph(const MaskedBranchGraphLayout &layout,
                                 void *stream,
                                 const MaskedGraphCapture &payload) {
  if (!payload || !layout.predicate || !layout.gate || !layout.dispatches ||
      !layout.branch_count ||
      (layout.conditional_type != 0 && layout.conditional_type != 2) ||
      (layout.conditional_type == 0 && layout.branch_count > 2) ||
      (layout.default_branch != std::numeric_limits<std::uint32_t>::max() &&
       layout.default_branch >= layout.branch_count))
    throw std::invalid_argument("Invalid CUDA masked branch graph layout");
  driver_graph_latch_branch(
      reinterpret_cast<void *>(layout.predicate),
      reinterpret_cast<void *>(layout.gate), layout.conditional_type,
      layout.branch_count, layout.default_branch, stream);
  for (std::size_t dispatch = 0; dispatch < layout.dispatches; ++dispatch)
    payload(dispatch);
}

struct NestedPredicateGraphControls::BuildState {
  NestedPredicateGraphLayout layout;
  std::vector<std::vector<std::uintptr_t>> nodes;
};

NestedPredicateGraphControls::NestedPredicateGraphControls() = default;
NestedPredicateGraphControls::~NestedPredicateGraphControls() = default;

std::uint32_t NestedPredicateGraphControls::initialize(
    NestedPredicateGraphLayout layout,
    bool diagnostics_enabled) {
  if (phase_ != Phase::empty || !layout.initial_dispatches || layout.inners.empty())
    return CUDA_ERROR_NOT_SUPPORTED;
  const auto cost = predicate_layout_cost(layout);
  if (!cost || cost->controllers_per_outer - 1 >
                   std::numeric_limits<std::uint32_t>::max() ||
      cost->expanded_controllers >
          std::numeric_limits<std::size_t>::max() /
              sizeof(CudaGraphPredicateGroupControl))
    return CUDA_ERROR_NOT_SUPPORTED;
  std::uint32_t error = CUDA_SUCCESS;
  if (cost->expanded_controllers &&
      !driver_graph_prepare_bounded_update(&error))
    return error == CUDA_SUCCESS ? CUDA_ERROR_NOT_SUPPORTED : error;

  // Publish ownership before any allocation so partial construction can be
  // retired by the enclosing owner. Reinitialization requires release().
  phase_ = Phase::initialized;
  build_ = std::make_unique<BuildState>();
  build_->layout = std::move(layout);
  host_controls.resize(cost->expanded_controllers);
  build_->nodes.resize(host_controls.size());
  if (host_controls.empty())
    return CUDA_SUCCESS;
  error = CUDADriver::get_instance().malloc.call(
      reinterpret_cast<void **>(&device_controls),
      host_controls.size() * sizeof(CudaGraphPredicateGroupControl));
  if (error != CUDA_SUCCESS)
    return error;
  device_bytes_ = host_controls.size() * sizeof(CudaGraphPredicateGroupControl);
  std::size_t index = 0;
  for (std::uint64_t outer = 0; outer < build_->layout.iterations; ++outer) {
    auto &parent = host_controls[index++];
    parent.predicate = build_->layout.predicate;
    parent.gate = build_->layout.gate;
    parent.telemetry_enabled = diagnostics_enabled ? 1u : 0u;
    parent.child_controls =
        reinterpret_cast<std::uintptr_t>(device_controls + index);
    parent.child_count =
        static_cast<std::uint32_t>(cost->controllers_per_outer - 1);
    for (const auto &inner : build_->layout.inners) {
      for (std::uint64_t iteration = 0; iteration < inner.iterations;
           ++iteration) {
        auto &control = host_controls[index++];
        control.predicate = inner.predicate;
        control.parent_gate = build_->layout.gate;
        control.gate = inner.gate;
        control.telemetry_enabled = diagnostics_enabled ? 1u : 0u;
      }
    }
  }
  return CUDA_SUCCESS;
}

std::uint32_t NestedPredicateGraphControls::capture(void *stream,
                                                   const Capture &payload) {
  if (phase_ != Phase::initialized || !build_ || !payload ||
      (!host_controls.empty() && !device_controls))
    return CUDA_ERROR_NOT_SUPPORTED;
  phase_ = Phase::capturing;
  // Reuse the scratch vector across all encoded dispatches, rather than
  // allocating it once for every repeated payload.
  std::vector<void *> nodes;
  auto append = [&](std::size_t dispatch, std::size_t control) {
    nodes.clear();
    const auto error = payload(dispatch, &nodes);
    if (error != CUDA_SUCCESS)
      return error;
    if (nodes.empty())
      return static_cast<std::uint32_t>(CUDA_ERROR_NOT_SUPPORTED);
    auto &destination = build_->nodes[control];
    for (auto *node : nodes) {
      if (!node)
        return static_cast<std::uint32_t>(CUDA_ERROR_NOT_SUPPORTED);
      destination.push_back(reinterpret_cast<std::uintptr_t>(node));
    }
    return static_cast<std::uint32_t>(CUDA_SUCCESS);
  };
  for (std::size_t i = 0; i < build_->layout.initial_dispatches; ++i) {
    const auto error = payload(i, nullptr);
    if (error != CUDA_SUCCESS)
      return error;
  }
  std::size_t index = 0;
  for (std::uint64_t outer = 0; outer < build_->layout.iterations; ++outer) {
    const auto parent = index++;
    auto error = driver_graph_update_predicate_group(device_controls + parent,
                                                    stream);
    if (error != CUDA_SUCCESS)
      return error;
    auto cursor = build_->layout.initial_dispatches;
    for (const auto &inner : build_->layout.inners) {
      for (; cursor < inner.body_begin; ++cursor) {
        error = append(cursor, parent);
        if (error != CUDA_SUCCESS)
          return error;
      }
      for (std::uint64_t iteration = 0; iteration < inner.iterations;
           ++iteration) {
        const auto child = index++;
        void *updater = nullptr;
        error = driver_graph_update_predicate_group(device_controls + child,
                                                    stream, &updater);
        if (error != CUDA_SUCCESS)
          return error;
        if (!updater)
          return CUDA_ERROR_NOT_SUPPORTED;
        build_->nodes[parent].push_back(
            reinterpret_cast<std::uintptr_t>(updater));
        for (auto dispatch = inner.body_begin; dispatch < inner.end; ++dispatch) {
          error = append(dispatch, child);
          if (error != CUDA_SUCCESS)
            return error;
        }
      }
      cursor = inner.end;
    }
    for (; cursor < build_->layout.dispatches; ++cursor) {
      error = append(cursor, parent);
      if (error != CUDA_SUCCESS)
        return error;
    }
  }
  phase_ = Phase::captured;
  return CUDA_SUCCESS;
}

std::uint32_t NestedPredicateGraphControls::finalize_nodes() {
  if (phase_ != Phase::captured || !build_ || device_nodes)
    return CUDA_ERROR_NOT_SUPPORTED;
  std::size_t count = 0;
  for (const auto &nodes : build_->nodes) {
    if (nodes.empty() ||
        nodes.size() > std::numeric_limits<std::uint32_t>::max() ||
        count > std::numeric_limits<std::size_t>::max() - nodes.size())
      return CUDA_ERROR_NOT_SUPPORTED;
    count += nodes.size();
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(std::uintptr_t) ||
      count > (std::numeric_limits<std::uint64_t>::max() - device_bytes_) /
                  sizeof(std::uintptr_t))
    return CUDA_ERROR_NOT_SUPPORTED;
  if (count) {
    const auto error = CUDADriver::get_instance().malloc.call(
        reinterpret_cast<void **>(&device_nodes), count * sizeof(std::uintptr_t));
    if (error != CUDA_SUCCESS)
      return error;
    device_bytes_ += count * sizeof(std::uintptr_t);
  }
  host_nodes.reserve(count);
  for (std::size_t i = 0; i < build_->nodes.size(); ++i) {
    auto &control = host_controls[i];
    control.device_nodes =
        reinterpret_cast<std::uintptr_t>(device_nodes + host_nodes.size());
    control.node_count = static_cast<std::uint32_t>(build_->nodes[i].size());
    host_nodes.insert(host_nodes.end(), build_->nodes[i].begin(),
                      build_->nodes[i].end());
  }
  build_.reset();
  phase_ = Phase::finalized;
  return CUDA_SUCCESS;
}

std::uint32_t NestedPredicateGraphControls::upload(bool retain_host_tables) {
  if (phase_ != Phase::finalized)
    return CUDA_ERROR_NOT_SUPPORTED;
  if (host_controls.empty()) {
    if (!retain_host_tables)
      phase_ = Phase::uploaded;
    return CUDA_SUCCESS;
  }
  auto &driver = CUDADriver::get_instance();
  auto error = driver.memcpy_host_to_device.call(
      device_nodes, host_nodes.data(),
      host_nodes.size() * sizeof(std::uintptr_t));
  if (error == CUDA_SUCCESS)
    error = driver.memcpy_host_to_device.call(
        device_controls, host_controls.data(),
        host_controls.size() * sizeof(CudaGraphPredicateGroupControl));
  if (error == CUDA_SUCCESS && !retain_host_tables) {
    std::vector<CudaGraphPredicateGroupControl>().swap(host_controls);
    std::vector<std::uintptr_t>().swap(host_nodes);
    phase_ = Phase::uploaded;
  }
  return error;
}

void NestedPredicateGraphControls::release(bool backend_safe,
                                            void *cleanup_stream) {
  if (backend_safe && device_controls)
    free_retired_allocation(device_controls, cleanup_stream);
  device_controls = nullptr;
  if (backend_safe && device_nodes)
    free_retired_allocation(device_nodes, cleanup_stream);
  device_nodes = nullptr;
  host_controls.clear();
  host_nodes.clear();
  build_.reset();
  device_bytes_ = 0;
  phase_ = Phase::empty;
}

std::uint64_t NestedPredicateGraphControls::device_bytes() const {
  return device_bytes_;
}

}  // namespace taichi::lang::cuda
