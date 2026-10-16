#include "taichi/runtime/cuda/fixed_graph.h"
#include "taichi/program/runtime_fault.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include "taichi/rhi/cuda/graph_region_builder.h"
#include "taichi/rhi/cuda/primitives/graph_ptx.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#include "taichi/runtime/cuda/bounded_graph.h"
#include "taichi/runtime/cuda/parallel_graph.h"
#include "taichi/runtime/cuda/predicate_graph.h"

namespace taichi::lang::cuda {
struct FixedGraph::State {
  struct SelectedControl {
    NestedPredicateGraphControls controls;
    bool masked{false};
    std::uint32_t *gates{nullptr};
    std::size_t gate_bytes{0};
    void release(bool backend_safe) {
      controls.release(backend_safe);
      if (backend_safe && gates)
        CUDADriver::get_instance().mem_free(gates);
      gates = nullptr;
    }
  };
  std::shared_ptr<RuntimeFaultDomain> fault_domain;
  std::shared_ptr<void> operands;
  bool binding_frame{false};
  CUgraph graph{nullptr};
  CUgraphExec executable{nullptr};
  void *stream{nullptr};
  void *arguments{nullptr};
  CudaGraphConditionalControl64 *controls{nullptr};
  BoundedGraphControls bounded;
  std::vector<std::unique_ptr<ParallelGraphCapture>> parallel;
  std::vector<std::unique_ptr<SelectedControl>> selected_controls;
  std::size_t control_bytes{0};
  std::size_t bytes{0};
  std::size_t nodes{0};
  std::size_t capture_bytes{0};
  std::vector<std::shared_ptr<CudaGraphCaptureResources>> capture_resources;
  std::vector<std::uint8_t> host_arguments;
  bool capturing{false};
  bool submitted{false};
  void release_selected(bool backend_safe) {
    for (const auto &region : selected_controls)
      region->release(backend_safe);
  }
  ~State() {
    bool backend_safe = false;
    if (fault_domain &&
        (!fault_domain->backend_calls_safe() ||
         fault_domain->state() == RuntimeLifecycleState::kFinalized)) {
      for (const auto &resource : capture_resources)
        resource->release(false);
      bounded.release(false);
      release_selected(false);
      for (const auto &region : parallel)
        region->release(false);
      return;
    }
    try {
      auto &cuda = CUDAContext::get_instance();
      auto guard = cuda.get_guard();
      auto lock = cuda.get_submission_lock_guard();
      auto &driver = CUDADriver::get_instance();
      backend_safe = true;
      bool may_call = true;
      auto observe = [&](std::uint32_t result, const char *operation) {
        if (result == CUDA_SUCCESS)
          return;
        backend_safe = false;
        if (classify_cuda_driver_error(result) ==
            BackendErrorClassification::kFatal)
          may_call = false;
        if (fault_domain)
          fault_domain->report_backend_error(
              BackendRuntimeError(Arch::cuda, result, operation,
                                  "Fixed CUDA graph retirement failed"),
              0);
      };
      // Cleanup also handles construction failure without publishing a partial
      // executable. No default-stream wait is needed for an unused graph.
      if (capturing)
        observe(driver.stream_end_capture.call(stream, &graph), "end capture");
      if (may_call && submitted)
        observe(driver.stream_synchronize.call(nullptr), "wait graph replay");
      if (may_call && stream)
        observe(driver.stream_synchronize.call(stream),
                "wait graph preparation");
      if (may_call && executable)
        observe(driver.graph_exec_destroy.call(executable),
                "destroy graph executable");
      if (may_call && graph)
        observe(driver.graph_destroy.call(graph), "destroy graph");
      for (const auto &resource : capture_resources)
        resource->release(backend_safe);
      capture_resources.clear();
      bounded.release(backend_safe);
    release_selected(backend_safe);
      for (const auto &region : parallel)
        region->release(backend_safe);
      if (backend_safe && controls)
        observe(driver.mem_free.call(controls), "free graph control");
      if (backend_safe && arguments)
        observe(driver.mem_free.call(arguments), "free graph arguments");
      if (may_call && stream)
        observe(driver.stream_destroy.call(stream), "destroy graph stream");
    } catch (const BackendRuntimeError &error) {
      if (fault_domain)
        fault_domain->report_backend_error(error, 0);
      for (const auto &resource : capture_resources)
        resource->release(false);
      capture_resources.clear();
      bounded.release(false);
      release_selected(false);
      for (const auto &region : parallel)
        region->release(false);
    } catch (...) {
      // Cleanup must not throw while unwinding a driver failure.
      for (const auto &resource : capture_resources)
        resource->release(false);
      capture_resources.clear();
      bounded.release(false);
      release_selected(false);
      for (const auto &region : parallel)
        region->release(false);
    }
  }
};

bool FixedGraph::available() {
  auto &driver = CUDADriver::get_instance();
  return driver.stream_begin_capture.available() &&
         driver.stream_end_capture.available() &&
         driver.graph_instantiate_with_flags.available() &&
         driver.graph_launch.available();
}

FixedGraph::FixedGraph(const std::vector<Dispatch> &dispatches,
                       std::shared_ptr<RuntimeFaultDomain> fault_domain)
    : FixedGraph(std::vector<Operation>(dispatches.begin(), dispatches.end()),
                 std::move(fault_domain)) {
}

FixedGraph::FixedGraph(const std::vector<Operation> &operations,
                       std::shared_ptr<RuntimeFaultDomain> fault_domain)
    : FixedGraph(operations, std::move(fault_domain), {}, false) {
}

FixedGraph::FixedGraph(const std::vector<Operation> &operations,
                       std::shared_ptr<RuntimeFaultDomain> fault_domain,
                       std::shared_ptr<void> operands,
                       bool binding_frame)
    : state_(std::make_unique<State>()) {
  state_->fault_domain = std::move(fault_domain);
  state_->operands = std::move(operands);
  state_->binding_frame = binding_frame;
  if (state_->fault_domain)
    state_->fault_domain->throw_if_submission_disallowed("prepare CUDA graph");
  try {
    initialize(operations);
  } catch (const BackendRuntimeError &error) {
    if (state_->fault_domain)
      state_->fault_domain->report_backend_error(error, 0);
    throw;
  } catch (const GraphBuildError &error) {
    if (state_->fault_domain)
      state_->fault_domain->report_backend_error(
          BackendRuntimeError(Arch::cuda, error.code(), error.stage(),
                              error.what()),
          0);
    throw;
  }
}

std::shared_ptr<FixedGraph> FixedGraph::capture_binding_frame(
    const std::vector<Operation> &operations,
    std::shared_ptr<void> operands,
    std::shared_ptr<RuntimeFaultDomain> fault_domain) {
  if (!operands)
    throw std::invalid_argument(
        "CUDA binding frame requires operand ownership");
  return std::shared_ptr<FixedGraph>(new FixedGraph(
      operations, std::move(fault_domain), std::move(operands), true));
}

void *FixedGraph::binding_source() const {
  if (!state_->binding_frame || !state_->graph)
    throw std::invalid_argument("CUDA graph is not a captured binding frame");
  return state_->graph;
}

void FixedGraph::initialize(const std::vector<Operation> &operations) {
  TI_ERROR_IF(operations.empty() || !available(),
              "Fixed CUDA Graph requires operations and driver graph support");
  auto &cuda = CUDAContext::get_instance();
  auto guard = cuda.get_guard();
  auto lock = cuda.get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  auto &state = *state_;
  if (state.binding_frame) {
    if (!driver.graph_node_get_type.available() ||
        !driver.graph_exec_update_v1.available())
      throw std::invalid_argument("CUDA binding-frame updates are unavailable");
    for (const auto &operation : operations) {
      const auto *dispatch = std::get_if<Dispatch>(&operation);
      if (!dispatch || dispatch->bounded_extent || dispatch->adaptive_block_dim)
        throw std::invalid_argument(
            "CUDA binding frames require a flat fixed dispatch topology");
    }
  }
  std::vector<const Dispatch *> dispatches;
  std::unordered_map<const Dispatch *, std::size_t> dispatch_index;
  std::unordered_map<const While *, std::size_t> loop_index;
  std::unordered_map<const Branch *, std::size_t> branch_index;
  std::unordered_map<const Parallel *, std::size_t> parallel_index;
  std::vector<const Parallel *> parallel_regions;
  // Each occurrence owns its control storage, even for a reused descriptor.
  // Masked packets contain gate addresses and cannot share across occurrences.
  std::unordered_map<const Operation *, std::size_t> selected_index;
  std::vector<const While *> selected_regions;
  struct SelectedBuild {
    NestedPredicateGraphLayout layout;
    std::vector<std::size_t> packets;
  };
  std::vector<SelectedBuild> selected_builds;
  std::unordered_set<const void *> active_controls;
  std::vector<CudaGraphConditionalControl64> controls;
  std::vector<CudaGraphConditionalControl> branch_controls;
  std::function<void(const std::vector<Operation> &)> collect;
  collect = [&](const std::vector<Operation> &sequence) {
    for (const auto &operation : sequence) {
      if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
        if (dispatch_index.emplace(dispatch, dispatches.size()).second)
          dispatches.push_back(dispatch);
      } else if (const auto *value =
                     std::get_if<std::shared_ptr<const Branch>>(&operation)) {
        const auto &branch = *value;
        if (!branch || !branch->selector)
          throw std::invalid_argument("Fixed CUDA branch requires a selector");
        branch->selection.validate(branch->bodies.size());
        if (branch->bodies.size() > std::numeric_limits<std::uint32_t>::max())
          throw std::invalid_argument("CUDA branch count exceeds its node ABI");
        if (active_controls.count(branch.get()))
          throw std::invalid_argument(
              "Fixed CUDA control regions contain a cycle");
        if (!branch_index.emplace(branch.get(), branch_controls.size()).second)
          continue;
        active_controls.insert(branch.get());
        CudaGraphConditionalControl control;
        control.predicate = reinterpret_cast<std::uintptr_t>(branch->selector);
        control.iteration =
            branch->selection.kind == BranchSelection::Kind::nonzero ? 0 : 2;
        control.max_iterations = branch->bodies.size();
        control.continue_while_nonzero = branch->selection.has_default
                                             ? branch->bodies.size() - 1
                                             : ~std::uint32_t{0};
        branch_controls.push_back(control);
        collect(branch->condition);
        for (const auto &body : branch->bodies)
          collect(body);
        active_controls.erase(branch.get());
      } else if (const auto *value =
                     std::get_if<std::shared_ptr<const Parallel>>(&operation)) {
        const auto &parallel = *value;
        if (!parallel || parallel->branches.size() < 2 ||
            !active_controls.empty())
          throw std::invalid_argument(
              "Fixed CUDA parallel requires a flat region and at least two "
              "branches");
        if (!parallel_index.emplace(parallel.get(), parallel_regions.size())
                 .second)
          continue;
        parallel_regions.push_back(parallel.get());
        for (const auto &branch : parallel->branches) {
          if (branch.empty())
            throw std::invalid_argument(
                "Fixed CUDA parallel requires nonempty branches");
          for (const auto &item : branch) {
            const auto *dispatch = std::get_if<Dispatch>(&item);
            if (!dispatch || dispatch->bounded_extent ||
                dispatch->adaptive_block_dim)
              throw std::invalid_argument(
                  "Fixed CUDA parallel requires ordinary dispatch branches");
          }
          collect(branch);
        }
      } else {
        const auto &loop = std::get<std::shared_ptr<const While>>(operation);
        if (!loop || !loop->predicate || loop->body.empty())
          throw std::invalid_argument(
              "Fixed CUDA while requires a predicate and nonempty body");
        if (loop->recipe) {
          const auto flat = [](const auto &region) {
            return std::all_of(region.begin(), region.end(), [](const auto &op) {
              const auto *dispatch = std::get_if<Dispatch>(&op);
              return dispatch && !dispatch->external &&
                     !dispatch->bounded_extent && !dispatch->adaptive_block_dim;
            });
          };
          if (!active_controls.empty() || !flat(loop->condition) ||
              !flat(loop->body))
            throw std::invalid_argument(
                "Selected CUDA control requires flat kernel "
                "regions");
          loop->recipe->validate(
              loop->condition.size(), loop->condition.size() + loop->body.size(),
              loop->continue_while_nonzero);
          for (const auto &inner : loop->recipe->inners)
            if (!inner.predicate)
              throw std::invalid_argument("Missing CUDA nested predicate");
          selected_index.emplace(&operation, selected_regions.size());
          selected_regions.push_back(loop.get());
          collect(loop->condition);
          if (loop->recipe->encoding == CudaControlEncoding::device_update)
            collect(loop->body);
          continue;
        }
        if (active_controls.count(loop.get()))
          throw std::invalid_argument(
              "Fixed CUDA control regions contain a cycle");
        if (!loop_index.emplace(loop.get(), controls.size()).second)
          continue;
        active_controls.insert(loop.get());
        CudaGraphConditionalControl64 control;
        control.predicate = reinterpret_cast<std::uintptr_t>(loop->predicate);
        control.max_iterations = loop->max_iterations;
        control.continue_while_nonzero = loop->continue_while_nonzero;
        controls.push_back(control);
        collect(loop->condition);
        collect(loop->body);
        active_controls.erase(loop.get());
      }
    }
  };
  collect(operations);
  const auto gated_begin = dispatches.size();
  std::vector<void *> packet_gates;
  bool has_device_update = false;
  for (const auto *loop : selected_regions) {
    const auto &recipe = *loop->recipe;
    const bool masked = recipe.encoding == CudaControlEncoding::masked;
    if (!masked &&
        (!driver.launch_kernel_ex.available() || !driver.graph_upload.available()))
      throw std::invalid_argument(
          "Selected CUDA device-update graph support is unavailable");
    if (recipe.inners.size() >=
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
      throw std::invalid_argument("CUDA nested gate storage size overflow");
    state.selected_controls.push_back(std::make_unique<State::SelectedControl>());
    auto &region = *state.selected_controls.back();
    region.masked = masked;
    region.gate_bytes = (recipe.inners.size() + 1) * sizeof(std::uint32_t);
    driver.malloc(reinterpret_cast<void **>(&region.gates), region.gate_bytes);
    selected_builds.emplace_back();
    auto &build = selected_builds.back();
    auto &layout = build.layout;
    layout.predicate = reinterpret_cast<std::uintptr_t>(loop->predicate);
    layout.gate = reinterpret_cast<std::uintptr_t>(region.gates);
    layout.iterations = loop->max_iterations;
    layout.initial_dispatches = loop->condition.size();
    layout.dispatches = layout.initial_dispatches + loop->body.size();
    for (std::size_t i = 0; i < recipe.inners.size(); ++i) {
      const auto &inner = recipe.inners[i];
      layout.inners.push_back(
          {reinterpret_cast<std::uintptr_t>(inner.predicate),
           reinterpret_cast<std::uintptr_t>(region.gates + i + 1),
           inner.max_iterations, inner.condition_begin, inner.body_begin,
           inner.end});
    }
    build.packets.reserve(layout.dispatches);
    for (const auto &item : loop->condition)
      build.packets.push_back(dispatch_index.at(&std::get<Dispatch>(item)));
    std::size_t child = 0;
    for (std::size_t i = 0; i < loop->body.size(); ++i) {
      const auto *dispatch = &std::get<Dispatch>(loop->body[i]);
      if (!masked) {
        build.packets.push_back(dispatch_index.at(dispatch));
        continue;
      }
      const auto index = layout.initial_dispatches + i;
      while (child < layout.inners.size() && index >= layout.inners[child].end)
        ++child;
      const auto gate = child < layout.inners.size() &&
                                index >= layout.inners[child].body_begin
                            ? layout.inners[child].gate
                            : layout.gate;
      build.packets.push_back(dispatches.size());
      dispatches.push_back(dispatch);
      packet_gates.push_back(reinterpret_cast<void *>(gate));
    }
    if (masked) {
      driver_graph_prepare_mask_latch();
    } else {
      has_device_update = true;
      const auto error = region.controls.initialize(std::move(layout), false);
      if (error != CUDA_SUCCESS)
        throw BackendRuntimeError(Arch::cuda, error,
                                  "prepare nested CUDA controls",
                                  "Selected device-update controls are unavailable");
    }
  }
  for (const auto *parallel : parallel_regions) {
    state.parallel.push_back(std::make_unique<ParallelGraphCapture>());
    state.parallel.back()->prepare(parallel->branches.size());
  }
  if (!controls.empty() || !branch_controls.empty()) {
    if (!GraphRegionBuilder::available())
      throw std::invalid_argument(
          "Fixed CUDA control requires native conditional graph support");
    driver_graph_prepare_conditional_setter();
  }
  std::vector<KernelLauncher::GraphLaunchPacket> packets(dispatches.size());
  std::vector<BoundedGraphDispatch> bounded;
  std::uint32_t grid_limit = 0;
  std::vector<std::size_t> offsets;
  auto &image = state.host_arguments;
  constexpr std::size_t alignment = 16;  // RuntimeContext argument ABI.
  for (std::size_t i = 0; i < dispatches.size(); ++i) {
    const auto &dispatch = *dispatches[i];
    if (dispatch.external) {
      offsets.push_back(0);
      continue;
    }
    const auto gate = i >= gated_begin ? packet_gates[i - gated_begin] : nullptr;
    TI_ERROR_IF(
        !dispatch.launcher || !dispatch.context ||
            !(gate
                  ? dispatch.launcher->prepare_cuda_graph_gated_packet(
                        dispatch.handle, *dispatch.context, packets[i],
                        gate, 1u)
              : dispatch.bounded_extent
                  ? dispatch.launcher->prepare_cuda_graph_bounded_packet(
                        dispatch.handle, *dispatch.context, packets[i],
                        dispatch.bounded_extent, dispatch.bounded_capacity)
                  : dispatch.launcher->prepare_cuda_graph_launch_packet(
                        dispatch.handle, *dispatch.context, packets[i])),
        "Unsupported fixed CUDA Graph dispatch");
    auto &packet = packets[i];
    if (dispatch.adaptive_block_dim) {
      TI_ERROR_IF(!controls.empty() || !branch_controls.empty() ||
                      !selected_regions.empty() ||
                      !packet.bounded_range ||
                      !dispatch.bounded_binding_identity ||
                      packet.offloaded_tasks.size() != 1 ||
                      packet.offloaded_tasks[0].block_dim !=
                          dispatch.adaptive_block_dim,
                  "Adaptive CUDA Graph requires a flat bounded kernel with "
                  "matching block dimension");
      const auto limit = dispatch.launcher->bounded_graph_grid_limit();
      TI_ERROR_IF(
          !limit || (grid_limit && grid_limit != limit),
          "Adaptive CUDA Graph requires a consistent consumer grid limit");
      grid_limit = limit;
      if (bounded.empty())
        bounded.resize(dispatches.size());
      auto &entry = bounded[i];
      entry.extent = reinterpret_cast<std::uintptr_t>(dispatch.bounded_extent);
      entry.capacity = dispatch.bounded_capacity;
      entry.block_dim = dispatch.adaptive_block_dim;
      entry.adaptive = true;
      if (i && dispatch.grouped_update) {
        const auto &previous = *dispatches[i - 1];
        entry.join_previous =
            previous.grouped_update &&
            previous.adaptive_block_dim == dispatch.adaptive_block_dim &&
            previous.bounded_capacity == dispatch.bounded_capacity &&
            previous.bounded_binding_identity ==
                dispatch.bounded_binding_identity;
      }
    }
    TI_ERROR_IF(
        !packet.bounded_range &&
            std::any_of(packet.offloaded_tasks.begin(),
                        packet.offloaded_tasks.end(),
                        [](const auto &task) { return task.one_to_one; }),
        "CUDA exact kernel requires its bounded launch recipe");
    TI_ERROR_IF(
        image.size() > std::numeric_limits<std::size_t>::max() - alignment,
        "Fixed CUDA argument image overflows address space");
    const auto offset = (image.size() + alignment - 1) & ~(alignment - 1);
    offsets.push_back(offset);
    TI_ERROR_IF(packet.device_arg_buffer_size >
                    std::numeric_limits<std::size_t>::max() - offset,
                "Fixed CUDA argument image overflows address space");
    if (packet.device_arg_buffer_size) {
      image.resize(offset + packet.device_arg_buffer_size);
      if (!packet.argument_prefix.empty())
        std::memcpy(image.data() + offset, packet.argument_prefix.data(),
                    packet.argument_prefix.size());
      if (packet.arg_buffer_size)
        std::memcpy(image.data() + offset + packet.arg_buffer_prefix_size,
                    packet.context.arg_buffer, packet.arg_buffer_size);
    }
  }
  if (grid_limit) {
    std::uint32_t error = CUDA_SUCCESS;
    if (!state.bounded.initialize(bounded, grid_limit, false, &error))
      throw BackendRuntimeError(Arch::cuda, error,
                                "prepare adaptive CUDA controls",
                                "Selected adaptive CUDA graph is unavailable");
  }
  state.bytes = image.size();
  driver.stream_create(&state.stream, CU_STREAM_NON_BLOCKING);
  if (state.bytes) {
    driver.malloc(&state.arguments, state.bytes);
    driver.memcpy_host_to_device_async(state.arguments, image.data(),
                                       image.size(), state.stream);
    // Only argument preparation submits here, never user mathematical work.
    driver.stream_synchronize(state.stream);
  }
  std::vector<std::uint8_t>().swap(state.host_arguments);
  for (std::size_t i = 0; i < packets.size(); ++i) {
    if (packets[i].device_arg_buffer_size) {
      packets[i].device_arg_buffer =
          static_cast<char *>(state.arguments) + offsets[i];
      packets[i].context.arg_buffer =
          static_cast<char *>(packets[i].device_arg_buffer) +
          packets[i].arg_buffer_prefix_size;
    }
  }
  auto record_dispatch_index = [&](std::size_t index, void *stream) {
    const auto &dispatch = *dispatches[index];
    if (dispatch.external) {
      const auto retain = [&] {
        if (dispatch.take_resources) {
          if (auto resource = dispatch.take_resources())
            state.capture_resources.push_back(std::move(resource));
        }
      };
      try {
        dispatch.external(stream);
      } catch (...) {
        retain();
        throw;
      }
      retain();
    } else {
      if (grid_limit) {
        const auto error = state.bounded.capture(index, *dispatch.launcher,
                                                 packets[index], stream);
        if (error != CUDA_SUCCESS)
          throw BackendRuntimeError(
              Arch::cuda, error, "capture adaptive CUDA kernel",
              "Selected adaptive CUDA kernel capture failed");
      } else {
        dispatch.launcher->capture_cuda_graph_launch(packets[index], stream);
      }
    }
  };
  const auto record_dispatch = [&](const Dispatch &dispatch, void *stream) {
    record_dispatch_index(dispatch_index.at(&dispatch), stream);
  };
  const auto record_operation = [&](const Operation &operation, void *stream) {
    if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
      record_dispatch(*dispatch, stream);
      return;
    }
    if (const auto *value =
            std::get_if<std::shared_ptr<const While>>(&operation)) {
      const auto index = selected_index.at(&operation);
      auto &region = *state.selected_controls[index];
      const auto &build = selected_builds[index];
      if (region.masked) {
        capture_masked_while_graph(
            build.layout, (*value)->continue_while_nonzero, stream,
            [&](std::size_t dispatch) {
              record_dispatch_index(build.packets[dispatch], stream);
            });
        return;
      }
      const auto error = region.controls.capture(
          stream, [&](std::size_t index, std::vector<void *> *nodes) {
            const auto packet_index = build.packets[index];
            const auto &dispatch = *dispatches[packet_index];
            if (!nodes) {
              record_dispatch_index(packet_index, stream);
              return static_cast<std::uint32_t>(CUDA_SUCCESS);
            }
            std::uint32_t result = CUDA_SUCCESS;
            if (!dispatch.launcher->capture_cuda_graph_updatable_launch(
                    packets[packet_index], stream, nodes,
                    &result))
              return result == CUDA_SUCCESS
                         ? static_cast<std::uint32_t>(CUDA_ERROR_NOT_SUPPORTED)
                         : result;
            return result;
          });
      if (error != CUDA_SUCCESS)
        throw BackendRuntimeError(Arch::cuda, error,
                                  "capture nested CUDA payload",
                                  "Selected device-update capture failed");
      return;
    }
    const auto &parallel =
        *std::get<std::shared_ptr<const Parallel>>(operation);
    const auto error = state.parallel[parallel_index.at(&parallel)]->capture(
        stream, [&](std::size_t index, void *branch_stream) {
          for (const auto &item : parallel.branches[index])
            record_dispatch(std::get<Dispatch>(item), branch_stream);
          return CUDA_SUCCESS;
        });
    if (error != CUDA_SUCCESS)
      throw BackendRuntimeError(Arch::cuda, error,
                                "capture fixed CUDA parallel",
                                "CUDA branch capture failed");
  };
  if (controls.empty() && branch_controls.empty()) {
    auto capture_lock = cuda.get_graph_capture_lock_guard();
    driver.stream_begin_capture(state.stream,
                                CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    state.capturing = true;
    for (const auto &operation : operations)
      record_operation(operation, state.stream);
    const auto error =
        driver.stream_end_capture.call(state.stream, &state.graph);
    state.capturing = false;
    if (error != CUDA_SUCCESS)
      throw BackendRuntimeError(
          Arch::cuda, error, "end fixed CUDA capture",
          driver.stream_end_capture.get_error_message(error));
  } else {
    if (controls.size() >
        std::numeric_limits<std::size_t>::max() / sizeof(controls[0]))
      throw std::invalid_argument("Fixed CUDA control storage overflow");
    const auto loop_bytes = controls.size() * sizeof(controls[0]);
    if (branch_controls.size() >
        (std::numeric_limits<std::size_t>::max() - loop_bytes) /
            sizeof(branch_controls[0]))
      throw std::invalid_argument("Fixed CUDA branch storage overflow");
    state.control_bytes =
        loop_bytes + branch_controls.size() * sizeof(branch_controls[0]);
    driver.malloc(reinterpret_cast<void **>(&state.controls),
                  state.control_bytes);
    auto *branch_storage = reinterpret_cast<CudaGraphConditionalControl *>(
        reinterpret_cast<std::uint8_t *>(state.controls) + loop_bytes);
    if (loop_bytes)
      driver.memcpy_host_to_device_async(state.controls, controls.data(),
                                         loop_bytes, state.stream);
    if (!branch_controls.empty())
      driver.memcpy_host_to_device_async(branch_storage, branch_controls.data(),
                                         state.control_bytes - loop_bytes,
                                         state.stream);
    driver.stream_synchronize(state.stream);
    driver.graph_create(&state.graph, 0);
    GraphRegionBuilder builder(state.graph, state.stream);
    using Dependencies = GraphRegionBuilder::Dependencies;
    std::function<Dependencies(CUgraph, Dependencies,
                               const std::vector<Operation> &)>
        record;
    record = [&](CUgraph graph, Dependencies dependencies,
                 const std::vector<Operation> &sequence) {
      std::size_t begin = 0;
      while (begin < sequence.size()) {
        auto end = begin;
        while (end < sequence.size() &&
               !std::holds_alternative<std::shared_ptr<const Branch>>(
                   sequence[end]) &&
               (!std::holds_alternative<std::shared_ptr<const While>>(
                    sequence[end]) ||
                std::get<std::shared_ptr<const While>>(sequence[end])
                    ->recipe))
          ++end;
        if (end != begin) {
          dependencies =
              builder.capture(graph, dependencies, [&](void *stream) {
                for (auto i = begin; i < end; ++i)
                  record_operation(sequence[i], stream);
              });
          begin = end;
          continue;
        }
        if (const auto *value =
                std::get_if<std::shared_ptr<const Branch>>(&sequence[begin])) {
          const auto &branch = **value;
          ++begin;
          auto *control = branch_storage + branch_index.at(&branch);
          const auto handle = builder.conditional_handle();
          dependencies =
              record(graph, std::move(dependencies), branch.condition);
          dependencies =
              builder.capture(graph, dependencies, [&](void *stream) {
                driver_graph_set_branch_conditional(control, handle, stream);
              });
          const auto node = builder.add_branch(
              graph, dependencies, handle,
              branch.selection.kind == BranchSelection::Kind::nonzero
                  ? GraphRegionBuilder::BranchKind::if_else
                  : GraphRegionBuilder::BranchKind::indexed,
              branch.bodies.size());
          for (std::size_t i = 0; i < node.bodies.size(); ++i) {
            record(node.bodies[i], {}, branch.bodies[i]);
            std::size_t count = 0;
            driver.graph_get_nodes(node.bodies[i], nullptr, &count);
            state.nodes += count;
          }
          dependencies = {node.node};
          continue;
        }
        const auto &loop =
            std::get<std::shared_ptr<const While>>(sequence[begin++]);
        auto *control = state.controls + loop_index.at(loop.get());
        const auto handle = builder.conditional_handle();
        dependencies = record(graph, std::move(dependencies), loop->condition);
        dependencies = builder.capture(graph, dependencies, [&](void *stream) {
          driver_graph_set_conditional64(control, handle, stream, true);
        });
        const auto node = builder.add_while(graph, dependencies, handle);
        auto tail = record(node.body, {}, loop->body);
        tail = record(node.body, std::move(tail), loop->condition);
        builder.capture(node.body, tail, [&](void *stream) {
          driver_graph_set_conditional64(control, handle, stream, false);
        });
        std::size_t count = 0;
        driver.graph_get_nodes(node.body, nullptr, &count);
        state.nodes += count;
        dependencies = {node.node};
      }
      return dependencies;
    };
    record(state.graph, {}, operations);
  }
  for (const auto &resource : state.capture_resources) {
    resource->finalize();
    const auto bytes = resource->requested_device_bytes();
    if (bytes > std::numeric_limits<std::size_t>::max() - state.capture_bytes)
      throw std::invalid_argument("Fixed CUDA capture storage size overflow");
    state.capture_bytes += bytes;
  }
  for (const auto &region : state.selected_controls) {
    if (!region->masked) {
      const auto error = region->controls.finalize_nodes();
      if (error != CUDA_SUCCESS)
        throw BackendRuntimeError(Arch::cuda, error,
                                  "finalize nested CUDA nodes",
                                  "Selected device-update node table failed");
    }
    const auto bytes = region->controls.device_bytes();
    if (bytes > std::numeric_limits<std::size_t>::max() - state.control_bytes ||
        region->gate_bytes >
            std::numeric_limits<std::size_t>::max() - state.control_bytes - bytes)
      throw std::invalid_argument("CUDA nested control storage size overflow");
    state.control_bytes += bytes + region->gate_bytes;
  }
  if (!state.binding_frame)
    driver.graph_instantiate_with_flags(&state.executable, state.graph, 0);
  if (grid_limit || has_device_update) {
    driver.graph_upload(state.executable, state.stream);
    driver.stream_synchronize(state.stream);
    if (grid_limit) {
      const auto error = state.bounded.upload();
      if (error != CUDA_SUCCESS)
        throw BackendRuntimeError(Arch::cuda, error,
                                  "upload adaptive CUDA controls",
                                  "Selected adaptive CUDA control upload failed");
    }
    for (const auto &region : state.selected_controls) {
      if (region->masked)
        continue;
      const auto error = region->controls.upload(false);
      if (error != CUDA_SUCCESS)
        throw BackendRuntimeError(Arch::cuda, error,
                                  "upload nested CUDA controls",
                                  "Selected device-update control upload failed");
    }
  }
  std::size_t root_nodes = 0;
  driver.graph_get_nodes(state.graph, nullptr, &root_nodes);
  state.nodes += root_nodes;
  if (state.binding_frame) {
    if (!root_nodes)
      throw std::invalid_argument("CUDA binding frame contains no commands");
    std::vector<void *> nodes(root_nodes);
    driver.graph_get_nodes(state.graph, nodes.data(), &root_nodes);
    for (auto *node : nodes) {
      std::uint32_t type = 0;
      driver.graph_node_get_type(node, &type);
      // Match the JIT frame contract: kernel, memcpy and memset only.
      if (type != 0 && type != 1 && type != 2)
        throw std::invalid_argument("Unsupported CUDA binding-frame node");
    }
  } else {
    driver.graph_destroy(state.graph);
    state.graph = nullptr;
  }
  driver.stream_destroy(state.stream);
  state.stream = nullptr;
}

FixedGraph::~FixedGraph() = default;
void FixedGraph::run() {
  if (state_->binding_frame)
    throw std::invalid_argument(
        "Captured CUDA binding frames require a replay family");
  if (state_->fault_domain)
    state_->fault_domain->throw_if_submission_disallowed("launch CUDA graph");
  auto &cuda = CUDAContext::get_instance();
  auto guard = cuda.get_guard();
  auto lock = cuda.get_submission_lock_guard();
  state_->submitted = true;
  try {
    CUDADriver::get_instance().graph_launch(state_->executable, nullptr);
  } catch (const BackendRuntimeError &error) {
    if (state_->fault_domain)
      state_->fault_domain->report_backend_error(error, 0);
    throw;
  }
}
std::size_t FixedGraph::argument_bytes() const {
  return state_->bytes;
}
std::size_t FixedGraph::control_bytes() const {
  return state_->control_bytes + state_->bounded.device_bytes();
}
std::size_t FixedGraph::capture_bytes() const {
  return state_->capture_bytes;
}
std::size_t FixedGraph::node_count() const {
  return state_->nodes;
}
}  // namespace taichi::lang::cuda
