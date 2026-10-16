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

namespace taichi::lang::cuda {
struct FixedGraph::State {
  std::shared_ptr<RuntimeFaultDomain> fault_domain;
  CUgraph graph{nullptr};
  CUgraphExec executable{nullptr};
  void *stream{nullptr};
  void *arguments{nullptr};
  CudaGraphConditionalControl64 *controls{nullptr};
  BoundedGraphControls bounded;
  std::size_t control_bytes{0};
  std::size_t bytes{0};
  std::size_t nodes{0};
  std::size_t capture_bytes{0};
  std::vector<std::shared_ptr<CudaGraphCaptureResources>> capture_resources;
  std::vector<std::uint8_t> host_arguments;
  bool capturing{false};
  bool submitted{false};
  ~State() {
    bool backend_safe = false;
    if (fault_domain &&
        (!fault_domain->backend_calls_safe() ||
         fault_domain->state() == RuntimeLifecycleState::kFinalized)) {
      for (const auto &resource : capture_resources)
        resource->release(false);
      bounded.release(false);
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
    } catch (...) {
      // Cleanup must not throw while unwinding a driver failure.
      for (const auto &resource : capture_resources)
        resource->release(false);
      capture_resources.clear();
      bounded.release(false);
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
    : state_(std::make_unique<State>()) {
  state_->fault_domain = std::move(fault_domain);
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

void FixedGraph::initialize(const std::vector<Operation> &operations) {
  TI_ERROR_IF(operations.empty() || !available(),
              "Fixed CUDA Graph requires operations and driver graph support");
  auto &cuda = CUDAContext::get_instance();
  auto guard = cuda.get_guard();
  auto lock = cuda.get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  auto &state = *state_;
  std::vector<const Dispatch *> dispatches;
  std::unordered_map<const Dispatch *, std::size_t> dispatch_index;
  std::unordered_map<const While *, std::size_t> loop_index;
  std::unordered_set<const While *> active_loops;
  std::vector<CudaGraphConditionalControl64> controls;
  std::function<void(const std::vector<Operation> &)> collect;
  collect = [&](const std::vector<Operation> &sequence) {
    for (const auto &operation : sequence) {
      if (const auto *dispatch = std::get_if<Dispatch>(&operation)) {
        if (dispatch_index.emplace(dispatch, dispatches.size()).second)
          dispatches.push_back(dispatch);
      } else {
        const auto &loop = std::get<std::shared_ptr<const While>>(operation);
        if (!loop || !loop->predicate || loop->body.empty())
          throw std::invalid_argument(
              "Fixed CUDA while requires a predicate and nonempty body");
        if (active_loops.count(loop.get()))
          throw std::invalid_argument(
              "Fixed CUDA control regions contain a cycle");
        if (!loop_index.emplace(loop.get(), controls.size()).second)
          continue;
        active_loops.insert(loop.get());
        CudaGraphConditionalControl64 control;
        control.predicate = reinterpret_cast<std::uintptr_t>(loop->predicate);
        control.max_iterations = loop->max_iterations;
        control.continue_while_nonzero = loop->continue_while_nonzero;
        controls.push_back(control);
        collect(loop->condition);
        collect(loop->body);
        active_loops.erase(loop.get());
      }
    }
  };
  collect(operations);
  if (!controls.empty()) {
    if (!GraphRegionBuilder::available())
      throw std::invalid_argument(
          "Fixed CUDA while requires native conditional graph support");
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
    TI_ERROR_IF(
        !dispatch.launcher || !dispatch.context ||
            !(dispatch.bounded_extent
                  ? dispatch.launcher->prepare_cuda_graph_bounded_packet(
                        dispatch.handle, *dispatch.context, packets[i],
                        dispatch.bounded_extent, dispatch.bounded_capacity)
                  : dispatch.launcher->prepare_cuda_graph_launch_packet(
                        dispatch.handle, *dispatch.context, packets[i])),
        "Unsupported fixed CUDA Graph dispatch");
    auto &packet = packets[i];
    if (dispatch.adaptive_block_dim) {
      TI_ERROR_IF(!controls.empty() || !packet.bounded_range ||
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
  auto record_dispatch = [&](const Dispatch &dispatch, void *stream) {
    const auto index = dispatch_index.at(&dispatch);
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
  if (controls.empty()) {
    auto capture_lock = cuda.get_graph_capture_lock_guard();
    driver.stream_begin_capture(state.stream,
                                CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    state.capturing = true;
    for (const auto *dispatch : dispatches)
      record_dispatch(*dispatch, state.stream);
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
    state.control_bytes = controls.size() * sizeof(controls[0]);
    driver.malloc(reinterpret_cast<void **>(&state.controls),
                  state.control_bytes);
    driver.memcpy_host_to_device_async(state.controls, controls.data(),
                                       state.control_bytes, state.stream);
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
               std::holds_alternative<Dispatch>(sequence[end]))
          ++end;
        if (end != begin) {
          dependencies =
              builder.capture(graph, dependencies, [&](void *stream) {
                for (auto i = begin; i < end; ++i)
                  record_dispatch(std::get<Dispatch>(sequence[i]), stream);
              });
          begin = end;
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
  driver.graph_instantiate_with_flags(&state.executable, state.graph, 0);
  if (grid_limit) {
    driver.graph_upload(state.executable, state.stream);
    driver.stream_synchronize(state.stream);
    const auto error = state.bounded.upload();
    if (error != CUDA_SUCCESS)
      throw BackendRuntimeError(Arch::cuda, error,
                                "upload adaptive CUDA controls",
                                "Selected adaptive CUDA control upload failed");
  }
  std::size_t root_nodes = 0;
  driver.graph_get_nodes(state.graph, nullptr, &root_nodes);
  state.nodes += root_nodes;
  driver.graph_destroy(state.graph);
  state.graph = nullptr;
  driver.stream_destroy(state.stream);
  state.stream = nullptr;
}

FixedGraph::~FixedGraph() = default;
void FixedGraph::run() {
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
