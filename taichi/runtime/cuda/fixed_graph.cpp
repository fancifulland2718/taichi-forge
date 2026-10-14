#include "taichi/runtime/cuda/fixed_graph.h"

#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include "taichi/rhi/cuda/graph_region_builder.h"
#include "taichi/rhi/cuda/primitives/graph_ptx.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/kernel_launcher.h"

namespace taichi::lang::cuda {
struct FixedGraph::State {
  CUgraph graph{nullptr};
  CUgraphExec executable{nullptr};
  void *stream{nullptr};
  void *arguments{nullptr};
  CudaGraphConditionalControl64 *controls{nullptr};
  std::size_t control_bytes{0};
  std::size_t bytes{0};
  std::size_t nodes{0};
  std::vector<std::uint8_t> host_arguments;
  bool capturing{false};
  bool submitted{false};
  ~State() {
    try {
      auto &cuda = CUDAContext::get_instance();
      auto guard = cuda.get_guard();
      auto lock = cuda.get_submission_lock_guard();
      auto &driver = CUDADriver::get_instance();
      // Cleanup also handles construction failure without publishing a partial
      // executable. No default-stream wait is needed for an unused graph.
      if (capturing)
        driver.stream_end_capture.call(stream, &graph);
      if (submitted)
        driver.stream_synchronize.call(nullptr);
      if (stream)
        driver.stream_synchronize.call(stream);
      if (executable)
        driver.graph_exec_destroy.call(executable);
      if (graph)
        driver.graph_destroy.call(graph);
      if (controls)
        driver.mem_free.call(controls);
      if (arguments)
        driver.mem_free.call(arguments);
      if (stream)
        driver.stream_destroy.call(stream);
    } catch (...) {
      // Cleanup must not throw while unwinding a driver failure.
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

FixedGraph::FixedGraph(const std::vector<Dispatch> &dispatches)
    : FixedGraph(std::vector<Operation>(dispatches.begin(), dispatches.end())) {
}

FixedGraph::FixedGraph(const std::vector<Operation> &operations)
    : state_(std::make_unique<State>()) {
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
  std::vector<std::size_t> offsets;
  auto &image = state.host_arguments;
  constexpr std::size_t alignment = 16;  // RuntimeContext argument ABI.
  for (std::size_t i = 0; i < dispatches.size(); ++i) {
    const auto &dispatch = *dispatches[i];
    if (dispatch.external) {
      offsets.push_back(0);
      continue;
    }
    TI_ERROR_IF(!dispatch.launcher || !dispatch.context ||
                    !dispatch.launcher->prepare_cuda_graph_launch_packet(
                        dispatch.handle, *dispatch.context, packets[i]),
                "Unsupported fixed CUDA Graph dispatch");
    auto &packet = packets[i];
    TI_ERROR_IF(
        image.size() > std::numeric_limits<std::size_t>::max() - alignment,
        "Fixed CUDA argument image overflows address space");
    const auto offset = (image.size() + alignment - 1) & ~(alignment - 1);
    offsets.push_back(offset);
    TI_ERROR_IF(packet.arg_buffer_size >
                    std::numeric_limits<std::size_t>::max() - offset,
                "Fixed CUDA argument image overflows address space");
    if (packet.arg_buffer_size) {
      image.resize(offset + packet.arg_buffer_size);
      std::memcpy(image.data() + offset, packet.context.arg_buffer,
                  packet.arg_buffer_size);
    }
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
    if (packets[i].arg_buffer_size) {
      packets[i].context.arg_buffer =
          static_cast<char *>(state.arguments) + offsets[i];
      packets[i].device_arg_buffer = packets[i].context.arg_buffer;
    }
  }
  auto record_dispatch = [&](const Dispatch &dispatch, void *stream) {
    const auto index = dispatch_index.at(&dispatch);
    if (dispatch.external)
      dispatch.external(stream);
    else
      dispatch.launcher->capture_cuda_graph_launch(packets[index], stream);
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
    TI_ERROR_IF(error != CUDA_SUCCESS, "Fixed CUDA Graph capture failed: {}",
                error);
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
  driver.graph_instantiate_with_flags(&state.executable, state.graph, 0);
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
  auto &cuda = CUDAContext::get_instance();
  auto guard = cuda.get_guard();
  auto lock = cuda.get_submission_lock_guard();
  state_->submitted = true;
  CUDADriver::get_instance().graph_launch(state_->executable, nullptr);
}
std::size_t FixedGraph::argument_bytes() const {
  return state_->bytes;
}
std::size_t FixedGraph::control_bytes() const {
  return state_->control_bytes;
}
std::size_t FixedGraph::node_count() const {
  return state_->nodes;
}
}  // namespace taichi::lang::cuda
