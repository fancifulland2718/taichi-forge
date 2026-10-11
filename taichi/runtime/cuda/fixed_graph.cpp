#include "taichi/runtime/cuda/fixed_graph.h"

#include <cstring>
#include <limits>
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/kernel_launcher.h"

namespace taichi::lang::cuda {
struct FixedGraph::State {
  CUgraph graph{nullptr};
  CUgraphExec executable{nullptr};
  void *stream{nullptr};
  void *arguments{nullptr};
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
    : state_(std::make_unique<State>()) {
  TI_ERROR_IF(
      dispatches.empty() || !available(),
      "Fixed CUDA Graph requires nonempty dispatches and driver graph support");
  auto &cuda = CUDAContext::get_instance();
  auto guard = cuda.get_guard();
  auto lock = cuda.get_submission_lock_guard();
  auto &driver = CUDADriver::get_instance();
  auto &state = *state_;
  std::vector<KernelLauncher::GraphLaunchPacket> packets(dispatches.size());
  std::vector<std::size_t> offsets;
  auto &image = state.host_arguments;
  constexpr std::size_t alignment = 16;  // RuntimeContext argument ABI.
  for (std::size_t i = 0; i < dispatches.size(); ++i) {
    const auto &dispatch = dispatches[i];
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
  auto capture_lock = cuda.get_graph_capture_lock_guard();
  driver.stream_begin_capture(state.stream,
                              CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
  state.capturing = true;
  for (std::size_t i = 0; i < packets.size(); ++i)
    dispatches[i].launcher->capture_cuda_graph_launch(packets[i], state.stream);
  const auto error = driver.stream_end_capture.call(state.stream, &state.graph);
  state.capturing = false;
  TI_ERROR_IF(error != CUDA_SUCCESS, "Fixed CUDA Graph capture failed: {}",
              error);
  driver.graph_instantiate_with_flags(&state.executable, state.graph, 0);
  driver.graph_get_nodes(state.graph, nullptr, &state.nodes);
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
std::size_t FixedGraph::node_count() const {
  return state_->nodes;
}
}  // namespace taichi::lang::cuda
