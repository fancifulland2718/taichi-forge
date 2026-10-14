#include "taichi/rhi/cuda/graph_region_builder.h"

#include "taichi/rhi/cuda/cuda_context.h"

namespace taichi::lang::cuda {
namespace {
void check(std::uint32_t code, const char *stage) {
  if (code != CUDA_SUCCESS)
    throw GraphBuildError(code, stage);
}

// stream_end_capture returns the borrowed graph. Even a callback exception
// must end capture without destroying that graph or masking the exception.
class BorrowedCapture {
 public:
  explicit BorrowedCapture(void *stream) : stream_(stream) {
  }
  ~BorrowedCapture() {
    if (active_) {
      CUgraph graph = nullptr;
      CUDADriver::get_instance().stream_end_capture.call(stream_, &graph);
    }
  }
  CUgraph end() {
    active_ = false;
    CUgraph graph = nullptr;
    check(CUDADriver::get_instance().stream_end_capture.call(stream_, &graph),
          "region capture end");
    return graph;
  }

 private:
  void *stream_;
  bool active_{true};
};
}  // namespace

GraphBuildError::GraphBuildError(std::uint32_t code, const char *stage)
    : std::runtime_error(std::string(stage) + ": CUDA error " +
                         std::to_string(code)),
      code_(code),
      stage_(stage) {
}

bool GraphRegionBuilder::available() {
  auto &driver = CUDADriver::get_instance();
  return driver.stream_begin_capture_to_graph.available() &&
         driver.stream_end_capture.available() &&
         driver.stream_get_capture_info_v2.available() &&
         driver.graph_conditional_handle_create.available() &&
         driver.graph_add_node.available();
}

GraphRegionBuilder::GraphRegionBuilder(CUgraph root, void *stream)
    : root_(root), stream_(stream) {
  if (!root || !stream)
    throw std::invalid_argument(
        "Graph region assembly requires a root and capture stream");
  if (!available())
    throw GraphBuildError(CUDA_ERROR_NOT_SUPPORTED,
                          "region builder availability");
  check(CUDADriver::get_instance().context_get_current.call(&context_),
        "region context query");
  if (!context_)
    throw std::invalid_argument(
        "Graph region assembly requires a current CUDA context");
}

GraphRegionBuilder::Dependencies GraphRegionBuilder::capture(
    CUgraph destination,
    const Dependencies &dependencies,
    const Recorder &record) const {
  auto lock = CUDAContext::get_instance().get_graph_capture_lock_guard();
  auto &driver = CUDADriver::get_instance();
  check(driver.stream_begin_capture_to_graph.call(
            stream_, destination, dependencies.data(), nullptr,
            dependencies.size(), CU_STREAM_CAPTURE_MODE_RELAXED),
        "region capture begin");
  BorrowedCapture guard(stream_);
  record(stream_);
  std::uint32_t status = 0;
  const void **frontier = nullptr;
  std::size_t count = 0;
  check(driver.stream_get_capture_info_v2.call(stream_, &status, nullptr,
                                               nullptr, &frontier, &count),
        "region capture dependencies");
  if (count && !frontier)
    throw std::runtime_error(
        "CUDA capture returned a missing dependency frontier");
  Dependencies result;
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i)
    result.push_back(const_cast<void *>(frontier[i]));
  if (guard.end() != destination)
    throw std::runtime_error(
        "CUDA capture returned a different borrowed graph");
  return result;
}

std::uint64_t GraphRegionBuilder::conditional_handle() const {
  constexpr unsigned int assign_default = 1;
  std::uint64_t handle = 0;
  check(CUDADriver::get_instance().graph_conditional_handle_create.call(
            &handle, root_, context_, 0, assign_default),
        "conditional handle create");
  if (!handle)
    throw std::runtime_error("CUDA returned an empty conditional handle");
  return handle;
}

GraphRegionBuilder::While GraphRegionBuilder::add_while(
    CUgraph destination,
    const Dependencies &dependencies,
    std::uint64_t handle) const {
  TaichiCudaGraphNodeParams params{};
  params.type = 13;  // CU_GRAPH_NODE_TYPE_CONDITIONAL.
  params.parameters.conditional.handle = handle;
  params.parameters.conditional.type = 1;  // CU_GRAPH_COND_TYPE_WHILE.
  params.parameters.conditional.size = 1;
  params.parameters.conditional.context = context_;
  void *node = nullptr;
  check(CUDADriver::get_instance().graph_add_node.call(
            &node, destination, dependencies.data(), dependencies.size(),
            &params),
        "conditional while node create");
  if (!node || !params.parameters.conditional.ph_graph_out ||
      !params.parameters.conditional.ph_graph_out[0])
    throw std::runtime_error("CUDA returned a missing conditional body");
  return {node, params.parameters.conditional.ph_graph_out[0]};
}
}  // namespace taichi::lang::cuda
