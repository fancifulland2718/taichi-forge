#include <gtest/gtest.h>

#ifdef TI_WITH_CUDA
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/parallel_graph.h"

namespace taichi::lang {
namespace {
class CudaParallelGraph : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CUDADriver::get_instance_without_context()
             .nvidia_extensions_available())
      GTEST_SKIP();
    CUDAContext::get_instance().make_current();
    auto &driver = CUDADriver::get_instance();
    if (!driver.stream_get_capture_info_v2.available())
      GTEST_SKIP();
    driver.stream_create(&stream, CU_STREAM_NON_BLOCKING);
    driver.malloc(&data, 32 * sizeof(std::uint32_t));
    driver.memsetd32(data, 99, 32);
  }
  void TearDown() override {
    if (!stream)
      return;
    auto &driver = CUDADriver::get_instance();
    if (capturing)
      driver.stream_end_capture.call(stream, &graph);
    driver.stream_synchronize.call(stream);
    if (executable)
      driver.graph_exec_destroy.call(executable);
    if (graph)
      driver.graph_destroy.call(graph);
    parallel.release(true);
    driver.mem_free.call(data);
    driver.stream_destroy.call(stream);
  }
  void *at(std::size_t index) {
    return static_cast<std::uint32_t *>(data) + index;
  }
  std::vector<const void *> frontier(void *s) {
    std::uint32_t status = 0;
    const void **nodes = nullptr;
    std::size_t count = 0;
    CUDADriver::get_instance().stream_get_capture_info_v2(
        s, &status, nullptr, nullptr, &nodes, &count);
    return count ? std::vector<const void *>(nodes, nodes + count)
                 : std::vector<const void *>{};
  }
  void begin() {
    CUDADriver::get_instance().stream_begin_capture(
        stream, CU_STREAM_CAPTURE_MODE_RELAXED);
    capturing = true;
  }
  void end() {
    CUDADriver::get_instance().stream_end_capture(stream, &graph);
    capturing = false;
  }
  void retire() {
    auto &driver = CUDADriver::get_instance();
    driver.stream_synchronize(stream);
    driver.graph_exec_destroy(executable);
    executable = nullptr;
    driver.graph_destroy(graph);
    graph = nullptr;
  }
  cuda::ParallelGraphCapture parallel;
  void *stream{nullptr}, *data{nullptr};
  CUgraph graph{nullptr};
  CUgraphExec executable{nullptr};
  bool capturing{false};
};

TEST_F(CudaParallelGraph, PreservesAllBranchTailsAndReusesCaptureResources) {
  auto &driver = CUDADriver::get_instance();
  // Same-count reuse and resize both retain the dependency contract.
  for (std::size_t branches : {2u, 8u, 8u, 3u}) {
    parallel.prepare(branches);
    begin();
    driver.memsetd32_async(data, static_cast<unsigned int>(branches), 1,
                           stream);
    std::vector<const void *> tails;
    ASSERT_EQ(
        parallel.capture(stream,
                         [&](std::size_t branch, void *s) -> std::uint32_t {
                           driver.memcpy_device_to_device_async(at(1 + branch),
                                                                data, 4, s);
                           driver.memcpy_device_to_device_async(
                               at(9 + branch), at(1 + branch), 4, s);
                           const auto dependencies = frontier(s);
                           EXPECT_EQ(dependencies.size(), 1);
                           if (dependencies.size() != 1)
                             return 1;
                           tails.push_back(dependencies.front());
                           return CUDA_SUCCESS;
                         }),
        CUDA_SUCCESS);
    const auto joined = frontier(stream);
    for (auto tail : tails)
      EXPECT_NE(std::find(joined.begin(), joined.end(), tail), joined.end());
    // Distinct terminal nodes in the root frontier prove it wasn't serialized.
    EXPECT_GE(joined.size(), branches);
    driver.memcpy_device_to_device_async(at(17), at(9), branches * 4, stream);
    end();
    driver.graph_instantiate_with_flags(&executable, graph, 0);
    std::array<std::uint32_t, 32> initial;
    initial.fill(99);
    driver.memcpy_host_to_device(data, initial.data(), sizeof(initial));
    for (int replay = 0; replay < 3; ++replay)
      driver.graph_launch(executable, stream);
    driver.stream_synchronize(stream);
    std::array<std::uint32_t, 32> result;
    driver.memcpy_device_to_host(result.data(), data, sizeof(result));
    for (std::size_t i = 0; i < branches; ++i)
      EXPECT_EQ(result[17 + i], branches);
    retire();
  }
}

TEST_F(CudaParallelGraph, CallbackFailureCanAbortAndRecapture) {
  auto &driver = CUDADriver::get_instance();
  constexpr std::uint32_t recorder_failure = 1;
  parallel.prepare(2);
  begin();
  const auto failure = parallel.capture(
      stream, [&](std::size_t branch, void *s) -> std::uint32_t {
        if (branch == 1)
          return recorder_failure;
        driver.memsetd32_async(at(1), 7, 1, s);
        return CUDA_SUCCESS;
      });
  ASSERT_EQ(failure, recorder_failure);
  // The caller aborts the whole capture; no partial graph is published.
  driver.stream_end_capture.call(stream, &graph);
  capturing = false;
  if (graph) {
    driver.graph_destroy(graph);
    graph = nullptr;
  }
  parallel.release(true);
  parallel.prepare(2);
  begin();
  EXPECT_THROW(
      parallel.capture(stream,
                       [&](std::size_t branch, void *s) -> std::uint32_t {
                         if (branch == 1)
                           throw std::runtime_error("recording rejected");
                         driver.memsetd32_async(at(1), 7, 1, s);
                         return CUDA_SUCCESS;
                       }),
      std::runtime_error);
  driver.stream_end_capture.call(stream, &graph);
  capturing = false;
  if (graph) {
    driver.graph_destroy(graph);
    graph = nullptr;
  }
  parallel.release(true);
  parallel.prepare(2);
  begin();
  ASSERT_EQ(parallel.capture(stream,
                             [&](std::size_t branch, void *s) -> std::uint32_t {
                               driver.memsetd32_async(at(branch), 41, 1, s);
                               return CUDA_SUCCESS;
                             }),
            CUDA_SUCCESS);
  end();
  driver.graph_instantiate_with_flags(&executable, graph, 0);
  driver.graph_launch(executable, stream);
  driver.stream_synchronize(stream);
  std::array<std::uint32_t, 2> result;
  driver.memcpy_device_to_host(result.data(), data, sizeof(result));
  EXPECT_EQ(result, (std::array<std::uint32_t, 2>{41, 41}));
}

TEST(CudaParallelGraphContract, RejectsUnpreparedAndDegenerateCaptures) {
  cuda::ParallelGraphCapture parallel;
  EXPECT_THROW(parallel.prepare(0), std::invalid_argument);
  EXPECT_THROW(parallel.prepare(1), std::invalid_argument);
  EXPECT_THROW(parallel.capture(nullptr, {}), std::invalid_argument);
  parallel.release(false);
  EXPECT_EQ(parallel.branch_count(), 0);
}
}  // namespace
}  // namespace taichi::lang
#endif
