#include "gtest/gtest.h"

#ifdef TI_WITH_CUDA
#include <array>
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/binding_frame_replay.h"

namespace taichi::lang::cuda {
namespace {

// Captured device work without a Program, JIT kernel, or Python owner.
struct Frame {
  CUgraph graph{nullptr};
  void *data{nullptr};

  explicit Frame(std::uint32_t value, bool extra_node = false) {
    auto &driver = CUDADriver::get_instance();
    driver.malloc(&data, 16 * sizeof(std::uint32_t));
    driver.memsetd32(data, 0, 16);
    void *stream = nullptr;
    driver.stream_create(&stream, CU_STREAM_NON_BLOCKING);
    driver.stream_begin_capture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    driver.memsetd32_async(data, value, 16, stream);
    if (extra_node)
      driver.memsetd32_async(data, value + 1, 1, stream);
    driver.stream_end_capture(stream, &graph);
    driver.stream_destroy(stream);
  }

  ~Frame() {
    auto &driver = CUDADriver::get_instance();
    driver.graph_destroy(graph);
    driver.mem_free(data);
  }

  void expect(std::uint32_t value) {
    auto &driver = CUDADriver::get_instance();
    driver.stream_synchronize(nullptr);
    std::array<std::uint32_t, 16> host{};
    driver.memcpy_device_to_host(host.data(), data, sizeof(host));
    for (auto actual : host)
      EXPECT_EQ(actual, value);
  }
};

TEST(CudaBindingFrameReplay, SharesOneExecutableAndRetainsInFlightFrames) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  auto &cuda = CUDAContext::get_instance();
  auto submission = cuda.get_submission_lock_guard();
  auto guard = cuda.get_guard();
  auto domain = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 77);
  BindingFrameReplay replay(77, domain);
  auto first = std::make_shared<Frame>(7);
  auto second = std::make_shared<Frame>(11);
  replay.prepare(first->graph, first);
  replay.prepare(second->graph, second);
  first->expect(0);
  second->expect(0);  // Preparation must not execute mathematical work.
  replay.run(first->graph, first);
  first->expect(7);
  const auto before = replay.snapshot();
  for (int i = 0; i < 3; ++i)
    replay.run(first->graph, first);
  const auto after = replay.snapshot();
  EXPECT_EQ(after.executables, 1);
  EXPECT_EQ(after.events.created, before.events.created);
  EXPECT_EQ(after.events.reused, before.events.reused);
  EXPECT_EQ(after.pending_frames, before.pending_frames);
  std::weak_ptr<Frame> retired = first;
  first.reset();
  EXPECT_FALSE(retired.expired());
  replay.run(second->graph, second);
  second->expect(11);
  replay.collect();
  EXPECT_TRUE(retired.expired());
  retired = second;
  second.reset();
  EXPECT_TRUE(replay.retire_backend(true));
  EXPECT_FALSE(retired.expired());  // Caller can invalidate all frames here.
  replay.release_frame_leases();
  EXPECT_TRUE(retired.expired());
  EXPECT_EQ(replay.snapshot().executables, 0);
  EXPECT_EQ(replay.snapshot().events.cached, 0);
  EXPECT_THROW(replay.run(nullptr, {}), std::runtime_error);
  EXPECT_TRUE(domain->submission_allowed());
}

TEST(CudaBindingFrameReplay, RejectsTopologyChangeWithoutReinstantiation) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  auto &cuda = CUDAContext::get_instance();
  auto submission = cuda.get_submission_lock_guard();
  auto guard = cuda.get_guard();
  auto domain = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 78);
  BindingFrameReplay replay(78, domain);
  auto first = std::make_shared<Frame>(7);
  auto changed = std::make_shared<Frame>(11, true);
  replay.prepare(first->graph, first);
  EXPECT_THROW(replay.prepare(changed->graph, changed), BackendRuntimeError);
  EXPECT_TRUE(replay.failed());
  EXPECT_TRUE(replay.is_active(first.get()));
  EXPECT_EQ(replay.snapshot().executables, 1);
  EXPECT_THROW(replay.run(first->graph, first), std::runtime_error);
  first->expect(0);
  changed->expect(0);
  EXPECT_TRUE(replay.retire_backend(true));
  replay.release_frame_leases();
  EXPECT_TRUE(domain->submission_allowed());
}

}  // namespace
}  // namespace taichi::lang::cuda
#endif
