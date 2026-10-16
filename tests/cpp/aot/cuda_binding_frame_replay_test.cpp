#include "gtest/gtest.h"

#ifdef TI_WITH_CUDA
#include <array>
#include "taichi/program/runtime_fault.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/binding_frame_replay.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"

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

struct DeviceFrame {
  std::array<DeviceAllocationUnique, 2> arrays;
  explicit DeviceFrame(CudaDevice &device) {
    for (auto &array : arrays) {
      auto allocation = device.allocate_memory_unique({64});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error("test allocation failed");
      array = std::move(allocation.first);
      CUDADriver::get_instance().memsetd32(device.get_memory_addr(*array), 0,
                                           16);
    }
  }
};

class CaptureCopy : public NativeOperation {
 public:
  CaptureCopy(CudaDevice &device,
              const DeviceFrame &frame,
              bool ordered = false)
      : device_(device),
        source_(device.get_memory_addr(*frame.arrays[0])),
        output_(device.get_memory_addr(*frame.arrays[1])),
        ordered_(ordered) {
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  NativeOperationMode execution_mode() const override {
    return ordered_ ? NativeOperationMode::cuda_ordered
                    : NativeOperationMode::cuda_capture;
  }
  void record_cuda(void *stream) override {
    ++recordings;
    CUDADriver::get_instance().memcpy_device_to_device_async(output_, source_,
                                                             64, stream);
  }
  int recordings{0};

 private:
  CudaDevice &device_;
  void *source_;
  void *output_;
  bool ordered_;
};

TEST(CudaBindingFrameReplay,
     FixedExecutionRetainsCapturedOperandsAndNativePlans) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  CudaDevice device;
  auto &cuda = CUDAContext::get_instance();
  auto submission = cuda.get_submission_lock_guard();
  auto guard = cuda.get_guard();
  auto domain = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 79);
  auto replay = std::make_shared<BindingFrameReplay>(79, domain);
  auto first = std::make_shared<DeviceFrame>(device);
  auto second = std::make_shared<DeviceFrame>(device);
  auto copy1 = std::make_shared<CaptureCopy>(device, *first);
  auto copy2 = std::make_shared<CaptureCopy>(device, *second);
  const auto operations = [](const DeviceFrame &frame,
                             std::shared_ptr<NativeOperation> copy, int value) {
    return std::vector<FixedExecution::Operation>{
        FixedExecution::BufferFill{frame.arrays[0]->get_ptr(), 64,
                                   static_cast<std::uint32_t>(value)},
        copy};
  };
  auto call1 = std::make_unique<FixedExecution>(Arch::cuda, device, nullptr,
                                                operations(*first, copy1, 7),
                                                nullptr, domain, replay, first);
  auto call2 = std::make_unique<FixedExecution>(
      Arch::cuda, device, nullptr, operations(*second, copy2, 11), nullptr,
      domain, replay, second);
  const auto expect = [&](const DeviceFrame &frame, std::uint32_t expected) {
    auto &driver = CUDADriver::get_instance();
    driver.stream_synchronize(nullptr);
    std::array<std::uint32_t, 16> values{};
    driver.memcpy_device_to_host(values.data(),
                                 device.get_memory_addr(*frame.arrays[1]), 64);
    for (auto value : values)
      EXPECT_EQ(value, expected);
  };
  expect(*first, 0);
  expect(*second, 0);
  for (int i = 0; i < 3; ++i) {
    call1->run();
    call2->run();
  }
  expect(*first, 7);
  expect(*second, 11);
  EXPECT_EQ(copy1->recordings, 1);
  EXPECT_EQ(copy2->recordings, 1);
  EXPECT_EQ(replay->snapshot().executables, 1);
  // A rejected owner/topology cannot change the already-published family.
  auto other = std::make_shared<RuntimeFaultDomain>(Arch::cuda, 80);
  EXPECT_THROW(
      FixedExecution(Arch::cuda, device, nullptr, operations(*first, copy1, 7),
                     nullptr, other, replay, first),
      std::invalid_argument);
  auto ordered = std::make_shared<CaptureCopy>(device, *first, true);
  EXPECT_THROW(FixedExecution(Arch::cuda, device, nullptr,
                              operations(*first, ordered, 7), nullptr, domain,
                              replay, first),
               std::invalid_argument);
  EXPECT_EQ(ordered->recordings, 0);
  ordered.reset();
  std::weak_ptr<DeviceFrame> retired = first;
  std::weak_ptr<NativeOperation> retired_copy = copy1;
  call1->run();
  call1.reset();
  copy1.reset();
  first.reset();
  EXPECT_FALSE(retired.expired());
  EXPECT_FALSE(retired_copy.expired());
  call2->run();
  expect(*second, 11);
  replay->collect();
  EXPECT_TRUE(retired.expired());
  EXPECT_TRUE(retired_copy.expired());
  retired = second;
  call2.reset();
  copy2.reset();
  second.reset();
  EXPECT_FALSE(retired.expired());
  EXPECT_TRUE(replay->retire_backend(true));
  replay->release_frame_leases();
  EXPECT_TRUE(retired.expired());
}

}  // namespace
}  // namespace taichi::lang::cuda
#endif
