#include <gtest/gtest.h>

#ifdef TI_WITH_CUDA
#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "taichi/program/primitive_workspace.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"

namespace taichi::lang {
namespace {
class CapturedBucket final : public NativeOperation {
 public:
  CapturedBucket(cuda::CudaDevice &device,
                 void *keys,
                 void *values,
                 void *offsets,
                 void *output,
                 int n,
                 int bins,
                 int words)
      : device_(device),
        keys_(keys),
        values_(values),
        offsets_(offsets),
        output_(output),
        n_(n),
        bins_(bins),
        words_(words),
        bytes_(cuda::driver_bucket_builder_workspace_bytes(n, bins)) {
    if (bytes_) {
      auto memory = device.allocate_memory_unique({bytes_});
      if (memory.second != RhiResult::success)
        throw std::runtime_error("Failed to allocate bucket test scratch");
      workspace_ = std::move(memory.first);
      scratch_ = device.get_memory_addr(*workspace_);
    }
    cuda::driver_prepare_bucket_builder();
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return bytes_;
  }
  void record_cuda(void *stream) override {
    cuda::driver_bucket_builder_with_workspace(
        keys_, values_, offsets_, output_, n_, bins_, words_, scratch_, stream);
  }

 private:
  cuda::CudaDevice &device_;
  void *keys_, *values_, *offsets_, *output_;
  int n_, bins_, words_;
  std::size_t bytes_;
  DeviceAllocationUnique workspace_;
  void *scratch_{nullptr};
};

void check_capture(int bins, int words) {
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  constexpr int n = 13;
  constexpr uint32_t sentinel = 0xbad0bad0;
  std::vector<int32_t> keys(n + 2, -77), offsets(bins + 3, -77), actual_offsets;
  std::vector<uint32_t> values((n + 2) * words, sentinel),
      output((n + 2) * words, sentinel), actual_output;
  auto key_mem = device.allocate_memory_unique({keys.size() * 4});
  auto value_mem = device.allocate_memory_unique({values.size() * 4});
  auto offset_mem = device.allocate_memory_unique({offsets.size() * 4});
  auto output_mem = device.allocate_memory_unique({output.size() * 4});
  ASSERT_EQ(key_mem.second, RhiResult::success);
  ASSERT_EQ(value_mem.second, RhiResult::success);
  ASSERT_EQ(offset_mem.second, RhiResult::success);
  ASSERT_EQ(output_mem.second, RhiResult::success);
  auto *key_ptr =
      static_cast<int32_t *>(device.get_memory_addr(*key_mem.first));
  auto *value_ptr =
      static_cast<uint32_t *>(device.get_memory_addr(*value_mem.first));
  auto *offset_ptr =
      static_cast<int32_t *>(device.get_memory_addr(*offset_mem.first));
  auto *output_ptr =
      static_cast<uint32_t *>(device.get_memory_addr(*output_mem.first));
  driver.memcpy_host_to_device(offset_ptr, offsets.data(), offsets.size() * 4);
  driver.memcpy_host_to_device(output_ptr, output.data(), output.size() * 4);
  auto make_full = [&] {
    return std::make_unique<FixedExecution>(
        Arch::cuda, device, nullptr,
        std::vector<FixedExecution::Operation>{std::make_shared<CapturedBucket>(
            device, key_ptr + 1, value_ptr + words, offset_ptr + 1,
            output_ptr + words, n, bins, words)});
  };
  auto full = make_full(), peer = make_full();
  FixedExecution empty(
      Arch::cuda, device, nullptr,
      {std::make_shared<CapturedBucket>(
          device, nullptr, nullptr, offset_ptr + 1, nullptr, 0, bins, words)});
  actual_offsets.resize(offsets.size());
  actual_output.resize(output.size());
  driver.memcpy_device_to_host(actual_offsets.data(), offset_ptr,
                               offsets.size() * 4);
  driver.memcpy_device_to_host(actual_output.data(), output_ptr,
                               output.size() * 4);
  EXPECT_EQ(actual_offsets, offsets);  // Preparation/capture execute no writes.
  EXPECT_EQ(actual_output, output);
  EXPECT_EQ(full->workspace_bytes(),
            cuda::driver_bucket_builder_workspace_bytes(n, bins));
  EXPECT_EQ(empty.workspace_bytes(), 0);
  for (int replay = 0; replay < 3; ++replay) {
    std::vector<int32_t> expected_offsets(bins + 3, -77);
    std::vector<int> counts(bins), seen(n);
    expected_offsets[1] = 0;
    for (int i = 0; i < n; ++i) {
      int key = (i * 37 + replay) % bins;
      if (i == 1)
        key = bins - 1;
      if (replay == 1 || i == 2)
        key = -1;
      if (i == 3)
        key = INT32_MAX;
      if (i == 4)
        key = INT32_MIN;
      keys[i + 1] = key;
      if (key >= 0 && key < bins)
        ++counts[key];
      for (int lane = 0; lane < words; ++lane)
        values[(i + 1) * words + lane] =
            lane == 0 ? 100 + i
                      : ((lane == 1 ? 0x7fc01234u : 0x80000000u) + i + replay);
    }
    for (int b = 0; b < bins; ++b)
      expected_offsets[b + 2] = expected_offsets[b + 1] + counts[b];
    driver.memcpy_host_to_device(key_ptr, keys.data(), keys.size() * 4);
    driver.memcpy_host_to_device(value_ptr, values.data(), values.size() * 4);
    driver.memcpy_host_to_device(output_ptr, output.data(), output.size() * 4);
    (replay == 1 ? peer : full)->run();
    driver.context_synchronize();
    driver.memcpy_device_to_host(actual_offsets.data(), offset_ptr,
                                 offsets.size() * 4);
    driver.memcpy_device_to_host(actual_output.data(), output_ptr,
                                 output.size() * 4);
    EXPECT_EQ(actual_offsets, expected_offsets);
    for (int b = 0; b < bins; ++b) {
      for (int pos = expected_offsets[b + 1]; pos < expected_offsets[b + 2];
           ++pos) {
        const auto id = actual_output[(pos + 1) * words] - 100;
        ASSERT_LT(id, n);
        EXPECT_EQ(keys[id + 1], b);
        ++seen[id];
        for (int lane = 0; lane < words; ++lane)
          EXPECT_EQ(actual_output[(pos + 1) * words + lane],
                    values[(id + 1) * words + lane]);
      }
    }
    for (int i = 0; i < n; ++i)
      EXPECT_EQ(seen[i], keys[i + 1] >= 0 && keys[i + 1] < bins ? 1 : 0);
    for (int lane = 0; lane < words; ++lane)
      EXPECT_EQ(actual_output[lane], sentinel);
    for (std::size_t i = (expected_offsets[bins + 1] + 1) * words;
         i < actual_output.size(); ++i)
      EXPECT_EQ(actual_output[i], sentinel);
  }
  full->run();
  full.reset();  // In-flight retirement must precede scratch destruction.
  peer->run();
  driver.context_synchronize();
  driver.memcpy_device_to_host(output.data(), output_ptr, output.size() * 4);
  empty.run();
  driver.context_synchronize();
  driver.memcpy_device_to_host(actual_offsets.data(), offset_ptr,
                               offsets.size() * 4);
  driver.memcpy_device_to_host(actual_output.data(), output_ptr,
                               output.size() * 4);
  for (int b = 0; b <= bins; ++b)
    EXPECT_EQ(actual_offsets[b + 1], 0);
  EXPECT_EQ(actual_offsets.front(), -77);
  EXPECT_EQ(actual_offsets.back(), -77);
  EXPECT_EQ(actual_output, output);
}
}  // namespace

TEST(BucketBuilderCuda, FixedCaptureBoundariesPayloadsAndIndependentOwners) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  for (int bins : {1, 5, 1024, 1025})
    for (int words : {1, 3})
      check_capture(bins, words);
}

TEST(BucketBuilderCuda, WorkspaceUsesCountsOnlyAndHasNoArbitraryBinCap) {
  EXPECT_EQ(cuda::driver_bucket_builder_workspace_bytes(0, 1025), 0);
  EXPECT_EQ(cuda::driver_bucket_builder_workspace_bytes(13, 1024), 4096);
  EXPECT_EQ(cuda::driver_bucket_builder_workspace_bytes(13, 1025), 4360);
  EXPECT_GT(cuda::driver_bucket_builder_workspace_bytes(INT32_MAX, INT32_MAX),
            std::size_t(INT32_MAX) * 4);
  EXPECT_ANY_THROW(cuda::driver_bucket_builder_workspace_bytes(-1, 1));
  EXPECT_ANY_THROW(cuda::driver_bucket_builder_workspace_bytes(1, 0));
}

TEST(BucketBuilderCuda, StridedCountsPayloadAndPaddingReuseArena) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  PrimitiveWorkspaceArena arena;
  constexpr int n = 9, bins = 1025, words = 3;
  constexpr uint32_t sentinel = 0xbad0bad0;
  const std::array<int32_t, n> original{0,   bins - 1, 512,  -1, 2,
                                        512, bins,     -100, 2};
  std::vector<int32_t> keys(2 * n + 1, -77), offsets(2 * (bins + 1) + 1, -77);
  std::vector<uint32_t> values(4 * n + 1, sentinel),
      output(4 * n + 1, sentinel);
  auto key_mem = device.allocate_memory_unique({keys.size() * 4});
  auto value_mem = device.allocate_memory_unique({values.size() * 4});
  auto offset_mem = device.allocate_memory_unique({offsets.size() * 4});
  auto output_mem = device.allocate_memory_unique({output.size() * 4});
  auto cursor_mem = device.allocate_memory_unique({bins * 4});
  ASSERT_EQ(key_mem.second, RhiResult::success);
  ASSERT_EQ(value_mem.second, RhiResult::success);
  ASSERT_EQ(offset_mem.second, RhiResult::success);
  ASSERT_EQ(output_mem.second, RhiResult::success);
  ASSERT_EQ(cursor_mem.second, RhiResult::success);
  void *key_ptr = device.get_memory_addr(*key_mem.first);
  void *value_ptr = device.get_memory_addr(*value_mem.first);
  void *offset_ptr = device.get_memory_addr(*offset_mem.first);
  void *output_ptr = device.get_memory_addr(*output_mem.first);
  void *cursor_ptr = device.get_memory_addr(*cursor_mem.first);
  for (int i = 0; i < n; ++i) {
    values[4 * i + 1] = 100 + i;
    values[4 * i + 2] = 0x7fc01234u + i;
    values[4 * i + 3] = 0x80000000u + i;
  }
  driver.memcpy_host_to_device(value_ptr, values.data(), values.size() * 4);
  driver.memcpy_host_to_device(offset_ptr, offsets.data(), offsets.size() * 4);
  for (int replay = 0; replay < 3; ++replay) {
    std::vector<int32_t> counts(bins), expected = offsets,
                                       actual(offsets.size());
    std::vector<uint32_t> result(output.size());
    for (int i = 0; i < n; ++i) {
      const auto key = replay == 1 ? -1 : original[i];
      keys[2 * i + 1] = key;
      if (key >= 0 && key < bins)
        ++counts[key];
    }
    expected[1] = 0;
    for (int b = 0; b < bins; ++b)
      expected[2 * (b + 1) + 1] = expected[2 * b + 1] + counts[b];
    driver.memcpy_host_to_device(key_ptr, keys.data(), keys.size() * 4);
    driver.memcpy_host_to_device(output_ptr, output.data(), output.size() * 4);
    EXPECT_GE(cuda::driver_bucket_builder_strided(
                  key_ptr, value_ptr, offset_ptr, output_ptr, cursor_ptr, n,
                  bins, words, 4, 8, 4, 16, 4, 8, 4, 16, nullptr, &arena),
              8);
    driver.context_synchronize();
    driver.memcpy_device_to_host(actual.data(), offset_ptr, actual.size() * 4);
    driver.memcpy_device_to_host(result.data(), output_ptr, result.size() * 4);
    EXPECT_EQ(actual, expected);
    std::array<int, n> seen{};
    for (int b = 0; b < bins; ++b) {
      for (int pos = expected[2 * b + 1]; pos < expected[2 * (b + 1) + 1];
           ++pos) {
        const auto id = result[4 * pos + 1] - 100;
        ASSERT_LT(id, n);
        EXPECT_EQ(keys[2 * id + 1], b);
        ++seen[id];
        for (int lane = 1; lane <= words; ++lane)
          EXPECT_EQ(result[4 * pos + lane], values[4 * id + lane]);
      }
    }
    for (int i = 0; i < n; ++i)
      EXPECT_EQ(seen[i],
                keys[2 * i + 1] >= 0 && keys[2 * i + 1] < bins ? 1 : 0);
    for (std::size_t i = 0; i < result.size(); i += 4)
      EXPECT_EQ(result[i], sentinel);
    for (std::size_t i = expected[2 * bins + 1] * 4 + 1; i < result.size(); ++i)
      EXPECT_EQ(result[i], sentinel);
  }
  EXPECT_EQ(arena
                .snapshot(PrimitiveWorkspaceBackend::cuda,
                          PrimitiveWorkspaceFamily::scan)
                .growth_events,
            1);
}

TEST(BucketBuilderCuda, InvalidViewsAndMissingScratchDoNotWrite) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  auto memory = device.allocate_memory_unique({16});
  ASSERT_EQ(memory.second, RhiResult::success);
  auto *ptr = device.get_memory_addr(*memory.first);
  std::array<int32_t, 4> initial{37, 37, 37, 91}, actual;
  driver.memcpy_host_to_device(ptr, initial.data(), sizeof(initial));
  EXPECT_ANY_THROW(cuda::driver_bucket_builder_with_workspace(
      ptr, ptr, ptr, ptr, 1, 1025, 1, nullptr, nullptr));
  EXPECT_ANY_THROW(cuda::driver_bucket_builder_strided(
      ptr, ptr, ptr, ptr, ptr, 1, 1025, 1, 0, 4, 0, 4, 0, 4, 0, 4, nullptr,
      nullptr));
  EXPECT_ANY_THROW(cuda::driver_bucket_builder_strided(
      ptr, ptr, ptr, ptr, ptr, 1, 1, 1,
      std::numeric_limits<std::size_t>::max() - 1, 4, 0, 4, 0, 4, 0, 4, nullptr,
      nullptr));
  driver.context_synchronize();
  driver.memcpy_device_to_host(actual.data(), ptr, sizeof(actual));
  EXPECT_EQ(actual, initial);
}
}  // namespace taichi::lang
#endif
