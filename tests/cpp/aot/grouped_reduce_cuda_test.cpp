#include <gtest/gtest.h>

#ifdef TI_WITH_CUDA
#include <array>
#include <functional>
#include <limits>
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"

namespace taichi::lang {
namespace {
using GroupValueType = cuda::CudaTransformValueType;

class CapturedGroupedReduce final : public NativeOperation {
 public:
  CapturedGroupedReduce(Device &device,
                        GroupValueType type,
                        std::function<void(void *)> record)
      : device_(device), record_(std::move(record)) {
    cuda::driver_prepare_grouped_reduce(type);
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void record_cuda(void *stream) override {
    record_(stream);
  }

 private:
  Device &device_;
  std::function<void(void *)> record_;
};

template <typename T>
void capture_grouped(GroupValueType type, bool strided) {
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  constexpr uint32_t n = 9, groups = 4;
  const size_t step = strided ? 2 : 1;
  std::array<int32_t, 21> keys;
  std::array<T, 21> values;
  std::array<T, 11> initial, actual;
  keys.fill(91);
  values.fill(T(37));
  initial.fill(T(37));
  const std::array<int32_t, n> valid{0,
                                     1,
                                     2,
                                     2,
                                     -1,
                                     5,
                                     std::numeric_limits<int32_t>::min(),
                                     std::numeric_limits<int32_t>::max(),
                                     0};
  for (uint32_t i = 0; i < n; ++i)
    values[1 + step * i] = T(i + 1);
  auto key_memory = device.allocate_memory_unique({sizeof(keys)});
  auto value_memory = device.allocate_memory_unique({sizeof(values)});
  auto output_memory = device.allocate_memory_unique({sizeof(initial)});
  ASSERT_EQ(key_memory.second, RhiResult::success);
  ASSERT_EQ(value_memory.second, RhiResult::success);
  ASSERT_EQ(output_memory.second, RhiResult::success);
  void *key_ptr = device.get_memory_addr(*key_memory.first);
  void *value_ptr = device.get_memory_addr(*value_memory.first);
  void *output_ptr = device.get_memory_addr(*output_memory.first);
  driver.memcpy_host_to_device(output_ptr, initial.data(), sizeof(initial));
  driver.memcpy_host_to_device(value_ptr, values.data(), sizeof(values));
  auto full_op =
      std::make_shared<CapturedGroupedReduce>(device, type, [=](void *stream) {
        size_t scratch;
        if (strided) {
          scratch = cuda::driver_grouped_reduce_strided(
              key_ptr, value_ptr, output_ptr, n, groups, type, sizeof(int32_t),
              step * sizeof(int32_t), sizeof(T), step * sizeof(T), sizeof(T),
              step * sizeof(T), stream);
        } else {
          scratch = cuda::driver_grouped_reduce_on_stream(
              static_cast<int32_t *>(key_ptr) + 1,
              static_cast<T *>(value_ptr) + 1, static_cast<T *>(output_ptr) + 1,
              n, groups, type, stream);
        }
        EXPECT_EQ(scratch, 0);
      });
  auto empty_op =
      std::make_shared<CapturedGroupedReduce>(device, type, [=](void *stream) {
        if (strided)
          cuda::driver_grouped_reduce_strided(
              nullptr, nullptr, output_ptr, 0, groups, type, sizeof(int32_t),
              step * sizeof(int32_t), sizeof(T), step * sizeof(T), sizeof(T),
              step * sizeof(T), stream);
        else
          cuda::driver_grouped_reduce_on_stream(
              nullptr, nullptr, static_cast<T *>(output_ptr) + 1, 0, groups,
              type, stream);
      });
  auto full = std::make_unique<FixedExecution>(
      Arch::cuda, device, nullptr,
      std::vector<FixedExecution::Operation>{full_op});
  FixedExecution empty(Arch::cuda, device, nullptr, {empty_op});
  driver.memcpy_device_to_host(actual.data(), output_ptr, sizeof(actual));
  EXPECT_EQ(actual, initial);  // Neither preparation nor capture executes zero.
  EXPECT_EQ(full->workspace_bytes(), 0);
  for (int replay = 0; replay < 3; ++replay) {
    std::array<T, 11> expected = initial;
    for (uint32_t g = 0; g < groups; ++g)
      expected[1 + step * g] = T{};
    for (uint32_t i = 0; i < n; ++i) {
      const int32_t key = replay == 1 ? -1 : valid[i];
      keys[1 + step * i] = key;
      if (key >= 0 && key < groups)
        expected[1 + step * key] += values[1 + step * i];
    }
    driver.memcpy_host_to_device(key_ptr, keys.data(), sizeof(keys));
    full->run();
    driver.context_synchronize();
    driver.memcpy_device_to_host(actual.data(), output_ptr, sizeof(actual));
    EXPECT_EQ(actual, expected);
  }
  empty.run();
  driver.context_synchronize();
  driver.memcpy_device_to_host(actual.data(), output_ptr, sizeof(actual));
  auto expected = initial;
  for (uint32_t g = 0; g < groups; ++g)
    expected[1 + step * g] = T{};
  EXPECT_EQ(actual, expected);
  full->run();
  full.reset();
  driver.context_synchronize();
  std::array<int32_t, 21> unchanged_keys;
  std::array<T, 21> unchanged_values;
  driver.memcpy_device_to_host(unchanged_keys.data(), key_ptr, sizeof(keys));
  driver.memcpy_device_to_host(unchanged_values.data(), value_ptr,
                               sizeof(values));
  EXPECT_EQ(unchanged_keys, keys);
  EXPECT_EQ(unchanged_values, values);
}
}  // namespace

TEST(GroupedReduceCuda, PackedAndStridedCaptureOverwriteEmptyGroupsAndRetire) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  for (bool strided : {false, true}) {
    capture_grouped<int32_t>(GroupValueType::i32, strided);
    capture_grouped<uint32_t>(GroupValueType::u32, strided);
    capture_grouped<int64_t>(GroupValueType::i64, strided);
    capture_grouped<uint64_t>(GroupValueType::u64, strided);
    capture_grouped<float>(GroupValueType::f32, strided);
    capture_grouped<double>(GroupValueType::f64, strided);
  }
}

TEST(GroupedReduceCuda, InvalidInputsAreRejectedBeforeClearingOutput) {
  EXPECT_ANY_THROW(cuda::driver_prepare_grouped_reduce(GroupValueType(99)));
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  auto memory = device.allocate_memory_unique({4 * sizeof(int32_t)});
  ASSERT_EQ(memory.second, RhiResult::success);
  auto *output = device.get_memory_addr(*memory.first);
  std::array<int32_t, 4> initial{37, 37, 37, 91};
  driver.memcpy_host_to_device(output, initial.data(), sizeof(initial));
  EXPECT_ANY_THROW(cuda::driver_grouped_reduce_on_stream(
      nullptr, output, output, 1, 3, GroupValueType::i32, nullptr));
  EXPECT_ANY_THROW(cuda::driver_grouped_reduce_strided(
      output, output, output, 1, 3, GroupValueType::i32, 0, 0, 0, 4, 0, 4,
      nullptr));
  EXPECT_ANY_THROW(cuda::driver_grouped_reduce_strided(
      output, output, output, -1, 3, GroupValueType::i32, 0, 4, 0, 4, 0, 4,
      nullptr));
  driver.context_synchronize();
  std::array<int32_t, 4> actual;
  driver.memcpy_device_to_host(actual.data(), output, sizeof(actual));
  EXPECT_EQ(actual, initial);
}
}  // namespace taichi::lang
#endif
