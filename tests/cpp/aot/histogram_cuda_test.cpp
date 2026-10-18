#include <gtest/gtest.h>

#ifdef TI_WITH_CUDA
#include <array>
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"

namespace taichi::lang {
namespace {
class Histogram final : public NativeOperation {
 public:
  Histogram(Device &device,
            void *values,
            void *bins,
            uint32_t n,
            cuda::CudaTransformValueType value_type,
            cuda::CudaTransformValueType bin_type)
      : device_(device),
        values_(values),
        bins_(bins),
        n_(n),
        value_type_(value_type),
        bin_type_(bin_type) {
    cuda::driver_prepare_histogram(value_type_, bin_type_);
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
    cuda::driver_histogram_on_stream(values_, bins_, n_, 6, value_type_,
                                     bin_type_, stream);
  }

 private:
  Device &device_;
  void *values_, *bins_;
  uint32_t n_;
  cuda::CudaTransformValueType value_type_, bin_type_;
};

template <typename Value, typename Counter>
void check_capture(cuda::CudaTransformValueType value_type,
                   cuda::CudaTransformValueType bin_type) {
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  std::array<Value, 13> values{0, 1, 1, 2, 2, 4, 5, Value(-1), 6, 99, 2, 3, 3};
  std::array<Counter, 8> initial, actual;
  initial.fill(37);
  auto src = device.allocate_memory_unique({sizeof(values)});
  auto dst = device.allocate_memory_unique({sizeof(initial)});
  ASSERT_EQ(src.second, RhiResult::success);
  ASSERT_EQ(dst.second, RhiResult::success);
  auto *source = device.get_memory_addr(*src.first);
  auto *output = device.get_memory_addr(*dst.first);
  auto *prefix = static_cast<char *>(output) + sizeof(Counter);
  driver.memcpy_host_to_device(output, initial.data(), sizeof(initial));
  auto op = std::make_shared<Histogram>(device, source, prefix, values.size(),
                                        value_type, bin_type);
  auto empty = std::make_shared<Histogram>(device, nullptr, prefix, 0,
                                           value_type, bin_type);
  auto full = std::make_unique<FixedExecution>(
      Arch::cuda, device, nullptr, std::vector<FixedExecution::Operation>{op});
  FixedExecution zero(Arch::cuda, device, nullptr, {empty});
  driver.memcpy_device_to_host(actual.data(), output, sizeof(actual));
  EXPECT_EQ(actual, initial);  // Preparation/capture must not clear bins.
  EXPECT_EQ(full->workspace_bytes(), 0);
  for (int replay = 0; replay < 2; ++replay) {
    values[0] = Value(replay);
    driver.memcpy_host_to_device(source, values.data(), sizeof(values));
    full->run();
    driver.context_synchronize();
    std::array<Counter, 8> expected{37, 0, 0, 0, 0, 0, 0, 37};
    for (auto value : values)
      if (value >= 0 && uint64_t(value) < 6)
        ++expected[std::size_t(value) + 1];
    driver.memcpy_device_to_host(actual.data(), output, sizeof(actual));
    EXPECT_EQ(actual, expected);
    zero.run();
    driver.context_synchronize();
    driver.memcpy_device_to_host(actual.data(), output, sizeof(actual));
    EXPECT_EQ(actual, (std::array<Counter, 8>{37, 0, 0, 0, 0, 0, 0, 37}));
  }
  full->run();
  full.reset();  // Retire an outstanding launch before allocations disappear.
  std::array<Value, 13> unchanged;
  driver.memcpy_device_to_host(unchanged.data(), source, sizeof(values));
  EXPECT_EQ(unchanged, values);
}
}  // namespace

TEST(HistogramCuda, ClearAndCountCapturePreservesPrefixesAndEmptyReplay) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  using Type = cuda::CudaTransformValueType;
  check_capture<int32_t, int32_t>(Type::i32, Type::i32);
  check_capture<uint32_t, int32_t>(Type::u32, Type::i32);
  check_capture<int32_t, int64_t>(Type::i32, Type::i64);
  check_capture<uint32_t, int64_t>(Type::u32, Type::i64);
}
}  // namespace taichi::lang
#endif
