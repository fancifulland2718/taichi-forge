#include <gtest/gtest.h>

#ifdef TI_WITH_CUDA
#include <array>
#include <cstring>
#include <functional>
#include <limits>
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/linear_ptx.h"
#include "taichi/runtime/fixed_execution.h"
#include "taichi/runtime/native_operation.h"

namespace taichi::lang {
namespace {
using DiagnosticType = cuda::CudaTransformValueType;
using Check = cuda::CudaCheckOp;
using Metric = cuda::CudaMetricOp;

class Diagnostic final : public NativeOperation {
 public:
  Diagnostic(Device &device, std::function<void(void *)> record)
      : device_(device), record_(std::move(record)) {
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

// Real FixedExecution capture, including an empty recording, with interleaved
// input and a scalar output surrounded by guards. Neither capture may write.
template <typename Value, typename Result, typename Prepare, typename Launch>
void exercise_capture(std::array<Value, 6> values,
                      std::array<Value, 6> other,
                      std::array<Result, 3> expected,
                      Prepare prepare,
                      Launch launch) {
  cuda::CudaDevice device;
  auto guard = CUDAContext::get_instance().get_guard();
  auto &driver = CUDADriver::get_instance();
  std::array<Value, 13> interleaved;
  interleaved.fill(Value(99));
  std::array<Result, 3> initial{37, 37, 37}, actual;
  auto src = device.allocate_memory_unique({sizeof(interleaved)});
  auto rhs = device.allocate_memory_unique({sizeof(other)});
  auto dst = device.allocate_memory_unique({sizeof(initial)});
  ASSERT_EQ(src.second, RhiResult::success);
  ASSERT_EQ(rhs.second, RhiResult::success);
  ASSERT_EQ(dst.second, RhiResult::success);
  auto *source = device.get_memory_addr(*src.first);
  auto *second = device.get_memory_addr(*rhs.first);
  auto *output = device.get_memory_addr(*dst.first);
  auto *scalar = static_cast<char *>(output) + sizeof(Result);
  driver.memcpy_host_to_device(output, initial.data(), sizeof(initial));
  prepare();
  auto op = std::make_shared<Diagnostic>(device, [=](void *stream) {
    EXPECT_EQ(launch(source, second, scalar, 6, stream), 0);
  });
  auto empty = std::make_shared<Diagnostic>(device, [=](void *stream) {
    EXPECT_EQ(launch(nullptr, nullptr, scalar, 0, stream), 0);
  });
  auto full = std::make_unique<FixedExecution>(
      Arch::cuda, device, nullptr, std::vector<FixedExecution::Operation>{op});
  FixedExecution zero(Arch::cuda, device, nullptr, {empty});
  driver.memcpy_device_to_host(actual.data(), output, sizeof(actual));
  EXPECT_EQ(actual, initial);
  EXPECT_EQ(full->workspace_bytes(), 0);
  for (int replay = 0; replay < 3; ++replay) {
    if (replay == 1) {
      values.fill(Value(0));
      other.fill(Value(0));
    } else if (replay == 2) {
      values[0] = Value(4);
    }
    for (size_t i = 0; i < values.size(); ++i)
      interleaved[1 + 2 * i] = values[i];
    driver.memcpy_host_to_device(source, interleaved.data(),
                                 sizeof(interleaved));
    driver.memcpy_host_to_device(second, other.data(), sizeof(other));
    full->run();
    driver.context_synchronize();
    driver.memcpy_device_to_host(actual.data(), output, sizeof(actual));
    EXPECT_EQ(actual, (std::array<Result, 3>{37, expected[replay], 37}));
    std::array<Value, 13> unchanged;
    std::array<Value, 6> other_unchanged;
    driver.memcpy_device_to_host(unchanged.data(), source, sizeof(unchanged));
    driver.memcpy_device_to_host(other_unchanged.data(), second, sizeof(other));
    EXPECT_EQ(
        std::memcmp(unchanged.data(), interleaved.data(), sizeof(unchanged)),
        0);
    EXPECT_EQ(std::memcmp(other_unchanged.data(), other.data(), sizeof(other)),
              0);
  }
  zero.run();
  driver.context_synchronize();
  driver.memcpy_device_to_host(actual.data(), output, sizeof(actual));
  EXPECT_EQ(actual, (std::array<Result, 3>{37, 0, 37}));
  full->run();
  full.reset();  // Retire the outstanding launch before allocations disappear.
  driver.memcpy_device_to_host(actual.data(), output, sizeof(actual));
  EXPECT_EQ(actual, (std::array<Result, 3>{37, expected.back(), 37}));
}

template <typename T>
void check_capture(DiagnosticType type,
                   Check op,
                   std::array<T, 6> values,
                   std::array<int32_t, 3> expected,
                   int lower = 0,
                   int upper = 0) {
  exercise_capture<T, int32_t>(
      values, {}, expected, [=] { cuda::driver_prepare_check_count(type, op); },
      [=](void *src, void *, void *dst, uint32_t n, void *stream) {
        return cuda::driver_check_count_on_stream(src, dst, n, type, sizeof(T),
                                                  2 * sizeof(T), op, lower,
                                                  upper, stream);
      });
}

template <typename T>
void metric_capture(DiagnosticType type) {
  const T inf = std::numeric_limits<T>::infinity();
  const T nan = std::numeric_limits<T>::quiet_NaN();
  for (auto op : {Metric::max_abs, Metric::max_abs_delta}) {
    auto prepare = [=] { cuda::driver_prepare_metric_reduce(type, op); };
    auto launch = [=](void *src, void *rhs, void *dst, uint32_t n,
                      void *stream) {
      return cuda::driver_metric_reduce_on_stream(
          src, op == Metric::max_abs ? nullptr : rhs, dst, n, type, sizeof(T),
          2 * sizeof(T), 0, sizeof(T), op, stream);
    };
    exercise_capture<T, T>({-0.0, 1, -2, 3, 0, 5}, {0, 1, 0, 0, 0, -2},
                           {op == Metric::max_abs ? T(5) : T(7), 0, 4}, prepare,
                           launch);
    exercise_capture<T, T>({-0.0, 1, -2, nan, 0, 5}, {}, {inf, 0, 4}, prepare,
                           launch);
    exercise_capture<T, T>({-0.0, 1, -2, inf, 0, 5}, {0, 0, 0, inf, 0, 0},
                           {inf, 0, 4}, prepare, launch);
  }
}
}  // namespace

TEST(DiagnosticCuda, CheckCaptureNumericTypesAndPredicates) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  check_capture<int32_t>(DiagnosticType::i32, Check::nonzero,
                         {0, 1, -1, 0, 2, 0}, {3, 0, 1});
  check_capture<uint32_t>(DiagnosticType::u32, Check::nonzero,
                          {0, 1, UINT32_MAX, 0, 2, 0}, {3, 0, 1});
  check_capture<int64_t>(DiagnosticType::i64, Check::nonzero,
                         {0, 1, -1, 0, 2, 0}, {3, 0, 1});
  check_capture<uint64_t>(DiagnosticType::u64, Check::nonzero,
                          {0, 1, UINT64_MAX, 0, 2, 0}, {3, 0, 1});
  check_capture<int32_t>(DiagnosticType::i32, Check::index_oob,
                         {0, 1, -1, 0, 2, 0}, {1, 0, 1}, -1, 2);
  check_capture<uint32_t>(DiagnosticType::u32, Check::index_oob,
                          {0, 1, UINT32_MAX, 0, 2, 0}, {6, 6, 6}, -3, -1);
  check_capture<uint64_t>(DiagnosticType::u64, Check::index_oob,
                          {0, 1, UINT64_MAX, 0, 2, 0}, {6, 6, 6}, -3, -1);
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::array<double, 6> floating{-0.0, 0.0, 1, -2, nan, inf};
  const std::array<float, 6> floating32{-0.0f, 0.0f,       1,
                                        -2,    float(nan), float(inf)};
  for (auto type : {DiagnosticType::f32, DiagnosticType::f64}) {
    for (auto entry :
         {std::make_pair(Check::nonzero, 4), std::make_pair(Check::zero, 2),
          std::make_pair(Check::nan, 1), std::make_pair(Check::inf, 1),
          std::make_pair(Check::not_finite, 2)}) {
      const std::array<int32_t, 3> expected{
          entry.second, entry.first == Check::zero ? 6 : 0,
          entry.first == Check::zero ? 5
                                     : (entry.first == Check::nonzero ? 1 : 0)};
      if (type == DiagnosticType::f32)
        check_capture(type, entry.first, floating32, expected);
      else
        check_capture(type, entry.first, floating, expected);
    }
  }
}

TEST(DiagnosticCuda, MetricCaptureOverwritesNonfiniteResultsAndEmptyOutput) {
  if (!CUDADriver::get_instance_without_context().detected())
    GTEST_SKIP();
  metric_capture<float>(DiagnosticType::f32);
  metric_capture<double>(DiagnosticType::f64);
}

TEST(DiagnosticCuda, PreparationRejectsUnsupportedSemantics) {
  EXPECT_ANY_THROW(
      cuda::driver_prepare_check_count(DiagnosticType(99), Check::nonzero));
  EXPECT_ANY_THROW(
      cuda::driver_prepare_check_count(DiagnosticType::i32, Check(99)));
  EXPECT_ANY_THROW(
      cuda::driver_prepare_check_count(DiagnosticType::f32, Check::index_oob));
  EXPECT_ANY_THROW(
      cuda::driver_prepare_metric_reduce(DiagnosticType::i32, Metric::max_abs));
  EXPECT_ANY_THROW(
      cuda::driver_prepare_metric_reduce(DiagnosticType::f64, Metric(99)));
}
}  // namespace taichi::lang
#endif
