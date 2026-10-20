#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include "taichi/rhi/cpu/reduce.h"

namespace taichi::lang {

TEST(ReduceCore, SelectedPartitionsPreserveFloatingOrderWithFewerWorkers) {
  const float values[]{1e20f, 1, -1e20f, 1, 1e20f, 1, -1e20f, 1};
  float result, partial[4];
  ThreadPool pool(1);
  cpu::reduce(values, &result, 0, 8, 1, partial, &pool);
  EXPECT_EQ(result, 1);
  cpu::reduce(values, &result, 0, 8, 2, partial, &pool);
  EXPECT_EQ(result, 2);
  cpu::reduce(values, &result, 0, 8, 4, partial, &pool);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(cpu::reduce_partitions(262145, 4), 4);
  EXPECT_EQ(cpu::reduce_partitions(262145, 2), 1);
  const auto n = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(cpu::partition_boundary(n, 4, 4), n);
  EXPECT_EQ(cpu::partition_boundary(n, 4, 2), n / 2);
}

TEST(ReduceCore, IntegerOverflowNanAndZeroMatchNativeSemantics) {
  const std::int32_t values[]{std::numeric_limits<std::int32_t>::max(), 1, -1};
  std::int32_t result, partial[2];
  ThreadPool pool(2);
  cpu::reduce(values, &result, 0, 3, 2, partial, &pool);
  EXPECT_EQ(result, std::numeric_limits<std::int32_t>::max());
  const float nan_values[]{std::numeric_limits<float>::quiet_NaN(), 2, -1};
  float value, scratch[2];
  cpu::reduce(nan_values, &value, 1, 3, 1, scratch, &pool);
  EXPECT_EQ(value, -1);
  cpu::reduce(nan_values, &value, 2, 3, 1, scratch, &pool);
  EXPECT_EQ(value, 2);
  const float zeros[]{0.f, -0.f};
  cpu::reduce(zeros, &value, 1, 2, 2, scratch, &pool);
  EXPECT_FALSE(std::signbit(value));
  value = 73;
  EXPECT_THROW(cpu::reduce(zeros, &value, 0, 2, 2, scratch, nullptr),
               std::invalid_argument);
  EXPECT_EQ(value, 73);
}

TEST(ReduceCore, StridedPartitionsPreserveOrderAndOnlyWriteFirstOutput) {
  struct Record { float tag, value, tail; };
  const Record values[]{{37, 1e20f, 91}, {37, 1, 91}, {37, -1e20f, 91}, {37, 1, 91},
                        {37, 1e20f, 91}, {37, 1, 91}, {37, -1e20f, 91}, {37, 1, 91}};
  float output[]{73, 91}, partial[4];
  ThreadPool pool(1);
  for (auto partitions : {1, 2, 4}) {
    cpu::reduce_strided<float>(reinterpret_cast<const std::uint8_t *>(values),
                              output, 0, 8, offsetof(Record, value), sizeof(Record),
                              partitions, partial, &pool);
    EXPECT_EQ(output[0], partitions == 1 ? 1 : partitions == 2 ? 2 : 0);
    EXPECT_EQ(output[1], 91);
  }
  output[0] = 73;
  EXPECT_THROW(cpu::reduce_strided<float>(
                   reinterpret_cast<const std::uint8_t *>(values), output, 0, 8,
                   4, sizeof(Record), 2, partial, nullptr),
               std::invalid_argument);
  EXPECT_EQ(output[0], 73);
  EXPECT_THROW(cpu::reduce_strided<float>(
                   reinterpret_cast<const std::uint8_t *>(values), output, 0,
                   std::numeric_limits<std::size_t>::max(), 4, sizeof(Record),
                   1, partial, &pool),
               std::invalid_argument);
  EXPECT_EQ(output[0], 73);
}

TEST(ReduceCore, StridedIntegerOverflowAndFloatingExtremaMatchPacked) {
  const std::int32_t integers[]{37, std::numeric_limits<std::int32_t>::max(), 91,
                                37, 1, 91};
  std::int32_t integer, partial[2];
  ThreadPool pool(2);
  for (auto partitions : {1, 2}) {
    cpu::reduce_strided<std::int32_t>(
        reinterpret_cast<const std::uint8_t *>(integers), &integer, 0, 2,
        4, 12, partitions, partial, &pool);
    EXPECT_EQ(integer, std::numeric_limits<std::int32_t>::min());
  }
  const float values[]{std::numeric_limits<float>::quiet_NaN(), 2, -1, 0.f, -0.f};
  float interleaved[15], actual, expected, scratch[2];
  for (int i = 0; i < 5; ++i) {
    interleaved[3 * i] = 37;
    interleaved[3 * i + 1] = values[i];
    interleaved[3 * i + 2] = 91;
  }
  for (auto op : {1, 2})
    for (auto partitions : {1, 2}) {
      cpu::reduce(values, &expected, op, 5, partitions, scratch, &pool);
      cpu::reduce_strided<float>(
          reinterpret_cast<const std::uint8_t *>(interleaved), &actual, op,
          5, 4, 12, partitions, scratch, &pool);
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(std::signbit(actual), std::signbit(expected));
    }
}

}  // namespace taichi::lang
