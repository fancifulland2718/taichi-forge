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

}  // namespace taichi::lang
