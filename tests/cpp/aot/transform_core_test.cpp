#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include "taichi/rhi/cpu/transform.h"

namespace taichi::lang {

TEST(TransformCore, ModularStorageAndInPlaceTailWithFewerWorkers) {
  ThreadPool pool(2);
  std::array<std::uint32_t, 6> values{0,           1,           0x7fffffffu,
                                      0x80000000u, 0xffffffffu, 91};
  const auto before = values;
  cpu::transform(values.data(), values.data(), 5, std::uint32_t{2},
                 std::uint32_t{0xfffffffdu}, 7, &pool);
  for (int i = 0; i < 5; ++i) {
    const auto expected =
        static_cast<std::uint32_t>(std::uint64_t{before[i]} * 2 + 0xfffffffdu);
    EXPECT_EQ(values[i], expected);
  }
  EXPECT_EQ(values[5], 91);
  std::uint64_t wide = std::numeric_limits<std::uint64_t>::max();
  cpu::transform(&wide, &wide, 1, std::uint64_t{2}, std::uint64_t{3}, 1,
                 nullptr);
  EXPECT_EQ(wide, 1);
}

TEST(TransformCore, FloatingSpecialValuesAndPartitionIndependence) {
  ThreadPool pool(2);
  const std::array<float, 5> values{
      1.25f, -2.5f, std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(), -0.0f};
  std::array<float, 5> serial{}, parallel{};
  cpu::transform(values.data(), serial.data(), values.size(), 2.f, -0.f, 1,
                 nullptr);
  cpu::transform(values.data(), parallel.data(), values.size(), 2.f, -0.f, 3,
                 &pool);
  for (const auto &result : {serial, parallel}) {
    EXPECT_EQ(result[0], 2.5f);
    EXPECT_EQ(result[1], -5.f);
    EXPECT_TRUE(std::isnan(result[2]));
    EXPECT_EQ(result[3], std::numeric_limits<float>::infinity());
    EXPECT_EQ(result[4], 0.f);
    EXPECT_TRUE(std::signbit(result[4]));
  }
}

TEST(TransformCore, InvalidSchedulerDoesNotWriteAndEmptyRangeIsNoOp) {
  float value = 7, output = 91;
  EXPECT_THROW(cpu::transform(&value, &output, 1, 2.f, 3.f, 2, nullptr),
               std::invalid_argument);
  EXPECT_EQ(output, 91);
  EXPECT_NO_THROW(
      cpu::transform<float>(nullptr, nullptr, 0, 2.f, 3.f, 2, nullptr));
  EXPECT_EQ(cpu::simple_loop_partitions(262143, 4), 1);
  EXPECT_EQ(cpu::simple_loop_partitions(262144, 4), 4);
  EXPECT_EQ(cpu::simple_loop_partitions(262145, 2), 2);
  EXPECT_EQ(
      cpu::simple_loop_partitions(std::numeric_limits<std::size_t>::max(), 17),
      17);
}

TEST(TransformCore, PackedFieldAndArrayKeepTheirSchedulingPolicies) {
  // The same scalar work may use different selected producer policies. These
  // are algorithm crossover points, not restrictions on accepted input size.
  EXPECT_EQ(cpu::transform_partitions(65536, 4, false), 1);
  EXPECT_EQ(cpu::transform_partitions(65536, 4, true), 2);
  EXPECT_EQ(cpu::transform_partitions(262144, 4, false), 4);
  EXPECT_EQ(cpu::transform_partitions(262144, 4, true), 4);
}

TEST(TransformCore, StridedMembersPreserveNeighborsAcrossWorkerPartitions) {
  struct Record {
    std::uint32_t tag, value, other, tail;
  };
  std::array<Record, 17> records;
  for (std::uint32_t i = 0; i < records.size(); ++i)
    records[i] = {0xa5a5a5a5u, 0xffffffffu - i, 0, 91};
  const auto before = records;
  auto *bytes = reinterpret_cast<std::uint8_t *>(records.data());
  ThreadPool pool(2);
  cpu::transform_strided<std::uint32_t>(
      bytes, bytes, 16, offsetof(Record, value), sizeof(Record),
      offsetof(Record, other), sizeof(Record), 2, 3, 7, &pool);
  for (int i = 0; i < records.size(); ++i) {
    EXPECT_EQ(records[i].tag, before[i].tag);
    EXPECT_EQ(records[i].tail, before[i].tail);
    EXPECT_EQ(records[i].value, before[i].value);
    EXPECT_EQ(records[i].other, i < 16 ? before[i].value * 2u + 3u : 0u);
  }
  cpu::transform_strided<std::uint32_t>(
      bytes, bytes, 16, offsetof(Record, other), sizeof(Record),
      offsetof(Record, other), sizeof(Record), 2, 1, 1, nullptr);
  for (int i = 0; i < 16; ++i)
    EXPECT_EQ(records[i].other, (before[i].value * 2u + 3u) * 2u + 1u);
  EXPECT_NO_THROW(cpu::transform_strided<float>(
      nullptr, nullptr, 0, 4, 20, 8, 20, 2, 3, 2, nullptr));
}

}  // namespace taichi::lang
