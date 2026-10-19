#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "taichi/rhi/cpu/bucket_builder.h"

namespace taichi::lang {
namespace {
template <typename T>
void check_typed_bucket() {
  ThreadPool pool(2);
  constexpr std::size_t n = 13, bins = 5;
  const std::array<int32_t, n> original{2, -1, 0, 2, 4, 99, 3,
                                        0, 2,  1, 4, 3, 0};
  std::array<T, n> values;
  for (std::size_t i = 0; i < n; ++i)
    values[i] = T(i + 1);
  for (int partitions : {1, 3, 17}) {
    std::vector<int32_t> scratch(
        cpu::bucket_builder_workspace_bytes(bins, partitions) /
        sizeof(int32_t));
    const auto *storage = scratch.data();
    for (int replay = 0; replay < 3; ++replay) {
      auto keys = original;
      if (replay == 1)
        keys.fill(-1);
      std::array<int32_t, bins + 2> offsets, expected_offsets;
      offsets.fill(-77);
      expected_offsets.fill(-77);
      std::array<T, n + 1> output, expected;
      output.fill(T(99));
      expected.fill(T(99));
      int32_t cursor = 0;
      expected_offsets[0] = 0;
      for (std::size_t bin = 0; bin < bins; ++bin) {
        for (std::size_t i = 0; i < n; ++i)
          if (keys[i] == int32_t(bin))
            expected[cursor++] = values[i];
        expected_offsets[bin + 1] = cursor;
      }
      std::fill(scratch.begin(), scratch.end(), 0);
      cpu::bucket_builder(keys.data(), values.data(), offsets.data(),
                          output.data(), n, bins, partitions, scratch.data(),
                          &pool);
      EXPECT_EQ(output, expected);
      EXPECT_EQ(offsets, expected_offsets);
      EXPECT_EQ(scratch.data(), storage);
      // Empty replay overwrites only the offsets prefix, leaving payloads and
      // both output tails intact after a previous successful non-empty replay.
      cpu::bucket_builder<T>(nullptr, nullptr, offsets.data(), nullptr, 0, bins,
                             partitions, nullptr, nullptr);
      EXPECT_EQ(output, expected);
      EXPECT_EQ(offsets.back(), -77);
      for (std::size_t i = 0; i <= bins; ++i)
        EXPECT_EQ(offsets[i], 0);
    }
  }
}
}  // namespace

TEST(BucketBuilderCore, StableTypedPayloadsAndRepeatedScratch) {
  check_typed_bucket<int32_t>();
  check_typed_bucket<uint32_t>();
  check_typed_bucket<int64_t>();
  check_typed_bucket<uint64_t>();
  check_typed_bucket<float>();
  check_typed_bucket<double>();
}

TEST(BucketBuilderCore, RawPayloadBitsAndUntouchedTail) {
  ThreadPool pool(2);
  const std::array<int32_t, 5> keys{1, 0, -1, 1, 0};
  using Payload = std::array<uint32_t, 3>;
  const std::array<Payload, 5> values{{{0x7fc01234, 0x80000000, 1},
                                       {0xffffffff, 2, 3},
                                       {0, 0, 0},
                                       {0x7fa00001, 4, 5},
                                       {6, 7, 8}}};
  for (int partitions : {1, 3, 17}) {
    std::vector<int32_t> scratch(
        cpu::bucket_builder_workspace_bytes(2, partitions) / sizeof(int32_t));
    for (int replay = 0; replay < 2; ++replay) {
      const Payload sentinel{37, 37, 37};
      std::array<Payload, 5> output;
      output.fill(sentinel);
      std::array<int32_t, 4> offsets{-77, -77, -77, -77};
      std::fill(scratch.begin(), scratch.end(), 0);
      cpu::bucket_builder_bytes(
          keys.data(), reinterpret_cast<const uint8_t *>(values.data()),
          offsets.data(), reinterpret_cast<uint8_t *>(output.data()),
          keys.size(), 2, sizeof(Payload), partitions, scratch.data(), &pool);
      EXPECT_EQ(output, (std::array<Payload, 5>{values[1], values[4], values[0],
                                                values[3], sentinel}));
      EXPECT_EQ(offsets, (std::array<int32_t, 4>{0, 2, 4, -77}));
    }
  }
}

TEST(BucketBuilderCore, InvalidStorageAndArithmeticLeaveOutputsUntouched) {
  int32_t key = 0, value = 42, offsets = -77, output = -77, scratch = 0;
  const auto maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(cpu::bucket_builder_workspace_bytes(maximum, 1),
               std::invalid_argument);
  EXPECT_THROW(cpu::bucket_builder_workspace_bytes(1, 0),
               std::invalid_argument);
  EXPECT_THROW(cpu::bucket_builder(&key, &value, &offsets, &output, 1, 1, 2,
                                   &scratch, nullptr),
               std::invalid_argument);
  EXPECT_THROW(
      cpu::bucket_builder(&key, &value, &offsets, &output,
                          std::size_t(INT32_MAX) + 1, 1, 1, &scratch, nullptr),
      std::invalid_argument);
  EXPECT_THROW(
      cpu::bucket_builder_bytes(&key, reinterpret_cast<const uint8_t *>(&value),
                                &offsets, reinterpret_cast<uint8_t *>(&output),
                                2, 1, maximum, 1, &scratch, nullptr),
      std::invalid_argument);
  EXPECT_EQ(offsets, -77);
  EXPECT_EQ(output, -77);
  EXPECT_EQ(scratch, 0);
}

TEST(BucketBuilderCore, SerialFallbackPreservesCapacityAndZeroBins) {
  const auto bins = cpu::max_local_aggregation_workspace_bytes / 8 + 1;
  EXPECT_EQ(cpu::bucket_builder_partitions(262151, bins, 4, 32768), 1);
  EXPECT_EQ(cpu::bucket_builder_workspace_bytes(bins, 1), bins * 4);
  EXPECT_EQ(cpu::bucket_builder_partitions(262151, 257, 4, 32768), 4);
  int32_t value = 42, offsets[2]{-77, -77}, output = -77;
  cpu::bucket_builder<int32_t>(nullptr, &value, offsets, &output, 1, 0, 3,
                               nullptr, nullptr);
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], -77);
  EXPECT_EQ(output, -77);
}
}  // namespace taichi::lang
