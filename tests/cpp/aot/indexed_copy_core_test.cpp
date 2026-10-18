#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <vector>
#include "taichi/rhi/cpu/indexed_copy.h"
#include "taichi/rhi/cpu/scatter_validation.h"

namespace taichi::lang {

TEST(IndexedCopyCore,
     ScatterValidationReusesStorageAfterChangedAndInvalidIndices) {
  for (const std::size_t bound : {std::size_t{7}, std::size_t{1} << 31}) {
    cpu::PlainScatterIndexValidator validator(5, bound);
    const auto bytes = validator.workspace_bytes();
    EXPECT_GT(bytes, 0);
    std::array<int32_t, 5> indices{0, 2, 5, -1, 6};
    EXPECT_TRUE(validator.unique(indices.data()));
    indices[4] = 2;
    EXPECT_FALSE(validator.unique(indices.data()));
    indices = {6, 4, 1, -1, std::numeric_limits<int32_t>::max()};
    EXPECT_TRUE(validator.unique(indices.data()));
    indices[0] = std::numeric_limits<int32_t>::max();
    EXPECT_EQ(validator.unique(indices.data()), bound == 7);
    EXPECT_EQ(validator.workspace_bytes(), bytes);
  }
  cpu::PlainScatterIndexValidator empty(17, 0);
  EXPECT_EQ(empty.workspace_bytes(), 0);
  EXPECT_TRUE(empty.unique(nullptr));
}

TEST(IndexedCopyCore, SparseScatterValidationHandlesStridedKeysAndCollisions) {
  std::array<int32_t, 33> indices;
  cpu::PlainScatterIndexValidator validator(indices.size(),
                                            std::size_t{1} << 30);
  for (int pass = 0; pass < 17; ++pass) {
    for (int i = 0; i < indices.size(); ++i)
      indices[i] = (i * 65536 + pass * 7) % (1 << 30);
    EXPECT_TRUE(validator.unique(indices.data()));
    indices.back() = indices[pass];
    EXPECT_FALSE(validator.unique(indices.data()));
  }
  EXPECT_LE(validator.workspace_bytes(), 16 * indices.size());
}

TEST(IndexedCopyCore, RawPayloadsAndInvalidIndicesWithFewerWorkers) {
  ThreadPool pool(2);
  const std::array<int32_t, 7> indices{
      3, -1, 0, 4, 2, 3, std::numeric_limits<int32_t>::min()};
  for (const int words : {1, 2, 3, 4, 9}) {
    std::vector<uint32_t> source(4 * words);
    for (int i = 0; i < source.size(); ++i)
      source[i] = 0x80000000u + i;  // Includes negative zero as f32 bits.
    source.back() = 0x7fc00001u;    // Preserve NaN payloads, not just values.
    std::vector<uint32_t> expected(7 * words + 1, 0);
    expected.back() = 91;
    for (int lane = 0; lane < words; ++lane) {
      expected[lane] = source[3 * words + lane];
      expected[2 * words + lane] = source[lane];
      expected[4 * words + lane] = source[2 * words + lane];
      expected[5 * words + lane] = source[3 * words + lane];
    }
    for (int parts : {1, 11}) {
      std::vector<uint32_t> output(expected.size(), 91);
      cpu::indexed_copy(
          reinterpret_cast<const uint8_t *>(source.data()), indices.data(),
          reinterpret_cast<uint8_t *>(output.data()), indices.size(), 4,
          words * sizeof(uint32_t), false, parts, parts > 1 ? &pool : nullptr);
      EXPECT_EQ(output, expected)
          << words << " words, " << parts << " partitions";
    }
  }
}

TEST(IndexedCopyCore, UniqueScatterKeepsUnwrittenTargetsAndTail) {
  ThreadPool pool(2);
  const std::array<int32_t, 5> indices{4, -1, 0, 8, 2};
  const std::array<uint64_t, 5> source{0xffffffffffffffffull, 17,
                                       0x8000000000000000ull, 19,
                                       0x7ff8000000000001ull};
  const std::array<uint64_t, 7> expected{source[2], 91, source[4], 91,
                                         source[0], 91, 91};
  for (int parts : {1, 9}) {
    std::array<uint64_t, 7> output;
    output.fill(91);
    cpu::indexed_copy(
        reinterpret_cast<const uint8_t *>(source.data()), indices.data(),
        reinterpret_cast<uint8_t *>(output.data()), indices.size(), 6,
        sizeof(uint64_t), true, parts, parts > 1 ? &pool : nullptr);
    EXPECT_EQ(output, expected);
  }
}

TEST(IndexedCopyCore, SchedulerFailureDoesNotWriteAndCapacityDoesNotOverflow) {
  uint32_t source = 7, output = 91;
  int32_t index = 0;
  EXPECT_THROW(cpu::indexed_copy(reinterpret_cast<const uint8_t *>(&source),
                                 &index, reinterpret_cast<uint8_t *>(&output),
                                 1, 1, sizeof(source), false, 2, nullptr),
               std::invalid_argument);
  EXPECT_EQ(output, 91);
  EXPECT_NO_THROW(
      cpu::indexed_copy(nullptr, nullptr, nullptr, 0, 0, 4, false, 3, nullptr));
  EXPECT_EQ(cpu::indexed_copy_partitions(65535, 8, false), 1);
  EXPECT_EQ(cpu::indexed_copy_partitions(65536, 8, false), 2);
  EXPECT_EQ(cpu::indexed_copy_partitions(65537, 8, false), 3);
  EXPECT_EQ(cpu::indexed_copy_partitions(262144, 8, true), 4);
  EXPECT_EQ(cpu::indexed_copy_partitions(std::numeric_limits<size_t>::max(), 17,
                                         false),
            17);
}

}  // namespace taichi::lang
