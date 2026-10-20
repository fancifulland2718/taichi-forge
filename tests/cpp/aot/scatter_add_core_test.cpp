#include <gtest/gtest.h>

#include <array>
#include <vector>
#include "taichi/rhi/cpu/aggregation.h"
#include "taichi/rhi/cpu/scatter_add.h"

namespace taichi::lang {
namespace {
template <typename T>
void check_accumulation() {
  ThreadPool pool(2);
  const std::array<int32_t, 13> indices{
      0,  2, 2,
      -1, 6, 1,
      0,  4, std::numeric_limits<int32_t>::min(),
      3,  2, std::numeric_limits<int32_t>::max(),
      4};
  std::array<T, 13> src;
  for (int i = 0; i < src.size(); ++i)
    src[i] = T(i + 1);
  for (const int partitions : {1, 3, 17}) {
    std::array<T, 7> output, expected;
    output.fill(T(37));
    expected = output;
    std::vector<T> partial(cpu::scatter_add_workspace_bytes<T>(6, partitions) /
                           sizeof(T));
    const auto *storage = partial.data();
    for (int replay = 0; replay < 2; ++replay) {
      std::fill(partial.begin(), partial.end(), T{});
      cpu::scatter_add(src.data(), indices.data(), output.data(), src.size(), 6,
                       partitions, partial.data(), &pool);
      for (int i = 0; i < src.size(); ++i)
        if (indices[i] >= 0 && indices[i] < 6)
          expected[indices[i]] += src[i];
      EXPECT_EQ(output, expected);
      EXPECT_EQ(storage, partial.data());
    }
  }
}
}  // namespace

TEST(GroupedReduceCore, OverwritesOldResultsAndEmptyGroupsWithFewerWorkers) {
  ThreadPool pool(2);
  const std::array<int32_t, 7> keys{
      2, 0, -1, 2, 4, std::numeric_limits<int32_t>::min(), 0};
  const std::array<int32_t, 7> values{
      std::numeric_limits<int32_t>::max(), 7, 13, 1, 17, 19, 3};
  for (int partitions : {1, 3, 11}) {
    std::array<int32_t, 5> output{37, 37, 37, 37, 91};
    std::vector<int32_t> partial(
        cpu::aggregation_workspace_bytes<int32_t>(4, partitions) / 4);
    const auto *storage = partial.data();
    cpu::grouped_reduce(keys.data(), values.data(), output.data(), keys.size(),
                        4, partitions, partial.data(), &pool);
    EXPECT_EQ(output, (std::array<int32_t, 5>{
                          10, 0, std::numeric_limits<int32_t>::min(), 0, 91}));
    std::fill(partial.begin(), partial.end(), 0);
    const std::array<int32_t, 7> invalid{-1, -1, -1, -1, -1, -1, -1};
    cpu::grouped_reduce(invalid.data(), values.data(), output.data(),
                        keys.size(), 4, partitions, partial.data(), &pool);
    EXPECT_EQ(output, (std::array<int32_t, 5>{0, 0, 0, 0, 91}));
    EXPECT_EQ(storage, partial.data());
  }
}

TEST(GroupedReduceCore, FloatOrderDoesNotReadThePreviousDestination) {
  ThreadPool pool(1);
  const std::array<float, 4> values{1e20f, 1.0f, -1e20f, 1.0f};
  const std::array<int32_t, 4> keys{};
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  float serial = nan, parallel = nan;
  std::array<float, 2> partial{};
  cpu::grouped_reduce(keys.data(), values.data(), &serial, 4, 1, 1,
                      static_cast<float *>(nullptr), nullptr);
  cpu::grouped_reduce(keys.data(), values.data(), &parallel, 4, 1, 2,
                      partial.data(), &pool);
  EXPECT_EQ(serial, 1.0f);
  EXPECT_EQ(parallel, 0.0f);
}

TEST(GroupedReduceCore, EmptyInputsClearOnlyTheirOutputAndFailuresDoNotWrite) {
  std::array<double, 4> output{37, 37, 37, 91};
  cpu::grouped_reduce<double>(nullptr, nullptr, output.data(), 0, 3, 1, nullptr,
                              nullptr);
  EXPECT_EQ(output, (std::array<double, 4>{0, 0, 0, 91}));
  output.fill(37);
  const int32_t key = 0;
  const double value = 7;
  EXPECT_THROW(cpu::grouped_reduce<double>(&key, &value, output.data(), 1, 3, 2,
                                           nullptr, nullptr),
               std::invalid_argument);
  EXPECT_EQ(output, (std::array<double, 4>{37, 37, 37, 37}));
}

TEST(ScatterAddCore, MutableAccumulationTailAndFewerWorkers) {
  check_accumulation<int32_t>();
  check_accumulation<uint32_t>();
  check_accumulation<int64_t>();
  check_accumulation<uint64_t>();
  check_accumulation<float>();
  check_accumulation<double>();
}

TEST(ScatterAddCore, SignedAdditionWrapsInSerialAndParallelPaths) {
  ThreadPool pool(2);
  const std::array<int32_t, 3> indices{0, 0, 1};
  const std::array<int64_t, 3> src{std::numeric_limits<int64_t>::max(), 1, -1};
  for (int partitions : {1, 4}) {
    std::array<int64_t, 3> output{1, std::numeric_limits<int64_t>::min(), 91};
    std::vector<int64_t> partial(partitions * 2, 0);
    cpu::scatter_add(src.data(), indices.data(), output.data(), src.size(), 2,
                     partitions, partial.data(), &pool);
    EXPECT_EQ(output[0], std::numeric_limits<int64_t>::min() + 1);
    EXPECT_EQ(output[1], std::numeric_limits<int64_t>::max());
    EXPECT_EQ(output[2], 91);
  }
}

TEST(ScatterAddCore, FloatOrderFollowsPartitionsAndPreservesDestination) {
  ThreadPool pool(1);
  const std::array<float, 4> src{1e20f, 1.0f, -1e20f, 1.0f};
  const std::array<int32_t, 4> indices{};
  float serial = 3.0f, parallel = 3.0f;
  std::array<float, 2> partial{};
  cpu::scatter_add(src.data(), indices.data(), &serial, 4, 1, 1,
                   static_cast<float *>(nullptr), nullptr);
  cpu::scatter_add(src.data(), indices.data(), &parallel, 4, 1, 2,
                   partial.data(), &pool);
  EXPECT_EQ(serial, 1.0f);
  EXPECT_EQ(parallel, 3.0f);
}

TEST(ScatterAddCore, InvalidSchedulerAndRealOverflowFailBeforeWrites) {
  float src = 7, output = 91;
  int32_t index = 0;
  EXPECT_THROW(cpu::scatter_add(&src, &index, &output, 1, 1, 2,
                                static_cast<float *>(nullptr), nullptr),
               std::invalid_argument);
  EXPECT_EQ(output, 91);
  EXPECT_THROW(cpu::scatter_add_workspace_bytes<double>(1, 0),
               std::invalid_argument);
  const auto maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(cpu::scatter_add_workspace_bytes<double>(maximum / 8, 2),
               std::invalid_argument);
  EXPECT_EQ(cpu::scatter_add_workspace_bytes<double>(maximum, 1), 0);
  EXPECT_NO_THROW(cpu::scatter_add<float>(nullptr, nullptr, nullptr, 0, 17, 1,
                                          nullptr, nullptr));
}

TEST(ScatterAddCore, OptionalWorkspaceBudgetDoesNotLimitInputCapacity) {
  EXPECT_EQ(cpu::scatter_add_partitions<float>(262147, 1024, 4), 4);
  EXPECT_EQ(cpu::scatter_add_partitions<float>(4099, 257, 4), 1);
  const auto maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(cpu::aggregation_target_threads(maximum, 2048, 4), 4);
  const auto partitions =
      cpu::scatter_add_partitions<double>(maximum, maximum, 4);
  EXPECT_EQ(partitions, 1);
  EXPECT_EQ(cpu::scatter_add_workspace_bytes<double>(maximum, partitions), 0);
}

TEST(ScatterAddCore, PackedLanesPreserveNeighborsAndMutableAccumulation) {
  ThreadPool pool(1);
  constexpr std::size_t n = 13, bound = 4, lanes = 2, stride = 4;
  std::array<int32_t, n> index{0, 2, -1, 4, 1, 2, 0, 3, 1, 4, -1, 3, 0};
  std::array<int32_t, n * stride> src;
  for (int i = 0; i < src.size(); ++i) src[i] = i + 1;
  for (int parts : {1, 3, 17}) {
    std::array<int32_t, (bound + 1) * stride> dst, expected;
    dst.fill(37);
    expected = dst;
    std::vector<int32_t> partial(cpu::scatter_add_workspace_bytes<int32_t>(bound * lanes, parts) / 4);
    for (int replay = 0; replay < 2; ++replay) {
      std::fill(partial.begin(), partial.end(), 0);
      cpu::packed_scatter_add(
          reinterpret_cast<const uint8_t *>(src.data()), 4, stride * 4,
          index.data(), reinterpret_cast<uint8_t *>(dst.data()), 4, stride * 4,
          n, bound, lanes, parts, partial.data(), &pool);
      for (int i = 0; i < n; ++i)
        if (index[i] >= 0 && index[i] < bound)
          for (int lane = 0; lane < lanes; ++lane)
            expected[index[i] * stride + 1 + lane] += src[i * stride + 1 + lane];
      EXPECT_EQ(dst, expected);
    }
  }
}

TEST(ScatterAddCore, PackedFloatOrderRetainsProducerRecordPartitions) {
  ThreadPool pool(1);
  const std::array<float, 8> src{1e20f, 1, 1, 2, -1e20f, 3, 1, 4};
  const std::array<int32_t, 4> indices{};
  for (int partitions : {1, 2}) {
    std::array<float, 2> dst{37, 37};
    std::vector<float> partial(cpu::scatter_add_workspace_bytes<float>(2, partitions) / 4);
    cpu::packed_scatter_add(
        reinterpret_cast<const uint8_t *>(src.data()), 0, 8, indices.data(),
        reinterpret_cast<uint8_t *>(dst.data()), 0, 8, 4, 1, 2, partitions,
        partial.data(), &pool);
    EXPECT_EQ(dst[0], partitions == 1 ? 1 : 37);
    EXPECT_EQ(dst[1], 47);
  }
}

TEST(ScatterAddCore, PackedWorkspacePolicyAndInvalidSchedulerDoNotWrite) {
  // Shape metadata alone exercises wide lanes and optional workspace fallback.
  EXPECT_EQ(cpu::packed_scatter_add_partitions<float>(262147, 1024, 65, 4), 4);
  EXPECT_EQ(cpu::packed_scatter_add_partitions<double>(262147, 1024, 4097, 4), 1);
  const auto maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW(cpu::packed_scatter_add_scalars(maximum, 2), std::invalid_argument);
  const std::array<float, 2> src{1, 2};
  std::array<float, 2> dst{37, 41};
  int32_t index = 0;
  EXPECT_THROW(cpu::packed_scatter_add(
      reinterpret_cast<const uint8_t *>(src.data()), 0, 8, &index,
      reinterpret_cast<uint8_t *>(dst.data()), 0, 8, 1, 1, 2, 2,
      static_cast<float *>(nullptr), nullptr), std::invalid_argument);
  EXPECT_EQ(dst, (std::array<float, 2>{37, 41}));
}
}  // namespace taichi::lang
