#include <gtest/gtest.h>

#include <array>
#include <vector>
#include "taichi/rhi/cpu/histogram.h"

namespace taichi::lang {
namespace {
template <typename ValueT, typename CounterT>
void check_histogram() {
  ThreadPool pool(2);
  std::array<ValueT, 13> values{0, 2,  2, ValueT(-1), 6,  1, 0,
                                4, 99, 3, 2,          99, 4};
  for (int partitions : {1, 3, 17}) {
    std::vector<CounterT> partial(
        cpu::aggregation_workspace_bytes<CounterT>(6, partitions) /
        sizeof(CounterT));
    const auto *storage = partial.data();
    for (int replay = 0; replay < 2; ++replay) {
      values[0] = ValueT(replay);
      std::array<CounterT, 7> actual, expected{};
      actual.fill(37);
      expected[6] = 37;
      for (auto value : values)
        if (value >= 0 && std::size_t(value) < 6)
          ++expected[value];
      std::fill(partial.begin(), partial.end(), CounterT{});
      cpu::histogram(values.data(), actual.data(), values.size(), 6, partitions,
                     partial.data(), &pool);
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(storage, partial.data());
      std::fill(partial.begin(), partial.end(), CounterT{});
      cpu::histogram(static_cast<const ValueT *>(nullptr), actual.data(), 0, 6,
                     partitions, partial.data(), &pool);
      EXPECT_EQ(actual, (std::array<CounterT, 7>{0, 0, 0, 0, 0, 0, 37}));
    }
  }
}
}  // namespace

TEST(HistogramCore, RepeatedCountsEmptyPrefixAndUntouchedTail) {
  check_histogram<int32_t, int32_t>();
  check_histogram<uint32_t, int32_t>();
  check_histogram<int32_t, int64_t>();
  check_histogram<uint32_t, int64_t>();
}

TEST(HistogramCore, SchedulerFailureDoesNotClearOutput) {
  int32_t value = 0, output = 37;
  EXPECT_THROW(cpu::histogram(&value, &output, 1, 1, 2,
                              static_cast<int32_t *>(nullptr), nullptr),
               std::invalid_argument);
  EXPECT_EQ(output, 37);
  EXPECT_THROW(cpu::histogram(&value, &output, 1, 1, 0,
                              static_cast<int32_t *>(nullptr), nullptr),
               std::invalid_argument);
  EXPECT_EQ(output, 37);
}

TEST(HistogramCore, ParallelStrategyDoesNotLimitCapacity) {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(cpu::histogram_partitions(maximum, 1024, 4), 4);
  EXPECT_EQ(cpu::histogram_partitions(262147, 8193, 4), 1);
  std::vector<int64_t> bins(8193, 37);
  std::array<uint32_t, 3> values{8192, 8193, UINT32_MAX};
  cpu::histogram(values.data(), bins.data(), values.size(), bins.size(), 1,
                 static_cast<int64_t *>(nullptr), nullptr);
  EXPECT_EQ(bins.back(), 1);
  EXPECT_TRUE(
      std::all_of(bins.begin(), bins.end() - 1, [](auto x) { return x == 0; }));
  EXPECT_THROW(cpu::aggregation_workspace_bytes<int64_t>(maximum / 8, 2),
               std::invalid_argument);
  EXPECT_EQ(cpu::aggregation_workspace_bytes<int64_t>(maximum, 1), 0);
}
}  // namespace taichi::lang
