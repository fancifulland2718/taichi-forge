#include <gtest/gtest.h>
#include <array>
#include <vector>
#include "taichi/rhi/cpu/diagnostic_reduce.h"

namespace taichi::lang {
namespace {
template <typename T>
void check_counts() {
  ThreadPool pool(2);
  std::array<T, 5> values{0, 1, 0, 2, T(-1)};
  std::array<T, 10> interleaved{91, 0, 91, 1, 91, 0, 91, 2, 91, T(-1)};
  cpu::DiagnosticPackedRead<T> packed{values.data()};
  cpu::DiagnosticStridedRead<T> strided{
      reinterpret_cast<const uint8_t *>(interleaved.data()), sizeof(T),
      sizeof(T) * 2};
  for (int parts : {1, 3, 17}) {
    std::vector<uint64_t> partial(parts, UINT64_MAX);
    const auto *storage = partial.data();
    for (int replay = 0; replay < 2; ++replay) {
      values[0] = interleaved[1] = T(replay);
      for (int op = 0; op < 6; ++op) {
        const uint64_t expected = op == 0   ? 3 + replay
                                  : op == 1 ? 2 - replay
                                  : op == 5 ? 1
                                            : 0;
        EXPECT_EQ(cpu::check_count(packed, values.size(), op, 0, 3, parts,
                                   partial.data(), &pool),
                  expected);
        EXPECT_EQ(cpu::check_count(strided, values.size(), op, 0, 3, parts,
                                   partial.data(), &pool),
                  expected);
      }
      EXPECT_EQ(storage, partial.data());
      EXPECT_EQ(
          cpu::check_count(packed, 0, 0, 0, 0, parts, partial.data(), &pool),
          0);
    }
    for (std::size_t i = 0; i < 5; ++i)
      EXPECT_EQ(interleaved[i * 2], T(91));
  }
}

struct FullCountRange {
  using value_type = uint64_t;
  uint64_t operator()(std::size_t begin, std::size_t end) const {
    return end - begin;
  }
  static uint64_t combine(uint64_t a, uint64_t b) {
    return a + b;
  }
};
}  // namespace

TEST(DiagnosticReduceCore,
     PackedStridedAndEmptyTasksOverwriteReusablePartials) {
  check_counts<int32_t>();
  check_counts<uint32_t>();
  check_counts<int64_t>();
  check_counts<uint64_t>();
  // Floating index_oob is intentionally not a supported public predicate.
}

TEST(DiagnosticReduceCore,
     FloatingPredicatesMetricsAndChangingNonfiniteValues) {
  ThreadPool pool(2);
  std::array<double, 7> values{-0.0, 0.0, 1.0, -2.0, NAN, INFINITY, -INFINITY};
  cpu::DiagnosticPackedRead<double> read{values.data()}, none{nullptr};
  std::array<uint64_t, 11> count_partial;
  std::array<double, 11> metric_partial;
  for (int parts : {1, 11}) {
    const std::array<uint64_t, 5> expected{5, 2, 1, 2, 3};
    for (int op = 0; op < 5; ++op)
      EXPECT_EQ(cpu::check_count(read, values.size(), op, 0, 0, parts,
                                 count_partial.data(), &pool),
                expected[op]);
    EXPECT_EQ(cpu::metric_reduce(read, none, values.size(), 0, parts,
                                 metric_partial.data(), &pool),
              INFINITY);
    // Inf-Inf is NaN and maps to +Inf, not a falsely converged zero delta.
    EXPECT_EQ(cpu::metric_reduce(read, read, values.size(), 1, parts,
                                 metric_partial.data(), &pool),
              INFINITY);
    EXPECT_EQ(cpu::metric_reduce(read, none, 4, 0, parts, metric_partial.data(),
                                 &pool),
              2.0);
    EXPECT_EQ(cpu::metric_reduce(read, read, 4, 1, parts, metric_partial.data(),
                                 &pool),
              0.0);
    EXPECT_EQ(cpu::metric_reduce(none, none, 0, 0, parts, metric_partial.data(),
                                 &pool),
              0.0);
  }
}

TEST(DiagnosticReduceCore, WideCountsPreserveTruthWithoutInputCapacityLimits) {
  ThreadPool pool(2);
  std::array<uint64_t, 7> partial;
  const auto maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(cpu::diagnostic_reduce(FullCountRange{}, maximum, 7, partial.data(),
                                   &pool),
            maximum);
  EXPECT_EQ(cpu::encode_check_count_i32(0), 0);
  EXPECT_EQ(cpu::encode_check_count_i32(INT32_MAX), INT32_MAX);
  for (uint64_t value :
       {uint64_t(INT32_MAX) + 1, uint64_t(1) << 32, UINT64_MAX})
    EXPECT_LT(cpu::encode_check_count_i32(value), 0);
}

TEST(DiagnosticReduceCore, InvalidSchedulerAndInputsDoNotTouchPartialStorage) {
  const double value = 1;
  cpu::DiagnosticPackedRead<double> read{&value}, none{nullptr};
  uint64_t partial = 37;
  EXPECT_THROW(cpu::check_count(read, 1, 0, 0, 0, 0, &partial, nullptr),
               std::invalid_argument);
  EXPECT_THROW(cpu::check_count(read, 1, 0, 0, 0, 2, &partial, nullptr),
               std::invalid_argument);
  EXPECT_THROW(cpu::check_count(none, 1, 0, 0, 0, 1, &partial, nullptr),
               std::invalid_argument);
  EXPECT_EQ(partial, 37);
  EXPECT_THROW(cpu::metric_reduce(read, none, 1, 1, 1,
                                  static_cast<double *>(nullptr), nullptr),
               std::invalid_argument);
}
}  // namespace taichi::lang
