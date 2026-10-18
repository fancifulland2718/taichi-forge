#include <gtest/gtest.h>

#include <array>
#include <vector>
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
}  // namespace taichi::lang
