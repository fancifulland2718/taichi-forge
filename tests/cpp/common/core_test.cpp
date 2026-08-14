#include "gtest/gtest.h"

#include "taichi/common/core.h"
#include "taichi/inc/constants.h"

namespace taichi {

TEST(CoreTest, Basic) {
  EXPECT_EQ(trim_string("hello taichi  "), "hello taichi");
}

TEST(CoreTest, DifferentSizeUnionCastInitializesUnusedBytes) {
  EXPECT_EQ(taichi_union_cast_with_different_sizes<std::uint64_t>(
                std::int32_t{-1}),
            std::uint64_t{0xffffffffu});
  EXPECT_EQ(taichi_union_cast_with_different_sizes<std::uint64_t>(
                std::uint16_t{0x1234u}),
            std::uint64_t{0x1234u});

  const float input = -3.5f;
  const auto bits =
      taichi_union_cast_with_different_sizes<std::uint64_t>(input);
  std::uint32_t expected = 0;
  std::memcpy(&expected, &input, sizeof(expected));
  EXPECT_EQ(bits, static_cast<std::uint64_t>(expected));
}

}  // namespace taichi
