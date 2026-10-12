#include "gtest/gtest.h"
#include "taichi/common/json.h"

namespace liong::json {
TEST(Json, PreservesFloatingValuesDuringParsingAndRoundtrip) {
  for (bool complete : {false, true}) {
    auto read = [=](const std::string &text) {
      return complete ? parse_complete(text) : parse(text);
    };
    const auto value = read(
        R"({"alpha":-0.75,"beta":1.25,"small":3.5e-7,"large":1e100,"integer":2})");
    EXPECT_EQ(value["alpha"].ty, L_JSON_FLOAT);
    EXPECT_DOUBLE_EQ(static_cast<double>(value["alpha"]), -0.75);
    EXPECT_FLOAT_EQ(static_cast<float>(value["beta"]), 1.25f);
    EXPECT_DOUBLE_EQ(value["small"].num_float, 3.5e-7);
    EXPECT_DOUBLE_EQ(value["large"].num_float, 1e100);
    EXPECT_EQ(value["integer"].ty, L_JSON_INT);
    EXPECT_EQ(static_cast<int>(value["integer"]), 2);
    const auto restored = read(print(value));
    EXPECT_DOUBLE_EQ(restored["alpha"].num_float, -0.75);
    EXPECT_DOUBLE_EQ(restored["beta"].num_float, 1.25);
    EXPECT_DOUBLE_EQ(restored["large"].num_float, 1e100);
  }
}
}  // namespace liong::json
