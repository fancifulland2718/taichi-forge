#include "gtest/gtest.h"
#include "taichi/common/json.h"
#include <limits>

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

TEST(Json, ValidatesNumberGrammarAndRepresentableRange) {
  for (bool complete : {false, true}) {
    auto read = [=](const std::string &text) {
      return complete ? parse_complete(text) : parse(text);
    };
    const auto value = read(
        R"([0,-0,9223372036854775807,-9223372036854775808,1E+2,-2.5E-2,1e2])");
    EXPECT_EQ(value[2].num_int, std::numeric_limits<int64_t>::max());
    EXPECT_EQ(value[3].num_int, std::numeric_limits<int64_t>::min());
    EXPECT_DOUBLE_EQ(value[4].num_float, 100.0);
    EXPECT_DOUBLE_EQ(value[5].num_float, -0.025);
    EXPECT_DOUBLE_EQ(value[6].num_float, 100.0);
    EXPECT_EQ(read("0").num_int, 0);
    EXPECT_EQ(read("-1").num_int, -1);
    EXPECT_DOUBLE_EQ(read("1E2").num_float, 100.0);
    for (const auto *number : {"+1",
                               "-",
                               "01",
                               "-01",
                               "1.",
                               "1e",
                               "1e+",
                               "1e-",
                               "1-2",
                               "1+2",
                               "1e2-3",
                               "1e++2",
                               "9223372036854775808",
                               "-9223372036854775809",
                               "18446744073709551616",
                               "1e999",
                               "-1e999",
                               "NaN",
                               "Infinity",
                               ".5"}) {
      EXPECT_THROW(read(number), JsonException) << number;
      EXPECT_THROW(read(std::string("{\"value\":") + number + "}"),
                   JsonException)
          << number;
    }
  }
}
}  // namespace liong::json
