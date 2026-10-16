#include <gtest/gtest.h>

#include <array>
#include <limits>

#include "taichi/runtime/control_encoding_cost.h"

namespace taichi::lang {
namespace {

TEST(ControlEncodingCost, MatchesEnumeratedCommandsAndObservations) {
  const std::array<std::uint64_t, 2> payloads{2, 3};
  for (std::uint64_t outer = 0; outer <= 3; ++outer) {
    for (std::uint64_t first = 0; first <= 3; ++first) {
      for (std::uint64_t second = 0; second <= 3; ++second) {
        const std::array<std::uint64_t, 2> budgets{first, second};
        NestedControlCostAccumulator estimate(11, 2, outer);
        ASSERT_TRUE(estimate.add_inner(payloads[0], first));
        ASSERT_TRUE(estimate.add_inner(payloads[1], second));
        const auto cost = estimate.finish();
        ASSERT_TRUE(cost);
        for (bool observations : {false, true}) {
          std::uint64_t payload = 2, controls = 0;
          std::uint64_t snapshots = observations ? 2 : 0;
          for (std::uint64_t iteration = 0; iteration < outer; ++iteration) {
            ++controls;
            // Four ordinary units outside the repeated inner payloads.
            for (int unit = 0; unit < 4; ++unit)
              ++payload;
            for (std::size_t child = 0; child < budgets.size(); ++child) {
              if (observations)
                snapshots += 2;
              for (std::uint64_t inner = 0; inner < budgets[child]; ++inner) {
                ++controls;
                for (std::uint64_t unit = 0; unit < payloads[child]; ++unit)
                  ++payload;
              }
            }
          }
          EXPECT_EQ(cost->expanded_payload_units, payload);
          EXPECT_EQ(cost->expanded_controllers, controls);
          EXPECT_EQ(cost->total_actions(observations),
                    payload + controls + snapshots);
        }
      }
    }
  }
}

TEST(ControlEncodingCost, RejectsOverlappingPayloadAndKeepsFailureSticky) {
  NestedControlCostAccumulator invalid_initial(1, 2, 1);
  EXPECT_FALSE(invalid_initial.finish());
  NestedControlCostAccumulator overlap(5, 2, 1);
  EXPECT_TRUE(overlap.add_inner(2, 1));
  EXPECT_FALSE(overlap.add_inner(2, 1));
  EXPECT_FALSE(overlap.add_inner(0, 0));
  EXPECT_FALSE(overlap.finish());
}

TEST(ControlEncodingCost, RejectsEveryIndependentArithmeticOverflow) {
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  NestedControlCostAccumulator product(2, 0, 1);
  EXPECT_FALSE(product.add_inner(2, max));
  EXPECT_FALSE(product.finish());

  NestedControlCostAccumulator sum(3, 0, 1);
  ASSERT_TRUE(sum.add_inner(1, max / 2));
  ASSERT_TRUE(sum.add_inner(1, max / 2));
  EXPECT_FALSE(sum.add_inner(1, 2));
  EXPECT_FALSE(sum.finish());

  NestedControlCostAccumulator controllers(0, 0, 1);
  EXPECT_FALSE(controllers.add_inner(0, max));
  EXPECT_FALSE(controllers.finish());

  NestedControlCostAccumulator outer_product(2, 0, 2);
  ASSERT_TRUE(outer_product.add_inner(1, max - 1));
  EXPECT_FALSE(outer_product.finish());

  NestedControlCostAccumulator initial_sum(3, 1, 1);
  ASSERT_TRUE(initial_sum.add_inner(1, max - 1));
  EXPECT_FALSE(initial_sum.finish());

  NestedControlCostAccumulator outer_controllers(0, 0, max);
  ASSERT_TRUE(outer_controllers.add_inner(0, 1));
  EXPECT_FALSE(outer_controllers.finish());
}

TEST(ControlEncodingCost, DistinguishesPayloadControlAndObservationCapacity) {
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  NestedControlCostAccumulator payload(2, 0, 1);
  ASSERT_TRUE(payload.add_inner(1, max - 1));
  auto cost = payload.finish();
  ASSERT_TRUE(cost);
  EXPECT_EQ(cost->expanded_payload_units, max);
  EXPECT_EQ(cost->expanded_controllers, max);
  EXPECT_FALSE(cost->total_actions(false));

  NestedControlCostAccumulator observation(0, 0, max / 2);
  ASSERT_TRUE(observation.add_inner(0, 0));
  cost = observation.finish();
  ASSERT_TRUE(cost);
  EXPECT_EQ(cost->total_actions(false), max / 2);
  EXPECT_FALSE(cost->total_actions(true));

  // No command allocation: wide counts are arithmetic inputs, not a test load.
  NestedControlCostAccumulator wide(2, 1, std::uint64_t{1} << 40);
  ASSERT_TRUE(wide.add_inner(1, 0));
  cost = wide.finish();
  ASSERT_TRUE(cost);
  EXPECT_EQ(cost->expanded_payload_units, 1);
  EXPECT_EQ(cost->expanded_controllers, std::uint64_t{1} << 40);
  EXPECT_EQ(cost->inner_regions, 1);
}

}  // namespace
}  // namespace taichi::lang
