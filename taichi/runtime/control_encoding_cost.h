#pragma once

#include <cstdint>
#include <limits>
#include <optional>

namespace taichi::lang {

namespace control_encoding_detail {
inline bool add_product(std::uint64_t &sum,
                        std::uint64_t count,
                        std::uint64_t value) {
  if (count && value > (std::numeric_limits<std::uint64_t>::max() - sum) /
                           count)
    return false;
  sum += count * value;
  return true;
}
}  // namespace control_encoding_detail

// A depth-two expanded program contains one initial condition and an outer
// body with ordered inner regions. Payload units must have one consistent
// meaning per estimate: dispatches during early admission, or actual tasks
// after backend preparation. This is arithmetic, not a route admission policy.
struct NestedControlEncodingCost {
  std::uint64_t outer_iterations{0};
  std::uint64_t inner_regions{0};
  std::uint64_t payload_units_per_outer{0};
  std::uint64_t expanded_payload_units{0};
  std::uint64_t controllers_per_outer{0};
  std::uint64_t expanded_controllers{0};

  // One controller per encoded outer/inner iteration. Optional observations
  // bracket the whole outer loop and each inner invocation (not iteration).
  std::optional<std::uint64_t> total_actions(bool observations) const {
    using control_encoding_detail::add_product;
    auto actions = expanded_payload_units;
    if (!add_product(actions, 1, expanded_controllers))
      return std::nullopt;
    if (observations &&
        (!add_product(actions, 1, 2) ||
         !add_product(actions, outer_iterations, inner_regions) ||
         !add_product(actions, outer_iterations, inner_regions)))
      return std::nullopt;
    return actions;
  }
};

// Streaming accumulation avoids a second traversal or an allocated list.
// Once invalid, the accumulator stays invalid and finish() cannot expose a
// wrapped count. Callers validate ordering and control-resource semantics.
class NestedControlCostAccumulator {
 public:
  NestedControlCostAccumulator(std::uint64_t total_units,
                               std::uint64_t initial_units,
                               std::uint64_t outer_iterations)
      : initial_units_(initial_units),
        static_units_(total_units >= initial_units ? total_units - initial_units
                                                   : 0),
        valid_(initial_units <= total_units) {
    cost_.outer_iterations = outer_iterations;
    cost_.controllers_per_outer = 1;
  }

  bool add_inner(std::uint64_t repeated_units, std::uint64_t iterations) {
    using control_encoding_detail::add_product;
    if (!valid_ || repeated_units > static_units_ ||
        !add_product(repeated_units_, iterations, repeated_units) ||
        !add_product(cost_.controllers_per_outer, 1, iterations) ||
        !add_product(cost_.inner_regions, 1, 1)) {
      valid_ = false;
      return false;
    }
    static_units_ -= repeated_units;
    return true;
  }

  std::optional<NestedControlEncodingCost> finish() const {
    using control_encoding_detail::add_product;
    if (!valid_)
      return std::nullopt;
    auto cost = cost_;
    cost.payload_units_per_outer = static_units_;
    cost.expanded_payload_units = initial_units_;
    if (!add_product(cost.payload_units_per_outer, 1, repeated_units_) ||
        !add_product(cost.expanded_payload_units, cost.outer_iterations,
                     cost.payload_units_per_outer) ||
        !add_product(cost.expanded_controllers, cost.outer_iterations,
                     cost.controllers_per_outer))
      return std::nullopt;
    return cost;
  }

 private:
  NestedControlEncodingCost cost_;
  std::uint64_t initial_units_{0};
  std::uint64_t static_units_{0};
  std::uint64_t repeated_units_{0};
  bool valid_{true};
};

}  // namespace taichi::lang
