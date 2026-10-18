#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace taichi::lang {
// Cold physical submission choice. Boundaries index the flat replay sequence,
// excluding initialization. Neither native handles nor code are duplicated.
struct VulkanSubmissionRecipe {
  enum class Kind { segmented, independent_graphics };
  Kind kind{Kind::segmented};
  std::vector<std::size_t> ends;

  void validate(std::size_t operations) const {
    if (kind == Kind::independent_graphics) {
      if (!ends.empty() || operations < 3)
        throw std::invalid_argument(
            "Independent Vulkan submission needs prefix, graphics and consumer");
      return;
    }
    if (kind != Kind::segmented || ends.empty())
      throw std::invalid_argument("Invalid segmented Vulkan submission");
    std::size_t previous = 0;
    for (const auto end : ends) {
      if (end <= previous || end > operations)
        throw std::invalid_argument(
            "Vulkan submission ends must increase within the replay sequence");
      previous = end;
    }
    if (previous != operations)
      throw std::invalid_argument(
          "Vulkan submission ends must cover the replay sequence");
  }
};
}  // namespace taichi::lang
