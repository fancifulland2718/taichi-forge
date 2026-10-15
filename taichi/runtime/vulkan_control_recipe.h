#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace taichi::lang {
// Shared physical strategy identifiers; no device, compiler or live handles.
enum class VulkanControlStrategy : std::uint32_t {
  automatic = 0,
  compact = 1,
  chained = 2,
  conditional = 3,
  coarse_conditional = 4,
};

struct VulkanControlRecipe {
  struct Chunk {
    std::uint32_t iterations;
    VulkanControlStrategy strategy;
  };
  std::vector<Chunk> chunks;

  void validate(std::uint64_t budget) const {
    // The shader/command descriptor uses uint32 per chunk. Neither the full
    // budget nor the number of chunks inherits the JIT replay-ring capacity.
    auto remaining = budget;
    for (const auto &chunk : chunks) {
      if (!chunk.iterations || chunk.iterations > remaining ||
          chunk.strategy < VulkanControlStrategy::compact ||
          chunk.strategy > VulkanControlStrategy::coarse_conditional)
        throw std::invalid_argument("Invalid selected Vulkan control chunk");
      remaining -= chunk.iterations;
    }
    if (remaining)
      throw std::invalid_argument("Vulkan control chunks must cover the loop budget");
  }
};

// A selected depth-2 program owns one copy of its flat physical dispatches.
// Its condition region runs once; body ranges already include their selected
// tail conditions. Boundaries index the concatenation of condition and body.
// Only the binding representation changes between persistent and bound plans.
template <typename Binding>
struct VulkanNestedControlRecipe {
  struct Inner {
    Binding predicate;
    Binding counter;
    std::optional<Binding> status;
    std::size_t condition_begin{0};
    std::size_t body_begin{0};
    std::size_t end{0};
    std::uint32_t max_iterations{0};
    std::uint32_t chunk_size{0};
  };
  std::vector<Inner> inners;

  void validate(std::size_t initial,
                std::size_t dispatches,
                std::uint64_t budget) const {
    if (!initial || initial >= dispatches || inners.empty() ||
        budget > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("Invalid selected Vulkan nested control");
    auto cursor = initial;
    for (const auto &inner : inners) {
      if (cursor > inner.condition_begin ||
          inner.condition_begin >= inner.body_begin ||
          inner.body_begin >= inner.end || inner.end >= dispatches ||
          (inner.max_iterations == 0
               ? inner.chunk_size != 0
               : !inner.chunk_size || inner.chunk_size > inner.max_iterations))
        throw std::invalid_argument(
            "Invalid Vulkan nested dispatch boundaries or budget");
      cursor = inner.end;
    }
  }
};
}  // namespace taichi::lang
