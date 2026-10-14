#pragma once

#include <cstdint>
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
}  // namespace taichi::lang
