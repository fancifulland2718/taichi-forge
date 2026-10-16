#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace taichi::lang {
// Semantic selection shared by persistent descriptions and backend builders.
// The optional fallback is the last body. A missing match executes no body.
// Device-specific body-count limits belong to the corresponding backend.
struct BranchSelection {
  enum class Kind { nonzero, indexed };
  Kind kind{Kind::nonzero};
  bool has_default{false};

  void validate(std::size_t body_count) const {
    if (body_count <= std::size_t(has_default) ||
        (kind != Kind::nonzero && kind != Kind::indexed) ||
        (kind == Kind::nonzero && body_count != 1 + std::size_t(has_default)))
      throw std::invalid_argument("Invalid branch selection or body layout");
  }

  // Called only with a validated layout. body_count denotes no selected body.
  std::size_t select(std::int32_t value, std::size_t body_count) const {
    const auto cases = body_count - std::size_t(has_default);
    if (kind == Kind::nonzero) {
      if (value != 0)
        return 0;
    } else if (value >= 0 && std::size_t(value) < cases) {
      return std::size_t(value);
    }
    return has_default ? cases : body_count;
  }
};
}  // namespace taichi::lang
