#pragma once

namespace taichi::lang {

// Invalidated at owner close/reset, before releasing allocations. Submitted
// commands retain native references; this is not a replay registry lookup.
class PreparedResourceLease {
 public:
  virtual ~PreparedResourceLease() = default;
  virtual void clear() noexcept = 0;
};

}  // namespace taichi::lang
