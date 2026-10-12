#pragma once

#include <cstddef>
#include "taichi/rhi/device.h"

namespace taichi::lang {

// An owner-local prepared object. Operations can retain resources, and resources
// can retain their dependencies. Destruction follows backend recording retirement.
class NativeResource {
 public:
  virtual ~NativeResource() = default;
  virtual Arch arch() const = 0;
  virtual Device &device() const = 0;
  // Own known scratch only; dependencies are accounted once by their owner.
  virtual std::size_t workspace_bytes() const = 0;
};

enum class NativeObjectKind { operation, resource };

}  // namespace taichi::lang
