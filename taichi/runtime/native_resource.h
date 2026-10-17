#pragma once

#include <cstddef>
#include <string>
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
  // Known persistent device payload owned by this object, excluding inputs,
  // scratch, host code/descriptors and driver-private residency. Zero does
  // not assert that opaque provider/driver allocations consume no memory.
  virtual std::size_t device_payload_bytes() const {
    return 0;
  }
  // Cold factory input ownership after successful construction. False means
  // all use of this input has completed; no pending work or retained pointer
  // may reference it. Unknown providers conservatively retain every input.
  virtual bool retains_input(const std::string &) const {
    return true;
  }
};

enum class NativeObjectKind { operation, resource };

}  // namespace taichi::lang
