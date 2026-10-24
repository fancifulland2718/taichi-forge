#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "taichi/rhi/device.h"

namespace taichi::lang {

class NativeResource;
// Cold construction dependencies, separate from replay access permissions.
// Listed allocations may still have RHI initialization in flight. Resources
// name the complete retained construction closure; the caller also proves
// private ownership or obtains an explicit producer loan for each dependency.
struct NativePreparationDependencies {
  std::vector<DeviceAllocation> allocations;
  std::vector<const NativeResource *> resources;
};

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
  // nullopt preserves the ordinary producer prefix for unknown factories.
  // An empty value certifies no pending construction work. Explicit recorded
  // initializers still execute before replay. Never queried on warm submission.
  virtual std::optional<NativePreparationDependencies>
  preparation_dependencies() const {
    return std::nullopt;
  }
  // Cold factory input ownership after successful construction. False means
  // all use of this input has completed; no pending work or retained pointer
  // may reference it. Unknown providers conservatively retain every input.
  virtual bool retains_input(const std::string &) const {
    return true;
  }
};

// Complete cold access facts for a stateful resource closure. A TLAS writer
// writes its root while reading its BLAS dependencies; retaining either owner
// alone does not authorize concurrent mutation.
struct NativeResourceUse {
  std::shared_ptr<NativeResource> resource;
  bool writes{false};
};

enum class NativeObjectKind { operation, resource };

}  // namespace taichi::lang
