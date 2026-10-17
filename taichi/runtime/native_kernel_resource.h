#pragma once

#include <vector>
#include "taichi/rhi/device.h"

namespace taichi::lang {
class CallableBase;
class LaunchContextBuilder;
struct NativeRecordingState;

// A cold-qualified kernel parameter binding. Concrete bindings own their
// provider resources and any host descriptors referenced by the launch packet.
// Fixed execution retains these owners until every recorded use has retired.
class NativeKernelResourceBinding {
 public:
  virtual ~NativeKernelResourceBinding() = default;
  virtual const std::vector<int> &argument() const = 0;
  virtual void validate(Arch arch,
                        Device &device,
                        const CallableBase &kernel,
                        const LaunchContextBuilder &context,
                        const NativeRecordingState &state) const = 0;
};
}  // namespace taichi::lang
