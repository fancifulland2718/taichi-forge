#pragma once

#include <vector>
#include <memory>
#include "taichi/rhi/device.h"

namespace taichi::lang {
class CallableBase;
class LaunchContextBuilder;
struct NativeRecordingState;
class NativeResource;

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
  // Mutable non-buffer dependencies need the same per-instance ownership proof
  // as native operations. Ordinary image/array binding contracts stay separate.
  virtual const std::vector<std::shared_ptr<NativeResource>> &
  vulkan_stateful_resources() const {
    static const std::vector<std::shared_ptr<NativeResource>> empty;
    return empty;
  }
};

std::shared_ptr<NativeKernelResourceBinding> prepare_native_kernel_resource(
    std::shared_ptr<NativeResource> resource,
    std::vector<int> argument,
    Arch arch,
    Device &device,
    const CallableBase &kernel,
    LaunchContextBuilder &context);
}  // namespace taichi::lang
