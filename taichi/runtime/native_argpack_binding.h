#pragma once

#include "taichi/runtime/native_argpack_storage.h"
#include "taichi/runtime/native_kernel_resource.h"

namespace taichi::lang {

// One cold binding owns every host allocation descriptor referenced by the
// launch packet, as well as the immutable storage. Module and lifted-resource
// lifetimes remain with the enclosing ordinary/prepared invocation owner.
class NativeArgPackKernelBinding final : public NativeKernelResourceBinding {
 public:
  NativeArgPackKernelBinding(std::shared_ptr<NativeArgPackStorage> storage,
                             std::vector<int> argument,
                             Arch arch,
                             Device &device,
                             const CallableBase &kernel,
                             LaunchContextBuilder &context);
  NativeArgPackKernelBinding(const NativeArgPackKernelBinding &) = delete;
  NativeArgPackKernelBinding &operator=(const NativeArgPackKernelBinding &) = delete;
  const std::vector<int> &argument() const override {
    return argument_;
  }
  void validate(Arch arch,
                Device &device,
                const CallableBase &kernel,
                const LaunchContextBuilder &context,
                const NativeRecordingState &state) const override;

 private:
  const std::shared_ptr<NativeArgPackStorage> storage_;
  const std::vector<int> argument_;
  const CallableBase *kernel_;
  const LaunchContextBuilder *context_;
  std::vector<std::vector<int>> paths_;
  std::vector<DeviceAllocation> allocations_;
};

}  // namespace taichi::lang
