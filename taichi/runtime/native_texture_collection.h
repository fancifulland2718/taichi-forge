#pragma once

#include "taichi/runtime/native_image.h"

namespace taichi::lang {

// Immutable descriptor order with one retained owner per distinct image.
// No image storage or executor is allocated; GFX records the existing table.
class NativeTextureCollection final : public NativeResource {
 public:
  NativeTextureCollection(
      Device &device,
      int dimensions,
      const std::vector<std::shared_ptr<NativeSampledImage>> &members);
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  int dimensions() const {
    return dimensions_;
  }
  uint64_t snapshot_id() const {
    return snapshot_id_;
  }
  const std::vector<DeviceAllocation> &allocations() const {
    return allocations_;
  }
  void validate_sampling(const NativeRecordingState &state) const;

 private:
  Device &device_;
  const int dimensions_;
  const uint64_t snapshot_id_;
  std::vector<DeviceAllocation> allocations_;
  std::vector<std::shared_ptr<NativeSampledImage>> owners_;
};

class NativeTextureCollectionKernelBinding final
    : public NativeKernelResourceBinding {
 public:
  NativeTextureCollectionKernelBinding(
      std::shared_ptr<NativeTextureCollection> collection,
      std::vector<int> argument,
      Arch arch,
      Device &device,
      const CallableBase &kernel,
      LaunchContextBuilder &context);
  const std::vector<int> &argument() const override {
    return argument_;
  }
  void validate(Arch arch,
                Device &device,
                const CallableBase &kernel,
                const LaunchContextBuilder &context,
                const NativeRecordingState &state) const override;

 private:
  void validate_parameter(Arch arch,
                          Device &device,
                          const CallableBase &kernel) const;
  const std::shared_ptr<NativeTextureCollection> collection_;
  const std::vector<int> argument_;
  const CallableBase *kernel_;
  const LaunchContextBuilder *context_;
};
}  // namespace taichi::lang
