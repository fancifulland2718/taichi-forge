#pragma once

#include "taichi/runtime/native_operation.h"

namespace taichi::lang {
class CallableBase;
class LaunchContextBuilder;

// Provider-neutral managed sampled image. Backend handles are private to
// same-device prepared consumers; serialized plans contain only descriptions.
// A prepared execution retains this owner until all sampling and updates
// retire.
class NativeSampledImage : public NativeResource {
 public:
  NativeSampledImage(ImageParams parameters, bool initialized)
      : parameters_(parameters), initialized_(initialized) {
  }
  const ImageParams &parameters() const {
    return parameters_;
  }
  bool retains_input(const std::string &port) const override {
    // Both qualified factories finish the initial upload before returning.
    return port != "pixels";
  }
  void validate_sampling(const NativeRecordingState &state) const {
    if (!initialized_ && !state.initialized.count(this))
      throw std::invalid_argument(
          "Sampled image requires initial pixels or an unconditional upload");
  }
  virtual std::uint64_t cuda_sampled_object() const {
    throw std::invalid_argument("Sampled image has no CUDA texture object");
  }
  virtual void upload_cuda(std::uint64_t, void *) {
    throw std::invalid_argument("Sampled image has no CUDA upload");
  }
  virtual DeviceAllocation gfx_sampled_allocation() const {
    throw std::invalid_argument("Sampled image has no GFX allocation");
  }
  // Packed texel bytes; CUDA array allocation/residency is driver-private.
  virtual std::size_t logical_bytes() const = 0;

 private:
  const ImageParams parameters_;
  const bool initialized_;
};

// Cold kernel binding, never a public raw image handle. The backend copies the
// sampled object into its launch packet; the enclosing execution retains this
// owner until the packet's last submission retires.
class NativeImageKernelBinding {
 public:
  NativeImageKernelBinding(std::shared_ptr<NativeSampledImage> image,
                           std::vector<int> argument,
                           Arch arch,
                           Device &device,
                           const CallableBase &kernel,
                           LaunchContextBuilder &context);
  NativeImageKernelBinding(const NativeImageKernelBinding &) = delete;
  NativeImageKernelBinding &operator=(const NativeImageKernelBinding &) =
      delete;
  const std::vector<int> &argument() const {
    return argument_;
  }
  void validate(Arch arch,
                Device &device,
                const CallableBase &kernel,
                const LaunchContextBuilder &context,
                const NativeRecordingState &state) const;

 private:
  void validate_parameter(Arch arch,
                          Device &device,
                          const CallableBase &kernel) const;
  const std::shared_ptr<NativeSampledImage> image_;
  const std::vector<int> argument_;
  const CallableBase *kernel_;
  const LaunchContextBuilder *context_;
  std::uint64_t cuda_object_{0};
  DeviceAllocation gfx_image_{kDeviceNullAllocation};
};

}  // namespace taichi::lang
