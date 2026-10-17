#pragma once

#include "taichi/runtime/native_operation.h"

namespace taichi::lang {

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
  // Packed texel bytes; CUDA array allocation/residency is driver-private.
  virtual std::size_t logical_bytes() const = 0;

 private:
  const ImageParams parameters_;
  const bool initialized_;
};

}  // namespace taichi::lang
