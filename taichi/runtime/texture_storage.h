#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "taichi/rhi/device.h"

namespace taichi::lang {

// Checked logical mip layout, shared by owned and borrowed image resources.
// These sizes describe packed texels, not driver allocation/residency.
ImageParams image_mip_parameters(const ImageParams &params, uint32_t level);
std::size_t image_logical_bytes(const ImageParams &params);

// Backend storage shared by managed JIT textures and native AOT resources.
// The enclosing owner retains its CUDA context, serializes device updates and
// retires all sampling work before destruction. This object does not create a
// Program, own a submission queue, or change generation/admission policy.
class CudaTextureStorage {
 public:
  explicit CudaTextureStorage(const ImageParams &params);
  ~CudaTextureStorage();
  CudaTextureStorage(const CudaTextureStorage &) = delete;
  CudaTextureStorage &operator=(const CudaTextureStorage &) = delete;

  std::uint64_t sampled_object() const {
    return sampled_object_;
  }
  const std::uint64_t *sampled_object_address() const {
    return &sampled_object_;
  }
  // Logical texel bytes, not driver allocation/residency accounting.
  std::size_t texel_bytes() const {
    return texel_bytes_;
  }
  std::size_t logical_bytes() const {
    return logical_bytes_;
  }

  // Complete packed upload, preserving Forge's logical ndarray axis order.
  // Caller validates source ownership/shape/size and orders prior producers.
  void upload_device(std::uint64_t source);
  // Enqueue the same packed transfer on the enclosing execution's stream.
  // Source and image remain leased through its completion.
  void upload_device_async(std::uint64_t source, void *stream);

 private:
  void upload(std::uint64_t source, void *stream, bool asynchronous);
  void *array_{nullptr};
  std::uint64_t sampled_object_{0};
  std::array<std::size_t, 3> physical_extent_{};
  std::size_t texel_bytes_{0};
  std::size_t logical_bytes_{0};
};

}  // namespace taichi::lang
