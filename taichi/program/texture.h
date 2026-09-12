#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"
#include "taichi/program/runtime_resource_registry.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {

class Program;
class Ndarray;
class SNode;

std::pair<DataType, uint32_t> buffer_format2type_channels(BufferFormat format);
// Combined samplers expose normalized/floating formats as f32. This includes
// sampled-only formats not accepted by the storage-image load/store contract.
bool is_float_sampled_texture_format(BufferFormat format);
// Storage images expose 32-bit shader values even when the backing image uses
// 8- or 16-bit integer channels. Normalized and floating-point formats expose
// f32, signed integer formats expose i32, and unsigned integer formats expose
// u32. This is the shared frontend/SPIR-V contract for image load/store.
DataType buffer_format2storage_image_sampled_type(BufferFormat format);
BufferFormat type_channels2buffer_format(const DataType &type,
                                         uint32_t num_channels);

class TI_DLL_EXPORT Texture {
 public:
  /* Constructs a Texture managed by Program.
   * Texture object allocation and deallocation is handled by Program.
   */
  explicit Texture(Program *prog,
                   BufferFormat format,
                   int width,
                   int height,
                   int depth = 1,
                   ImageSamplerConfig sampler_config = {},
                   ImageDimension dimension = ImageDimension::d2D,
                   int mip_levels = 1);

  /* Constructs a Texture from an existing DeviceAllocation
   * It doesn't handle the allocation and deallocation.
   */
  explicit Texture(DeviceAllocation &devalloc,
                   BufferFormat format,
                   int width,
                   int height,
                   int depth = 1);

  intptr_t get_device_allocation_ptr_as_int() const;

  bool is_cuda_texture() const noexcept;

  std::uint64_t get_cuda_texture_object() const;

  void from_ndarray(Ndarray *ndarray);

  void from_snode(SNode *snode);

  DeviceAllocation get_device_allocation() const {
    return texture_alloc_;
  }

  Program *owning_program() const noexcept {
    return prog_;
  }

  RuntimeResourceHandle runtime_resource_handle() const noexcept {
    return runtime_resource_handle_;
  }

  ~Texture();

  BufferFormat get_buffer_format() const {
    return format_;
  }

  std::array<int, 3> get_size() const {
    return {width_, height_, depth_};
  }

  int get_mip_levels() const {
    return mip_levels_;
  }

  ImageDimension get_dimension() const noexcept {
    return dimension_;
  }

  std::array<int, 3> get_mip_size(int level) const;

 private:
  struct CudaTextureResource;

  friend class Program;

  void bind_runtime_resource_handle(RuntimeResourceHandle handle) noexcept {
    runtime_resource_handle_ = handle;
  }

  DeviceAllocation texture_alloc_{kDeviceNullAllocation};
  std::unique_ptr<CudaTextureResource> cuda_texture_;
  DataType dtype_;
  BufferFormat format_;
  int num_channels_{4};
  int width_;
  int height_;
  int depth_;
  int mip_levels_{1};
  ImageDimension dimension_{ImageDimension::d2D};
  ImageSamplerConfig sampler_config_;

  Program *prog_{nullptr};
  // Immutable after Program registry publication. Kernel/Graph/GGUI binding
  // copies this identity and validates it before dereferencing texture_alloc_.
  RuntimeResourceHandle runtime_resource_handle_;
};

// Immutable membership snapshot. The Python owner retains the Texture objects;
// launch/Graph admission retains their existing generation-qualified leases.
class TI_DLL_EXPORT TextureCollection {
 public:
  struct Member {
    const Texture *texture;
    RuntimeResourceHandle handle;
    DeviceAllocation allocation;
  };

  explicit TextureCollection(const std::vector<Texture *> &textures);
  Program *owning_program() const noexcept { return owner_; }
  int num_dimensions() const noexcept { return num_dimensions_; }
  int capacity() const noexcept { return static_cast<int>(members_.size()); }
  std::uint64_t snapshot_id() const noexcept { return snapshot_id_; }
  const std::vector<Member> &members() const noexcept { return members_; }

 private:
  Program *owner_{nullptr};
  int num_dimensions_{0};
  std::vector<Member> members_;
  // Process-monotonic immutable identity. Graph caches use this instead of the
  // C++/Python object address, which may be reused after collection retirement.
  std::uint64_t snapshot_id_{0};
};

}  // namespace taichi::lang
