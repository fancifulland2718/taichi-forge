#include "taichi/program/texture.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/rhi/device.h"
#include "taichi/ir/snode.h"

#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#endif

namespace taichi::lang {

struct Texture::CudaTextureResource {
#ifdef TI_WITH_CUDA
  CUarray array{nullptr};
  CUtexObject sampled_object{0};
#else
  void *array{nullptr};
  std::uint64_t sampled_object{0};
#endif
};

#ifdef TI_WITH_CUDA
namespace {

CUaddress_mode cuda_address_mode(ImageAddressMode mode) {
  switch (mode) {
    case ImageAddressMode::repeat:
      return CU_TR_ADDRESS_MODE_WRAP;
    case ImageAddressMode::mirrored_repeat:
      return CU_TR_ADDRESS_MODE_MIRROR;
    case ImageAddressMode::clamp_to_edge:
      return CU_TR_ADDRESS_MODE_CLAMP;
  }
  TI_ERROR("Unsupported CUDA texture address mode");
}

CUfilter_mode cuda_filter_mode(ImageFilter filter) {
  switch (filter) {
    case ImageFilter::nearest:
      return CU_TR_FILTER_MODE_POINT;
    case ImageFilter::linear:
      return CU_TR_FILTER_MODE_LINEAR;
  }
  TI_ERROR("Unsupported CUDA texture filter mode");
}

bool cuda_sampled_format_supported(BufferFormat format) {
  switch (format) {
    case BufferFormat::r8:
    case BufferFormat::rg8:
    case BufferFormat::rgba8:
    case BufferFormat::r16:
    case BufferFormat::rg16:
    case BufferFormat::rgba16:
    case BufferFormat::r16f:
    case BufferFormat::rg16f:
    case BufferFormat::rgba16f:
    case BufferFormat::r32f:
    case BufferFormat::rg32f:
    case BufferFormat::rgba32f:
      return true;
    default:
      return false;
  }
}

CUarray_format cuda_array_format(DataType type) {
  if (type == PrimitiveType::u8) {
    return CU_AD_FORMAT_UNSIGNED_INT8;
  }
  if (type == PrimitiveType::u16) {
    return CU_AD_FORMAT_UNSIGNED_INT16;
  }
  if (type == PrimitiveType::f16) {
    return CU_AD_FORMAT_HALF;
  }
  if (type == PrimitiveType::f32) {
    return CU_AD_FORMAT_FLOAT;
  }
  TI_ERROR("Unsupported CUDA sampled texture channel type {}",
           type.to_string());
}

}  // namespace
#endif

// FIXME: (penguinliong) We might have to differentiate buffer formats and
// texture formats at some point because formats like `rgb10a2` are not easily
// represented by primitive types.
std::pair<DataType, uint32_t> buffer_format2type_channels(BufferFormat format) {
  switch (format) {
    case BufferFormat::r8:
      return std::make_pair(PrimitiveType::u8, 1);
    case BufferFormat::rg8:
      return std::make_pair(PrimitiveType::u8, 2);
    case BufferFormat::rgba8:
      return std::make_pair(PrimitiveType::u8, 4);
    case BufferFormat::rgba8srgb:
      return std::make_pair(PrimitiveType::u8, 4);
    case BufferFormat::bgra8:
      return std::make_pair(PrimitiveType::u8, 4);
    case BufferFormat::bgra8srgb:
      return std::make_pair(PrimitiveType::u8, 4);
    case BufferFormat::r8u:
      return std::make_pair(PrimitiveType::u8, 1);
    case BufferFormat::rg8u:
      return std::make_pair(PrimitiveType::u8, 2);
    case BufferFormat::rgba8u:
      return std::make_pair(PrimitiveType::u8, 4);
    case BufferFormat::r8i:
      return std::make_pair(PrimitiveType::i8, 1);
    case BufferFormat::rg8i:
      return std::make_pair(PrimitiveType::i8, 2);
    case BufferFormat::rgba8i:
      return std::make_pair(PrimitiveType::i8, 4);
    case BufferFormat::r16:
      return std::make_pair(PrimitiveType::u16, 1);
    case BufferFormat::rg16:
      return std::make_pair(PrimitiveType::u16, 2);
    case BufferFormat::rgb16:
      return std::make_pair(PrimitiveType::u16, 3);
    case BufferFormat::rgba16:
      return std::make_pair(PrimitiveType::u16, 4);
    case BufferFormat::r16u:
      return std::make_pair(PrimitiveType::u16, 1);
    case BufferFormat::rg16u:
      return std::make_pair(PrimitiveType::u16, 2);
    case BufferFormat::rgb16u:
      return std::make_pair(PrimitiveType::u16, 3);
    case BufferFormat::rgba16u:
      return std::make_pair(PrimitiveType::u16, 4);
    case BufferFormat::r16i:
      return std::make_pair(PrimitiveType::i16, 1);
    case BufferFormat::rg16i:
      return std::make_pair(PrimitiveType::i16, 2);
    case BufferFormat::rgb16i:
      return std::make_pair(PrimitiveType::i16, 3);
    case BufferFormat::rgba16i:
      return std::make_pair(PrimitiveType::i16, 4);
    case BufferFormat::r16f:
      return std::make_pair(PrimitiveType::f16, 1);
    case BufferFormat::rg16f:
      return std::make_pair(PrimitiveType::f16, 2);
    case BufferFormat::rgb16f:
      return std::make_pair(PrimitiveType::f16, 3);
    case BufferFormat::rgba16f:
      return std::make_pair(PrimitiveType::f16, 4);
    case BufferFormat::r32u:
      return std::make_pair(PrimitiveType::u32, 1);
    case BufferFormat::rg32u:
      return std::make_pair(PrimitiveType::u32, 2);
    case BufferFormat::rgb32u:
      return std::make_pair(PrimitiveType::u32, 3);
    case BufferFormat::rgba32u:
      return std::make_pair(PrimitiveType::u32, 4);
    case BufferFormat::r32i:
      return std::make_pair(PrimitiveType::i32, 1);
    case BufferFormat::rg32i:
      return std::make_pair(PrimitiveType::i32, 2);
    case BufferFormat::rgb32i:
      return std::make_pair(PrimitiveType::i32, 3);
    case BufferFormat::rgba32i:
      return std::make_pair(PrimitiveType::i32, 4);
    case BufferFormat::r32f:
      return std::make_pair(PrimitiveType::f32, 1);
    case BufferFormat::rg32f:
      return std::make_pair(PrimitiveType::f32, 2);
    case BufferFormat::rgb32f:
      return std::make_pair(PrimitiveType::f32, 3);
    case BufferFormat::rgba32f:
      return std::make_pair(PrimitiveType::f32, 4);
    case BufferFormat::depth32f:
      return std::make_pair(PrimitiveType::f32, 1);
    default:
      TI_ERROR("Invalid buffer format");
      return {};
  }
}

DataType buffer_format2storage_image_sampled_type(BufferFormat format) {
  switch (format) {
    case BufferFormat::r8u:
    case BufferFormat::rg8u:
    case BufferFormat::rgba8u:
    case BufferFormat::r16u:
    case BufferFormat::rg16u:
    case BufferFormat::rgba16u:
    case BufferFormat::r32u:
    case BufferFormat::rg32u:
    case BufferFormat::rgba32u:
      return PrimitiveType::u32;
    case BufferFormat::r8i:
    case BufferFormat::rg8i:
    case BufferFormat::rgba8i:
    case BufferFormat::r16i:
    case BufferFormat::rg16i:
    case BufferFormat::rgba16i:
    case BufferFormat::r32i:
    case BufferFormat::rg32i:
    case BufferFormat::rgba32i:
      return PrimitiveType::i32;
    case BufferFormat::r8:
    case BufferFormat::rg8:
    case BufferFormat::rgba8:
    case BufferFormat::rgba8srgb:
    case BufferFormat::r16:
    case BufferFormat::rg16:
    case BufferFormat::rgba16:
    case BufferFormat::r16f:
    case BufferFormat::rg16f:
    case BufferFormat::rgba16f:
    case BufferFormat::r32f:
    case BufferFormat::rg32f:
    case BufferFormat::rgba32f:
    case BufferFormat::depth16:
    case BufferFormat::depth32f:
      return PrimitiveType::f32;
    default:
      TI_ERROR("Buffer format {} is not a supported storage image format",
               static_cast<int>(format));
  }
}

BufferFormat type_channels2buffer_format(const DataType &type,
                                         uint32_t num_channels) {
  BufferFormat format;
  if (type == PrimitiveType::f16) {
    if (num_channels == 1) {
      format = BufferFormat::r16f;
    } else if (num_channels == 2) {
      format = BufferFormat::rg16f;
    } else if (num_channels == 4) {
      format = BufferFormat::rgba16f;
    } else {
      TI_ERROR("Invalid texture channels");
    }
  } else if (type == PrimitiveType::u16) {
    if (num_channels == 1) {
      format = BufferFormat::r16;
    } else if (num_channels == 2) {
      format = BufferFormat::rg16;
    } else if (num_channels == 4) {
      format = BufferFormat::rgba16;
    } else {
      TI_ERROR("Invalid texture channels");
    }
  } else if (type == PrimitiveType::u8) {
    if (num_channels == 1) {
      format = BufferFormat::r8;
    } else if (num_channels == 2) {
      format = BufferFormat::rg8;
    } else if (num_channels == 4) {
      format = BufferFormat::rgba8;
    } else {
      TI_ERROR("Invalid texture channels");
    }
  } else if (type == PrimitiveType::f32) {
    if (num_channels == 1) {
      format = BufferFormat::r32f;
    } else if (num_channels == 2) {
      format = BufferFormat::rg32f;
    } else if (num_channels == 3) {
      format = BufferFormat::rgb32f;
    } else if (num_channels == 4) {
      format = BufferFormat::rgba32f;
    } else {
      TI_ERROR("Invalid texture channels");
    }
  } else {
    TI_ERROR("Invalid texture dtype");
  }
  return format;
}

Texture::Texture(Program *prog,
                 BufferFormat format,
                 int width,
                 int height,
                 int depth,
                 ImageSamplerConfig sampler_config,
                 ImageDimension dimension)
    : format_(format),
      width_(width),
      height_(height),
      depth_(depth),
      dimension_(dimension),
      sampler_config_(sampler_config),
      prog_(prog) {
  auto [type, num_channels] = buffer_format2type_channels(format);
  dtype_ = type;
  num_channels_ = static_cast<int>(num_channels);
  TI_ASSERT(num_channels > 0 && num_channels <= 4);

#ifdef TI_WITH_CUDA
  if (prog_->compile_config().arch == Arch::cuda) {
    auto &driver = CUDADriver::get_instance();
    TI_ERROR_IF(
        driver.get_provider() != CUDADriverProvider::nvidia_cuda,
        "CUDA texture resources require the NVIDIA CUDA Driver provider");
    TI_ERROR_IF(!driver.array_3d_create.available() ||
                    !driver.array_destroy.available() ||
                    !driver.tex_object_create.available() ||
                    !driver.tex_object_destroy.available() ||
                    !driver.memcpy_3d.available(),
                "CUDA texture resources require array, texture-object, and "
                "3D-copy Driver API symbols");
    TI_ERROR_IF(!cuda_sampled_format_supported(format),
                "CUDA sampled textures do not support BufferFormat {} under "
                "the vec4-f32 sampling contract",
                static_cast<int>(format));
    TI_ERROR_IF(num_channels == 3,
                "CUDA sampled textures support one, two, or four channels");
    TI_ERROR_IF(sampler_config.min_filter != sampler_config.mag_filter,
                "CUDA sampled textures require matching min and mag filters");

    auto resource = std::make_unique<CudaTextureResource>();
    CUDA_ARRAY3D_DESCRIPTOR array_desc{};
    array_desc.Width = static_cast<std::size_t>(width);
    array_desc.Height =
        dimension == ImageDimension::d1D ? 0 : static_cast<std::size_t>(height);
    array_desc.Depth =
        dimension == ImageDimension::d3D ? static_cast<std::size_t>(depth) : 0;
    array_desc.Format = cuda_array_format(type);
    array_desc.NumChannels = num_channels;
    array_desc.Flags = 0;

    auto context_guard = CUDAContext::get_instance().get_guard();
    driver.array_3d_create(&resource->array, &array_desc);
    try {
      CUDA_RESOURCE_DESC resource_desc{};
      resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
      resource_desc.res.array.hArray = resource->array;

      CUDA_TEXTURE_DESC texture_desc{};
      texture_desc.addressMode[0] =
          cuda_address_mode(sampler_config.address_mode_u);
      texture_desc.addressMode[1] =
          cuda_address_mode(sampler_config.address_mode_v);
      texture_desc.addressMode[2] =
          cuda_address_mode(sampler_config.address_mode_w);
      texture_desc.filterMode = cuda_filter_mode(sampler_config.min_filter);
      texture_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
      texture_desc.maxAnisotropy = 1;
      texture_desc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
      driver.tex_object_create(&resource->sampled_object, &resource_desc,
                               &texture_desc, nullptr);
    } catch (...) {
      driver.array_destroy.call_with_warning(resource->array);
      throw;
    }
    cuda_texture_ = std::move(resource);
    TI_TRACE("CUDA texture created, format={}, w={}, h={}, d={}",
             type.to_string(), width, height, depth);
    return;
  }
#endif

  GraphicsDevice *device =
      static_cast<GraphicsDevice *>(prog_->get_graphics_device());
  TI_TRACE("Create image, gfx device {}, format={}, w={}, h={}, d={}",
           (void *)device, type.to_string(), num_channels, width, height,
           depth);

  ImageParams img_params{};
  img_params.dimension = dimension;
  img_params.format = format;
  img_params.x = width;
  img_params.y = height;
  img_params.z = depth;
  img_params.initial_layout = ImageLayout::undefined;
  img_params.sampler_config = sampler_config;
  if (format == BufferFormat::depth16 ||
      format == BufferFormat::depth24stencil8 ||
      format == BufferFormat::depth32f) {
    // Depth attachments are sampled/read after rendering but are not storage
    // images. Advertising VK_IMAGE_USAGE_STORAGE_BIT for depth formats is not
    // portable and can make image creation fail before the graphics API runs.
    img_params.usage =
        ImageAllocUsage::Sampled | ImageAllocUsage::Attachment;
  }
  texture_alloc_ = prog_->allocate_texture(img_params);

  format_ = img_params.format;

  TI_TRACE("image created, gfx device {}", (void *)device);
}

Texture::Texture(DeviceAllocation &devalloc,
                 BufferFormat format,
                 int width,
                 int height,
                 int depth)
    : texture_alloc_(devalloc),
      format_(format),
      width_(width),
      height_(height),
      depth_(depth) {
  format_ = format;
}

intptr_t Texture::get_device_allocation_ptr_as_int() const {
  if (cuda_texture_) {
    return reinterpret_cast<intptr_t>(&cuda_texture_->sampled_object);
  }
  return reinterpret_cast<intptr_t>(&texture_alloc_);
}

bool Texture::is_cuda_texture() const noexcept {
  return cuda_texture_ != nullptr;
}

std::uint64_t Texture::get_cuda_texture_object() const {
  TI_ERROR_IF(!cuda_texture_, "Texture is not backed by a CUDA texture object");
  return cuda_texture_->sampled_object;
}

void Texture::from_ndarray(Ndarray *ndarray) {
  TI_ERROR_IF(!ndarray, "Texture upload received a null Ndarray");
  TI_ERROR_IF(!prog_,
              "Texture upload from Ndarray requires a Program-owned Texture");
  TI_ERROR_IF(ndarray->owning_program() != prog_,
              "Texture upload source Ndarray belongs to another Program");
  auto resource_guard = prog_->acquire_runtime_resource_submission_guard();
  auto texture_lease = prog_->acquire_texture_external_lease(this);
  prog_->validate_ndarrays_for_external_submission({ndarray});

  auto semaphore = prog_->flush();

#ifdef TI_WITH_CUDA
  if (cuda_texture_) {
    (void)semaphore;
    TI_ERROR_IF(ndarray->shape.size() !=
                    static_cast<std::size_t>(dimension_ == ImageDimension::d1D
                                                 ? 1
                                                 : dimension_ ==
                                                           ImageDimension::d2D
                                                       ? 2
                                                       : 3),
                "CUDA texture upload dimensionality does not match the source "
                "Ndarray");
    TI_ERROR_IF(ndarray->shape[0] != width_ ||
                    (ndarray->shape.size() > 1 &&
                     ndarray->shape[1] != height_) ||
                    (ndarray->shape.size() > 2 &&
                     ndarray->shape[2] != depth_),
                "CUDA texture upload shape does not match the Texture");
    const std::size_t texel_bytes =
        static_cast<std::size_t>(data_type_size(dtype_)) * num_channels_;
    TI_ERROR_IF(ndarray->get_element_size() != texel_bytes,
                "CUDA texture upload requires a source element size of {} "
                "bytes, got {}",
                texel_bytes, ndarray->get_element_size());

    auto *device =
        static_cast<cuda::CudaDevice *>(prog_->get_compute_device());
    auto source = device->get_alloc_info(ndarray->ndarray_alloc_).ptr;
    CUDA_MEMCPY3D copy{};
    copy.srcMemoryType = ::CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = reinterpret_cast<CUdeviceptr>(source);
    copy.srcPitch = static_cast<std::size_t>(width_) * texel_bytes;
    copy.srcHeight = static_cast<std::size_t>(height_);
    copy.dstMemoryType = ::CU_MEMORYTYPE_ARRAY;
    copy.dstArray = cuda_texture_->array;
    copy.WidthInBytes = static_cast<std::size_t>(width_) * texel_bytes;
    copy.Height = static_cast<std::size_t>(height_);
    copy.Depth = static_cast<std::size_t>(depth_);
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().memcpy_3d(&copy);
    return;
  }
#endif

  GraphicsDevice *device =
      static_cast<GraphicsDevice *>(prog_->get_graphics_device());

  device->image_transition(texture_alloc_, ImageLayout::undefined,
                           ImageLayout::transfer_dst);

  Stream *stream = device->get_compute_stream();
  auto [cmdlist, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);

  BufferImageCopyParams params;
  params.buffer_row_length = ndarray->shape[0];
  params.buffer_image_height = ndarray->shape[1];
  params.image_mip_level = 0;
  params.image_extent.x = width_;
  params.image_extent.y = height_;
  params.image_extent.z = depth_;

  cmdlist->buffer_barrier(ndarray->ndarray_alloc_);
  cmdlist->buffer_to_image(texture_alloc_, ndarray->ndarray_alloc_.get_ptr(0),
                           ImageLayout::transfer_dst, params);

  stream->submit_synced(cmdlist.get(), {semaphore});
}

DevicePtr get_device_ptr(taichi::lang::Program *program, SNode *snode) {
  SNode *dense_parent = snode->parent;
  SNode *root = dense_parent->parent;

  int tree_id = root->get_snode_tree_id();
  DevicePtr root_ptr = program->get_snode_tree_device_ptr(tree_id);

  return root_ptr.get_ptr(program->get_field_in_tree_offset(tree_id, snode));
}

void Texture::from_snode(SNode *snode) {
  TI_ERROR_IF(!snode, "Texture upload received a null SNode");
  TI_ERROR_IF(!prog_,
              "Texture upload from SNode requires a Program-owned Texture");
  auto tree_guard = prog_->acquire_snode_tree_lifecycle_read_guard();
  auto resource_guard = prog_->acquire_runtime_resource_submission_guard();
  auto texture_lease = prog_->acquire_texture_external_lease(this);
  auto semaphore = prog_->flush();

#ifdef TI_WITH_CUDA
  if (cuda_texture_) {
    (void)semaphore;
    TI_ERROR_IF(!snode->is_path_all_dense,
                "CUDA texture upload requires a dense Field");
    const std::size_t texel_bytes =
        static_cast<std::size_t>(data_type_size(dtype_)) * num_channels_;
    TI_ERROR_IF(snode->cell_size_bytes != texel_bytes,
                "CUDA texture upload requires a Field cell size of {} bytes, "
                "got {}",
                texel_bytes, snode->cell_size_bytes);
    TI_ERROR_IF(snode->shape_along_axis(0) != width_ ||
                    (dimension_ != ImageDimension::d1D &&
                     snode->shape_along_axis(1) != height_) ||
                    (dimension_ == ImageDimension::d3D &&
                     snode->shape_along_axis(2) != depth_),
                "CUDA texture upload shape does not match the Field");

    DevicePtr source_ptr = get_device_ptr(prog_, snode);
    DeviceAllocation source_alloc{source_ptr.device, source_ptr.alloc_id};
    auto *device =
        static_cast<cuda::CudaDevice *>(prog_->get_compute_device());
    auto *source = static_cast<std::uint8_t *>(
                       device->get_alloc_info(source_alloc).ptr) +
                   source_ptr.offset;
    CUDA_MEMCPY3D copy{};
    copy.srcMemoryType = ::CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = reinterpret_cast<CUdeviceptr>(source);
    copy.srcPitch = static_cast<std::size_t>(width_) * texel_bytes;
    copy.srcHeight = static_cast<std::size_t>(height_);
    copy.dstMemoryType = ::CU_MEMORYTYPE_ARRAY;
    copy.dstArray = cuda_texture_->array;
    copy.WidthInBytes = static_cast<std::size_t>(width_) * texel_bytes;
    copy.Height = static_cast<std::size_t>(height_);
    copy.Depth = static_cast<std::size_t>(depth_);
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().memcpy_3d(&copy);
    return;
  }
#endif

  TI_ASSERT(snode->is_path_all_dense);

  GraphicsDevice *device =
      static_cast<GraphicsDevice *>(prog_->get_graphics_device());

  device->image_transition(texture_alloc_, ImageLayout::undefined,
                           ImageLayout::transfer_dst);

  DevicePtr devptr = get_device_ptr(prog_, snode);

  Stream *stream = device->get_compute_stream();
  auto [cmdlist, res] = stream->new_command_list_unique();
  TI_ASSERT(res == RhiResult::success);

  BufferImageCopyParams params;
  params.buffer_row_length = snode->shape_along_axis(0);
  params.buffer_image_height = snode->shape_along_axis(1);
  params.image_mip_level = 0;
  params.image_extent.x = width_;
  params.image_extent.y = height_;
  params.image_extent.z = depth_;

  cmdlist->buffer_barrier(devptr);
  cmdlist->buffer_to_image(texture_alloc_, devptr, ImageLayout::transfer_dst,
                           params);

  stream->submit_synced(cmdlist.get(), {semaphore});
}

Texture::~Texture() {
#ifdef TI_WITH_CUDA
  if (cuda_texture_) {
    auto context_guard = CUDAContext::get_instance().get_guard();
    auto &driver = CUDADriver::get_instance();
    if (cuda_texture_->sampled_object != 0 &&
        driver.tex_object_destroy.available()) {
      driver.tex_object_destroy.call_with_warning(
          cuda_texture_->sampled_object);
    }
    if (cuda_texture_->array != nullptr && driver.array_destroy.available()) {
      driver.array_destroy.call_with_warning(cuda_texture_->array);
    }
    return;
  }
#endif
  if (prog_) {
    GraphicsDevice *device =
        static_cast<GraphicsDevice *>(prog_->get_graphics_device());
    device->destroy_image(texture_alloc_);
  }
}

}  // namespace taichi::lang
