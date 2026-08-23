#include "taichi/program/texture.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "taichi/rhi/device.h"
#include "taichi/ir/snode.h"

namespace taichi::lang {

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
                 ImageSamplerConfig sampler_config)
    : format_(format),
      width_(width),
      height_(height),
      depth_(depth),
      prog_(prog) {
  GraphicsDevice *device =
      static_cast<GraphicsDevice *>(prog_->get_graphics_device());

  auto [type, num_channels] = buffer_format2type_channels(format);
  TI_TRACE("Create image, gfx device {}, format={}, w={}, h={}, d={}",
           (void *)device, type.to_string(), num_channels, width, height,
           depth);

  TI_ASSERT(num_channels > 0 && num_channels <= 4);

  ImageParams img_params{};
  img_params.dimension = depth > 1 ? ImageDimension::d3D : ImageDimension::d2D;
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
  return reinterpret_cast<intptr_t>(&texture_alloc_);
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
  if (prog_) {
    GraphicsDevice *device =
        static_cast<GraphicsDevice *>(prog_->get_graphics_device());
    device->destroy_image(texture_alloc_);
  }
}

}  // namespace taichi::lang
