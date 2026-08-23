#include "taichi/program/program.h"

#include <array>
#include <limits>

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
namespace {

bool is_depth_format(BufferFormat format) {
  return format == BufferFormat::depth16 ||
         format == BufferFormat::depth24stencil8 ||
         format == BufferFormat::depth32f;
}

std::array<std::uint32_t, 3> checked_image_coordinates(
    const std::vector<int> &values,
    const char *name,
    bool require_positive) {
  TI_ERROR_IF(values.size() != 3,
              "Vulkan image {} must have exactly three components.", name);
  std::array<std::uint32_t, 3> result{};
  for (std::size_t axis = 0; axis < values.size(); ++axis) {
    TI_ERROR_IF(values[axis] < 0 || (require_positive && values[axis] == 0),
                "Vulkan image {} components must be {}.", name,
                require_positive ? "positive" : "nonnegative");
    result[axis] = static_cast<std::uint32_t>(values[axis]);
  }
  return result;
}

void validate_color_texture(Program *program,
                            Texture *texture,
                            const char *name) {
  TI_ERROR_IF(texture == nullptr, "Vulkan image {} is null.", name);
  TI_ERROR_IF(texture->owning_program() != program,
              "Vulkan image {} must belong to the active runtime.", name);
  TI_ERROR_IF(is_depth_format(texture->get_buffer_format()),
              "Vulkan image transfers currently support color formats only.");
}

void validate_texture_region(Texture *texture,
                             const std::array<std::uint32_t, 3> &offset,
                             const std::array<std::uint32_t, 3> &extent,
                             std::uint32_t mip_level,
                             std::uint32_t base_layer,
                             std::uint32_t layer_count,
                             const char *name) {
  TI_ERROR_IF(mip_level != 0,
              "Vulkan {} mip level is unsupported by the current Texture "
              "resource model.",
              name);
  TI_ERROR_IF(base_layer != 0 || layer_count != 1,
              "Vulkan {} array layers are unsupported by the current Texture "
              "resource model.",
              name);
  const auto size = texture->get_size();
  for (std::size_t axis = 0; axis < 3; ++axis) {
    TI_ERROR_IF(static_cast<std::uint64_t>(offset[axis]) + extent[axis] >
                    static_cast<std::uint64_t>(size[axis]),
                "Vulkan {} region exceeds the texture extent.", name);
  }
}

std::size_t texture_texel_bytes(Texture *texture) {
  const auto [type, channels] =
      buffer_format2type_channels(texture->get_buffer_format());
  return static_cast<std::size_t>(data_type_size(type)) * channels;
}

std::size_t ndarray_storage_bytes(Ndarray *ndarray, const char *name) {
  const std::size_t elements = ndarray->get_nelement();
  const std::size_t element_bytes = ndarray->get_element_size();
  const auto max_size = (std::numeric_limits<std::size_t>::max)();
  TI_ERROR_IF(element_bytes != 0 && elements > max_size / element_bytes,
              "Vulkan {} ndarray byte size overflows size_t.", name);
  return elements * element_bytes;
}

std::size_t required_buffer_image_bytes(
    std::size_t buffer_offset,
    std::uint32_t buffer_row_length,
    std::uint32_t buffer_image_height,
    const std::array<std::uint32_t, 3> &extent,
    std::size_t texel_bytes) {
  const std::size_t row_length =
      buffer_row_length == 0 ? extent[0] : buffer_row_length;
  const std::size_t image_height =
      buffer_image_height == 0 ? extent[1] : buffer_image_height;
  TI_ERROR_IF(row_length < extent[0] || image_height < extent[1],
              "Vulkan buffer-image row length and image height must cover the "
              "copy extent.");
  TI_ERROR_IF(texel_bytes == 0 || buffer_offset % texel_bytes != 0,
              "Vulkan buffer-image offset must be aligned to the image texel "
              "block size.");
  const auto max_size = (std::numeric_limits<std::size_t>::max)();
  const auto max_vk_pitch =
      static_cast<std::size_t>((std::numeric_limits<std::int32_t>::max)());
  TI_ERROR_IF(row_length > max_vk_pitch / texel_bytes,
              "Vulkan buffer-image row pitch exceeds the 2^31-1 byte limit.");
  TI_ERROR_IF(extent[2] > max_size / image_height ||
                  extent[2] * image_height > max_size / row_length,
              "Vulkan buffer-image storage size overflow.");
  const std::size_t full_slices =
      static_cast<std::size_t>(extent[2] - 1) * image_height * row_length;
  const std::size_t full_rows =
      static_cast<std::size_t>(extent[1] - 1) * row_length;
  TI_ERROR_IF(full_slices > max_size - full_rows,
              "Vulkan buffer-image storage size overflow.");
  const std::size_t completed_rows = full_slices + full_rows;
  TI_ERROR_IF(extent[0] > max_size - completed_rows,
              "Vulkan buffer-image storage size overflow.");
  const std::size_t texels = completed_rows + extent[0];
  TI_ERROR_IF(texels > max_size / texel_bytes ||
                  buffer_offset > max_size - texels * texel_bytes,
              "Vulkan buffer-image byte range overflow.");
  return buffer_offset + texels * texel_bytes;
}

BufferImageCopyParams make_buffer_image_params(
    std::uint32_t buffer_row_length,
    std::uint32_t buffer_image_height,
    const std::array<std::uint32_t, 3> &offset,
    const std::array<std::uint32_t, 3> &extent,
    std::uint32_t mip_level,
    std::uint32_t base_layer,
    std::uint32_t layer_count) {
  BufferImageCopyParams params{};
  params.buffer_row_length = buffer_row_length;
  params.buffer_image_height = buffer_image_height;
  params.image_mip_level = mip_level;
  params.image_offset = {offset[0], offset[1], offset[2]};
  params.image_extent = {extent[0], extent[1], extent[2]};
  params.image_base_layer = base_layer;
  params.image_layer_count = layer_count;
  return params;
}

}  // namespace

void Program::vulkan_copy_texture(Texture *destination, Texture *source) {
  TI_ERROR_IF(source == nullptr || destination == nullptr,
              "Vulkan texture copy received a null Texture.");
  const auto extent = source->get_size();
  TI_ERROR_IF(destination->get_size() != extent,
              "Vulkan texture copy requires identical source and destination "
              "extents.");
  vulkan_copy_texture_region(destination, source, {0, 0, 0}, {0, 0, 0},
                             {extent[0], extent[1], extent[2]}, 0, 0, 0, 0,
                             1);
}

void Program::vulkan_copy_texture_region(
    Texture *destination,
    Texture *source,
    std::vector<int> source_offset_values,
    std::vector<int> destination_offset_values,
    std::vector<int> extent_values,
    std::uint32_t source_mip_level,
    std::uint32_t destination_mip_level,
    std::uint32_t source_base_layer,
    std::uint32_t destination_base_layer,
    std::uint32_t layer_count) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan || !program_impl_,
              "Vulkan texture copy requires the Vulkan backend.");
  validate_color_texture(this, source, "copy source");
  validate_color_texture(this, destination, "copy destination");
  TI_ERROR_IF(destination == source,
              "Vulkan texture copy source and destination must not alias.");
  TI_ERROR_IF(destination->get_buffer_format() != source->get_buffer_format(),
              "Vulkan texture copy requires identical source and destination "
              "formats.");
  const auto source_offset = checked_image_coordinates(
      source_offset_values, "source offset", false);
  const auto destination_offset = checked_image_coordinates(
      destination_offset_values, "destination offset", false);
  const auto extent =
      checked_image_coordinates(extent_values, "copy extent", true);
  validate_texture_region(source, source_offset, extent, source_mip_level,
                          source_base_layer, layer_count, "copy source");
  validate_texture_region(destination, destination_offset, extent,
                          destination_mip_level, destination_base_layer,
                          layer_count, "copy destination");

  auto leases = acquire_texture_leases({destination, source});
  const DeviceAllocation destination_allocation =
      destination->get_device_allocation();
  const DeviceAllocation source_allocation = source->get_device_allocation();
  ImageCopyParams params{};
  params.width = extent[0];
  params.height = extent[1];
  params.depth = extent[2];
  params.source_offset = {static_cast<std::int32_t>(source_offset[0]),
                          static_cast<std::int32_t>(source_offset[1]),
                          static_cast<std::int32_t>(source_offset[2])};
  params.destination_offset = {
      static_cast<std::int32_t>(destination_offset[0]),
      static_cast<std::int32_t>(destination_offset[1]),
      static_cast<std::int32_t>(destination_offset[2])};
  params.source_mip_level = source_mip_level;
  params.destination_mip_level = destination_mip_level;
  params.source_base_layer = source_base_layer;
  params.destination_base_layer = destination_base_layer;
  params.layer_count = layer_count;
  enqueue_compute_op_lambda(
      [destination_allocation, source_allocation,
       params](Device *, CommandList *commands) {
        commands->copy_image(destination_allocation, source_allocation,
                             ImageLayout::transfer_dst,
                             ImageLayout::transfer_src, params);
      },
      {{destination_allocation, ImageLayout::transfer_dst,
        ImageLayout::shader_read},
       {source_allocation, ImageLayout::transfer_src,
        ImageLayout::shader_read}});
  mark_runtime_submission_pending();
  pin_texture_launch_leases(leases);
}

void Program::vulkan_copy_ndarray_to_texture(
    Texture *destination,
    Ndarray *source,
    std::size_t buffer_offset,
    std::uint32_t buffer_row_length,
    std::uint32_t buffer_image_height,
    std::vector<int> image_offset_values,
    std::vector<int> image_extent_values,
    std::uint32_t image_mip_level,
    std::uint32_t image_base_layer,
    std::uint32_t image_layer_count) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan || !program_impl_,
              "Vulkan buffer-to-image copy requires the Vulkan backend.");
  validate_color_texture(this, destination, "buffer-copy destination");
  TI_ERROR_IF(source == nullptr || source->owning_program() != this,
              "Vulkan buffer-copy source must be an active-runtime ndarray.");
  const auto image_offset = checked_image_coordinates(
      image_offset_values, "buffer-copy image offset", false);
  const auto image_extent = checked_image_coordinates(
      image_extent_values, "buffer-copy image extent", true);
  validate_texture_region(destination, image_offset, image_extent,
                          image_mip_level, image_base_layer, image_layer_count,
                          "buffer-copy destination");
  const std::size_t required_bytes = required_buffer_image_bytes(
      buffer_offset, buffer_row_length, buffer_image_height, image_extent,
      texture_texel_bytes(destination));
  const std::size_t available_bytes =
      ndarray_storage_bytes(source, "buffer-to-image source");
  TI_ERROR_IF(required_bytes > available_bytes,
              "Vulkan buffer-to-image source ndarray is too small for the "
              "declared layout and region.");

  auto texture_leases = acquire_texture_leases({destination});
  auto ndarray_leases = acquire_ndarray_leases({source});
  const auto destination_allocation = destination->get_device_allocation();
  const auto source_allocation = source->get_device_allocation();
  const auto params = make_buffer_image_params(
      buffer_row_length, buffer_image_height, image_offset, image_extent,
      image_mip_level, image_base_layer, image_layer_count);
  enqueue_compute_op_lambda(
      [destination_allocation, source_allocation, buffer_offset,
       params](Device *, CommandList *commands) {
        commands->buffer_barrier(source_allocation);
        commands->buffer_to_image(
            destination_allocation, source_allocation.get_ptr(buffer_offset),
            ImageLayout::transfer_dst, params);
      },
      {{destination_allocation, ImageLayout::transfer_dst,
        ImageLayout::shader_read}});
  mark_runtime_submission_pending();
  pin_texture_launch_leases(texture_leases);
  pin_ndarray_launch_leases(ndarray_leases);
}

void Program::vulkan_copy_texture_to_ndarray(
    Ndarray *destination,
    Texture *source,
    std::size_t buffer_offset,
    std::uint32_t buffer_row_length,
    std::uint32_t buffer_image_height,
    std::vector<int> image_offset_values,
    std::vector<int> image_extent_values,
    std::uint32_t image_mip_level,
    std::uint32_t image_base_layer,
    std::uint32_t image_layer_count) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan || !program_impl_,
              "Vulkan image-to-buffer copy requires the Vulkan backend.");
  validate_color_texture(this, source, "buffer-copy source");
  TI_ERROR_IF(destination == nullptr || destination->owning_program() != this,
              "Vulkan buffer-copy destination must be an active-runtime "
              "ndarray.");
  const auto image_offset = checked_image_coordinates(
      image_offset_values, "buffer-copy image offset", false);
  const auto image_extent = checked_image_coordinates(
      image_extent_values, "buffer-copy image extent", true);
  validate_texture_region(source, image_offset, image_extent, image_mip_level,
                          image_base_layer, image_layer_count,
                          "buffer-copy source");
  const std::size_t required_bytes = required_buffer_image_bytes(
      buffer_offset, buffer_row_length, buffer_image_height, image_extent,
      texture_texel_bytes(source));
  const std::size_t available_bytes =
      ndarray_storage_bytes(destination, "image-to-buffer destination");
  TI_ERROR_IF(required_bytes > available_bytes,
              "Vulkan image-to-buffer destination ndarray is too small for "
              "the declared layout and region.");

  auto texture_leases = acquire_texture_leases({source});
  auto ndarray_leases = acquire_ndarray_leases({destination});
  const auto destination_allocation = destination->get_device_allocation();
  const auto source_allocation = source->get_device_allocation();
  const auto params = make_buffer_image_params(
      buffer_row_length, buffer_image_height, image_offset, image_extent,
      image_mip_level, image_base_layer, image_layer_count);
  enqueue_compute_op_lambda(
      [destination_allocation, source_allocation, buffer_offset,
       params](Device *, CommandList *commands) {
        commands->image_to_buffer(destination_allocation.get_ptr(buffer_offset),
                                  source_allocation,
                                  ImageLayout::transfer_src, params);
        commands->buffer_barrier(destination_allocation);
      },
      {{source_allocation, ImageLayout::transfer_src,
        ImageLayout::shader_read}});
  mark_runtime_submission_pending();
  pin_texture_launch_leases(texture_leases);
  pin_ndarray_launch_leases(ndarray_leases);
}

void Program::vulkan_blit_texture(
    Texture *destination,
    Texture *source,
    std::vector<int> source_offset_values,
    std::vector<int> source_extent_values,
    std::vector<int> destination_offset_values,
    std::vector<int> destination_extent_values,
    std::uint32_t source_mip_level,
    std::uint32_t destination_mip_level,
    std::uint32_t source_base_layer,
    std::uint32_t destination_base_layer,
    std::uint32_t layer_count,
    bool linear_filter) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan || !program_impl_,
              "Vulkan image blit requires the Vulkan backend.");
  validate_color_texture(this, source, "blit source");
  validate_color_texture(this, destination, "blit destination");
  TI_ERROR_IF(source == destination,
              "Vulkan image blit source and destination must not alias.");
  TI_ERROR_IF(source->get_buffer_format() != destination->get_buffer_format(),
              "Vulkan image blit currently requires identical formats.");
  const auto source_offset = checked_image_coordinates(
      source_offset_values, "blit source offset", false);
  const auto source_extent = checked_image_coordinates(
      source_extent_values, "blit source extent", true);
  const auto destination_offset = checked_image_coordinates(
      destination_offset_values, "blit destination offset", false);
  const auto destination_extent = checked_image_coordinates(
      destination_extent_values, "blit destination extent", true);
  validate_texture_region(source, source_offset, source_extent,
                          source_mip_level, source_base_layer, layer_count,
                          "blit source");
  validate_texture_region(destination, destination_offset, destination_extent,
                          destination_mip_level, destination_base_layer,
                          layer_count, "blit destination");
  auto *device = static_cast<vulkan::VulkanDevice *>(get_compute_device());
  TI_ERROR_IF(device == nullptr ||
                  !device->image_blit_supported(source->get_buffer_format(),
                                                destination->get_buffer_format(),
                                                linear_filter),
              "Vulkan image blit is unsupported for the selected format and "
              "filter.");

  auto leases = acquire_texture_leases({destination, source});
  const auto destination_allocation = destination->get_device_allocation();
  const auto source_allocation = source->get_device_allocation();
  ImageBlitParams params{};
  params.source.offset = {static_cast<std::int32_t>(source_offset[0]),
                          static_cast<std::int32_t>(source_offset[1]),
                          static_cast<std::int32_t>(source_offset[2])};
  params.source.extent = {source_extent[0], source_extent[1],
                          source_extent[2]};
  params.source.mip_level = source_mip_level;
  params.source.base_layer = source_base_layer;
  params.destination.offset = {
      static_cast<std::int32_t>(destination_offset[0]),
      static_cast<std::int32_t>(destination_offset[1]),
      static_cast<std::int32_t>(destination_offset[2])};
  params.destination.extent = {destination_extent[0], destination_extent[1],
                               destination_extent[2]};
  params.destination.mip_level = destination_mip_level;
  params.destination.base_layer = destination_base_layer;
  params.layer_count = layer_count;
  params.linear_filter = linear_filter;
  enqueue_graphics_op_lambda(
      [destination_allocation, source_allocation,
       params](GraphicsDevice *, CommandList *commands) {
        commands->blit_image(destination_allocation, source_allocation,
                             ImageLayout::transfer_dst,
                             ImageLayout::transfer_src, params);
      },
      {{destination_allocation, ImageLayout::transfer_dst,
        ImageLayout::shader_read},
       {source_allocation, ImageLayout::transfer_src,
        ImageLayout::shader_read}});
  mark_runtime_submission_pending();
  pin_texture_launch_leases(leases);
}

std::size_t Program::debug_vulkan_image_sampler_cache_size() {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan || !program_impl_,
              "Vulkan image sampler statistics require the Vulkan backend.");
  auto *device = static_cast<vulkan::VulkanDevice *>(get_compute_device());
  TI_ERROR_IF(device == nullptr, "Vulkan image sampler device is unavailable.");
  return device->image_sampler_cache_size();
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

void Program::vulkan_copy_texture(Texture *, Texture *) {
  TI_ERROR("Vulkan texture copy is unavailable in this build.");
}

void Program::vulkan_copy_texture_region(Texture *,
                                         Texture *,
                                         std::vector<int>,
                                         std::vector<int>,
                                         std::vector<int>,
                                         std::uint32_t,
                                         std::uint32_t,
                                         std::uint32_t,
                                         std::uint32_t,
                                         std::uint32_t) {
  TI_ERROR("Vulkan texture-region copy is unavailable in this build.");
}

void Program::vulkan_copy_ndarray_to_texture(Texture *,
                                             Ndarray *,
                                             std::size_t,
                                             std::uint32_t,
                                             std::uint32_t,
                                             std::vector<int>,
                                             std::vector<int>,
                                             std::uint32_t,
                                             std::uint32_t,
                                             std::uint32_t) {
  TI_ERROR("Vulkan buffer-to-image copy is unavailable in this build.");
}

void Program::vulkan_copy_texture_to_ndarray(Ndarray *,
                                             Texture *,
                                             std::size_t,
                                             std::uint32_t,
                                             std::uint32_t,
                                             std::vector<int>,
                                             std::vector<int>,
                                             std::uint32_t,
                                             std::uint32_t,
                                             std::uint32_t) {
  TI_ERROR("Vulkan image-to-buffer copy is unavailable in this build.");
}

void Program::vulkan_blit_texture(Texture *,
                                  Texture *,
                                  std::vector<int>,
                                  std::vector<int>,
                                  std::vector<int>,
                                  std::vector<int>,
                                  std::uint32_t,
                                  std::uint32_t,
                                  std::uint32_t,
                                  std::uint32_t,
                                  std::uint32_t,
                                  bool) {
  TI_ERROR("Vulkan image blit is unavailable in this build.");
}

std::size_t Program::debug_vulkan_image_sampler_cache_size() {
  TI_ERROR("Vulkan image sampler statistics are unavailable in this build.");
}

}  // namespace taichi::lang

#endif
