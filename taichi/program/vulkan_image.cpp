#include "taichi/program/program.h"

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {

void Program::vulkan_copy_texture(Texture *destination, Texture *source) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::vulkan || !program_impl_,
              "Vulkan texture copy requires the Vulkan backend.");
  TI_ERROR_IF(destination == nullptr || source == nullptr,
              "Vulkan texture copy received a null Texture.");
  TI_ERROR_IF(destination->owning_program() != this ||
                  source->owning_program() != this,
              "Vulkan texture copy resources must belong to the active "
              "runtime.");
  TI_ERROR_IF(destination == source,
              "Vulkan texture copy source and destination must not alias.");
  TI_ERROR_IF(destination->get_buffer_format() != source->get_buffer_format(),
              "Vulkan texture copy requires identical source and destination "
              "formats.");
  const auto extent = source->get_size();
  TI_ERROR_IF(destination->get_size() != extent,
              "Vulkan texture copy requires identical source and destination "
              "extents.");
  const auto format = source->get_buffer_format();
  TI_ERROR_IF(format == BufferFormat::depth16 ||
                  format == BufferFormat::depth24stencil8 ||
                  format == BufferFormat::depth32f,
              "Vulkan texture copy currently supports color formats only.");

  auto leases = acquire_texture_leases({destination, source});
  const DeviceAllocation destination_allocation =
      destination->get_device_allocation();
  const DeviceAllocation source_allocation = source->get_device_allocation();
  ImageCopyParams params{};
  params.width = static_cast<std::uint32_t>(extent[0]);
  params.height = static_cast<std::uint32_t>(extent[1]);
  params.depth = static_cast<std::uint32_t>(extent[2]);
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

std::size_t Program::debug_vulkan_image_sampler_cache_size() {
  TI_ERROR("Vulkan image sampler statistics are unavailable in this build.");
}

}  // namespace taichi::lang

#endif
