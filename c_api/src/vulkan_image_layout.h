#pragma once

#include <optional>

#include "taichi/rhi/public_device.h"
#include "vulkan/vulkan_core.h"

namespace capi {

// VkImageLayout and TiImageLayout have different numeric values. Only import
// layouts that the runtime can represent without losing the external state.
// GENERAL and attachment layouts use the conservative writable internal form.
inline std::optional<taichi::lang::ImageLayout> decode_vulkan_image_import_layout(
    VkImageLayout layout) {
  using taichi::lang::ImageLayout;
  switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
      return ImageLayout::undefined;
    case VK_IMAGE_LAYOUT_GENERAL:
      return ImageLayout::shader_read_write;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return ImageLayout::color_attachment;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      return ImageLayout::depth_attachment;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
      return ImageLayout::depth_attachment_read;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return ImageLayout::shader_read;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      return ImageLayout::transfer_src;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return ImageLayout::transfer_dst;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
      return ImageLayout::present_src;
    default:
      return std::nullopt;
  }
}

}  // namespace capi
