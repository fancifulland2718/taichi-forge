#if defined(TI_WITH_VULKAN) && defined(TI_TEST_RUNTIME_C_API)
#undef TI_WITH_VULKAN
#define TI_WITH_VULKAN 1

#include <array>
#include "gtest/gtest.h"
#include "c_api/src/taichi_core_impl.h"
#include "c_api/src/vulkan_image_layout.h"
#include "taichi/cpp/taichi.hpp"
#include "c_api/tests/gtest_fixture.h"
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace {
using taichi::lang::ImageLayout;

TEST(VulkanImageImportLayout, DecodesExplicitVulkanValues) {
  const std::pair<VkImageLayout, ImageLayout> cases[] = {
      {VK_IMAGE_LAYOUT_UNDEFINED, ImageLayout::undefined},
      {VK_IMAGE_LAYOUT_GENERAL, ImageLayout::shader_read_write},
      {VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, ImageLayout::color_attachment},
      {VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
       ImageLayout::depth_attachment},
      {VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
       ImageLayout::depth_attachment_read},
      {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, ImageLayout::shader_read},
      {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ImageLayout::transfer_src},
      {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ImageLayout::transfer_dst},
      {VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, ImageLayout::present_src},
  };
  for (const auto &[input, expected] : cases) {
    SCOPED_TRACE(static_cast<int>(input));
    const auto actual = capi::decode_vulkan_image_import_layout(input);
    ASSERT_TRUE(actual.has_value());
    EXPECT_EQ(*actual, expected);
  }
}

TEST(VulkanImageImportLayout, RejectsLayoutsWithoutExactInternalRepresentation) {
  for (const auto layout : {
           VK_IMAGE_LAYOUT_PREINITIALIZED,
           VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
           VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
           VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
           VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
           static_cast<VkImageLayout>(123456789),
           VK_IMAGE_LAYOUT_MAX_ENUM,
       }) {
    SCOPED_TRACE(static_cast<int>(layout));
    EXPECT_FALSE(capi::decode_vulkan_image_import_layout(layout));
  }
}

// Observe this test's private device table, never Volk's global entry points.
// Restored before device teardown. The callbacks allocate nothing and forward
// to the real driver, so the pixel test exercises the actual submitted work.
class ImportCalls {
 public:
  explicit ImportCalls(taichi::lang::vulkan::VulkanDevice &device)
      : table_(const_cast<VolkDeviceTable &>(device.vk_device().functions())),
        barrier_(table_.vkCmdPipelineBarrier),
        create_view_(table_.vkCreateImageView) {
    EXPECT_EQ(active_, nullptr);
    active_ = this;
    table_.vkCmdPipelineBarrier = barrier;
    table_.vkCreateImageView = create_view;
  }
  ~ImportCalls() {
    table_.vkCmdPipelineBarrier = barrier_;
    table_.vkCreateImageView = create_view_;
    active_ = nullptr;
  }
  VkImage observed_image{VK_NULL_HANDLE};
  VkImageLayout first_old{VK_IMAGE_LAYOUT_UNDEFINED};
  VkImageLayout first_new{VK_IMAGE_LAYOUT_UNDEFINED};
  uint32_t image_barriers{0};
  uint32_t view_creations{0};

 private:
  static VKAPI_ATTR void VKAPI_CALL barrier(
      VkCommandBuffer commands,
      VkPipelineStageFlags source,
      VkPipelineStageFlags destination,
      VkDependencyFlags flags,
      uint32_t memory_count,
      const VkMemoryBarrier *memory,
      uint32_t buffer_count,
      const VkBufferMemoryBarrier *buffers,
      uint32_t image_count,
      const VkImageMemoryBarrier *images) {
    auto &self = *active_;
    for (uint32_t i = 0; i < image_count; ++i) {
      if (images[i].image == self.observed_image) {
        if (self.image_barriers++ == 0) {
          self.first_old = images[i].oldLayout;
          self.first_new = images[i].newLayout;
        }
      }
    }
    self.barrier_(commands, source, destination, flags, memory_count, memory,
                  buffer_count, buffers, image_count, images);
  }
  static VKAPI_ATTR VkResult VKAPI_CALL create_view(
      VkDevice device,
      const VkImageViewCreateInfo *info,
      const VkAllocationCallbacks *allocator,
      VkImageView *view) {
    auto &self = *active_;
    ++self.view_creations;
    return self.create_view_(device, info, allocator, view);
  }
  VolkDeviceTable &table_;
  PFN_vkCmdPipelineBarrier barrier_;
  PFN_vkCreateImageView create_view_;
  inline static thread_local ImportCalls *active_{nullptr};
};

TEST_F(CapiTest, VulkanImageImportRejectsLayoutBeforeCreatingView) {
  if (!ti::is_arch_available(TI_ARCH_VULKAN)) GTEST_SKIP();
  ti::Runtime runtime(TI_ARCH_VULKAN);
  auto texture = runtime.allocate_texture2d(2, 2, TI_FORMAT_RGBA8, TI_NULL_HANDLE);
  ASSERT_TAICHI_SUCCESS();
  ASSERT_TRUE(texture.is_valid());
  TiVulkanImageInteropInfo info{};
  ti_export_vulkan_image(runtime, texture.image(), &info);
  ASSERT_TAICHI_SUCCESS();
  auto &device = static_cast<taichi::lang::vulkan::VulkanDevice &>(
      reinterpret_cast<Runtime *>(runtime.runtime())->get());
  ImportCalls calls(device);
  for (auto layout : {VK_IMAGE_LAYOUT_PREINITIALIZED,
                      VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
                      static_cast<VkImageLayout>(123456789),
                      VK_IMAGE_LAYOUT_MAX_ENUM}) {
    EXPECT_EQ(ti_import_vulkan_image(runtime, &info, VK_IMAGE_VIEW_TYPE_2D,
                                     layout),
              nullptr);
    EXPECT_TAICHI_ERROR(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "layout");
    EXPECT_EQ(calls.view_creations, 0u);
  }
  // A rejected import must leave the runtime available for a valid import.
  auto handle = ti_import_vulkan_image(runtime, &info, VK_IMAGE_VIEW_TYPE_2D,
                                       VK_IMAGE_LAYOUT_UNDEFINED);
  ASSERT_NE(handle, nullptr);
  ti::Image imported(runtime, handle, TI_IMAGE_DIMENSION_2D, {2, 2, 1, 1}, 1,
                     TI_FORMAT_RGBA8, true);
  ASSERT_TAICHI_SUCCESS();
  EXPECT_EQ(calls.view_creations, 1u);
}

TEST_F(CapiTest, VulkanImageImportPreservesLayoutAndPixels) {
  using namespace taichi::lang;
  if (!ti::is_arch_available(TI_ARCH_VULKAN)) GTEST_SKIP();
  ti::Runtime runtime(TI_ARCH_VULKAN);
  auto &native_runtime = *reinterpret_cast<Runtime *>(runtime.runtime());
  Device &device = native_runtime.get();
  auto *stream = device.get_compute_stream();
  const std::array<uint32_t, 4> expected{
      0xff132537u, 0xffabcdefu, 0xff649321u, 0xff102040u};
  auto [upload, upload_result] = device.allocate_memory_unique(
      {sizeof(expected), false, false, false, AllocUsage::Storage});
  auto [readback, readback_result] = device.allocate_memory_unique(
      {sizeof(expected), false, false, false, AllocUsage::Storage});
  ASSERT_EQ(upload_result, RhiResult::success);
  ASSERT_EQ(readback_result, RhiResult::success);
  auto source = upload->get_ptr();
  const void *pixels = expected.data();
  size_t bytes = sizeof(expected);
  ASSERT_EQ(device.upload_data(&source, &pixels, &bytes, 1), RhiResult::success);
  BufferImageCopyParams copy;
  copy.buffer_row_length = copy.buffer_image_height = 2;
  copy.image_extent = {2, 2, 1};
  const std::pair<VkImageLayout, ImageLayout> cases[] = {
      {VK_IMAGE_LAYOUT_GENERAL, ImageLayout::shader_read_write},
      {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, ImageLayout::shader_read},
      {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ImageLayout::transfer_src},
      {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ImageLayout::transfer_dst},
  };
  for (const auto &[external_layout, initial] : cases) {
    SCOPED_TRACE(static_cast<int>(external_layout));
    auto owner = runtime.allocate_texture2d(2, 2, TI_FORMAT_RGBA8, TI_NULL_HANDLE);
    ASSERT_TRUE(owner.is_valid());
    runtime.wait();
    ASSERT_TAICHI_SUCCESS();
    const auto image = devimg2devalloc(native_runtime, owner.image());
    auto [seed, seed_result] = stream->new_command_list_unique();
    ASSERT_EQ(seed_result, RhiResult::success);
    seed->buffer_barrier(source);
    seed->image_transition(image, ImageLayout::undefined, ImageLayout::transfer_dst);
    seed->buffer_to_image(image, source, ImageLayout::transfer_dst, copy);
    seed->image_transition(image, ImageLayout::transfer_dst, initial);
    stream->submit_synced(seed.get());
    // The external producer has set the real layout. Do not repair either
    // tracker with ti_track_image_ext; only the imported handle is used below.
    TiVulkanImageInteropInfo info{};
    ti_export_vulkan_image(runtime, owner.image(), &info);
    ASSERT_TAICHI_SUCCESS();
    ImportCalls calls(static_cast<vulkan::VulkanDevice &>(device));
    calls.observed_image = info.image;
    auto handle = ti_import_vulkan_image(runtime, &info, VK_IMAGE_VIEW_TYPE_2D,
                                         external_layout);
    ASSERT_NE(handle, nullptr);
    ti::Image imported(runtime, handle, TI_IMAGE_DIMENSION_2D, {2, 2, 1, 1}, 1,
                       TI_FORMAT_RGBA8, true);
    ASSERT_TAICHI_SUCCESS();
    ti_transition_image(runtime, imported, TI_IMAGE_LAYOUT_TRANSFER_SRC);
    runtime.wait();
    ASSERT_TAICHI_SUCCESS();
    ASSERT_GT(calls.image_barriers, 0u);
    EXPECT_EQ(calls.first_old, external_layout);
    EXPECT_EQ(calls.first_new, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    auto [download, download_result] = stream->new_command_list_unique();
    ASSERT_EQ(download_result, RhiResult::success);
    download->image_to_buffer(readback->get_ptr(),
                              devimg2devalloc(native_runtime, imported),
                              ImageLayout::transfer_src, copy);
    download->buffer_barrier(readback->get_ptr());
    stream->submit_synced(download.get());
    std::array<uint32_t, 4> actual{};
    auto readback_source = readback->get_ptr();
    void *destination = actual.data();
    ASSERT_EQ(device.readback_data(&readback_source, &destination, &bytes),
              RhiResult::success);
    EXPECT_EQ(actual, expected);
    // imported is destroyed before owner; all commands have completed.
  }
}
}  // namespace
#endif
