#include "gtest/gtest.h"

#include <array>
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_device_creator.h"
#include "taichi/rhi/vulkan/vulkan_loader.h"
#include "taichi/runtime/vulkan_buffer_commands.h"

namespace taichi::lang {

TEST(VulkanBufferCommands, CopyValidatesImportedPhysicalAliases) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto *device = creator.device();
  auto [allocation, result] = device->allocate_memory_unique(
      {64, false, false, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  const auto [memory, offset, capacity] =
      device->get_vkmemory_offset_size(*allocation);
  const auto alias = device->import_vkbuffer(device->get_vkbuffer(*allocation),
                                             64, memory, offset);
  EXPECT_NE(alias.alloc_id, allocation->alloc_id);
  EXPECT_THROW(
      validate_vulkan_buffer_copy(allocation->get_ptr(8), alias.get_ptr(0), 16),
      std::invalid_argument);
  EXPECT_NO_THROW(validate_vulkan_buffer_copy(allocation->get_ptr(16),
                                              alias.get_ptr(0), 16));
  const auto unknown = device->import_vkbuffer(
      device->get_vkbuffer(*allocation), 64, VK_NULL_HANDLE, 0);
  EXPECT_THROW(validate_vulkan_buffer_copy(allocation->get_ptr(8),
                                           unknown.get_ptr(0), 16),
               std::invalid_argument);
  EXPECT_NO_THROW(validate_vulkan_buffer_copy(allocation->get_ptr(16),
                                              unknown.get_ptr(0), 16));
  device->dealloc_memory(unknown);
  device->dealloc_memory(alias);
}

TEST(VulkanBufferCommands, OrderedFillCopyAndBarriersPreserveUntouchedBytes) {
  if (!vulkan::is_vulkan_api_available())
    GTEST_SKIP();
  vulkan::VulkanDeviceCreator creator(vulkan::VulkanDeviceCreator::Params{});
  auto *device = creator.device();
  auto [allocation, result] = device->allocate_memory_unique(
      {64, false, true, false, AllocUsage::Storage});
  ASSERT_EQ(result, RhiResult::success);
  using Kind = VulkanBufferCommandKind;
  const auto base = allocation->get_ptr();
  const auto source = resolve_vulkan_buffer_command_range(base, 64, 4, 16);
  const auto destination =
      resolve_vulkan_buffer_command_range(base, 64, 32, 16);
  ASSERT_NO_THROW(validate_vulkan_buffer_copy(destination, source, 16));
  auto *stream = device->get_compute_stream();
  for (std::uint32_t value : {13u, 37u}) {
    const std::vector<PreparedVulkanBufferCommands::Command> commands{
        {Kind::kFillU32, base, kDeviceNullPtr, 64, 99},
        {Kind::kBufferBarrier, base, kDeviceNullPtr, 0, 0},
        {Kind::kFillU32, source, kDeviceNullPtr, 16, value},
        {Kind::kMemoryBarrier, kDeviceNullPtr, kDeviceNullPtr, 0, 0},
        {Kind::kCopy, destination, source, 16, 0},
        {Kind::kBufferBarrier, base, kDeviceNullPtr, 0, 0},
    };
    auto [list, recorded] = stream->new_command_list_unique();
    ASSERT_EQ(recorded, RhiResult::success);
    record_vulkan_buffer_commands(*list, commands);
    ASSERT_TRUE(stream->submit_synced(list.get()));
    void *mapped = nullptr;
    ASSERT_EQ(device->map(*allocation, &mapped), RhiResult::success);
    const auto *words = static_cast<const std::uint32_t *>(mapped);
    for (unsigned i = 0; i < 16; ++i)
      EXPECT_EQ(words[i],
                (i >= 1 && i < 5) || (i >= 8 && i < 12) ? value : 99u);
    device->unmap(*allocation);
  }
}

}  // namespace taichi::lang
