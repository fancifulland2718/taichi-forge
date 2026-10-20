#include "gtest/gtest.h"

#include <limits>
#include "taichi/runtime/vulkan_buffer_commands.h"

namespace taichi::lang {

TEST(VulkanBufferCommands, FiniteWindowsRejectOverflowBeforeAlignment) {
  auto pointer = kDeviceNullPtr;
  pointer.offset = 12;
  EXPECT_EQ(resolve_vulkan_buffer_command_range(pointer, 20, 4, 16).offset,
            16u);
  EXPECT_THROW(resolve_vulkan_buffer_command_range(pointer, 20, 8, 16),
               std::invalid_argument);
  EXPECT_THROW(resolve_vulkan_buffer_command_range(pointer, 20, 0, 0),
               std::invalid_argument);
  EXPECT_THROW(resolve_vulkan_buffer_command_range(pointer, 20, 1, 4),
               std::invalid_argument);
  EXPECT_THROW(resolve_vulkan_buffer_command_range(pointer, 20, 0, 3),
               std::invalid_argument);
  const auto maximum = (std::numeric_limits<std::uint64_t>::max)();
  pointer.offset = maximum - 3;
  EXPECT_THROW(resolve_vulkan_buffer_command_range(pointer, 16, 4, 4),
               std::invalid_argument);
  EXPECT_THROW(resolve_vulkan_buffer_command_range(pointer, 16, 0, 4),
               std::invalid_argument);
}

}  // namespace taichi::lang
