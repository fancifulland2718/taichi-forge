#include "gtest/gtest.h"

#include <limits>
#include "taichi/runtime/vulkan_buffer_commands.h"
#include "taichi/runtime/native_operation_registry.h"

namespace taichi::lang {

TEST(VulkanBufferCommands, SchemaChecksPortsAndCommandSpecificFields) {
  NativeOperationDescription d;
  d.provider = "vulkan_rhi";
  d.operation = "buffer_commands";
  d.bindings = {{"buffer_0", "input"}, {"buffer_1", "output"}};
  const auto &registry = native_operation_registry();
  d.parameters = liong::json::parse_complete(
      R"({"buffer_count":2,"commands":[{"kind":"copy","destination":1,"source":0,"bytes":16},{"kind":"buffer_barrier","destination":1},{"kind":"memory_barrier"}]})");
  EXPECT_EQ(registry.decode(d).kind(), "vulkan_buffer_commands");
  for (
      const auto *parameters :
      {R"({"buffer_count":3,"commands":[{"kind":"memory_barrier"}]})",
       R"({"buffer_count":2,"commands":[]})",
       R"({"buffer_count":2,"commands":[{"kind":"copy","destination":2,"source":0,"bytes":16}]})",
       R"({"buffer_count":2,"commands":[{"kind":"copy","destination":1,"source":0,"bytes":0}]})",
       R"({"buffer_count":2,"commands":[{"kind":"copy","destination":1,"source":0,"bytes":6}]})",
       R"({"buffer_count":2,"commands":[{"kind":"copy","destination":1,"source":0,"bytes":16,"value":0}]})",
       R"({"buffer_count":2,"commands":[{"kind":"fill_u32","destination":1,"bytes":16,"value":4294967296}]})",
       R"({"buffer_count":2,"commands":[{"kind":"buffer_barrier","destination":0,"bytes":4}]})",
       R"({"buffer_count":2,"commands":[{"kind":"memory_barrier","destination":0}]})",
       R"({"buffer_count":2,"commands":[{"kind":"fill_u32","destination":true,"bytes":16,"value":0}]})"}) {
    d.parameters = liong::json::parse_complete(parameters);
    EXPECT_THROW(registry.decode(d), std::invalid_argument);
  }
  d.bindings.clear();
  d.parameters = liong::json::parse_complete(
      R"({"buffer_count":0,"commands":[{"kind":"memory_barrier"}]})");
  EXPECT_NO_THROW(registry.decode(d));
  auto &commands = d.parameters["commands"].arr.inner;
  commands.resize(4097, commands.front());
  EXPECT_NO_THROW(registry.decode(d));
}

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
