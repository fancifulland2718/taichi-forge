#include "taichi/runtime/vulkan_buffer_commands.h"

#include <limits>
#include <stdexcept>
#include "taichi/runtime/native_array_contract.h"

namespace taichi::lang {

DevicePtr resolve_vulkan_buffer_command_range(DevicePtr pointer,
                                              std::uint64_t window_bytes,
                                              std::uint64_t offset,
                                              std::uint64_t bytes) {
  if (!bytes)
    throw std::invalid_argument(
        "Vulkan buffer fill/copy requires a nonzero byte count");
  if (offset > window_bytes || bytes > window_bytes - offset)
    throw std::invalid_argument("Vulkan buffer command range exceeds " +
                                std::to_string(window_bytes) + " bytes");
  const auto maximum = (std::numeric_limits<decltype(pointer.offset)>::max)();
  if (offset > maximum - pointer.offset ||
      bytes > maximum - pointer.offset - offset)
    throw std::invalid_argument("Vulkan buffer command range overflows");
  pointer.offset += offset;
  if ((pointer.offset & 3u) || (bytes & 3u))
    throw std::invalid_argument(
        "Vulkan buffer command offset and size must be four-byte aligned");
  return pointer;
}

void validate_vulkan_buffer_copy(DevicePtr destination,
                                 DevicePtr source,
                                 std::uint64_t bytes) {
  const auto write =
      resolve_native_storage_range(Arch::vulkan, destination, bytes);
  const auto read = resolve_native_storage_range(Arch::vulkan, source, bytes);
  if (!write.proven_disjoint(read))
    throw std::invalid_argument(
        "Vulkan buffer copy regions overlap or lack physical separation");
}

void record_vulkan_buffer_commands(
    CommandList &command_list,
    const std::vector<PreparedVulkanBufferCommands::Command> &commands) {
  for (const auto &command : commands) {
    switch (command.kind) {
      case VulkanBufferCommandKind::kFillU32:
        command_list.buffer_fill(command.destination, command.bytes,
                                 command.value);
        break;
      case VulkanBufferCommandKind::kCopy:
        command_list.buffer_copy(command.destination, command.source,
                                 command.bytes);
        break;
      case VulkanBufferCommandKind::kBufferBarrier:
        command_list.buffer_barrier(command.destination);
        break;
      case VulkanBufferCommandKind::kMemoryBarrier:
        command_list.memory_barrier();
        break;
      default:
        throw std::invalid_argument("Invalid prepared Vulkan buffer command");
    }
  }
}

}  // namespace taichi::lang
