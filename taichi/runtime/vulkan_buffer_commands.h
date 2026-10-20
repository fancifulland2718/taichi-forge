#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "taichi/rhi/device.h"

namespace taichi::lang {

class PreparedNativeStorage;

enum class VulkanBufferCommandKind : std::uint8_t {
  kFillU32,
  kCopy,
  kBufferBarrier,
  kMemoryBarrier,
};

struct PreparedVulkanBufferCommands {
  struct Command {
    VulkanBufferCommandKind kind;
    DevicePtr destination;
    DevicePtr source;
    std::size_t bytes;
    std::uint32_t value;
  };
  std::shared_ptr<PreparedNativeStorage> storage;
  std::shared_ptr<const std::vector<Command>> commands;
};

// Cold bounds/alignment validation of a borrowed finite window. The adapter
// separately proves physical storage and retains its owner through completion.
DevicePtr resolve_vulkan_buffer_command_range(DevicePtr pointer,
                                              std::uint64_t window_bytes,
                                              std::uint64_t offset,
                                              std::uint64_t bytes);

// Distinct allocation wrappers can refer to the same Vulkan memory or buffer.
// Only one copy's read/write ranges must be disjoint; consecutive commands may
// intentionally access overlapping ranges with explicit barriers between them.
void validate_vulkan_buffer_copy(DevicePtr destination,
                                 DevicePtr source,
                                 std::uint64_t bytes);

// No ownership, submission, or implicit barriers are added here. Both adapters
// record the same prepared sequence into their existing enclosing command list.
void record_vulkan_buffer_commands(
    CommandList &command_list,
    const std::vector<PreparedVulkanBufferCommands::Command> &commands);

}  // namespace taichi::lang
