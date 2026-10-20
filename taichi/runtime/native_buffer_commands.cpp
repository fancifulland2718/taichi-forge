#include "taichi/runtime/vulkan_buffer_commands.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi::lang {
namespace {
using Kind = VulkanBufferCommandKind;
struct BufferCommand {
  Kind kind;
  std::size_t destination{0}, source{0};
  std::uint64_t destination_offset{0}, source_offset{0}, bytes{0};
  std::uint32_t value{0};
};
struct BufferSequence {
  static constexpr const char *kind = "vulkan_buffer_commands";
  static constexpr std::uint32_t version = 1;
  std::vector<BufferCommand> commands;
  std::vector<NativeArrayBinding> buffers;
};

auto decode_sequence(const liong::json::JsonValue &parameters) {
  using namespace operation_json;
  fields(parameters, {"buffer_count", "commands"});
  const auto count = integer(parameters["buffer_count"]);
  const auto &commands = parameters["commands"];
  if (!commands.is_arr() || commands.arr.inner.empty())
    throw std::invalid_argument("Vulkan buffer sequence requires commands");
  BufferSequence d;
  d.commands.reserve(commands.arr.inner.size());
  const auto index = [&](const JsonValue &value) {
    const auto n = integer(value);
    if (n >= count || n > (std::numeric_limits<std::size_t>::max)())
      throw std::invalid_argument(
          "Vulkan buffer command index is out of range");
    return std::size_t(n);
  };
  for (const auto &p : commands.arr.inner) {
    if (!p.is_obj())
      throw std::invalid_argument("Vulkan buffer command must be an object");
    const auto kind = name(p["kind"]);
    BufferCommand command{};
    if (kind == "memory_barrier") {
      fields(p, {"kind"});
      command.kind = Kind::kMemoryBarrier;
    } else if (kind == "buffer_barrier") {
      fields(p, {"kind", "destination"});
      command.kind = Kind::kBufferBarrier;
      command.destination = index(p["destination"]);
    } else if (kind == "fill_u32" || kind == "copy") {
      if (kind == "fill_u32") {
        fields(p,
               {"kind", "destination", "destination_offset", "bytes", "value"});
        command.kind = Kind::kFillU32;
        const auto value = integer(p["value"]);
        if (value > UINT32_MAX)
          throw std::invalid_argument("Vulkan fill value must fit uint32");
        command.value = std::uint32_t(value);
      } else {
        fields(p, {"kind", "destination", "destination_offset", "source",
                   "source_offset", "bytes"});
        command.kind = Kind::kCopy;
        command.source = index(p["source"]);
        command.source_offset =
            optional_integer(p, "source_offset").value_or(0);
      }
      command.destination = index(p["destination"]);
      command.destination_offset =
          optional_integer(p, "destination_offset").value_or(0);
      command.bytes = integer(p["bytes"]);
      if (!command.bytes || (command.bytes & 3u) ||
          (command.destination_offset & 3u) || (command.source_offset & 3u) ||
          command.bytes > (std::numeric_limits<std::size_t>::max)())
        throw std::invalid_argument(
            "Vulkan transfer ranges must be nonempty and four-byte aligned");
    } else {
      throw std::invalid_argument("Unknown Vulkan buffer command kind");
    }
    d.commands.push_back(command);
  }
  // The registry bounds this count by the actual port inventory before decode.
  return [d = std::move(d),
          count](const NativeOperationBindings &bindings) mutable {
    auto result = d;
    result.buffers.reserve(std::size_t(count));
    for (std::uint64_t i = 0; i < count; ++i)
      result.buffers.push_back(
          bindings.at("buffer_" + std::to_string(i)).array());
    return result;
  };
}

void validate_sequence(Device &device, const BufferSequence &d) {
  for (const auto &buffer : d.buffers)
    if (buffer.pointer.device != &device || buffer.strides)
      throw std::invalid_argument(
          "Vulkan buffer commands require local packed storage");
}

#ifdef TI_WITH_VULKAN
class VulkanBufferSequence : public NativeOperation {
 public:
  VulkanBufferSequence(Device &device, const BufferSequence &d)
      : device_(dynamic_cast<vulkan::VulkanDevice &>(device)) {
    std::vector<NativeStorageRange> ranges;
    std::vector<VkBufferUsageFlags> usages;
    ranges.reserve(d.buffers.size());
    usages.reserve(d.buffers.size());
    buffers_.reserve(d.buffers.size());
    for (const auto &buffer : d.buffers) {
      ranges.push_back(resolve_native_storage_range(
          Arch::vulkan, buffer.pointer, buffer.bytes));
      usages.push_back(device_.get_vkbuffer(buffer.pointer)->usage);
      buffers_.push_back(buffer.pointer);
    }
    const auto transfer = [&](std::size_t index, std::uint64_t offset,
                              std::uint64_t bytes, VkBufferUsageFlags usage) {
      if (!(usages.at(index) & usage))
        throw std::invalid_argument(
            "Vulkan buffer command requires transfer usage");
      const auto &buffer = d.buffers.at(index);
      return resolve_vulkan_buffer_command_range(buffer.pointer, buffer.bytes,
                                                 offset, bytes);
    };
    const auto subrange = [&](std::size_t index, std::uint64_t offset,
                              std::uint64_t bytes) {
      auto result = ranges.at(index);
      result.memory_offset += offset;
      result.buffer_offset += offset;
      result.bytes = bytes;
      return result;
    };
    commands_.reserve(d.commands.size());
    for (const auto &c : d.commands) {
      PreparedVulkanBufferCommands::Command command{
          c.kind, kDeviceNullPtr, kDeviceNullPtr, std::size_t(c.bytes),
          c.value};
      if (c.kind == Kind::kFillU32 || c.kind == Kind::kCopy) {
        command.destination =
            transfer(c.destination, c.destination_offset, c.bytes,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (c.kind == Kind::kCopy) {
          command.source = transfer(c.source, c.source_offset, c.bytes,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
          if (!subrange(c.destination, c.destination_offset, c.bytes)
                   .proven_disjoint(
                       subrange(c.source, c.source_offset, c.bytes)))
            throw std::invalid_argument(
                "Vulkan buffer copy regions overlap or lack physical "
                "separation");
        }
      } else if (c.kind == Kind::kBufferBarrier) {
        command.destination = d.buffers.at(c.destination).pointer;
      }
      commands_.push_back(command);
    }
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  NativeVulkanRecordingRequirements vulkan_recording_requirements()
      const override {
    NativeVulkanRecordingRequirements result;
    result.buffers = buffers_;
    return result;
  }
  void record_vulkan(Device *, CommandList *commands) override {
    record_vulkan_buffer_commands(*commands, commands_);
  }

 private:
  vulkan::VulkanDevice &device_;
  std::vector<DeviceAllocation> buffers_;
  std::vector<PreparedVulkanBufferCommands::Command> commands_;
};
#endif
}  // namespace

void register_vulkan_buffer_operations(NativeOperationRegistry &registry) {
  registry.define<BufferSequence>(validate_sequence);
  registry.register_schema<BufferSequence>("vulkan_rhi", "buffer_commands", {},
                                           decode_sequence, {}, {},
                                           {{"buffer_count", {"buffer"}}});
#ifdef TI_WITH_VULKAN
  registry.register_factory<BufferSequence>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const BufferSequence &d) {
        return std::make_shared<VulkanBufferSequence>(device, d);
      });
#endif
}
}  // namespace taichi::lang
