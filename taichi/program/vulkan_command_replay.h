#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "taichi/program/program.h"
#include "taichi/rhi/device.h"
#include "taichi/util/environ_config.h"

namespace taichi::lang {

struct VulkanNativeCommandRecordingContext {
  Program *program{nullptr};
  Device *device{nullptr};
  CommandList *cmdlist{nullptr};
};

inline thread_local VulkanNativeCommandRecordingContext
    *vulkan_native_command_recording_context = nullptr;

class VulkanNativeCommandRecordingScope {
 public:
  VulkanNativeCommandRecordingScope(Program *program,
                                    Device *device,
                                    CommandList *cmdlist)
      : context_{program, device, cmdlist},
        previous_(vulkan_native_command_recording_context) {
    TI_ERROR_IF(previous_ != nullptr,
                "Nested Vulkan native command recording is unsupported.");
    vulkan_native_command_recording_context = &context_;
  }

  ~VulkanNativeCommandRecordingScope() {
    TI_ASSERT(vulkan_native_command_recording_context == &context_);
    vulkan_native_command_recording_context = previous_;
  }

  VulkanNativeCommandRecordingScope(
      const VulkanNativeCommandRecordingScope &) = delete;
  VulkanNativeCommandRecordingScope &operator=(
      const VulkanNativeCommandRecordingScope &) = delete;

 private:
  VulkanNativeCommandRecordingContext context_;
  VulkanNativeCommandRecordingContext *previous_{nullptr};
};

template <typename RecordFn>
bool try_record_vulkan_native_command(Program *program, RecordFn &&record) {
  auto *context = vulkan_native_command_recording_context;
  if (context == nullptr || context->program != program) {
    return false;
  }
  record(context->device, context->cmdlist);
  return true;
}

inline bool vulkan_native_command_replay_enabled() {
  return get_environ_config("TI_VULKAN_NATIVE_COMMAND_REPLAY", 1) != 0;
}

struct VulkanCommandReplayKey {
  std::array<uint64_t, 128> words{};
  uint32_t size{0};

  void push(uint64_t word) {
    TI_ASSERT_INFO(size < words.size(),
                   "Vulkan native command replay key is too small.");
    words[size++] = word;
  }

  void push_ptr(const void *ptr) {
    push(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
  }

  bool operator==(const VulkanCommandReplayKey &other) const {
    if (size != other.size) {
      return false;
    }
    for (uint32_t i = 0; i < size; ++i) {
      if (words[i] != other.words[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const VulkanCommandReplayKey &other) const {
    return !(*this == other);
  }
};

struct VulkanCommandReplayCache {
  enum class LastPath : uint8_t {
    none,
    fallback,
    record,
    replay,
    nested_record,
  };

  struct Entry {
    VulkanCommandReplayKey key;
    std::unique_ptr<CommandList> cmdlist;
  };

  Device *device{nullptr};
  // Last-key replay avoids reusing command buffers whose descriptor sets may
  // have been rebound for a newer key.
  Entry entry;
  LastPath last_path{LastPath::none};
  uint64_t records{0};
  uint64_t replays{0};

  void reset() {
    entry.cmdlist = nullptr;
    device = nullptr;
    last_path = LastPath::none;
  }

  template <typename RecordFn>
  bool submit_or_record(Program *program,
                        Device *dev,
                        const VulkanCommandReplayKey &new_key,
                        bool profiler_scopes,
                        RecordFn &&record) {
    if (try_record_vulkan_native_command(program,
                                         std::forward<RecordFn>(record))) {
      last_path = LastPath::nested_record;
      return true;
    }
    static const bool replay_enabled = vulkan_native_command_replay_enabled();
    if (!replay_enabled || profiler_scopes) {
      reset();
      last_path = LastPath::fallback;
      return false;
    }
    if (device != dev) {
      reset();
      device = dev;
    }
    if (program->has_pending_gfx_command_list()) {
      reset();
      last_path = LastPath::fallback;
      return false;
    }

    Stream *stream = dev->get_compute_stream();
    static const std::vector<StreamSemaphore> kNoWaits;
    if (entry.cmdlist && entry.key == new_key) {
      stream->submit(entry.cmdlist.get(), kNoWaits);
      last_path = LastPath::replay;
      ++replays;
      return true;
    }

    auto [recorded_cmdlist, res] = stream->new_command_list_unique();
    TI_ERROR_IF(res != RhiResult::success,
                "Vulkan native command replay could not allocate a command "
                "list: RhiResult({})",
                res);
    record(dev, recorded_cmdlist.get());
    entry.key = new_key;
    entry.cmdlist = std::move(recorded_cmdlist);
    stream->submit(entry.cmdlist.get(), kNoWaits);
    last_path = LastPath::record;
    ++records;
    return true;
  }
};

}  // namespace taichi::lang
