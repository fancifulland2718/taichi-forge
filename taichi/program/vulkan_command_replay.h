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
  struct Entry {
    VulkanCommandReplayKey key;
    std::unique_ptr<CommandList> cmdlist;
  };

  Device *device{nullptr};
  // Last-key replay avoids reusing command buffers whose descriptor sets may
  // have been rebound for a newer key.
  Entry entry;

  void reset() {
    entry.cmdlist = nullptr;
    device = nullptr;
  }

  template <typename RecordFn>
  bool submit_or_record(Program *program,
                        Device *dev,
                        const VulkanCommandReplayKey &new_key,
                        bool profiler_scopes,
                        RecordFn &&record) {
    static const bool replay_enabled = vulkan_native_command_replay_enabled();
    if (!replay_enabled || profiler_scopes) {
      reset();
      return false;
    }
    if (device != dev) {
      reset();
      device = dev;
    }
    if (program->has_pending_gfx_command_list()) {
      reset();
      return false;
    }

    Stream *stream = dev->get_compute_stream();
    static const std::vector<StreamSemaphore> kNoWaits;
    if (entry.cmdlist && entry.key == new_key) {
      stream->submit(entry.cmdlist.get(), kNoWaits);
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
    return true;
  }
};

}  // namespace taichi::lang
