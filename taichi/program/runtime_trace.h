#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

#include "taichi/common/core.h"
#include "taichi/program/runtime_statistics.h"

namespace taichi::lang {

enum class RuntimeTraceEventKind : std::uint16_t {
  kKernelSubmission,
  kGraphSubmission,
  kGraphBackendSubmission,
  kNativeSubmission,
  kProgramSynchronize,
  kHostToDeviceTransfer,
  kDeviceToHostTransfer,
  kDeviceToDeviceTransfer,
  kCudaVulkanDirectTransfer,
  kCudaVulkanFallbackTransfer,
};

struct RuntimeTraceSnapshot {
  std::uint64_t program_domain{0};
  std::uint64_t session{0};
  bool enabled{false};
  std::uint64_t max_threads{0};
  std::uint64_t events_per_thread{0};
  std::uint64_t event_capacity{0};
  std::uint64_t allocated_bytes{0};
  std::uint64_t recorded_events{0};
  std::uint64_t dropped_events{0};
};

// Program-owned, opt-in runtime trace recorder. The disabled path is one
// predictable atomic load. Enabling allocates a fixed number of thread shards
// and fixed-size POD event slots; the hot path never grows either allocation.
class TI_DLL_EXPORT RuntimeTraceRecorder final {
 public:
  static constexpr std::size_t kDefaultMaxThreads = 16;
  static constexpr std::size_t kDefaultEventsPerThread = 4096;
  static constexpr std::size_t kMaximumThreads = 64;
  static constexpr std::size_t kMaximumTotalEvents = 1u << 20;

  class Scope final {
   public:
    Scope(RuntimeTraceRecorder *recorder,
          RuntimeTraceEventKind kind,
          std::uint64_t value = 0) noexcept;
    ~Scope();

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    void mark_failed() noexcept {
      failed_ = true;
    }

   private:
    RuntimeTraceRecorder *recorder_{nullptr};
    void *slot_{nullptr};
    std::uint64_t start_ns_{0};
    std::uint64_t value_{0};
    RuntimeTraceEventKind kind_{RuntimeTraceEventKind::kKernelSubmission};
    std::uint32_t thread_index_{0};
    bool failed_{false};
  };

  RuntimeTraceRecorder(RuntimeStatistics &statistics,
                       std::uint64_t program_domain) noexcept;
  ~RuntimeTraceRecorder();

  RuntimeTraceRecorder(const RuntimeTraceRecorder &) = delete;
  RuntimeTraceRecorder &operator=(const RuntimeTraceRecorder &) = delete;

  RuntimeTraceSnapshot start(
      std::size_t max_threads = kDefaultMaxThreads,
      std::size_t events_per_thread = kDefaultEventsPerThread);
  RuntimeTraceSnapshot stop();
  RuntimeTraceSnapshot snapshot() const;
  bool export_chrome_trace(const std::string &path) const;

  TI_FORCE_INLINE bool enabled() const noexcept {
    return enabled_.load(std::memory_order_relaxed);
  }

  void record_instant(RuntimeTraceEventKind kind,
                      std::uint64_t value = 0) noexcept;

 private:
  struct RuntimeTraceEvent;
  struct ThreadBuffer;

  void *reserve(std::uint32_t &thread_index) noexcept;
  void finish(void *slot,
              std::uint32_t thread_index,
              RuntimeTraceEventKind kind,
              std::uint64_t start_ns,
              std::uint64_t duration_ns,
              std::uint64_t value,
              bool failed) noexcept;
  RuntimeTraceSnapshot snapshot_locked() const noexcept;
  static std::uint64_t timestamp_ns() noexcept;

  RuntimeStatistics &statistics_;
  const std::uint64_t program_domain_;
  mutable std::mutex control_mutex_;
  mutable std::shared_mutex session_gate_;
  std::unique_ptr<ThreadBuffer[]> thread_buffers_;
  std::unique_ptr<RuntimeTraceEvent[]> events_;
  std::size_t max_threads_{0};
  std::size_t events_per_thread_{0};
  std::size_t event_capacity_{0};
  std::uint64_t allocated_bytes_{0};
  std::uint64_t session_start_ns_{0};
  std::atomic<std::uint64_t> session_{0};
  std::atomic<std::uint64_t> recorded_events_{0};
  std::atomic<std::uint64_t> dropped_events_{0};
  std::atomic<std::uint64_t> active_scopes_{0};
  std::atomic<bool> enabled_{false};
};

RuntimeTraceEventKind runtime_trace_kind(
    RuntimeSubmissionKind kind) noexcept;
RuntimeTraceEventKind runtime_trace_kind(RuntimeTransferKind kind) noexcept;

}  // namespace taichi::lang
