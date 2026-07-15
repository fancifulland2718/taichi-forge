#include "taichi/program/runtime_trace.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <thread>
#include <vector>

namespace taichi::lang {

namespace {

std::atomic<std::uint64_t> next_runtime_trace_thread_token{1};

std::uint64_t runtime_trace_thread_token() noexcept {
  thread_local const std::uint64_t token = [] {
    const std::uint64_t result = next_runtime_trace_thread_token.fetch_add(
        1, std::memory_order_relaxed);
    TI_ASSERT(result != 0 &&
              result != (std::numeric_limits<std::uint64_t>::max)());
    return result;
  }();
  return token;
}

const char *runtime_trace_event_name(RuntimeTraceEventKind kind) noexcept {
  switch (kind) {
    case RuntimeTraceEventKind::kKernelSubmission:
      return "runtime.kernel.submit";
    case RuntimeTraceEventKind::kGraphSubmission:
      return "runtime.graph.submit";
    case RuntimeTraceEventKind::kGraphBackendSubmission:
      return "runtime.graph.backend_submit";
    case RuntimeTraceEventKind::kNativeSubmission:
      return "runtime.native.submit";
    case RuntimeTraceEventKind::kProgramSynchronize:
      return "runtime.synchronize";
    case RuntimeTraceEventKind::kHostToDeviceTransfer:
      return "runtime.transfer.h2d";
    case RuntimeTraceEventKind::kDeviceToHostTransfer:
      return "runtime.transfer.d2h";
    case RuntimeTraceEventKind::kDeviceToDeviceTransfer:
      return "runtime.transfer.d2d";
    case RuntimeTraceEventKind::kCudaVulkanDirectTransfer:
      return "runtime.transfer.cuda_vulkan_direct";
    case RuntimeTraceEventKind::kCudaVulkanFallbackTransfer:
      return "runtime.transfer.cuda_vulkan_fallback";
  }
  return "runtime.unknown";
}

const char *runtime_trace_event_category(RuntimeTraceEventKind kind) noexcept {
  switch (kind) {
    case RuntimeTraceEventKind::kKernelSubmission:
    case RuntimeTraceEventKind::kGraphSubmission:
    case RuntimeTraceEventKind::kGraphBackendSubmission:
    case RuntimeTraceEventKind::kNativeSubmission:
      return "taichi.runtime.submission";
    case RuntimeTraceEventKind::kProgramSynchronize:
      return "taichi.runtime.synchronization";
    case RuntimeTraceEventKind::kHostToDeviceTransfer:
    case RuntimeTraceEventKind::kDeviceToHostTransfer:
    case RuntimeTraceEventKind::kDeviceToDeviceTransfer:
    case RuntimeTraceEventKind::kCudaVulkanDirectTransfer:
    case RuntimeTraceEventKind::kCudaVulkanFallbackTransfer:
      return "taichi.runtime.transfer";
  }
  return "taichi.runtime";
}

bool runtime_trace_event_has_bytes(RuntimeTraceEventKind kind) noexcept {
  switch (kind) {
    case RuntimeTraceEventKind::kHostToDeviceTransfer:
    case RuntimeTraceEventKind::kDeviceToHostTransfer:
    case RuntimeTraceEventKind::kDeviceToDeviceTransfer:
    case RuntimeTraceEventKind::kCudaVulkanDirectTransfer:
    case RuntimeTraceEventKind::kCudaVulkanFallbackTransfer:
      return true;
    default:
      return false;
  }
}

}  // namespace

struct RuntimeTraceRecorder::RuntimeTraceEvent {
  std::uint64_t start_ns;
  std::uint64_t duration_ns;
  std::uint64_t value;
  std::uint32_t thread_index;
  RuntimeTraceEventKind kind;
  std::uint8_t failed;
  std::uint8_t reserved;
};

struct RuntimeTraceRecorder::ThreadBuffer {
  std::atomic<std::uint64_t> owner_token{0};
  std::uint64_t next_index{0};
};

RuntimeTraceRecorder::RuntimeTraceRecorder(RuntimeStatistics &statistics,
                                           std::uint64_t program_domain) noexcept
    : statistics_(statistics), program_domain_(program_domain) {
}

RuntimeTraceRecorder::~RuntimeTraceRecorder() {
  stop();
}

std::uint64_t RuntimeTraceRecorder::timestamp_ns() noexcept {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

RuntimeTraceSnapshot RuntimeTraceRecorder::start(
    std::size_t max_threads,
    std::size_t events_per_thread) {
  static_assert(sizeof(RuntimeTraceEvent) <= 32);
  static_assert(sizeof(ThreadBuffer) <= 32);
  TI_ERROR_IF(max_threads == 0 || max_threads > kMaximumThreads,
              "Runtime trace max_threads must be in [1, {}], got {}",
              kMaximumThreads, max_threads);
  TI_ERROR_IF(events_per_thread == 0 ||
                  events_per_thread > kMaximumTotalEvents,
              "Runtime trace events_per_thread must be in [1, {}], got {}",
              kMaximumTotalEvents, events_per_thread);
  TI_ERROR_IF(events_per_thread > kMaximumTotalEvents / max_threads,
              "Runtime trace capacity exceeds the {} event safety bound",
              kMaximumTotalEvents);

  const std::size_t event_capacity = max_threads * events_per_thread;
  auto thread_buffers = std::make_unique<ThreadBuffer[]>(max_threads);
  auto events = std::unique_ptr<RuntimeTraceEvent[]>(
      new RuntimeTraceEvent[event_capacity]);

  std::lock_guard<std::mutex> control_lock(control_mutex_);
  TI_ERROR_IF(enabled_.load(std::memory_order_acquire),
              "Runtime trace is already enabled");
  std::unique_lock<std::shared_mutex> session_lock(session_gate_);
  TI_ERROR_IF(active_scopes_.load(std::memory_order_acquire) != 0,
              "Runtime trace still has active event scopes");
  const std::uint64_t current_session =
      session_.load(std::memory_order_relaxed);
  TI_ERROR_IF(current_session ==
                  (std::numeric_limits<std::uint64_t>::max)(),
              "Runtime trace session counter exhausted");

  thread_buffers_ = std::move(thread_buffers);
  events_ = std::move(events);
  max_threads_ = max_threads;
  events_per_thread_ = events_per_thread;
  event_capacity_ = event_capacity;
  allocated_bytes_ =
      static_cast<std::uint64_t>(event_capacity * sizeof(RuntimeTraceEvent) +
                                 max_threads * sizeof(ThreadBuffer));
  session_start_ns_ = timestamp_ns();
  recorded_events_.store(0, std::memory_order_relaxed);
  dropped_events_.store(0, std::memory_order_relaxed);
  session_.store(current_session + 1, std::memory_order_release);
  enabled_.store(true, std::memory_order_release);
  return snapshot_locked();
}

RuntimeTraceSnapshot RuntimeTraceRecorder::stop() {
  std::lock_guard<std::mutex> control_lock(control_mutex_);
  enabled_.store(false, std::memory_order_release);
  {
    // Pair with the shared gate in reserve(). Once acquired, no event can be
    // between its enabled/session checks and active-scope publication.
    std::unique_lock<std::shared_mutex> session_lock(session_gate_);
  }
  while (active_scopes_.load(std::memory_order_acquire) != 0) {
    std::this_thread::yield();
  }
  return snapshot_locked();
}

RuntimeTraceSnapshot RuntimeTraceRecorder::snapshot_locked() const noexcept {
  RuntimeTraceSnapshot result;
  result.program_domain = program_domain_;
  result.session = session_.load(std::memory_order_acquire);
  result.enabled = enabled_.load(std::memory_order_acquire);
  result.max_threads = max_threads_;
  result.events_per_thread = events_per_thread_;
  result.event_capacity = event_capacity_;
  result.allocated_bytes = allocated_bytes_;
  result.recorded_events = recorded_events_.load(std::memory_order_relaxed);
  result.dropped_events = dropped_events_.load(std::memory_order_relaxed);
  return result;
}

RuntimeTraceSnapshot RuntimeTraceRecorder::snapshot() const {
  std::lock_guard<std::mutex> control_lock(control_mutex_);
  return snapshot_locked();
}

void *RuntimeTraceRecorder::reserve(std::uint32_t &thread_index) noexcept {
  if (!enabled_.load(std::memory_order_acquire)) {
    return nullptr;
  }
  const std::uint64_t expected_session =
      session_.load(std::memory_order_acquire);

  std::shared_lock<std::shared_mutex> session_lock(session_gate_);
  if (!enabled_.load(std::memory_order_relaxed) ||
      session_.load(std::memory_order_relaxed) != expected_session) {
    return nullptr;
  }

  struct ThreadCache {
    const RuntimeTraceRecorder *recorder{nullptr};
    std::uint64_t program_domain{0};
    std::uint64_t session{0};
    std::uint64_t token{0};
    std::size_t index{0};
  };
  thread_local ThreadCache cache;

  const std::uint64_t token = runtime_trace_thread_token();
  std::size_t buffer_index = max_threads_;
  if (cache.recorder == this && cache.program_domain == program_domain_ &&
      cache.session == expected_session && cache.token == token &&
      cache.index < max_threads_ &&
      thread_buffers_[cache.index].owner_token.load(
          std::memory_order_relaxed) == token) {
    buffer_index = cache.index;
  } else {
    for (std::size_t i = 0; i < max_threads_; ++i) {
      if (thread_buffers_[i].owner_token.load(std::memory_order_relaxed) ==
          token) {
        buffer_index = i;
        break;
      }
    }
    if (buffer_index == max_threads_) {
      for (std::size_t i = 0; i < max_threads_; ++i) {
        std::uint64_t unclaimed = 0;
        if (thread_buffers_[i].owner_token.compare_exchange_strong(
                unclaimed, token, std::memory_order_relaxed,
                std::memory_order_relaxed)) {
          buffer_index = i;
          break;
        }
      }
    }
    if (buffer_index != max_threads_) {
      cache = {this, program_domain_, expected_session, token, buffer_index};
    }
  }

  if (buffer_index == max_threads_) {
    dropped_events_.fetch_add(1, std::memory_order_relaxed);
    statistics_.record_trace_events(0, 1);
    return nullptr;
  }

  ThreadBuffer &buffer = thread_buffers_[buffer_index];
  const std::uint64_t slot_index = buffer.next_index++;
  if (slot_index >= events_per_thread_) {
    dropped_events_.fetch_add(1, std::memory_order_relaxed);
    statistics_.record_trace_events(0, 1);
    return nullptr;
  }

  active_scopes_.fetch_add(1, std::memory_order_release);
  thread_index = static_cast<std::uint32_t>(buffer_index);
  return &events_[buffer_index * events_per_thread_ + slot_index];
}

void RuntimeTraceRecorder::finish(void *slot,
                                  std::uint32_t thread_index,
                                  RuntimeTraceEventKind kind,
                                  std::uint64_t start_ns,
                                  std::uint64_t duration_ns,
                                  std::uint64_t value,
                                  bool failed) noexcept {
  auto *event = static_cast<RuntimeTraceEvent *>(slot);
  *event = {start_ns, duration_ns, value, thread_index, kind,
            static_cast<std::uint8_t>(failed), 0};
  recorded_events_.fetch_add(1, std::memory_order_relaxed);
  statistics_.record_trace_events(1, 0);
  const std::uint64_t previous =
      active_scopes_.fetch_sub(1, std::memory_order_release);
  TI_ASSERT(previous != 0);
}

RuntimeTraceRecorder::Scope::Scope(RuntimeTraceRecorder *recorder,
                                   RuntimeTraceEventKind kind,
                                   std::uint64_t value) noexcept
    : recorder_(recorder), value_(value), kind_(kind) {
  if (recorder_ != nullptr) {
    slot_ = recorder_->reserve(thread_index_);
    if (slot_ != nullptr) {
      start_ns_ = RuntimeTraceRecorder::timestamp_ns();
    }
  }
}

RuntimeTraceRecorder::Scope::~Scope() {
  if (slot_ == nullptr) {
    return;
  }
  const std::uint64_t end_ns = RuntimeTraceRecorder::timestamp_ns();
  recorder_->finish(slot_, thread_index_, kind_, start_ns_,
                    end_ns >= start_ns_ ? end_ns - start_ns_ : 0, value_,
                    failed_);
}

void RuntimeTraceRecorder::record_instant(RuntimeTraceEventKind kind,
                                          std::uint64_t value) noexcept {
  std::uint32_t thread_index = 0;
  void *slot = reserve(thread_index);
  if (slot == nullptr) {
    return;
  }
  finish(slot, thread_index, kind, timestamp_ns(), 0, value, false);
}

bool RuntimeTraceRecorder::export_chrome_trace(
    const std::string &path) const {
  std::lock_guard<std::mutex> control_lock(control_mutex_);
  TI_ERROR_IF(enabled_.load(std::memory_order_acquire),
              "Stop the runtime trace before exporting it");
  TI_ERROR_IF(active_scopes_.load(std::memory_order_acquire) != 0,
              "Runtime trace still has active event scopes");

  std::vector<RuntimeTraceEvent> merged;
  merged.reserve(static_cast<std::size_t>(
      recorded_events_.load(std::memory_order_relaxed)));
  for (std::size_t thread_index = 0; thread_index < max_threads_;
       ++thread_index) {
    const std::size_t count = static_cast<std::size_t>(std::min<std::uint64_t>(
        thread_buffers_[thread_index].next_index, events_per_thread_));
    const RuntimeTraceEvent *begin =
        events_.get() + thread_index * events_per_thread_;
    merged.insert(merged.end(), begin, begin + count);
  }
  std::sort(merged.begin(), merged.end(),
            [](const RuntimeTraceEvent &lhs, const RuntimeTraceEvent &rhs) {
              if (lhs.start_ns != rhs.start_ns) {
                return lhs.start_ns < rhs.start_ns;
              }
              return lhs.thread_index < rhs.thread_index;
            });

  std::ofstream output(path);
  if (!output) {
    return false;
  }
  output << R"JSON({
  "traceEvents": [
)JSON";
  output << std::fixed << std::setprecision(3);
  bool first = true;
  for (const RuntimeTraceEvent &event : merged) {
    if (!first) {
      output << ",\n";
    }
    first = false;
    const std::uint64_t relative_ns =
        event.start_ns >= session_start_ns_
            ? event.start_ns - session_start_ns_
            : 0;
    output << R"JSON(    {"name":")JSON"
           << runtime_trace_event_name(event.kind) << R"JSON(","cat":")JSON"
           << runtime_trace_event_category(event.kind)
           << R"JSON(","ph":"X","pid":1,"tid":)JSON"
           << event.thread_index + 1;
    output << R"JSON(,"ts":)JSON"
           << static_cast<double>(relative_ns) / 1000.0
           << R"JSON(,"dur":)JSON"
           << static_cast<double>(event.duration_ns) / 1000.0
           << R"JSON(,"args":{)JSON";
    bool has_arg = false;
    if (runtime_trace_event_has_bytes(event.kind)) {
      output << R"JSON("bytes":)JSON" << event.value;
      has_arg = true;
    }
    if (event.failed != 0) {
      if (has_arg) {
        output << ',';
      }
      output << R"JSON("failed":true)JSON";
    }
    output << "}}";
  }
  output << R"JSON(
  ],
  "displayTimeUnit": "ns",
  "taichiRuntimeTrace": {"schemaVersion": 1, "programDomain": )JSON"
         << program_domain_ << R"JSON(, "session": )JSON"
         << session_.load(std::memory_order_relaxed)
         << R"JSON(, "recordedEvents": )JSON"
         << recorded_events_.load(std::memory_order_relaxed)
         << R"JSON(, "droppedEvents": )JSON"
         << dropped_events_.load(std::memory_order_relaxed) << "}\n}\n";
  return static_cast<bool>(output);
}

RuntimeTraceEventKind runtime_trace_kind(RuntimeSubmissionKind kind) noexcept {
  switch (kind) {
    case RuntimeSubmissionKind::kKernel:
      return RuntimeTraceEventKind::kKernelSubmission;
    case RuntimeSubmissionKind::kGraph:
      return RuntimeTraceEventKind::kGraphSubmission;
    case RuntimeSubmissionKind::kGraphBackendSubmission:
      return RuntimeTraceEventKind::kGraphBackendSubmission;
    case RuntimeSubmissionKind::kNative:
      return RuntimeTraceEventKind::kNativeSubmission;
  }
  return RuntimeTraceEventKind::kKernelSubmission;
}

RuntimeTraceEventKind runtime_trace_kind(RuntimeTransferKind kind) noexcept {
  switch (kind) {
    case RuntimeTransferKind::kHostToDevice:
      return RuntimeTraceEventKind::kHostToDeviceTransfer;
    case RuntimeTransferKind::kDeviceToHost:
      return RuntimeTraceEventKind::kDeviceToHostTransfer;
    case RuntimeTransferKind::kDeviceToDevice:
      return RuntimeTraceEventKind::kDeviceToDeviceTransfer;
    case RuntimeTransferKind::kCudaVulkanDirect:
      return RuntimeTraceEventKind::kCudaVulkanDirectTransfer;
    case RuntimeTransferKind::kCudaVulkanFallback:
      return RuntimeTraceEventKind::kCudaVulkanFallbackTransfer;
  }
  return RuntimeTraceEventKind::kDeviceToDeviceTransfer;
}

}  // namespace taichi::lang
