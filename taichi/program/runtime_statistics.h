#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>

#include "taichi/common/core.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {

// Public runtime statistics snapshots use a versioned, backend-neutral POD
// contract. Backends may populate only the fields they can measure; an
// unavailable value is different from a measured zero.
constexpr std::uint32_t kRuntimeStatisticsSchemaVersion = 1;

struct RuntimeOptionalCounter {
  std::uint64_t value{0};
  bool available{false};
};

struct RuntimeSubmissionStatistics {
  std::uint64_t kernel_submissions{0};
  std::uint64_t graph_submissions{0};
  std::uint64_t graph_backend_replays{0};
  std::uint64_t native_submissions{0};
  std::uint64_t failed_submissions{0};
};

struct RuntimeSynchronizationStatistics {
  std::uint64_t program_syncs{0};
  std::uint64_t program_sync_wait_ns{0};
  std::uint64_t completion_polls{0};
  std::uint64_t completion_waits{0};
  std::uint64_t completion_wait_ns{0};
  RuntimeOptionalCounter backend_waits;
  RuntimeOptionalCounter backend_wait_ns;
  RuntimeOptionalCounter queue_lock_samples;
  RuntimeOptionalCounter queue_lock_contentions;
  RuntimeOptionalCounter queue_lock_wait_ns;
};

struct RuntimeMemoryStatistics {
  std::uint64_t live_resources{0};
  std::uint64_t retiring_resources{0};
  std::uint64_t inflight_resources{0};
  RuntimeOptionalCounter host_requested_live_bytes;
  RuntimeOptionalCounter host_raw_bytes;
  RuntimeOptionalCounter host_capacity_bytes;
  RuntimeOptionalCounter device_requested_live_bytes;
  RuntimeOptionalCounter device_raw_bytes;
  RuntimeOptionalCounter device_cached_bytes;
  RuntimeOptionalCounter cuda_mempool_reserved_bytes;
  RuntimeOptionalCounter cuda_mempool_used_bytes;
};

struct RuntimeTransferStatistics {
  std::uint64_t host_to_device_bytes{0};
  std::uint64_t device_to_host_bytes{0};
  std::uint64_t device_to_device_bytes{0};
  std::uint64_t cuda_vulkan_direct_bytes{0};
  std::uint64_t cuda_vulkan_fallback_bytes{0};
};

struct RuntimeGraphStatistics {
  std::uint64_t captures{0};
  std::uint64_t recaptures{0};
  std::uint64_t replays{0};
  std::uint64_t ordinary_fallbacks{0};
  std::uint64_t replay_slot_saturation_fallbacks{0};
};

struct RuntimeDisplayStatistics {
  std::uint64_t accepted_frames{0};
  std::uint64_t submitted_frames{0};
  std::uint64_t dropped_frames{0};
  std::uint64_t staged_frame_bytes{0};
};

struct RuntimeFaultStatistics {
  std::uint64_t first_fatal_faults{0};
  std::uint64_t rejected_submissions{0};
};

struct RuntimeTraceStatistics {
  std::uint64_t recorded_events{0};
  std::uint64_t dropped_events{0};
};

struct RuntimeStatisticsSnapshot {
  std::uint32_t schema_version{kRuntimeStatisticsSchemaVersion};
  Arch backend{Arch::x64};
  std::uint64_t program_domain{0};
  RuntimeSubmissionStatistics submission;
  RuntimeSynchronizationStatistics synchronization;
  RuntimeMemoryStatistics memory;
  RuntimeTransferStatistics transfer;
  RuntimeGraphStatistics graph;
  RuntimeDisplayStatistics display;
  RuntimeFaultStatistics fault;
  RuntimeTraceStatistics trace;
};

static_assert(std::is_standard_layout_v<RuntimeStatisticsSnapshot>);
static_assert(std::is_trivially_copyable_v<RuntimeStatisticsSnapshot>);

enum class RuntimeSubmissionKind : std::uint8_t {
  kKernel,
  kGraph,
  kGraphBackendReplay,
  kNative,
};

enum class RuntimeTransferKind : std::uint8_t {
  kHostToDevice,
  kDeviceToHost,
  kDeviceToDevice,
  kCudaVulkanDirect,
  kCudaVulkanFallback,
};

// Program-owned counter source. The regular path is allocation-free and uses
// relaxed atomics: snapshots need monotonic values, not cross-counter event
// ordering. External adapters fill current-value/optional fields after taking
// their own subsystem snapshots.
class TI_DLL_EXPORT RuntimeStatistics final {
 public:
  RuntimeStatistics(Arch backend, std::uint64_t program_domain) noexcept
      : backend_(backend), program_domain_(program_domain) {
  }

  RuntimeStatisticsSnapshot snapshot() const noexcept {
    RuntimeStatisticsSnapshot result;
    result.backend = backend_;
    result.program_domain = program_domain_;
    result.submission.kernel_submissions = load(kernel_submissions_);
    result.submission.graph_submissions = load(graph_submissions_);
    result.submission.graph_backend_replays = load(graph_backend_replays_);
    result.submission.native_submissions = load(native_submissions_);
    result.submission.failed_submissions = load(failed_submissions_);
    result.synchronization.program_syncs = load(program_syncs_);
    result.synchronization.program_sync_wait_ns = load(program_sync_wait_ns_);
    result.synchronization.completion_polls = load(completion_polls_);
    result.synchronization.completion_waits = load(completion_waits_);
    result.synchronization.completion_wait_ns = load(completion_wait_ns_);
    result.transfer.host_to_device_bytes = load(host_to_device_bytes_);
    result.transfer.device_to_host_bytes = load(device_to_host_bytes_);
    result.transfer.device_to_device_bytes = load(device_to_device_bytes_);
    result.transfer.cuda_vulkan_direct_bytes = load(cuda_vulkan_direct_bytes_);
    result.transfer.cuda_vulkan_fallback_bytes =
        load(cuda_vulkan_fallback_bytes_);
    result.graph.captures = load(graph_captures_);
    result.graph.recaptures = load(graph_recaptures_);
    result.graph.replays = load(graph_replays_);
    result.graph.ordinary_fallbacks = load(graph_ordinary_fallbacks_);
    result.graph.replay_slot_saturation_fallbacks =
        load(graph_replay_slot_saturation_fallbacks_);
    result.display.accepted_frames = load(display_accepted_frames_);
    result.display.submitted_frames = load(display_submitted_frames_);
    result.display.dropped_frames = load(display_dropped_frames_);
    result.display.staged_frame_bytes = load(display_staged_frame_bytes_);
    result.fault.first_fatal_faults = load(first_fatal_faults_);
    result.fault.rejected_submissions = load(rejected_submissions_);
    result.trace.recorded_events = load(trace_recorded_events_);
    result.trace.dropped_events = load(trace_dropped_events_);
    return result;
  }

  void record_submission(RuntimeSubmissionKind kind) noexcept {
    switch (kind) {
      case RuntimeSubmissionKind::kKernel:
        add(kernel_submissions_, 1);
        break;
      case RuntimeSubmissionKind::kGraph:
        add(graph_submissions_, 1);
        break;
      case RuntimeSubmissionKind::kGraphBackendReplay:
        add(graph_backend_replays_, 1);
        break;
      case RuntimeSubmissionKind::kNative:
        add(native_submissions_, 1);
        break;
    }
  }

  void record_submission_failure() noexcept {
    add(failed_submissions_, 1);
  }
  void record_program_sync(std::uint64_t wait_ns) noexcept {
    add(program_syncs_, 1);
    add(program_sync_wait_ns_, wait_ns);
  }
  void record_completion_poll() noexcept {
    add(completion_polls_, 1);
  }
  void record_completion_wait(std::uint64_t wait_ns) noexcept {
    add(completion_waits_, 1);
    add(completion_wait_ns_, wait_ns);
  }
  void record_transfer(RuntimeTransferKind kind, std::uint64_t bytes) noexcept {
    switch (kind) {
      case RuntimeTransferKind::kHostToDevice:
        add(host_to_device_bytes_, bytes);
        break;
      case RuntimeTransferKind::kDeviceToHost:
        add(device_to_host_bytes_, bytes);
        break;
      case RuntimeTransferKind::kDeviceToDevice:
        add(device_to_device_bytes_, bytes);
        break;
      case RuntimeTransferKind::kCudaVulkanDirect:
        add(cuda_vulkan_direct_bytes_, bytes);
        break;
      case RuntimeTransferKind::kCudaVulkanFallback:
        add(cuda_vulkan_fallback_bytes_, bytes);
        break;
    }
  }
  void record_graph_capture() noexcept {
    add(graph_captures_, 1);
  }
  void record_graph_recapture() noexcept {
    add(graph_recaptures_, 1);
  }
  void record_graph_replay() noexcept {
    add(graph_replays_, 1);
  }
  void record_graph_ordinary_fallback() noexcept {
    add(graph_ordinary_fallbacks_, 1);
  }
  void record_graph_slot_saturation_fallback() noexcept {
    add(graph_replay_slot_saturation_fallbacks_, 1);
  }
  void record_display(bool accepted,
                      bool submitted,
                      bool dropped,
                      std::uint64_t staged_bytes) noexcept {
    add(display_accepted_frames_, accepted ? 1 : 0);
    add(display_submitted_frames_, submitted ? 1 : 0);
    add(display_dropped_frames_, dropped ? 1 : 0);
    add(display_staged_frame_bytes_, staged_bytes);
  }
  void record_first_fatal_fault() noexcept {
    add(first_fatal_faults_, 1);
  }
  void record_rejected_submission() noexcept {
    add(rejected_submissions_, 1);
  }
  void record_trace_events(std::uint64_t recorded,
                           std::uint64_t dropped) noexcept {
    add(trace_recorded_events_, recorded);
    add(trace_dropped_events_, dropped);
  }

 private:
  static std::uint64_t load(const std::atomic<std::uint64_t> &value) noexcept {
    return value.load(std::memory_order_relaxed);
  }
  static void add(std::atomic<std::uint64_t> &value,
                  std::uint64_t amount) noexcept {
    if (amount != 0) {
      value.fetch_add(amount, std::memory_order_relaxed);
    }
  }

  const Arch backend_;
  const std::uint64_t program_domain_;
  std::atomic<std::uint64_t> kernel_submissions_{0};
  std::atomic<std::uint64_t> graph_submissions_{0};
  std::atomic<std::uint64_t> graph_backend_replays_{0};
  std::atomic<std::uint64_t> native_submissions_{0};
  std::atomic<std::uint64_t> failed_submissions_{0};
  std::atomic<std::uint64_t> program_syncs_{0};
  std::atomic<std::uint64_t> program_sync_wait_ns_{0};
  std::atomic<std::uint64_t> completion_polls_{0};
  std::atomic<std::uint64_t> completion_waits_{0};
  std::atomic<std::uint64_t> completion_wait_ns_{0};
  std::atomic<std::uint64_t> host_to_device_bytes_{0};
  std::atomic<std::uint64_t> device_to_host_bytes_{0};
  std::atomic<std::uint64_t> device_to_device_bytes_{0};
  std::atomic<std::uint64_t> cuda_vulkan_direct_bytes_{0};
  std::atomic<std::uint64_t> cuda_vulkan_fallback_bytes_{0};
  std::atomic<std::uint64_t> graph_captures_{0};
  std::atomic<std::uint64_t> graph_recaptures_{0};
  std::atomic<std::uint64_t> graph_replays_{0};
  std::atomic<std::uint64_t> graph_ordinary_fallbacks_{0};
  std::atomic<std::uint64_t> graph_replay_slot_saturation_fallbacks_{0};
  std::atomic<std::uint64_t> display_accepted_frames_{0};
  std::atomic<std::uint64_t> display_submitted_frames_{0};
  std::atomic<std::uint64_t> display_dropped_frames_{0};
  std::atomic<std::uint64_t> display_staged_frame_bytes_{0};
  std::atomic<std::uint64_t> first_fatal_faults_{0};
  std::atomic<std::uint64_t> rejected_submissions_{0};
  std::atomic<std::uint64_t> trace_recorded_events_{0};
  std::atomic<std::uint64_t> trace_dropped_events_{0};
};

}  // namespace taichi::lang
