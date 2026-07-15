#include "gtest/gtest.h"

#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "taichi/program/runtime_statistics.h"
#include "taichi/rhi/common/runtime_telemetry.h"

namespace taichi::lang {
namespace {

TEST(RuntimeStatistics, SnapshotSchemaStartsAtMeasuredZero) {
  RuntimeStatistics statistics(Arch::vulkan, 17);
  const RuntimeStatisticsSnapshot snapshot = statistics.snapshot();

  EXPECT_EQ(snapshot.schema_version, kRuntimeStatisticsSchemaVersion);
  EXPECT_EQ(snapshot.backend, Arch::vulkan);
  EXPECT_EQ(snapshot.program_domain, 17u);
  EXPECT_EQ(snapshot.submission.kernel_submissions, 0u);
  EXPECT_EQ(snapshot.synchronization.program_syncs, 0u);
  EXPECT_EQ(snapshot.fault.first_fatal_faults, 0u);
  EXPECT_FALSE(snapshot.memory.cuda_mempool_used_bytes.available);
  EXPECT_FALSE(snapshot.synchronization.backend_lock_samples.available);
  EXPECT_TRUE(std::is_standard_layout_v<RuntimeStatisticsSnapshot>);
  EXPECT_TRUE(std::is_trivially_copyable_v<RuntimeStatisticsSnapshot>);
  EXPECT_LE(sizeof(RuntimeStatistics), 512u);
}

TEST(RuntimeStatistics, ConcurrentCountersRemainExactAndMonotonic) {
  RuntimeStatistics statistics(Arch::cuda, 23);
  constexpr std::uint64_t kThreads = 8;
  constexpr std::uint64_t kIterations = 20000;
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (std::uint64_t thread = 0; thread < kThreads; ++thread) {
    workers.emplace_back([&] {
      for (std::uint64_t i = 0; i < kIterations; ++i) {
        statistics.record_submission(RuntimeSubmissionKind::kKernel);
        statistics.record_transfer(RuntimeTransferKind::kHostToDevice, 4);
        if ((i & 7u) == 0) {
          statistics.record_completion_poll();
        }
      }
    });
  }

  RuntimeStatisticsSnapshot previous = statistics.snapshot();
  while (previous.submission.kernel_submissions < kThreads * kIterations) {
    const RuntimeStatisticsSnapshot current = statistics.snapshot();
    EXPECT_GE(current.submission.kernel_submissions,
              previous.submission.kernel_submissions);
    EXPECT_GE(current.transfer.host_to_device_bytes,
              previous.transfer.host_to_device_bytes);
    previous = current;
    std::this_thread::yield();
  }
  for (auto &worker : workers) {
    worker.join();
  }

  const RuntimeStatisticsSnapshot snapshot = statistics.snapshot();
  EXPECT_EQ(snapshot.submission.kernel_submissions,
            kThreads * kIterations);
  EXPECT_EQ(snapshot.transfer.host_to_device_bytes,
            kThreads * kIterations * 4);
  EXPECT_EQ(snapshot.synchronization.completion_polls,
            kThreads * (kIterations / 8));
}

TEST(RuntimeStatistics, CategoriesDoNotAliasEachOther) {
  RuntimeStatistics statistics(Arch::x64, 31);
  statistics.record_submission(RuntimeSubmissionKind::kGraph);
  statistics.record_submission(RuntimeSubmissionKind::kGraphBackendSubmission);
  statistics.record_submission(RuntimeSubmissionKind::kNative);
  statistics.record_submission_failure();
  statistics.record_program_sync(19);
  statistics.record_completion_wait(23);
  statistics.record_graph_capture();
  statistics.record_graph_recapture();
  statistics.record_graph_replay();
  statistics.record_graph_ordinary_fallback();
  statistics.record_graph_slot_saturation_fallback();
  statistics.record_display(true, true, false, 4096);
  statistics.record_trace_events(7, 2);

  const RuntimeStatisticsSnapshot snapshot = statistics.snapshot();
  EXPECT_EQ(snapshot.submission.kernel_submissions, 0u);
  EXPECT_EQ(snapshot.submission.graph_submissions, 1u);
  EXPECT_EQ(snapshot.submission.graph_backend_submissions, 1u);
  EXPECT_EQ(snapshot.submission.native_submissions, 1u);
  EXPECT_EQ(snapshot.submission.failed_submissions, 1u);
  EXPECT_EQ(snapshot.synchronization.program_syncs, 1u);
  EXPECT_EQ(snapshot.synchronization.program_sync_wait_ns, 19u);
  EXPECT_EQ(snapshot.synchronization.completion_waits, 1u);
  EXPECT_EQ(snapshot.synchronization.completion_wait_ns, 23u);
  EXPECT_EQ(snapshot.graph.captures, 1u);
  EXPECT_EQ(snapshot.graph.recaptures, 1u);
  EXPECT_EQ(snapshot.graph.replays, 1u);
  EXPECT_EQ(snapshot.graph.ordinary_fallbacks, 1u);
  EXPECT_EQ(snapshot.graph.replay_slot_saturation_fallbacks, 1u);
  EXPECT_EQ(snapshot.display.accepted_frames, 1u);
  EXPECT_EQ(snapshot.display.submitted_frames, 1u);
  EXPECT_EQ(snapshot.display.dropped_frames, 0u);
  EXPECT_EQ(snapshot.display.accepted_frame_bytes, 4096u);
  EXPECT_EQ(snapshot.trace.recorded_events, 7u);
  EXPECT_EQ(snapshot.trace.dropped_events, 2u);
}

TEST(RuntimeTelemetry, SampledUncontendedLockHasBoundedExactSamples) {
  SampledLockTelemetry<std::mutex> telemetry;
  std::mutex mutex;
  for (std::uint64_t i = 0; i < 128; ++i) {
    auto lock = telemetry.acquire(mutex);
  }

  const auto snapshot = telemetry.snapshot();
  EXPECT_EQ(snapshot.sampled_acquisitions, 2u);
  EXPECT_EQ(snapshot.contended_acquisitions, 0u);
  EXPECT_EQ(snapshot.sampled_wait_ns, 0u);
}

TEST(RuntimeTelemetry, BackendWaitCounterAccumulatesExactValues) {
  BackendWaitTelemetry telemetry;
  telemetry.record(17);
  telemetry.record(23);

  const auto snapshot = telemetry.snapshot();
  EXPECT_EQ(snapshot.waits, 2u);
  EXPECT_EQ(snapshot.wait_ns, 40u);
}

}  // namespace
}  // namespace taichi::lang
