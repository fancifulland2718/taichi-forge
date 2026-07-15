#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "taichi/program/runtime_trace.h"

namespace taichi::lang {
namespace {

TEST(RuntimeTrace, DisabledPathOwnsNoEventBuffer) {
  RuntimeStatistics statistics(Arch::x64, 101);
  RuntimeTraceRecorder trace(statistics, 101);

  for (int i = 0; i < 32; ++i) {
    trace.record_instant(RuntimeTraceEventKind::kKernelSubmission);
  }

  const RuntimeTraceSnapshot snapshot = trace.snapshot();
  EXPECT_FALSE(snapshot.enabled);
  EXPECT_EQ(snapshot.session, 0);
  EXPECT_EQ(snapshot.event_capacity, 0);
  EXPECT_EQ(snapshot.allocated_bytes, 0);
  EXPECT_EQ(snapshot.recorded_events, 0);
  EXPECT_EQ(snapshot.dropped_events, 0);
  const RuntimeStatisticsSnapshot counters = statistics.snapshot();
  EXPECT_EQ(counters.trace.recorded_events, 0);
  EXPECT_EQ(counters.trace.dropped_events, 0);
}

TEST(RuntimeTrace, CapacityIsFixedAndOverflowIsObservable) {
  RuntimeStatistics statistics(Arch::cuda, 102);
  RuntimeTraceRecorder trace(statistics, 102);

  const RuntimeTraceSnapshot started = trace.start(1, 3);
  EXPECT_TRUE(started.enabled);
  EXPECT_EQ(started.event_capacity, 3);
  EXPECT_GT(started.allocated_bytes, 0);
  EXPECT_LE(started.allocated_bytes,
            started.event_capacity * 32 + started.max_threads * 32);
  for (int i = 0; i < 5; ++i) {
    trace.record_instant(RuntimeTraceEventKind::kKernelSubmission);
  }
  const RuntimeTraceSnapshot stopped = trace.stop();
  EXPECT_FALSE(stopped.enabled);
  EXPECT_EQ(stopped.recorded_events, 3);
  EXPECT_EQ(stopped.dropped_events, 2);

  const RuntimeStatisticsSnapshot counters = statistics.snapshot();
  EXPECT_EQ(counters.trace.recorded_events, 3);
  EXPECT_EQ(counters.trace.dropped_events, 2);
}

TEST(RuntimeTrace, ThreadShardsAndEventSlotsNeverGrow) {
  RuntimeStatistics statistics(Arch::vulkan, 103);
  RuntimeTraceRecorder trace(statistics, 103);
  trace.start(2, 4);

  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  for (int thread_index = 0; thread_index < 4; ++thread_index) {
    threads.emplace_back([&] {
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int event = 0; event < 6; ++event) {
        trace.record_instant(RuntimeTraceEventKind::kGraphSubmission);
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != 4) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);
  for (std::thread &thread : threads) {
    thread.join();
  }

  const RuntimeTraceSnapshot stopped = trace.stop();
  EXPECT_EQ(stopped.event_capacity, 8);
  EXPECT_EQ(stopped.recorded_events, 8);
  EXPECT_EQ(stopped.dropped_events, 16);
}

TEST(RuntimeTrace, NewSessionResetsBufferButNotProgramCounters) {
  RuntimeStatistics statistics(Arch::x64, 104);
  RuntimeTraceRecorder trace(statistics, 104);

  trace.start(1, 2);
  trace.record_instant(RuntimeTraceEventKind::kKernelSubmission);
  EXPECT_EQ(trace.stop().recorded_events, 1);

  const RuntimeTraceSnapshot second = trace.start(1, 1);
  EXPECT_EQ(second.session, 2);
  EXPECT_EQ(second.recorded_events, 0);
  trace.record_instant(RuntimeTraceEventKind::kNativeSubmission);
  trace.record_instant(RuntimeTraceEventKind::kNativeSubmission);
  const RuntimeTraceSnapshot stopped = trace.stop();
  EXPECT_EQ(stopped.recorded_events, 1);
  EXPECT_EQ(stopped.dropped_events, 1);

  const RuntimeStatisticsSnapshot counters = statistics.snapshot();
  EXPECT_EQ(counters.trace.recorded_events, 2);
  EXPECT_EQ(counters.trace.dropped_events, 1);
}

TEST(RuntimeTrace, ConcurrentStopFreezesAStableBoundedSnapshot) {
  RuntimeStatistics statistics(Arch::cuda, 107);
  RuntimeTraceRecorder trace(statistics, 107);
  trace.start(8, 128);

  std::atomic<bool> run{true};
  std::vector<std::thread> workers;
  for (int thread_index = 0; thread_index < 8; ++thread_index) {
    workers.emplace_back([&] {
      while (run.load(std::memory_order_acquire)) {
        RuntimeTraceRecorder::Scope scope(
            &trace, RuntimeTraceEventKind::kProgramSynchronize);
        std::this_thread::yield();
      }
    });
  }
  while (trace.snapshot().recorded_events < 64) {
    std::this_thread::yield();
  }

  const RuntimeTraceSnapshot stopped = trace.stop();
  run.store(false, std::memory_order_release);
  for (std::thread &worker : workers) {
    worker.join();
  }
  const RuntimeTraceSnapshot stable = trace.snapshot();
  EXPECT_FALSE(stable.enabled);
  EXPECT_EQ(stable.recorded_events, stopped.recorded_events);
  EXPECT_EQ(stable.dropped_events, stopped.dropped_events);
  EXPECT_LE(stable.recorded_events, stable.event_capacity);
  EXPECT_EQ(statistics.snapshot().trace.recorded_events,
            stable.recorded_events);
  EXPECT_EQ(statistics.snapshot().trace.dropped_events,
            stable.dropped_events);
}

TEST(RuntimeTrace, ChromeExportIsBoundedAndSelfDescribing) {
  RuntimeStatistics statistics(Arch::x64, 105);
  RuntimeTraceRecorder trace(statistics, 105);
  trace.start(1, 4);
  trace.record_instant(RuntimeTraceEventKind::kKernelSubmission);
  {
    RuntimeTraceRecorder::Scope transfer(
        &trace, RuntimeTraceEventKind::kHostToDeviceTransfer, 4096);
    transfer.mark_failed();
  }
  EXPECT_ANY_THROW(trace.export_chrome_trace("trace-must-be-stopped.json"));
  trace.stop();

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      "taichi-runtime-trace-foundation-test.json";
  ASSERT_TRUE(trace.export_chrome_trace(path.string()));
  std::ifstream input(path);
  const std::string content((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("runtime.kernel.submit"), std::string::npos);
  EXPECT_NE(content.find("runtime.transfer.h2d"), std::string::npos);
  EXPECT_NE(content.find(R"JSON("bytes":4096)JSON"), std::string::npos);
  EXPECT_NE(content.find(R"JSON("failed":true)JSON"), std::string::npos);
  EXPECT_NE(content.find(R"JSON("programDomain": 105)JSON"),
            std::string::npos);
  EXPECT_NE(content.find(R"JSON("droppedEvents": 0)JSON"),
            std::string::npos);
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

TEST(RuntimeTrace, RejectsUnboundedConfigurations) {
  RuntimeStatistics statistics(Arch::x64, 106);
  RuntimeTraceRecorder trace(statistics, 106);
  EXPECT_ANY_THROW(trace.start(0, 1));
  EXPECT_ANY_THROW(trace.start(RuntimeTraceRecorder::kMaximumThreads + 1, 1));
  EXPECT_ANY_THROW(trace.start(2,
                               RuntimeTraceRecorder::kMaximumTotalEvents));
  EXPECT_EQ(trace.snapshot().allocated_bytes, 0);
}

}  // namespace
}  // namespace taichi::lang
