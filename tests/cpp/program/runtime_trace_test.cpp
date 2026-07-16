#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "taichi/program/runtime_trace.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/system/profiler.h"
#include "taichi/system/timeline.h"

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

TEST(DiagnosticMemory, TimelineDropsEventsAfterGlobalBudget) {
  auto &timelines = ::taichi::Timelines::get_instance();
  timelines.set_event_capacity_for_testing(16);
  timelines.set_enabled(true);

  std::vector<std::thread> workers;
  for (int thread_index = 0; thread_index < 8; ++thread_index) {
    workers.emplace_back([thread_index] {
      auto &timeline = ::taichi::Timeline::get_this_thread_instance();
      timeline.set_name(std::to_string(thread_index));
      for (int event_index = 0; event_index < 4; ++event_index) {
        timeline.insert_event(
            {"bounded", true, ::taichi::Time::get_time(), "worker"});
      }
    });
  }
  for (std::thread &worker : workers) {
    worker.join();
  }

  EXPECT_EQ(timelines.recorded_event_count(), 16);
  EXPECT_EQ(timelines.dropped_event_count(), 16);
  timelines.set_enabled(false);
  timelines.set_event_capacity_for_testing(
      ::taichi::Timelines::kDefaultEventCapacity);
}

TEST(DiagnosticMemory, CompileTraceDropsEventsAfterBudget) {
  auto &profiling = ::taichi::Profiling::get_instance();
  profiling.set_trace_event_capacity_for_testing(8);

  for (int event_index = 0; event_index < 12; ++event_index) {
    profiling.record_trace_event(
        {"bounded", 0, static_cast<double>(event_index), 1.0});
  }

  EXPECT_EQ(profiling.trace_event_count(), 8);
  EXPECT_EQ(profiling.dropped_trace_event_count(), 4);
  profiling.set_trace_event_capacity_for_testing(
      ::taichi::Profiling::kDefaultTraceEventCapacity);
}

TEST(DiagnosticMemory, ExitedThreadProfilersAreRetired) {
  auto &profiling = ::taichi::Profiling::get_instance();
  ::taichi::Profiling::set_tracing_runtime_override(false);
  const auto baseline = profiling.live_profiler_count();

  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> workers;
  for (int thread_index = 0; thread_index < 32; ++thread_index) {
    workers.emplace_back([&] {
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      ::taichi::ScopedProfiler scope("transient worker");
    });
  }
  while (ready.load(std::memory_order_acquire) != 32) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);
  for (std::thread &worker : workers) {
    worker.join();
  }

  EXPECT_EQ(profiling.live_profiler_count(), baseline);
  ::taichi::Profiling::clear_tracing_runtime_override();
}

TEST(DiagnosticMemory, KernelProfilerRejectsUnboundedRawHistory) {
  auto profiler = make_profiler(Arch::x64, true);
  ASSERT_NE(profiler, nullptr);
  profiler->set_record_capacity_for_testing(2);

  profiler->insert_record("first", 1.0);
  profiler->insert_record("second", 2.0);
  EXPECT_EQ(profiler->record_count(), 2);
  EXPECT_ANY_THROW(profiler->insert_record("overflow", 3.0));
  EXPECT_EQ(profiler->record_count(), 2);

  profiler->clear();
  EXPECT_EQ(profiler->record_count(), 0);
}

}  // namespace
}  // namespace taichi::lang
