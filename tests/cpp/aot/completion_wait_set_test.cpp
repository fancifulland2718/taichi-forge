#include "taichi/runtime/completion_wait_set.h"

#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <future>
#include <thread>
#include "gtest/gtest.h"

namespace taichi::lang {
namespace {
using namespace std::chrono_literals;
using Status = CompletionWaitSet::Status;

struct Signal final : StreamSemaphoreObject {
  mutable std::atomic<int> polls{0};
  std::atomic<bool> ready{false}, failed{false};
  bool is_ready() const override {
    ++polls;
    if (failed)
      throw std::runtime_error("wait-set backend failure");
    return ready;
  }
  bool wait() const override {
    throw std::logic_error("wait-any must not wait a single pending member");
  }
};

RuntimeCompletion token(const std::shared_ptr<Signal> &signal, uint64_t sequence) {
  return RuntimeCompletion::from_stream_semaphore(Arch::vulkan, 11, sequence, signal);
}

TEST(CompletionWaitSet, FirstReadyPreservesInputOrderAndNativeFailure) {
  auto signal = std::make_shared<Signal>();
  const auto pending = token(signal, 1);
  const auto ready = RuntimeCompletion::completed(Arch::x64, 12, 2);
  CompletionWaitSet set({pending, ready, ready}, 1ms);
  const auto result = set.wait(0ns);
  EXPECT_EQ(result.status, Status::ready);
  EXPECT_EQ(result.index, 1);
  signal->ready = true;
  EXPECT_EQ(set.wait(0ns).index, 0);
  auto failed = std::make_shared<Signal>();
  CompletionWaitSet broken({token(failed, 3)}, 1ms);
  failed->failed = true;
  EXPECT_THROW(broken.wait(0ns), std::runtime_error);
  EXPECT_THROW(broken.wait(0ns), std::runtime_error);
  EXPECT_THROW(CompletionWaitSet({}, 1ms), std::invalid_argument);
  EXPECT_THROW(CompletionWaitSet({{}}, 1ms), std::invalid_argument);
  EXPECT_THROW(CompletionWaitSet({ready}, 0ns), std::invalid_argument);
  EXPECT_THROW(set.wait(-2ns), std::invalid_argument);
}

TEST(CompletionWaitSet, TimeoutSleepsAndFutureReadinessRemainsObservable) {
  auto signal = std::make_shared<Signal>();
  CompletionWaitSet set({token(signal, 1)}, 2ms);
  EXPECT_EQ(set.wait(10ms).status, Status::timeout);
  EXPECT_GE(signal->polls, 2);
  EXPECT_LT(signal->polls, 30);  // Detect spinning, not a production limit.
  signal->ready = true;
  EXPECT_EQ(set.wait(0ns).status, Status::ready);
}

TEST(CompletionWaitSet, CancellationWakesAllWaitersAndIsTerminal) {
  auto signal = std::make_shared<Signal>();
  CompletionWaitSet set({token(signal, 1)}, 10s);
  auto first = std::async(std::launch::async, [&] { return set.wait(-1ns); });
  auto second = std::async(std::launch::async, [&] { return set.wait(-1ns); });
  const auto deadline = std::chrono::steady_clock::now() + 1s;
  while (signal->polls < 2 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  set.cancel();
  EXPECT_EQ(first.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(second.wait_for(1s), std::future_status::ready);
  EXPECT_EQ(first.get().status, Status::cancelled);
  EXPECT_EQ(second.get().status, Status::cancelled);
  signal->ready = true;
  EXPECT_EQ(set.wait(0ns).status, Status::cancelled);
}

TEST(CompletionWaitSet, CompletionBetweenScansAndRetiredOwnerStayVisible) {
  auto signal = std::make_shared<Signal>();
  auto completion = token(signal, 1);
  CompletionWaitSet set({completion}, 1ms);
  auto waiting = std::async(std::launch::async, [&] { return set.wait(1s); });
  const auto deadline = std::chrono::steady_clock::now() + 1s;
  while (signal->polls == 0 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  completion.mark_completed();
  completion = {};
  signal.reset();
  EXPECT_EQ(waiting.get().status, Status::ready);
  EXPECT_EQ(set.wait(0ns).status, Status::ready);
}

TEST(CompletionWaitSet, ObservationPolicyMeasurement) {
#ifndef _WIN32
  GTEST_SKIP() << "This timing probe uses Windows thread CPU accounting";
#else
  if (!std::getenv("TI_COMPLETION_SET_BENCHMARK"))
    GTEST_SKIP() << "Opt-in observation latency/CPU measurement";
  const auto cpu_ns = [] {
    FILETIME creation, exit, kernel, user;
    if (!GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user))
      throw std::runtime_error("Unable to read waiter CPU time");
    const auto value = [](FILETIME time) {
      return (uint64_t(time.dwHighDateTime) << 32) | time.dwLowDateTime;
    };
    return (value(kernel) + value(user)) * 100;
  };
  const auto now_ns = [] {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  };
  for (const auto interval : {0ns, 250000ns, 1000000ns}) {
    const auto label = interval == 0ns ? "spin" :
                       interval == 250000ns ? "poll250us" : "poll1000us";
    std::vector<double> latency;
    const auto start_cpu = cpu_ns();
    const auto start_wall = now_ns();
    for (int sample = 0; sample < 32; ++sample) {
      auto signal = std::make_shared<Signal>();
      auto completion = token(signal, sample + 1);
      CompletionWaitSet set({completion}, interval == 0ns ? 1ms : interval);
      std::atomic<int64_t> published{0};
      auto producer = std::async(std::launch::async, [&] {
        std::this_thread::sleep_for(3ms);
        published = now_ns();
        signal->ready = true;
      });
      if (interval == 0ns) {
        while (!completion.done()) {}
      } else {
        EXPECT_EQ(set.wait(1s).status, Status::ready);
      }
      latency.push_back((now_ns() - published.load()) / 1000.0);
      producer.get();
    }
    const auto used_cpu = cpu_ns() - start_cpu;
    const auto used_wall = now_ns() - start_wall;
    std::sort(latency.begin(), latency.end());
    std::printf("completion-set %s n=32 cpu_ms=%.3f wall_ms=%.3f p50_us=%.3f p95_us=%.3f\n",
                label, used_cpu / 1e6, used_wall / 1e6, latency[15], latency[30]);
    RecordProperty(std::string(label) + "_cpu_ms", std::to_string(used_cpu / 1e6));
    RecordProperty(std::string(label) + "_p95_us", std::to_string(latency[30]));
  }
#endif
}
}  // namespace
}  // namespace taichi::lang
