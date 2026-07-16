/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "gtest/gtest.h"

#include "taichi/system/threading.h"

#ifdef TI_WITH_LLVM
#include "taichi/program/compile_config.h"
#include "taichi/rhi/arch.h"
#include "taichi/runtime/llvm/llvm_context.h"
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

namespace taichi {
namespace {

struct ConcurrentRunContext {
  std::atomic<int> completed{0};
};

void count_concurrent_run_task(void *context, int, int) {
  static_cast<ConcurrentRunContext *>(context)->completed.fetch_add(
      1, std::memory_order_relaxed);
}

struct NestedRunContext {
  ThreadPool *pool{nullptr};
  std::atomic<int> inner_completed{0};
  std::atomic<bool> inner_used_nonzero_thread{false};
};

void nested_run_task(void *context, int thread_id, int) {
  auto *nested = static_cast<NestedRunContext *>(context);
  if (thread_id != 0) {
    nested->inner_used_nonzero_thread.store(true, std::memory_order_relaxed);
  }
  nested->inner_completed.fetch_add(1, std::memory_order_relaxed);
}

void outer_run_task(void *context, int thread_id, int) {
  auto *nested = static_cast<NestedRunContext *>(context);
  (void)thread_id;
  nested->pool->run(8, 4, nested, nested_run_task);
}

void count_task(void *context, int, int) {
  static_cast<std::atomic<int> *>(context)->fetch_add(1,
                                                       std::memory_order_relaxed);
}

struct ParallelismContext {
  std::atomic<int> active{0};
  std::atomic<int> max_active{0};
  std::atomic<int> completed{0};
};

struct FifoContext {
  std::mutex mutex;
  std::condition_variable started_cv;
  std::condition_variable release_cv;
  bool first_started{false};
  bool release_first{false};
  std::vector<int> execution_order;
};

void first_fifo_task(void *context, int, int) {
  auto *state = static_cast<FifoContext *>(context);
  std::unique_lock<std::mutex> lock(state->mutex);
  if (!state->first_started) {
    state->first_started = true;
    state->started_cv.notify_one();
    state->release_cv.wait(lock, [&] { return state->release_first; });
  }
  state->execution_order.push_back(0);
}

void second_fifo_task(void *context, int, int) {
  auto *state = static_cast<FifoContext *>(context);
  std::lock_guard<std::mutex> lock(state->mutex);
  state->execution_order.push_back(1);
}

void bounded_parallel_task(void *context, int, int) {
  auto *state = static_cast<ParallelismContext *>(context);
  const int active = state->active.fetch_add(1, std::memory_order_relaxed) + 1;
  int observed = state->max_active.load(std::memory_order_relaxed);
  while (observed < active &&
         !state->max_active.compare_exchange_weak(
             observed, active, std::memory_order_relaxed)) {
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  state->completed.fetch_add(1, std::memory_order_relaxed);
  state->active.fetch_sub(1, std::memory_order_relaxed);
}

void throwing_task(void *, int, int task_id) {
  if (task_id == 2) {
    throw std::runtime_error("expected task failure");
  }
}

TEST(ThreadPoolTest, ConcurrentRunCallsKeepTheirContextsIsolated) {
  constexpr int kCallers = 4;
  constexpr int kSplits = 64;
  std::array<ConcurrentRunContext, kCallers> contexts;
  ThreadPool pool(4);
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::thread> callers;
  callers.reserve(kCallers);

  for (int i = 0; i < kCallers; i++) {
    callers.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      pool.run(kSplits, 4, &contexts[i], count_concurrent_run_task);
    });
  }
  while (ready.load(std::memory_order_acquire) != kCallers) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto &caller : callers) {
    caller.join();
  }

  for (const auto &context : contexts) {
    EXPECT_EQ(context.completed.load(std::memory_order_relaxed), kSplits);
  }
}

TEST(ThreadPoolTest, NestedRunUsesTheCurrentWorkerAsLogicalWorkerZero) {
  NestedRunContext context;
  ThreadPool pool(4);
  context.pool = &pool;

  pool.run(1, 1, &context, outer_run_task);

  EXPECT_EQ(context.inner_completed.load(std::memory_order_relaxed), 8);
  EXPECT_FALSE(
      context.inner_used_nonzero_thread.load(std::memory_order_relaxed));
}

TEST(ThreadPoolTest, NestedRunOnAnotherPoolUsesLogicalWorkerZero) {
  NestedRunContext context;
  ThreadPool outer_pool(2);
  ThreadPool nested_pool(2);
  context.pool = &nested_pool;

  outer_pool.run(1, 1, &context, outer_run_task);

  EXPECT_EQ(context.inner_completed.load(std::memory_order_relaxed), 8);
  EXPECT_FALSE(
      context.inner_used_nonzero_thread.load(std::memory_order_relaxed));
}

TEST(ThreadPoolTest, ZeroThreadRequestsAreClampedToOne) {
  std::atomic<int> completed{0};
  ThreadPool pool(0);

  pool.run(5, 0, &completed, count_task);

  EXPECT_EQ(completed.load(std::memory_order_relaxed), 5);
}

TEST(ThreadPoolTest, ConcurrentJobsRespectTheirOwnParallelismBudget) {
  ThreadPool pool(4);
  ParallelismContext first;
  ParallelismContext second;
  std::thread first_caller(
      [&] { pool.run(32, 2, &first, bounded_parallel_task); });
  std::thread second_caller(
      [&] { pool.run(32, 2, &second, bounded_parallel_task); });
  first_caller.join();
  second_caller.join();

  EXPECT_EQ(first.completed.load(std::memory_order_relaxed), 32);
  EXPECT_EQ(second.completed.load(std::memory_order_relaxed), 32);
  EXPECT_EQ(first.max_active.load(std::memory_order_relaxed), 2);
  EXPECT_EQ(second.max_active.load(std::memory_order_relaxed), 2);
}

TEST(ThreadPoolTest, ConcurrentJobsRunInFifoSubmissionOrder) {
  ThreadPool pool(2);
  FifoContext context;

  std::thread first([&] { pool.run(4, 1, &context, first_fifo_task); });
  {
    std::unique_lock<std::mutex> lock(context.mutex);
    context.started_cv.wait(lock, [&] { return context.first_started; });
  }
  std::thread second(
      [&] { pool.run(4, 1, &context, second_fifo_task); });
  {
    std::lock_guard<std::mutex> lock(context.mutex);
    context.release_first = true;
  }
  context.release_cv.notify_one();
  first.join();
  second.join();

  std::lock_guard<std::mutex> lock(context.mutex);
  ASSERT_EQ(context.execution_order.size(), 8);
  EXPECT_EQ(std::vector<int>(context.execution_order.begin(),
                             context.execution_order.begin() + 4),
            std::vector<int>(4, 0));
  EXPECT_EQ(std::vector<int>(context.execution_order.begin() + 4,
                             context.execution_order.end()),
            std::vector<int>(4, 1));
}

TEST(ThreadPoolTest, WorkerExceptionCompletesJobAndPoolRemainsUsable) {
  ThreadPool pool(2);
  EXPECT_THROW(pool.run(8, 2, nullptr, throwing_task), std::runtime_error);

  std::atomic<int> completed{0};
  pool.run(8, 2, &completed, count_task);
  EXPECT_EQ(completed.load(std::memory_order_relaxed), 8);
}

#ifdef TI_WITH_LLVM
TEST(LLVMContextThreadData, ReleasesExitedWorkerContexts) {
  lang::CompileConfig config;
  lang::TaichiLLVMContext context(config, host_arch());
  EXPECT_EQ(context.debug_thread_local_data_count(), 1u);

  constexpr int kThreadCount = 32;
  std::atomic<int> ready{0};
  std::atomic<bool> release{false};
  std::atomic<bool> valid{true};
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);

  for (int i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&] {
      if (context.get_this_thread_context() == nullptr) {
        valid.store(false, std::memory_order_relaxed);
      }
      ready.fetch_add(1, std::memory_order_release);
      while (!release.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != kThreadCount) {
    std::this_thread::yield();
  }
  EXPECT_EQ(context.debug_thread_local_data_count(), kThreadCount + 1);
  release.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }
  EXPECT_TRUE(valid.load(std::memory_order_relaxed));
  EXPECT_EQ(context.debug_thread_local_data_count(), 1u);
}
#endif

}  // namespace
}  // namespace taichi
