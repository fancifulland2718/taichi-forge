/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "gtest/gtest.h"

#include "taichi/system/threading.h"

#include <array>
#include <atomic>
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

}  // namespace
}  // namespace taichi
