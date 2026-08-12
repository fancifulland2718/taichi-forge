/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "taichi/common/core.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace taichi {

using RangeForTaskFunc = void(void *, int thread_id, int i);
using ParallelFor = void(int n, int num_threads, void *, RangeForTaskFunc func);

class ThreadPool {
 public:
  explicit ThreadPool(int max_num_threads);

  void run(int splits,
           int desired_num_threads,
           void *range_for_task_context,
           RangeForTaskFunc *func);

  static void static_run(ThreadPool *pool,
                         int splits,
                         int desired_num_threads,
                         void *range_for_task_context,
                         RangeForTaskFunc *func) {
    return pool->run(splits, desired_num_threads, range_for_task_context, func);
  }

  ~ThreadPool();

 private:
  struct Job {
    int splits{0};
    int desired_num_threads{1};
    void *range_for_task_context{nullptr};
    RangeForTaskFunc *func{nullptr};
    std::atomic<int> next_task{0};
    std::atomic<bool> cancelled{false};
    // Positive values count workers that may still access this stack-owned
    // Job. The final worker atomically changes 1 to -1 before completing it;
    // -1 is a closing sentinel that prevents a late join and therefore keeps
    // the Job alive until the closer releases ThreadPool::mutex_.
    std::atomic<int> active_workers{0};
    int joined_workers{0};
    bool completed{false};
    std::exception_ptr exception;
  };

  // Joins one worker to the active FIFO job. `mutex_` is paid once per worker,
  // not once per range chunk; chunks are claimed through Job::next_task.
  bool join_job_locked(Job **job);
  void activate_next_job_locked();
  void target();

  const int max_num_threads_;
  std::vector<std::thread> threads_;
  std::deque<Job *> pending_jobs_;
  Job *active_job_{nullptr};
  std::condition_variable worker_cv_;
  std::condition_variable completion_cv_;
  std::mutex mutex_;
  std::atomic<int> next_worker_id_{0};
  bool exiting_{false};
};

}  // namespace taichi
