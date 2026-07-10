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
#include <memory>
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
    int next_task{0};
    int active_workers{0};
    bool completed{false};
    bool cancelled{false};
    std::exception_ptr exception;
  };

  // Takes one task from the active FIFO job. `mutex_` must be held. Producers
  // may enqueue concurrently, but a full job owns the fixed worker budget
  // until completion: the measured workload does not justify interleaving two
  // saturated memory-bound jobs.
  bool take_task_locked(std::shared_ptr<Job> *job, int *task_id);
  void activate_next_job_locked();
  void target();

  const int max_num_threads_;
  std::vector<std::thread> threads_;
  std::deque<std::shared_ptr<Job>> pending_jobs_;
  std::shared_ptr<Job> active_job_;
  std::condition_variable worker_cv_;
  std::condition_variable completion_cv_;
  std::mutex mutex_;
  std::atomic<int> next_worker_id_{0};
  bool exiting_{false};
};

}  // namespace taichi
