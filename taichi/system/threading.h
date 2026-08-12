/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include "taichi/common/core.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace taichi {

using RangeForTaskFunc = void(void *, int thread_id, int i);
using ParallelFor = void(int n, int num_threads, void *, RangeForTaskFunc func);

struct ThreadPoolStatistics {
  bool enabled{false};
  std::uint64_t jobs_submitted{0};
  std::uint64_t jobs_completed{0};
  std::uint64_t queued_jobs{0};
  std::uint64_t nested_serial_jobs{0};
  std::uint64_t tasks_requested{0};
  std::uint64_t tasks_completed{0};
  std::uint64_t nested_serial_tasks{0};
  std::uint64_t requested_worker_slots{0};
  std::uint64_t joined_workers{0};
  std::uint64_t underfilled_jobs{0};
  std::uint64_t cancelled_jobs{0};
  std::uint64_t exception_jobs{0};
  std::uint64_t queue_wait_ns{0};
  std::uint64_t execution_ns{0};
  std::uint64_t submitter_wait_ns{0};
  std::uint64_t max_queue_depth{0};
  std::uint64_t max_requested_threads{0};
  std::uint64_t max_joined_workers{0};
};

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

  void set_telemetry_enabled(bool enabled) noexcept;
  ThreadPoolStatistics telemetry_statistics(bool reset = false) noexcept;

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
    std::atomic<int> completed_tasks{0};
    int joined_workers{0};
    bool completed{false};
    bool telemetry{false};
    std::uint64_t submitted_ns{0};
    std::uint64_t activated_ns{0};
    std::exception_ptr exception;
  };

  // Joins one worker to the active FIFO job. `mutex_` is paid once per worker,
  // not once per range chunk; chunks are claimed through Job::next_task.
  bool join_job_locked(Job **job);
  void activate_next_job_locked();
  void target();
  void record_job_completion(Job *job) noexcept;

  const int max_num_threads_;
  std::vector<std::thread> threads_;
  std::deque<Job *> pending_jobs_;
  Job *active_job_{nullptr};
  std::condition_variable worker_cv_;
  std::condition_variable completion_cv_;
  std::mutex mutex_;
  std::atomic<int> next_worker_id_{0};
  std::atomic<bool> telemetry_enabled_{false};
  std::atomic<std::uint64_t> telemetry_jobs_submitted_{0};
  std::atomic<std::uint64_t> telemetry_jobs_completed_{0};
  std::atomic<std::uint64_t> telemetry_queued_jobs_{0};
  std::atomic<std::uint64_t> telemetry_nested_serial_jobs_{0};
  std::atomic<std::uint64_t> telemetry_tasks_requested_{0};
  std::atomic<std::uint64_t> telemetry_tasks_completed_{0};
  std::atomic<std::uint64_t> telemetry_nested_serial_tasks_{0};
  std::atomic<std::uint64_t> telemetry_requested_worker_slots_{0};
  std::atomic<std::uint64_t> telemetry_joined_workers_{0};
  std::atomic<std::uint64_t> telemetry_underfilled_jobs_{0};
  std::atomic<std::uint64_t> telemetry_cancelled_jobs_{0};
  std::atomic<std::uint64_t> telemetry_exception_jobs_{0};
  std::atomic<std::uint64_t> telemetry_queue_wait_ns_{0};
  std::atomic<std::uint64_t> telemetry_execution_ns_{0};
  std::atomic<std::uint64_t> telemetry_submitter_wait_ns_{0};
  std::atomic<std::uint64_t> telemetry_max_queue_depth_{0};
  std::atomic<std::uint64_t> telemetry_max_requested_threads_{0};
  std::atomic<std::uint64_t> telemetry_max_joined_workers_{0};
  bool exiting_{false};
};

}  // namespace taichi
