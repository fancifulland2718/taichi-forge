/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/system/threading.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <thread>
#include <vector>

namespace taichi {

namespace {

thread_local ThreadPool *active_thread_pool = nullptr;

std::uint64_t steady_clock_now_ns() noexcept {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

void update_atomic_max(std::atomic<std::uint64_t> &target,
                       std::uint64_t value) noexcept {
  auto previous = target.load(std::memory_order_relaxed);
  while (previous < value &&
         !target.compare_exchange_weak(previous, value,
                                       std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
  }
}

std::uint64_t read_counter(std::atomic<std::uint64_t> &counter,
                           bool reset) noexcept {
  return reset ? counter.exchange(0, std::memory_order_relaxed)
               : counter.load(std::memory_order_relaxed);
}

class ScopedThreadPoolExecution {
 public:
  explicit ScopedThreadPoolExecution(ThreadPool *pool)
      : previous_pool_(active_thread_pool) {
    active_thread_pool = pool;
  }

  ~ScopedThreadPoolExecution() {
    active_thread_pool = previous_pool_;
  }

 private:
  ThreadPool *previous_pool_;
};

}  // namespace

bool test_threading() {
  auto tp = ThreadPool(20);
  for (int j = 0; j < 100; j++) {
    tp.run(10, j + 1, &j, [](void *j, int _thread_id, int i) {
      double ret = 0.0;
      for (int t = 0; t < 10000000; t++) {
        ret += t * 1e-20;
      }
      TI_P(int(i + ret + 10 * *(int *)j));
    });
  }
  return true;
}

ThreadPool::ThreadPool(int max_num_threads)
    : max_num_threads_(std::max(1, max_num_threads)) {
  threads_.reserve(static_cast<std::size_t>(max_num_threads_));
  for (int i = 0; i < max_num_threads_; i++) {
    threads_.emplace_back([this] { target(); });
  }
}

void ThreadPool::run(int splits,
                     int desired_num_threads,
                     void *range_for_task_context,
                     RangeForTaskFunc *func) {
  if (splits <= 0) {
    return;
  }
  TI_ASSERT(func != nullptr);

  const bool telemetry =
      telemetry_enabled_.load(std::memory_order_relaxed);
  const int selected_threads =
      std::clamp(desired_num_threads, 1, max_num_threads_);
  const std::uint64_t submit_started_ns =
      telemetry ? steady_clock_now_ns() : 0;
  if (telemetry) {
    telemetry_jobs_submitted_.fetch_add(1, std::memory_order_relaxed);
    telemetry_tasks_requested_.fetch_add(
        static_cast<std::uint64_t>(splits), std::memory_order_relaxed);
    telemetry_requested_worker_slots_.fetch_add(
        static_cast<std::uint64_t>(selected_threads),
        std::memory_order_relaxed);
    update_atomic_max(telemetry_max_requested_threads_,
                      static_cast<std::uint64_t>(selected_threads));
  }

  // A worker must not wait for another shared pool while it owns an outer job:
  // nested pools can form a lock cycle. Execute nested work on the current
  // host thread with logical worker 0, which is valid even when the nested
  // request asks for fewer workers than the outer one.
  if (active_thread_pool != nullptr) {
    int completed_tasks = 0;
    try {
      for (int task_id = 0; task_id < splits; task_id++) {
        func(range_for_task_context, /*thread_id=*/0, task_id);
        ++completed_tasks;
      }
    } catch (...) {
      if (telemetry) {
        const auto elapsed_ns = steady_clock_now_ns() - submit_started_ns;
        telemetry_jobs_completed_.fetch_add(1, std::memory_order_relaxed);
        telemetry_nested_serial_jobs_.fetch_add(1,
                                                std::memory_order_relaxed);
        telemetry_nested_serial_tasks_.fetch_add(
            static_cast<std::uint64_t>(completed_tasks),
            std::memory_order_relaxed);
        telemetry_tasks_completed_.fetch_add(
            static_cast<std::uint64_t>(completed_tasks),
            std::memory_order_relaxed);
        telemetry_cancelled_jobs_.fetch_add(1, std::memory_order_relaxed);
        telemetry_exception_jobs_.fetch_add(1, std::memory_order_relaxed);
        telemetry_execution_ns_.fetch_add(elapsed_ns,
                                          std::memory_order_relaxed);
        telemetry_submitter_wait_ns_.fetch_add(
            elapsed_ns, std::memory_order_relaxed);
      }
      throw;
    }
    if (telemetry) {
      const auto elapsed_ns = steady_clock_now_ns() - submit_started_ns;
      telemetry_jobs_completed_.fetch_add(1, std::memory_order_relaxed);
      telemetry_nested_serial_jobs_.fetch_add(1, std::memory_order_relaxed);
      telemetry_nested_serial_tasks_.fetch_add(
          static_cast<std::uint64_t>(splits), std::memory_order_relaxed);
      telemetry_tasks_completed_.fetch_add(
          static_cast<std::uint64_t>(splits), std::memory_order_relaxed);
      telemetry_execution_ns_.fetch_add(elapsed_ns,
                                        std::memory_order_relaxed);
      telemetry_submitter_wait_ns_.fetch_add(
          elapsed_ns, std::memory_order_relaxed);
    }
    return;
  }

  Job job;
  job.splits = splits;
  job.desired_num_threads = selected_threads;
  job.range_for_task_context = range_for_task_context;
  job.func = func;
  job.telemetry = telemetry;
  job.submitted_ns = submit_started_ns;

  std::exception_ptr exception;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    TI_ERROR_IF(exiting_, "ThreadPool is shutting down.");
    const bool queued = active_job_ != nullptr || !pending_jobs_.empty();
    pending_jobs_.push_back(&job);
    if (telemetry) {
      if (queued) {
        telemetry_queued_jobs_.fetch_add(1, std::memory_order_relaxed);
      }
      const auto queue_depth = static_cast<std::uint64_t>(
          pending_jobs_.size() + (active_job_ != nullptr ? 1 : 0));
      update_atomic_max(telemetry_max_queue_depth_, queue_depth);
    }
    activate_next_job_locked();
    // A new job can use up to its requested number of already-created
    // workers. Waking all is only a cold submit-side cost; workers that do not
    // obtain a task sleep again under the same mutex.
    worker_cv_.notify_all();
    completion_cv_.wait(lock, [&job] { return job.completed; });
    exception = job.exception;
  }
  if (telemetry) {
    telemetry_submitter_wait_ns_.fetch_add(
        steady_clock_now_ns() - submit_started_ns,
        std::memory_order_relaxed);
  }
  if (exception) {
    std::rethrow_exception(exception);
  }
}

void ThreadPool::activate_next_job_locked() {
  if (active_job_) {
    return;
  }
  while (!pending_jobs_.empty()) {
    auto job = pending_jobs_.front();
    pending_jobs_.pop_front();
    if (!job->completed && !job->cancelled.load(std::memory_order_relaxed)) {
      active_job_ = job;
      if (job->telemetry) {
        job->activated_ns = steady_clock_now_ns();
      }
      return;
    }
  }
}

void ThreadPool::set_telemetry_enabled(bool enabled) noexcept {
  telemetry_enabled_.store(enabled, std::memory_order_relaxed);
}

ThreadPoolStatistics ThreadPool::telemetry_statistics(bool reset) noexcept {
  ThreadPoolStatistics result;
  result.enabled = telemetry_enabled_.load(std::memory_order_relaxed);
  result.jobs_submitted = read_counter(telemetry_jobs_submitted_, reset);
  result.jobs_completed = read_counter(telemetry_jobs_completed_, reset);
  result.queued_jobs = read_counter(telemetry_queued_jobs_, reset);
  result.nested_serial_jobs =
      read_counter(telemetry_nested_serial_jobs_, reset);
  result.tasks_requested = read_counter(telemetry_tasks_requested_, reset);
  result.tasks_completed = read_counter(telemetry_tasks_completed_, reset);
  result.nested_serial_tasks =
      read_counter(telemetry_nested_serial_tasks_, reset);
  result.requested_worker_slots =
      read_counter(telemetry_requested_worker_slots_, reset);
  result.joined_workers = read_counter(telemetry_joined_workers_, reset);
  result.underfilled_jobs = read_counter(telemetry_underfilled_jobs_, reset);
  result.cancelled_jobs = read_counter(telemetry_cancelled_jobs_, reset);
  result.exception_jobs = read_counter(telemetry_exception_jobs_, reset);
  result.queue_wait_ns = read_counter(telemetry_queue_wait_ns_, reset);
  result.execution_ns = read_counter(telemetry_execution_ns_, reset);
  result.submitter_wait_ns =
      read_counter(telemetry_submitter_wait_ns_, reset);
  result.max_queue_depth =
      read_counter(telemetry_max_queue_depth_, reset);
  result.max_requested_threads =
      read_counter(telemetry_max_requested_threads_, reset);
  result.max_joined_workers =
      read_counter(telemetry_max_joined_workers_, reset);
  return result;
}

bool ThreadPool::join_job_locked(Job **job) {
  auto candidate = active_job_;
  if (!candidate || candidate->completed ||
      candidate->cancelled.load(std::memory_order_relaxed) ||
      candidate->next_task.load(std::memory_order_relaxed) >=
          candidate->splits ||
      candidate->joined_workers >= candidate->desired_num_threads) {
    return false;
  }

  int active_workers =
      candidate->active_workers.load(std::memory_order_relaxed);
  while (active_workers >= 0 &&
         !candidate->active_workers.compare_exchange_weak(
             active_workers, active_workers + 1, std::memory_order_acq_rel,
             std::memory_order_relaxed)) {
  }
  if (active_workers < 0) {
    return false;
  }
  candidate->joined_workers++;
  *job = candidate;
  return true;
}

void ThreadPool::target() {
  const int thread_id = next_worker_id_.fetch_add(1, std::memory_order_relaxed);
  while (true) {
    Job *job = nullptr;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      worker_cv_.wait(lock, [this] {
        return exiting_ ||
               (active_job_ &&
                !active_job_->cancelled.load(std::memory_order_relaxed) &&
                active_job_->next_task.load(std::memory_order_relaxed) <
                    active_job_->splits &&
                active_job_->joined_workers <
                    active_job_->desired_num_threads);
      });
      if (exiting_) {
        break;
      }
      if (!join_job_locked(&job)) {
        continue;
      }
      // Let another worker join before this one consumes the remaining atomic
      // chunks. The job stays isolated, but chunk claims no longer serialize
      // through the global pool mutex.
      worker_cv_.notify_one();
    }

    std::exception_ptr exception;
    int completed_tasks = 0;
    try {
      ScopedThreadPoolExecution execution(this);
      if (job->telemetry) {
        while (!job->cancelled.load(std::memory_order_relaxed)) {
          const int task_id =
              job->next_task.fetch_add(1, std::memory_order_relaxed);
          if (task_id >= job->splits) {
            break;
          }
          job->func(job->range_for_task_context, thread_id, task_id);
          ++completed_tasks;
        }
      } else {
        // Keep the default task loop free of telemetry branches and counter
        // updates. Observability must not change steady-state scheduling cost.
        while (!job->cancelled.load(std::memory_order_relaxed)) {
          const int task_id =
              job->next_task.fetch_add(1, std::memory_order_relaxed);
          if (task_id >= job->splits) {
            break;
          }
          job->func(job->range_for_task_context, thread_id, task_id);
        }
      }
    } catch (...) {
      exception = std::current_exception();
      job->cancelled.store(true, std::memory_order_relaxed);
    }

    if (job->telemetry && completed_tasks > 0) {
      // Aggregate once per participating worker, not once per range chunk.
      job->completed_tasks.fetch_add(completed_tasks,
                                     std::memory_order_relaxed);
    }

    if (exception) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!job->exception) {
        job->exception = exception;
      }
    }

    int active_workers = job->active_workers.load(std::memory_order_acquire);
    bool closes_job = false;
    while (true) {
      TI_ASSERT(active_workers > 0);
      const int next_active_workers = active_workers == 1 ? -1
                                                          : active_workers - 1;
      if (job->active_workers.compare_exchange_weak(
              active_workers, next_active_workers, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        closes_job = next_active_workers == -1;
        break;
      }
    }
    if (closes_job) {
      std::lock_guard<std::mutex> lock(mutex_);
      TI_ASSERT(job->active_workers.load(std::memory_order_acquire) == -1);
      const bool exhausted =
          job->next_task.load(std::memory_order_relaxed) >= job->splits;
      TI_ASSERT(job->cancelled.load(std::memory_order_relaxed) || exhausted);
      job->active_workers.store(0, std::memory_order_relaxed);
      record_job_completion(job);
      job->completed = true;
      TI_ASSERT(active_job_ == job);
      active_job_ = nullptr;
      activate_next_job_locked();
      completion_cv_.notify_all();
      // Idle workers already wait on the predicate. Waking every worker
      // after the final chunk competes with the submitter that must resume
      // Python and dominated small range launches. Only a queued successor
      // needs a worker wake-up here.
      if (active_job_ != nullptr) {
        worker_cv_.notify_all();
      }
    }
  }
}

void ThreadPool::record_job_completion(Job *job) noexcept {
  if (!job->telemetry) {
    return;
  }
  const auto completed_ns = steady_clock_now_ns();
  telemetry_jobs_completed_.fetch_add(1, std::memory_order_relaxed);
  telemetry_tasks_completed_.fetch_add(
      static_cast<std::uint64_t>(
          job->completed_tasks.load(std::memory_order_relaxed)),
      std::memory_order_relaxed);
  telemetry_joined_workers_.fetch_add(
      static_cast<std::uint64_t>(job->joined_workers),
      std::memory_order_relaxed);
  const int expected_workers =
      std::min(job->desired_num_threads, job->splits);
  if (job->joined_workers < expected_workers) {
    telemetry_underfilled_jobs_.fetch_add(1, std::memory_order_relaxed);
  }
  if (job->cancelled.load(std::memory_order_relaxed)) {
    telemetry_cancelled_jobs_.fetch_add(1, std::memory_order_relaxed);
  }
  if (job->exception) {
    telemetry_exception_jobs_.fetch_add(1, std::memory_order_relaxed);
  }
  telemetry_queue_wait_ns_.fetch_add(
      job->activated_ns - job->submitted_ns, std::memory_order_relaxed);
  telemetry_execution_ns_.fetch_add(
      completed_ns - job->activated_ns, std::memory_order_relaxed);
  update_atomic_max(telemetry_max_joined_workers_,
                    static_cast<std::uint64_t>(job->joined_workers));
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    exiting_ = true;
  }
  worker_cv_.notify_all();
  for (auto &th : threads_)
    th.join();
}

}  // namespace taichi
