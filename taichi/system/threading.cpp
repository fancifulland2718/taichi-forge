/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/system/threading.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <thread>
#include <vector>

namespace taichi {

namespace {

thread_local ThreadPool *active_thread_pool = nullptr;

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

  // A worker must not wait for another shared pool while it owns an outer job:
  // nested pools can form a lock cycle. Execute nested work on the current
  // host thread with logical worker 0, which is valid even when the nested
  // request asks for fewer workers than the outer one.
  if (active_thread_pool != nullptr) {
    for (int task_id = 0; task_id < splits; task_id++) {
      func(range_for_task_context, /*thread_id=*/0, task_id);
    }
    return;
  }

  Job job;
  job.splits = splits;
  job.desired_num_threads =
      std::clamp(desired_num_threads, 1, max_num_threads_);
  job.range_for_task_context = range_for_task_context;
  job.func = func;

  std::exception_ptr exception;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    TI_ERROR_IF(exiting_, "ThreadPool is shutting down.");
    pending_jobs_.push_back(&job);
    activate_next_job_locked();
    // A new job can use up to its requested number of already-created
    // workers. Waking all is only a cold submit-side cost; workers that do not
    // obtain a task sleep again under the same mutex.
    worker_cv_.notify_all();
    completion_cv_.wait(lock, [&job] { return job.completed; });
    exception = job.exception;
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
      return;
    }
  }
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
    try {
      ScopedThreadPoolExecution execution(this);
      while (!job->cancelled.load(std::memory_order_relaxed)) {
        const int task_id =
            job->next_task.fetch_add(1, std::memory_order_relaxed);
        if (task_id >= job->splits) {
          break;
        }
        job->func(job->range_for_task_context, thread_id, task_id);
      }
    } catch (...) {
      exception = std::current_exception();
      job->cancelled.store(true, std::memory_order_relaxed);
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
