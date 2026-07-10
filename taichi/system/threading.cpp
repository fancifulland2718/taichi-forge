/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/system/threading.h"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <memory>
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

  auto job = std::make_shared<Job>();
  job->splits = splits;
  job->desired_num_threads =
      std::clamp(desired_num_threads, 1, max_num_threads_);
  job->range_for_task_context = range_for_task_context;
  job->func = func;

  std::exception_ptr exception;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    TI_ERROR_IF(exiting_, "ThreadPool is shutting down.");
    pending_jobs_.push_back(job);
    activate_next_job_locked();
    // A new job can use up to its requested number of already-created
    // workers. Waking all is only a cold submit-side cost; workers that do not
    // obtain a task sleep again under the same mutex.
    worker_cv_.notify_all();
    completion_cv_.wait(lock, [&job] { return job->completed; });
    exception = job->exception;
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
    if (!job->completed && !job->cancelled) {
      active_job_ = std::move(job);
      return;
    }
  }
}

bool ThreadPool::take_task_locked(std::shared_ptr<Job> *job, int *task_id) {
  auto candidate = active_job_;
  if (!candidate || candidate->completed || candidate->cancelled ||
      candidate->next_task >= candidate->splits ||
      candidate->active_workers >= candidate->desired_num_threads) {
    return false;
  }

  *task_id = candidate->next_task++;
  candidate->active_workers++;
  *job = std::move(candidate);
  return true;
}

void ThreadPool::target() {
  const int thread_id = next_worker_id_.fetch_add(1, std::memory_order_relaxed);
  while (true) {
    std::shared_ptr<Job> job;
    int task_id = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      worker_cv_.wait(lock, [this] {
        return exiting_ ||
               (active_job_ && !active_job_->cancelled &&
                active_job_->next_task < active_job_->splits &&
                active_job_->active_workers <
                    active_job_->desired_num_threads);
      });
      if (exiting_) {
        break;
      }
      if (!take_task_locked(&job, &task_id)) {
        continue;
      }
      // The active job still has another slot in its requested parallelism
      // budget. Wake one peer so it does not silently degrade to one worker.
      worker_cv_.notify_one();
    }

    std::exception_ptr exception;
    try {
      ScopedThreadPoolExecution execution(this);
      job->func(job->range_for_task_context, thread_id, task_id);
    } catch (...) {
      exception = std::current_exception();
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      TI_ASSERT(job->active_workers > 0);
      job->active_workers--;
      if (exception && !job->exception) {
        job->exception = exception;
        job->cancelled = true;
      }
      if ((job->cancelled && job->active_workers == 0) ||
          (!job->cancelled && job->next_task == job->splits &&
           job->active_workers == 0)) {
        job->completed = true;
        TI_ASSERT(active_job_ == job);
        active_job_.reset();
        activate_next_job_locked();
        completion_cv_.notify_all();
        worker_cv_.notify_all();
      } else {
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
