#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <vector>
#include "taichi/program/runtime_completion.h"
#ifdef _WIN32
#include <windows.h>
#endif

namespace taichi::lang {

// A frozen collection of exact prefixes, independent of their runtime owners.
// The waiting caller polls backend-neutral completion state between cancellable
// sleeps. No worker per ticket, submission lock, stream callback or GPU work is
// added. poll_interval is an explicit observation latency/CPU cost policy.
class CompletionWaitSet {
 public:
  enum class Status { ready = 0, timeout = 1, cancelled = 2 };
  struct Result {
    Status status;
    std::size_t index{0};  // Meaningful only for ready.
  };
  using Duration = std::chrono::nanoseconds;

  CompletionWaitSet(std::vector<RuntimeCompletion> completions,
                    Duration poll_interval)
      : completions_(std::move(completions)), interval_(poll_interval) {
    if (completions_.empty())
      throw std::invalid_argument("Completion wait set must not be empty");
    if (interval_.count() <= 0)
      throw std::invalid_argument("Completion poll interval must be positive");
    for (const auto &completion : completions_)
      if (!completion.valid())
        throw std::invalid_argument("Completion wait set contains an invalid token");
#ifdef _WIN32
    cancellation_ = std::make_unique<WindowsHandle>(false);
#endif
  }

  // timeout == -1 means unbounded. Zero performs a single nonblocking scan.
  // Cancellation is terminal and never cancels submitted GPU work. Native
  // failures remain failures; they are never returned as timeouts/readiness.
  Result wait(Duration timeout) const {
    if (timeout.count() < -1)
      throw std::invalid_argument("Invalid completion wait timeout");
    const auto deadline = timeout.count() == -1
                              ? Clock::time_point::max()
                              : after(Clock::now(), timeout);
#ifdef _WIN32
    // One timer per active waiter, lazily created only if the first scan is
    // pending. Never one thread/handle per ticket or a process-wide timer tweak.
    std::unique_ptr<WindowsHandle> timer;
#endif
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cancelled_)
          return {Status::cancelled};
      }
      for (std::size_t index = 0; index < completions_.size(); ++index) {
        if (completions_[index].done()) {
          std::lock_guard<std::mutex> lock(mutex_);
          return cancelled_ ? Result{Status::cancelled}
                            : Result{Status::ready, index};
        }
      }
      std::unique_lock<std::mutex> lock(mutex_);
      if (cancelled_)
        return {Status::cancelled};
      const auto now = Clock::now();
      if (now >= deadline)
        return {Status::timeout};
      // Readiness racing this sleep is observed on the next scan; cancellation
      // uses the predicate under the same mutex and cannot lose its wakeup.
      const auto wake = std::min(deadline, after(now, interval_));
#ifdef _WIN32
      lock.unlock();
      if (!timer)
        timer = std::make_unique<WindowsHandle>(true);
      const auto remaining = wake - Clock::now();
      if (remaining <= Clock::duration::zero())
        continue;
      const auto nanoseconds = std::chrono::duration_cast<Duration>(remaining).count();
      LARGE_INTEGER due;
      // Win32 relative timers use negative 100 ns units. Round up without
      // overflowing a caller-provided int64 nanosecond interval.
      due.QuadPart = -(nanoseconds / 100 + (nanoseconds % 100 != 0));
      if (!SetWaitableTimer(timer->value, &due, 0, nullptr, nullptr, FALSE))
        throw std::system_error(GetLastError(), std::system_category(), "Arm completion wait timer");
      HANDLE objects[] = {cancellation_->value, timer->value};
      const auto observed = WaitForMultipleObjects(2, objects, FALSE, INFINITE);
      if (observed != WAIT_OBJECT_0 && observed != WAIT_OBJECT_0 + 1)
        throw std::system_error(GetLastError(), std::system_category(), "Wait completion set");
#else
      changed_.wait_until(lock, wake, [&] { return cancelled_; });
#endif
    }
  }

  void cancel() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = true;
    }
#ifdef _WIN32
    SetEvent(cancellation_->value);
#else
    changed_.notify_all();
#endif
  }

 private:
  using Clock = std::chrono::steady_clock;
  static Clock::time_point after(Clock::time_point now, Duration duration) {
    const auto remaining = Clock::time_point::max() - now;
    return duration >= remaining ? Clock::time_point::max() : now + duration;
  }
  const std::vector<RuntimeCompletion> completions_;
  const Duration interval_;
  mutable std::mutex mutex_;
#ifdef _WIN32
  struct WindowsHandle {
    HANDLE value;
    explicit WindowsHandle(bool timer) {
      if (timer) {
        value = CreateWaitableTimerExW(nullptr, nullptr,
                                       CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                       TIMER_MODIFY_STATE | SYNCHRONIZE);
        if (!value && GetLastError() == ERROR_INVALID_PARAMETER)
          value = CreateWaitableTimerExW(nullptr, nullptr, 0,
                                         TIMER_MODIFY_STATE | SYNCHRONIZE);
      } else {
        value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      }
      if (!value)
        throw std::system_error(GetLastError(), std::system_category(), "Create completion wait object");
    }
    ~WindowsHandle() { CloseHandle(value); }
  };
  std::unique_ptr<WindowsHandle> cancellation_;
#else
  mutable std::condition_variable changed_;
#endif
  bool cancelled_{false};
};

}  // namespace taichi::lang
