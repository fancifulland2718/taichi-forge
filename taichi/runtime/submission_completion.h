#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "taichi/program/runtime_completion.h"

namespace taichi::lang {

// The backend owner serializes this queue with submissions and retirement.
// User handles can disappear independently: pending completions must still be
// retired while their device exists. This does not lease application results.
class SubmissionCompletionQueue {
 public:
  void retain(const RuntimeCompletion &completion) {
    if (!completion.valid())
      throw std::invalid_argument("Cannot retain an invalid completion");
    if (completion.has_backend_work())
      pending_.push_back(completion);
  }

  // Amortize one old query per new submission, without a backlog/capacity cap.
  // Query before erasing so a first backend error remains tracked for teardown.
  void collect_one() {
    if (!pending_.empty() && pending_.front().done() &&
        !pending_.front().has_backend_work())
      pending_.pop_front();
  }

  void complete_synchronized() noexcept {
    for (const auto &completion : pending_)
      completion.mark_completed();
    pending_.clear();
  }

  // The owner must establish its existing failed-backend retirement boundary
  // before releasing primitives. A mere failed query is not such a boundary.
  void fail_and_release(const std::string &reason) noexcept {
    for (const auto &completion : pending_)
      completion.invalidate_and_release(reason);
    pending_.clear();
  }

  std::size_t size() const noexcept {
    return pending_.size();
  }

 private:
  std::deque<RuntimeCompletion> pending_;
};

// Opaque process-unique tokens are allocated by the C API's existing authority.
// Copy out the completion under this short lock; no device query/wait or final
// release runs under the handle-table lock. Completed handles can outlive the
// backend owner because retirement removes all native primitives from State.
class SubmissionCompletionHandles {
 public:
  using Handle = std::uintptr_t;

  void publish(Handle handle, RuntimeCompletion completion) {
    if (!handle || !completion.valid())
      throw std::invalid_argument("Invalid submission completion handle");
    std::lock_guard<std::mutex> lock(mutex_);
    // Keep the parameter's reference until after the lock is released, even
    // if insertion allocates and then fails or rejects a duplicate handle.
    if (!entries_.try_emplace(handle, completion).second)
      throw std::invalid_argument("Duplicate submission completion handle");
  }

  RuntimeCompletion get(Handle handle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(handle);
    if (found == entries_.end())
      throw std::invalid_argument("Submission completion is closed or unknown");
    return found->second;
  }

  void release(Handle handle) {
    RuntimeCompletion released;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto found = entries_.find(handle);
      if (found == entries_.end())
        throw std::invalid_argument("Submission completion is closed or unknown");
      released = std::move(found->second);
      entries_.erase(found);
    }
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<Handle, RuntimeCompletion> entries_;
};

}  // namespace taichi::lang
