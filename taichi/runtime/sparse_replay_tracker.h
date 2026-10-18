#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace taichi::lang {

// Tracks interference with host sparse-list caches, not resource ownership.
// The existing Graph/module owners retain these leases along with their trees.
// One tracker belongs to one launcher; IDs are physical runtime tree IDs.
// Acquiring/releasing a lease is cold. A warm sparse launch only needs an epoch
// comparison; tree intersections are recomputed when the epoch changes.
class SparseReplayTracker {
  struct State {
    std::mutex mutex;
    std::map<int, std::size_t> references;
    std::atomic<std::uint64_t> epoch{0};
  };

  struct Lease {
    std::shared_ptr<State> state;
    std::vector<int> trees;
    bool registered{false};

    ~Lease() {
      if (!registered)
        return;
      std::lock_guard<std::mutex> lock(state->mutex);
      for (int id : trees) {
        auto it = state->references.find(id);
        if (--it->second == 0)
          state->references.erase(it);
      }
      state->epoch.fetch_add(1, std::memory_order_release);
    }
  };

 public:
  struct Snapshot {
    std::uint64_t epoch;
    bool intersects;
  };

  SparseReplayTracker() : state_(std::make_shared<State>()) {
  }
  SparseReplayTracker(const SparseReplayTracker &) = delete;
  SparseReplayTracker &operator=(const SparseReplayTracker &) = delete;

  std::uint64_t epoch() const {
    return state_->epoch.load(std::memory_order_acquire);
  }

  Snapshot snapshot(const std::vector<int> &trees) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return {state_->epoch.load(std::memory_order_relaxed),
            std::any_of(trees.begin(), trees.end(), [&](int id) {
              return state_->references.count(id) != 0;
            })};
  }

  std::shared_ptr<void> acquire(std::vector<int> trees) {
    if (trees.empty())
      return {};
    std::sort(trees.begin(), trees.end());
    if (trees.front() < 0)
      throw std::invalid_argument("Sparse replay requires physical tree IDs");
    trees.erase(std::unique(trees.begin(), trees.end()), trees.end());
    auto lease = std::make_shared<Lease>();
    lease->state = state_;
    lease->trees = std::move(trees);
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::size_t acquired = 0;
    try {
      for (int id : lease->trees) {
        ++state_->references[id];
        ++acquired;
      }
    } catch (...) {
      for (std::size_t i = 0; i < acquired; ++i) {
        auto it = state_->references.find(lease->trees[i]);
        if (--it->second == 0)
          state_->references.erase(it);
      }
      throw;
    }
    lease->registered = true;
    state_->epoch.fetch_add(1, std::memory_order_release);
    return lease;
  }

 private:
  std::shared_ptr<State> state_;
};

}  // namespace taichi::lang
