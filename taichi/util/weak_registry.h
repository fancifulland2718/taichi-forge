#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace taichi::lang {

// A non-owning inventory with amortized cold-path maintenance. The caller
// provides synchronization; callbacks must not mutate the registry. A sweep
// is paid for by new entries proportional to inventory size, not a fixed
// capacity or a full scan on every insertion. Delayed release of pinned
// owners therefore cannot leave bookkeeping growing with all historical work.
template <typename T>
class WeakRegistry {
 public:
  void add(const std::shared_ptr<T> &value) {
    entries_.push_back(value);
    ++added_since_sweep_;
    if (added_since_sweep_ >= (entries_.size() + 1) / 2) {
      prune();
    }
  }

  void prune() {
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [](const auto &entry) {
                                    return entry.expired();
                                  }),
                   entries_.end());
    added_since_sweep_ = 0;
  }

  template <typename F>
  void for_each_live(F &&visit) {
    prune();
    for (const auto &entry : entries_) {
      if (auto owner = entry.lock()) {
        visit(owner);
      }
    }
  }

  std::size_t size() const noexcept {
    return entries_.size();
  }

 private:
  std::vector<std::weak_ptr<T>> entries_;
  std::size_t added_since_sweep_{0};
};

}  // namespace taichi::lang
