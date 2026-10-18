#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <vector>

#include "taichi/common/core.h"

namespace taichi::lang {

// One cold-path namespace per backend runtime, shared by JIT layouts and AOT
// modules. A layout claims its ID before compilation, not just allocation.
// The owning Program/module releases the claim after retiring its resources.
class SNodeTreeIdRegistry {
 public:
  bool try_reserve(int id) {
    TI_ASSERT(id >= 0);
    std::lock_guard<std::mutex> lock(mutex_);
    return ids_.insert(id).second;
  }

  bool contains(int id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ids_.count(id) != 0;
  }

  std::vector<int> reserve(std::size_t count,
                           const std::function<bool(int)> &unavailable = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int> result;
    result.reserve(count);
    for (std::int64_t candidate = 0; result.size() < count; ++candidate) {
      TI_ERROR_IF(candidate > std::numeric_limits<int>::max(),
                  "Runtime SNode tree ID space exhausted");
      const int id = static_cast<int>(candidate);
      if (!ids_.count(id) && (!unavailable || !unavailable(id)))
        result.push_back(id);
    }
    insert_locked(result);
    return result;
  }

  void reserve(const std::vector<int> &ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(!std::is_sorted(ids.begin(), ids.end()) ||
                    std::adjacent_find(ids.begin(), ids.end()) != ids.end(),
                "Runtime SNode tree reservations must be sorted and unique");
    for (int id : ids) {
      TI_ERROR_IF(id < 0 || ids_.count(id),
                  "SNode tree id {} conflicts with an active module or JIT "
                  "layout; load this artifact in a separate runtime.",
                  id);
    }
    insert_locked(ids);
  }

  void release(int id) {
    std::lock_guard<std::mutex> lock(mutex_);
    ids_.erase(id);
  }

  void release(const std::vector<int> &ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int id : ids)
      ids_.erase(id);
  }

 private:
  void insert_locked(const std::vector<int> &ids) {
    std::size_t inserted = 0;
    try {
      for (int id : ids) {
        ids_.insert(id);
        ++inserted;
      }
    } catch (...) {
      for (std::size_t i = 0; i < inserted; ++i)
        ids_.erase(ids[i]);
      throw;
    }
  }

  mutable std::mutex mutex_;
  std::set<int> ids_;
};

}  // namespace taichi::lang
