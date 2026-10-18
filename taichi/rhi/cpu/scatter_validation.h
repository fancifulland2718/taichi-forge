#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace taichi::lang::cpu {

// Fixed extents, mutable indices. A prepared owner can keep this storage across
// replays; every call clears and validates it before any destination is
// written. Dense index spaces use byte markers. Sparse spaces use retained
// open-addressed keys instead of allocating a hash node for every index on
// every replay.
class PlainScatterIndexValidator {
 public:
  PlainScatterIndexValidator(std::size_t n, std::size_t dst_items)
      : n_(n), bound_(dst_items) {
    if (!n || !dst_items)
      return;
    // The existing i32 index ABI cannot name larger destination positions.
    const auto addressable = std::min<std::size_t>(
        dst_items, std::size_t{std::numeric_limits<int32_t>::max()} + 1);
    if (n >= std::numeric_limits<std::size_t>::max() / 8 ||
        addressable <= n * 8) {
      dense_.resize(addressable);
    } else {
      // At most n distinct valid keys; keep the table at most half full.
      const auto maximum = sparse_.max_size();
      if (n > maximum / 2)
        throw std::length_error(
            "Scatter validation storage exceeds address space");
      const auto required = n * 2;
      std::size_t capacity = 1;
      while (capacity < required) {
        if (capacity > maximum / 2)
          throw std::length_error(
              "Scatter validation storage exceeds address space");
        capacity *= 2;
      }
      sparse_.resize(capacity);
    }
  }

  std::size_t workspace_bytes() const {
    return dense_.capacity() + sparse_.capacity() * sizeof(uint32_t);
  }

  bool unique(const int32_t *indices) {
    if (!n_ || !bound_)
      return true;
    if (!indices)
      throw std::invalid_argument("Scatter validation requires index storage");
    std::fill(dense_.begin(), dense_.end(), 0);
    std::fill(sparse_.begin(), sparse_.end(), 0);
    for (std::size_t i = 0; i < n_; ++i) {
      const auto index = indices[i];
      if (index < 0 || static_cast<std::size_t>(index) >= bound_)
        continue;
      if (!dense_.empty()) {
        if (dense_[index])
          return false;
        dense_[index] = 1;
      } else {
        // Zero marks an empty bucket; every nonnegative i32 maps to a nonzero
        // u32 key, including INT32_MAX. Mix high bits for strided index spaces.
        const uint32_t key = static_cast<uint32_t>(index) + 1;
        uint32_t hash = key;
        hash ^= hash >> 16;
        hash *= 0x7feb352du;
        hash ^= hash >> 15;
        hash *= 0x846ca68bu;
        hash ^= hash >> 16;
        const auto mask = sparse_.size() - 1;
        std::size_t bucket = hash & mask;
        while (sparse_[bucket] && sparse_[bucket] != key)
          bucket = (bucket + 1) & mask;
        if (sparse_[bucket] == key)
          return false;
        sparse_[bucket] = key;
      }
    }
    return true;
  }

 private:
  std::size_t n_, bound_;
  std::vector<uint8_t> dense_;
  std::vector<uint32_t> sparse_;
};

}  // namespace taichi::lang::cpu
