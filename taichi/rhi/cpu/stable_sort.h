#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace taichi::lang::cpu {

// Shared JIT/AOT CPU stable ordering. A prepared owner can retain the primary
// array. std::stable_sort may still allocate its own transient merge buffer,
// just as the original JIT implementation does; bytes() excludes that buffer.
inline uint32_t cpu_sortable_f32_key(float value, int nan_policy) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint32_t kSign = 0x80000000u;
  constexpr uint32_t kAbsMask = 0x7fffffffu;
  constexpr uint32_t kInfBits = 0x7f800000u;
  if (nan_policy == 0 && (bits & kAbsMask) > kInfBits) {
    return 0xffffffffu;
  }
  if (nan_policy == 0 && (bits & kAbsMask) == 0) {
    return kSign;
  }
  return (bits & kSign) ? ~bits : (bits ^ kSign);
}

inline uint64_t cpu_sortable_f64_key(double value, int nan_policy) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  constexpr uint64_t kSign = 0x8000000000000000ull;
  constexpr uint64_t kAbsMask = 0x7fffffffffffffffull;
  constexpr uint64_t kInfBits = 0x7ff0000000000000ull;
  if (nan_policy == 0 && (bits & kAbsMask) > kInfBits) {
    return 0xffffffffffffffffull;
  }
  if (nan_policy == 0 && (bits & kAbsMask) == 0) {
    return kSign;
  }
  return (bits & kSign) ? ~bits : (bits ^ kSign);
}

template <typename KeyT>
bool cpu_sort_key_before(KeyT lhs,
                         KeyT rhs,
                         bool descending,
                         int /*nan_policy*/) {
  if (lhs == rhs) {
    return false;
  }
  return descending ? lhs > rhs : lhs < rhs;
}

template <>
inline bool cpu_sort_key_before<float>(float lhs,
                                float rhs,
                                bool descending,
                                int nan_policy) {
  uint32_t lhs_bits = 0;
  uint32_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  constexpr uint32_t kAbsMask = 0x7fffffffu;
  constexpr uint32_t kInfBits = 0x7f800000u;
  const bool lhs_nan = (lhs_bits & kAbsMask) > kInfBits;
  const bool rhs_nan = (rhs_bits & kAbsMask) > kInfBits;
  if (nan_policy == 0 && (lhs_nan || rhs_nan)) {
    return !lhs_nan && rhs_nan;
  }
  if (nan_policy == 0 && (lhs_bits & kAbsMask) == 0 &&
      (rhs_bits & kAbsMask) == 0) {
    return false;
  }
  const uint32_t lhs_key = cpu_sortable_f32_key(lhs, nan_policy);
  const uint32_t rhs_key = cpu_sortable_f32_key(rhs, nan_policy);
  if (lhs_key == rhs_key) {
    return false;
  }
  return descending ? lhs_key > rhs_key : lhs_key < rhs_key;
}

template <>
inline bool cpu_sort_key_before<double>(double lhs,
                                 double rhs,
                                 bool descending,
                                 int nan_policy) {
  uint64_t lhs_bits = 0;
  uint64_t rhs_bits = 0;
  std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
  std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
  constexpr uint64_t kAbsMask = 0x7fffffffffffffffull;
  constexpr uint64_t kInfBits = 0x7ff0000000000000ull;
  const bool lhs_nan = (lhs_bits & kAbsMask) > kInfBits;
  const bool rhs_nan = (rhs_bits & kAbsMask) > kInfBits;
  if (nan_policy == 0 && (lhs_nan || rhs_nan)) {
    return !lhs_nan && rhs_nan;
  }
  if (nan_policy == 0 && (lhs_bits & kAbsMask) == 0 &&
      (rhs_bits & kAbsMask) == 0) {
    return false;
  }
  const uint64_t lhs_key = cpu_sortable_f64_key(lhs, nan_policy);
  const uint64_t rhs_key = cpu_sortable_f64_key(rhs, nan_policy);
  if (lhs_key == rhs_key) {
    return false;
  }
  return descending ? lhs_key > rhs_key : lhs_key < rhs_key;
}

template <typename KeyT, typename ValueT,
          bool Floating = std::is_floating_point_v<KeyT>>
struct StableSortItem {
  KeyT key;
  ValueT value;
};

template <typename KeyT, typename ValueT>
struct StableSortItem<KeyT, ValueT, true> {
  KeyT key;
  ValueT value;
  std::conditional_t<sizeof(KeyT) == 4, uint32_t, uint64_t> order;
};

template <typename KeyT, typename ValueT>
struct StableSortWorkspace {
  using Item = StableSortItem<KeyT, ValueT>;
  std::vector<Item> items;
  std::vector<KeyT> sorted_keys;

  void prepare(std::size_t n, bool with_values) {
    if (n <= 1) return;
    if (with_values) items.resize(n);
    else sorted_keys.resize(n);
  }
  std::size_t bytes() const {
    return items.size() * sizeof(Item) + sorted_keys.size() * sizeof(KeyT);
  }
};

template <typename KeyT, typename ValueT>
std::size_t cpu_stable_sort_impl(KeyT *keys,
                                 ValueT *values,
                                 std::size_t n,
                                 bool descending,
                                 int nan_policy,
                                 StableSortWorkspace<KeyT, ValueT> *prepared = nullptr) {
  if (n <= 1) {
    return 0;
  }
  StableSortWorkspace<KeyT, ValueT> local;
  auto &workspace = prepared ? *prepared : local;
  workspace.prepare(n, values != nullptr);
  if (values) {
    using Item = typename StableSortWorkspace<KeyT, ValueT>::Item;
    auto &items = workspace.items;
    for (std::size_t i = 0; i < n; ++i) {
      items[i].key = keys[i];
      std::memcpy(&items[i].value, values + i, sizeof(ValueT));
      if constexpr (std::is_floating_point_v<KeyT>) {
        auto &order = items[i].order;
        if constexpr (sizeof(KeyT) == 4) {
          order = cpu_sortable_f32_key(keys[i], nan_policy);
        } else {
          order = cpu_sortable_f64_key(keys[i], nan_policy);
        }
        // Compute NaN/signed-zero ordering once per input, rather than on
        // every comparison. NaNs-last remains last for descending sorts too.
        if (descending &&
            !(nan_policy == 0 &&
              order == std::numeric_limits<std::decay_t<decltype(order)>>::max())) {
          order = ~order;
        }
      }
    }
    std::stable_sort(items.begin(), items.end(), [&](const Item &lhs,
                                                     const Item &rhs) {
      if constexpr (std::is_floating_point_v<KeyT>) {
        return lhs.order < rhs.order;
      } else {
        return cpu_sort_key_before<KeyT>(
            lhs.key, rhs.key, descending, nan_policy);
      }
    });
    for (std::size_t i = 0; i < n; ++i) {
      keys[i] = items[i].key;
      std::memcpy(values + i, &items[i].value, sizeof(ValueT));
    }
    return items.size() * sizeof(Item);
  }

  auto &sorted_keys = workspace.sorted_keys;
  std::copy(keys, keys + n, sorted_keys.begin());
  std::stable_sort(sorted_keys.begin(), sorted_keys.end(), [&](KeyT lhs,
                                                               KeyT rhs) {
    return cpu_sort_key_before<KeyT>(lhs, rhs, descending, nan_policy);
  });
  std::memcpy(keys, sorted_keys.data(), n * sizeof(KeyT));
  return sorted_keys.size() * sizeof(KeyT);
}

}  // namespace taichi::lang::cpu
