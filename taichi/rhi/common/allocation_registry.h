/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "taichi/common/core.h"
#include "taichi/rhi/public_device.h"

namespace taichi::lang {

// Thread-safe ownership for backend-private allocation records. The public
// DeviceAllocationId remains an opaque uint64_t: its low 32 bits select a slot
// and its high 32 bits are a nonzero generation. A released slot is not reused
// until every Lease that could still access its record has gone away.
//
// Record must expose an immutable `size` member. Resource-specific mutable
// state (for example CUDA mapping state) remains the backend owner's
// responsibility and must be synchronized independently.
template <typename Record>
class AllocationRegistry {
  static_assert(std::is_nothrow_move_constructible<Record>::value,
                "AllocationRegistry records must be noexcept-movable");

 public:
  enum class State {
    kLive,
    kRetiring,
    kReleased,
  };

  struct Stats {
    size_t live{0};
    size_t retiring{0};
    size_t released{0};
  };

 private:
  struct Slot {
    uint32_t generation{1};
    State state{State::kReleased};
    std::atomic<uint32_t> lease_count{0};
    std::optional<Record> record;
    uint32_t next_retiring{std::numeric_limits<uint32_t>::max()};
  };

 public:
  class Lease {
   public:
    Lease() = default;
    ~Lease() {
      reset();
    }
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
    Lease(Lease &&other) noexcept
        : slot_(std::exchange(other.slot_, nullptr)) {
    }
    Lease &operator=(Lease &&other) noexcept {
      if (this != &other) {
        reset();
        slot_ = std::exchange(other.slot_, nullptr);
      }
      return *this;
    }

    explicit operator bool() const {
      return slot_ != nullptr;
    }

    Record *get() const {
      return slot_ ? &*slot_->record : nullptr;
    }

    Record &operator*() const {
      return *slot_->record;
    }

    Record *operator->() const {
      return get();
    }

   private:
    friend class AllocationRegistry;

    explicit Lease(Slot *slot) : slot_(slot) {
    }

    void reset() {
      if (slot_) {
        slot_->lease_count.fetch_sub(1, std::memory_order_release);
        slot_ = nullptr;
      }
    }

    // Leases are deliberately short lived and may not outlive their registry.
    // They only keep a record alive while an RHI operation uses it; device
    // shutdown must join those operations before destroying the registry.
    Slot *slot_{nullptr};
  };

  template <typename... Args>
  std::pair<RhiResult, DeviceAllocationId> emplace(Args &&...args) {
    std::vector<Record> retired;
    std::lock_guard<std::mutex> lock(mutex_);
    collect_retired_locked(retired);

    try {
      const auto reusable_slot = find_released_slot_locked();
      if (reusable_slot.has_value()) {
        const uint32_t index = *reusable_slot;
        auto &slot = slots_[index];
        // A failed record constructor must leave this slot available for a
        // later allocation. Pop it only after construction succeeds.
        slot->record.emplace(std::forward<Args>(args)...);
        slot->generation = next_generation(slot->generation);
        slot->state = State::kLive;
        if (!free_slots_.empty() && free_slots_.back() == index) {
          free_slots_.pop_back();
        }
        return {RhiResult::success, encode(index, slot->generation)};
      }

      const size_t slot_count = slots_.size();
      if (slot_count > std::numeric_limits<uint32_t>::max()) {
        return {RhiResult::out_of_memory, 0};
      }
      auto slot = std::make_unique<Slot>();
      slot->record.emplace(std::forward<Args>(args)...);
      slot->state = State::kLive;
      const auto handle =
          encode(static_cast<uint32_t>(slot_count), slot->generation);
      slots_.push_back(std::move(slot));
      return {RhiResult::success, handle};
    } catch (const std::bad_alloc &) {
      return {RhiResult::out_of_memory, 0};
    } catch (...) {
      return {RhiResult::error, 0};
    }
  }

  std::pair<RhiResult, Lease> acquire(DeviceAllocationId handle,
                                      uint64_t offset = 0,
                                      uint64_t size = 0) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *slot = find_live_slot_locked(handle);
    if (!slot || !is_valid_range(slot->record->size, offset, size)) {
      return {RhiResult::invalid_usage, Lease{}};
    }
    slot->lease_count.fetch_add(1, std::memory_order_relaxed);
    return {RhiResult::success, Lease(slot)};
  }

  RhiResult retire(DeviceAllocationId handle) {
    std::vector<Record> retired;
    std::lock_guard<std::mutex> lock(mutex_);
    auto *slot = find_live_slot_locked(handle);
    if (!slot) {
      return RhiResult::invalid_usage;
    }
    const uint32_t index = decode_index(handle);
    slot->state = State::kRetiring;
    slot->next_retiring = retiring_head_;
    retiring_head_ = index;
    collect_retired_locked(retired);
    return RhiResult::success;
  }

  // Destroys records whose retirement is no longer protected by a Lease and
  // makes their slots eligible for reuse. Backends may call this after a batch
  // of retirements; emplace() also performs collection opportunistically.
  std::vector<Record> collect_retired() {
    std::vector<Record> retired;
    std::lock_guard<std::mutex> lock(mutex_);
    collect_retired_locked(retired);
    return retired;
  }

  std::optional<State> state(DeviceAllocationId handle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto index = decode_index(handle);
    if (index >= slots_.size()) {
      return std::nullopt;
    }
    const auto &slot = slots_[index];
    if (!slot || slot->generation != decode_generation(handle)) {
      return std::nullopt;
    }
    return slot->state;
  }

  Stats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats result;
    for (const auto &slot : slots_) {
      if (!slot) {
        continue;
      }
      switch (slot->state) {
        case State::kLive:
          ++result.live;
          break;
        case State::kRetiring:
          ++result.retiring;
          break;
        case State::kReleased:
          ++result.released;
          break;
      }
    }
    return result;
  }

  void clear() {
    std::vector<Record> retired;
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t index = 0; index < slots_.size(); ++index) {
      auto &slot = slots_[index];
      if (slot && slot->state == State::kLive) {
        slot->state = State::kRetiring;
        slot->next_retiring = retiring_head_;
        retiring_head_ = static_cast<uint32_t>(index);
      }
    }
    collect_retired_locked(retired);
  }

  static bool is_valid_range(uint64_t allocation_size,
                             uint64_t offset,
                             uint64_t size) {
    return offset <= allocation_size && size <= allocation_size - offset;
  }

 private:
  static constexpr uint64_t kIndexMask =
      std::numeric_limits<uint32_t>::max();

  static DeviceAllocationId encode(uint32_t index, uint32_t generation) {
    return (static_cast<DeviceAllocationId>(generation) << 32) | index;
  }

  static uint32_t decode_index(DeviceAllocationId handle) {
    return static_cast<uint32_t>(handle & kIndexMask);
  }

  static uint32_t decode_generation(DeviceAllocationId handle) {
    return static_cast<uint32_t>(handle >> 32);
  }

  static uint32_t next_generation(uint32_t generation) {
    TI_ASSERT(generation < std::numeric_limits<uint32_t>::max());
    return generation + 1;
  }

  Slot *find_live_slot_locked(DeviceAllocationId handle) const {
    const auto index = decode_index(handle);
    if (decode_generation(handle) == 0 || index >= slots_.size()) {
      return nullptr;
    }
    const auto &slot = slots_[index];
    if (!slot || slot->generation != decode_generation(handle) ||
        slot->state != State::kLive || !slot->record) {
      return nullptr;
    }
    return slot.get();
  }

  size_t collect_retired_locked(std::vector<Record> &retired) {
    size_t collected = 0;
    uint32_t *link = &retiring_head_;
    while (*link != kInvalidIndex) {
      const uint32_t index = *link;
      TI_ASSERT(index < slots_.size());
      auto &slot = slots_[index];
      TI_ASSERT(slot && slot->state == State::kRetiring && slot->record);
      // Once a slot starts retiring, no new lease can be acquired. Existing
      // leases increment this counter while holding the registry mutex, so a
      // zero count is sufficient to destroy the record outside their reach.
      if (slot->lease_count.load(std::memory_order_acquire) == 0) {
        const uint32_t next_retiring = slot->next_retiring;
        // Record is noexcept-movable, so allocation is the only operation
        // here that may throw. Keep the entry linked until emplace succeeds.
        retired.emplace_back(std::move(*slot->record));
        *link = next_retiring;
        slot->next_retiring = kInvalidIndex;
        slot->record.reset();
        slot->state = State::kReleased;
        // Do not wrap generation: a handle that may still exist must never
        // become valid again after enough reuse cycles.
        if (slot->generation != std::numeric_limits<uint32_t>::max()) {
          try {
            free_slots_.push_back(static_cast<uint32_t>(index));
          } catch (...) {
            // This only delays slot reuse. The released record is already
            // safe, and emplace() has a linear fallback for this rare OOM
            // path.
          }
        }
        ++collected;
      } else {
        link = &slot->next_retiring;
      }
    }
    return collected;
  }

  std::optional<uint32_t> find_released_slot_locked() const {
    if (!free_slots_.empty()) {
      return free_slots_.back();
    }
    // If recording a free index ran out of host memory, preserve correctness
    // and reuse an already released slot by scanning. This path is cold and
    // only occurs after a metadata-allocation failure.
    for (size_t index = 0; index < slots_.size(); ++index) {
      const auto &slot = slots_[index];
      if (slot && slot->state == State::kReleased && !slot->record &&
          slot->generation != std::numeric_limits<uint32_t>::max()) {
        return static_cast<uint32_t>(index);
      }
    }
    return std::nullopt;
  }

  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::vector<uint32_t> free_slots_;
  static constexpr uint32_t kInvalidIndex =
      std::numeric_limits<uint32_t>::max();
  uint32_t retiring_head_{kInvalidIndex};
};

}  // namespace taichi::lang
