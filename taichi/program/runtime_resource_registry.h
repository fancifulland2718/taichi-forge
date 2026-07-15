/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace taichi::lang {

// Copyable, registry-domain-qualified identity captured by launch contexts.
// Keeping this type independent of Resource lets a context validate the exact
// high-level generation without storing a raw pointer as its identity.
struct RuntimeResourceHandle {
  using Kind = std::uint32_t;

  std::uint64_t domain{0};
  Kind kind{0};
  std::uint32_t index{(std::numeric_limits<std::uint32_t>::max)()};
  std::uint32_t generation{0};

  explicit operator bool() const noexcept {
    return domain != 0 && kind != 0 &&
           index != (std::numeric_limits<std::uint32_t>::max)() &&
           generation != 0;
  }

  friend bool operator==(const RuntimeResourceHandle &lhs,
                         const RuntimeResourceHandle &rhs) noexcept {
    return lhs.domain == rhs.domain && lhs.kind == rhs.kind &&
           lhs.index == rhs.index && lhs.generation == rhs.generation;
  }

  friend bool operator!=(const RuntimeResourceHandle &lhs,
                         const RuntimeResourceHandle &rhs) noexcept {
    return !(lhs == rhs);
  }
};

template <typename Resource>
struct RuntimeResourceNoopFinalizer {
  void operator()(Resource &) const noexcept {
  }
};

// Backend-neutral ownership state machine for high-level runtime resources.
//
// A handle is qualified by registry domain, kind, slot and generation. A raw
// object address is never an identity. Retiring an entry rejects new
// handle-based acquire operations, while an existing lease (or ownership
// cloned from it) keeps its resource alive. Resource finalization and
// destruction always happen after the registry mutex has been released.
//
// Finalizer may report teardown failures by throwing; the registry catches the
// exception after committing the entry to kReleased. Resource destructors must
// remain noexcept. Finalizer can run on the thread dropping the last lease and
// therefore must synchronize any shared state it owns.
//
// The kind order passed to finalize applies to entries whose leases are already
// drained. A delayed lease must also retain every resource dependency it uses;
// the registry deliberately does not infer or retain an arbitrary dependency
// graph.
template <typename Resource,
          typename Finalizer = RuntimeResourceNoopFinalizer<Resource>>
class RuntimeResourceRegistry {
  static_assert(std::is_nothrow_destructible<Resource>::value);

 public:
  using Kind = std::uint32_t;
  using Handle = RuntimeResourceHandle;

  enum class Result {
    kSuccess,
    kInvalidArgument,
    kInvalidHandle,
    kClosed,
    kOutOfMemory,
    kResourceError,
  };

  enum class State {
    kLive,
    kRetiring,
    kReleased,
  };

  struct Stats {
    std::size_t slots{0};
    std::size_t live{0};
    std::size_t retiring{0};
    std::size_t released{0};
    std::uint64_t leases{0};
    std::uint64_t created_total{0};
    std::uint64_t retired_total{0};
    std::uint64_t released_total{0};
    std::uint64_t release_errors{0};
    bool closed{false};
  };

 private:
  static constexpr std::uint32_t kInvalidIndex =
      (std::numeric_limits<std::uint32_t>::max)();

  struct Slot {
    std::uint32_t generation{1};
    Kind kind{0};
    std::atomic<State> state{State::kReleased};
    std::atomic<std::uint32_t> lease_count{0};
    std::unique_ptr<Resource> resource;
    std::uint32_t next_free{kInvalidIndex};
  };

  struct Control {
    explicit Control(std::uint64_t domain, Finalizer finalizer)
        : domain(domain), finalizer(std::move(finalizer)) {
    }

    ~Control() {
      for (auto &slot : slots) {
        if (slot && slot->resource) {
          destroy_resource(std::move(slot->resource));
        }
      }
    }

    std::pair<Result, Handle> insert(Kind kind,
                                     std::unique_ptr<Resource> resource) {
      if (domain == 0 || kind == 0 || !resource) {
        return {Result::kInvalidArgument, {}};
      }

      try {
        std::lock_guard<std::mutex> lock(mutex);
        if (closed.load(std::memory_order_acquire)) {
          return {Result::kClosed, {}};
        }

        std::uint32_t index = kInvalidIndex;
        Slot *slot = nullptr;
        if (free_head != kInvalidIndex) {
          index = free_head;
          slot = slots[index].get();
          free_head = slot->next_free;
          slot->next_free = kInvalidIndex;
          ++slot->generation;
        } else {
          if (slots.size() >= kInvalidIndex) {
            return {Result::kOutOfMemory, {}};
          }
          if (slots.size() == slots.capacity()) {
            const std::size_t old_capacity = slots.capacity();
            const std::size_t new_capacity =
                old_capacity == 0
                    ? 8
                    : old_capacity > kInvalidIndex / 2
                          ? static_cast<std::size_t>(kInvalidIndex)
                          : old_capacity * 2;
            slots.reserve(new_capacity);
          }
          auto new_slot = std::make_unique<Slot>();
          index = static_cast<std::uint32_t>(slots.size());
          slot = new_slot.get();
          slots.push_back(std::move(new_slot));
        }

        slot->kind = kind;
        slot->resource = std::move(resource);
        slot->lease_count.store(0, std::memory_order_relaxed);
        slot->state.store(State::kLive, std::memory_order_release);
        ++created_total;
        return {Result::kSuccess,
                Handle{domain, kind, index, slot->generation}};
      } catch (const std::bad_alloc &) {
        return {Result::kOutOfMemory, {}};
      } catch (...) {
        return {Result::kResourceError, {}};
      }
    }

    Result retire(Handle handle) {
      std::unique_ptr<Resource> retired;
      {
        std::lock_guard<std::mutex> lock(mutex);
        Slot *slot = find_live_slot_locked(handle);
        if (!slot) {
          return Result::kInvalidHandle;
        }
        slot->state.store(State::kRetiring, std::memory_order_release);
        ++retired_total;
        retired = collect_ready_locked(handle.index, slot);
      }
      destroy_resource(std::move(retired));
      return Result::kSuccess;
    }

    void finalize(std::initializer_list<Kind> kind_order) noexcept {
      std::size_t slot_count = 0;
      {
        std::lock_guard<std::mutex> lock(mutex);
        closed.store(true, std::memory_order_release);
        slot_count = slots.size();
        for (auto &slot : slots) {
          if (slot &&
              slot->state.load(std::memory_order_relaxed) == State::kLive) {
            slot->state.store(State::kRetiring, std::memory_order_release);
            ++retired_total;
          }
        }
      }

      for (Kind kind : kind_order) {
        for (std::size_t i = slot_count; i > 0; --i) {
          destroy_ready_at(static_cast<std::uint32_t>(i - 1), kind, true);
        }
      }
      for (std::size_t i = slot_count; i > 0; --i) {
        destroy_ready_at(static_cast<std::uint32_t>(i - 1), 0, false);
      }
    }

    void release_lease(Slot *slot,
                       std::uint32_t index,
                       std::uint32_t generation) noexcept {
      const std::uint32_t previous =
          slot->lease_count.fetch_sub(1, std::memory_order_acq_rel);
      if (previous == 0) {
        std::terminate();
      }
      if (previous != 1 ||
          slot->state.load(std::memory_order_acquire) != State::kRetiring) {
        return;
      }

      std::unique_ptr<Resource> retired;
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (index < slots.size() && slots[index].get() == slot &&
            slot->generation == generation) {
          retired = collect_ready_locked(index, slot);
        }
      }
      destroy_resource(std::move(retired));
    }

    std::optional<State> state(Handle handle) const {
      std::lock_guard<std::mutex> lock(mutex);
      Slot *slot = find_slot_locked(handle);
      if (!slot) {
        return std::nullopt;
      }
      return slot->state.load(std::memory_order_acquire);
    }

    Stats stats() const {
      std::lock_guard<std::mutex> lock(mutex);
      Stats result;
      result.slots = slots.size();
      result.created_total = created_total;
      result.retired_total = retired_total;
      result.released_total = released_total;
      result.release_errors =
          release_errors.load(std::memory_order_relaxed);
      result.closed = closed.load(std::memory_order_acquire);
      for (const auto &slot : slots) {
        if (!slot) {
          continue;
        }
        result.leases +=
            slot->lease_count.load(std::memory_order_acquire);
        switch (slot->state.load(std::memory_order_acquire)) {
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

    Slot *find_slot_locked(Handle handle) const noexcept {
      if (!handle || handle.domain != domain || handle.index >= slots.size()) {
        return nullptr;
      }
      Slot *slot = slots[handle.index].get();
      if (!slot || slot->generation != handle.generation ||
          slot->kind != handle.kind) {
        return nullptr;
      }
      return slot;
    }

    Slot *find_live_slot_locked(Handle handle) const noexcept {
      Slot *slot = find_slot_locked(handle);
      if (!slot || !slot->resource ||
          slot->state.load(std::memory_order_acquire) != State::kLive) {
        return nullptr;
      }
      return slot;
    }

    std::unique_ptr<Resource> collect_ready_locked(std::uint32_t index,
                                                   Slot *slot) noexcept {
      if (!slot || !slot->resource ||
          slot->state.load(std::memory_order_acquire) != State::kRetiring ||
          slot->lease_count.load(std::memory_order_acquire) != 0) {
        return nullptr;
      }

      auto resource = std::move(slot->resource);
      slot->state.store(State::kReleased, std::memory_order_release);
      ++released_total;
      if (slot->generation !=
          (std::numeric_limits<std::uint32_t>::max)()) {
        slot->next_free = free_head;
        free_head = index;
      }
      return resource;
    }

    void destroy_ready_at(std::uint32_t index,
                          Kind kind,
                          bool match_kind) noexcept {
      std::unique_ptr<Resource> retired;
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (index >= slots.size()) {
          return;
        }
        Slot *slot = slots[index].get();
        if (!slot || (match_kind && slot->kind != kind)) {
          return;
        }
        retired = collect_ready_locked(index, slot);
      }
      destroy_resource(std::move(retired));
    }

    void destroy_resource(std::unique_ptr<Resource> resource) noexcept {
      if (!resource) {
        return;
      }
      try {
        finalizer(*resource);
      } catch (...) {
        release_errors.fetch_add(1, std::memory_order_relaxed);
      }
      resource.reset();
    }

    const std::uint64_t domain;
    Finalizer finalizer;
    mutable std::mutex mutex;
    std::vector<std::unique_ptr<Slot>> slots;
    std::uint32_t free_head{kInvalidIndex};
    std::uint64_t created_total{0};
    std::uint64_t retired_total{0};
    std::uint64_t released_total{0};
    std::atomic<std::uint64_t> release_errors{0};
    std::atomic<bool> closed{false};
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
        : control_(std::move(other.control_)),
          slot_(std::exchange(other.slot_, nullptr)),
          resource_(std::exchange(other.resource_, nullptr)),
          handle_(std::exchange(other.handle_, Handle{})) {
    }
    Lease &operator=(Lease &&other) noexcept {
      if (this != &other) {
        reset();
        control_ = std::move(other.control_);
        slot_ = std::exchange(other.slot_, nullptr);
        resource_ = std::exchange(other.resource_, nullptr);
        handle_ = std::exchange(other.handle_, Handle{});
      }
      return *this;
    }

    explicit operator bool() const noexcept {
      return resource_ != nullptr;
    }

    Resource *get() const noexcept {
      return resource_;
    }

    Resource &operator*() const noexcept {
      return *resource_;
    }

    Resource *operator->() const noexcept {
      return resource_;
    }

    Handle handle() const noexcept {
      return handle_;
    }

    // Subdivide ownership that this Lease already holds without returning to
    // the registry lookup mutex. This remains valid after retire, unlike
    // handle-based acquire(). Access to the source Lease object itself must be
    // externally synchronized against reset/move/destruction.
    Lease clone() const noexcept {
      if (!control_ || !slot_ || !resource_) {
        return {};
      }
      std::uint32_t leases =
          slot_->lease_count.load(std::memory_order_relaxed);
      do {
        if (leases == 0 ||
            leases == (std::numeric_limits<std::uint32_t>::max)()) {
          return {};
        }
      } while (!slot_->lease_count.compare_exchange_weak(
          leases, leases + 1, std::memory_order_acq_rel,
          std::memory_order_relaxed));
      return Lease(control_, slot_, resource_, handle_);
    }

    void reset() noexcept {
      auto control = std::move(control_);
      Slot *slot = std::exchange(slot_, nullptr);
      resource_ = nullptr;
      const Handle handle = std::exchange(handle_, Handle{});
      if (control && slot) {
        control->release_lease(slot, handle.index, handle.generation);
      }
    }

   private:
    friend class RuntimeResourceRegistry;

    Lease(std::shared_ptr<Control> control,
          Slot *slot,
          Resource *resource,
          Handle handle)
        : control_(std::move(control)),
          slot_(slot),
          resource_(resource),
          handle_(handle) {
    }

    std::shared_ptr<Control> control_;
    Slot *slot_{nullptr};
    Resource *resource_{nullptr};
    Handle handle_;
  };

  explicit RuntimeResourceRegistry(std::uint64_t domain,
                                   Finalizer finalizer = Finalizer{})
      : control_(
            std::make_shared<Control>(domain, std::move(finalizer))) {
  }

  ~RuntimeResourceRegistry() {
    finalize();
  }

  RuntimeResourceRegistry(const RuntimeResourceRegistry &) = delete;
  RuntimeResourceRegistry &operator=(const RuntimeResourceRegistry &) = delete;
  RuntimeResourceRegistry(RuntimeResourceRegistry &&) = delete;
  RuntimeResourceRegistry &operator=(RuntimeResourceRegistry &&) = delete;

  std::uint64_t domain() const noexcept {
    return control_->domain;
  }

  template <typename... Args>
  std::pair<Result, Handle> emplace(Kind kind, Args &&...args) noexcept {
    if (control_->domain == 0 || kind == 0) {
      return {Result::kInvalidArgument, {}};
    }
    if (control_->closed.load(std::memory_order_acquire)) {
      return {Result::kClosed, {}};
    }
    try {
      return insert(kind,
                    std::make_unique<Resource>(std::forward<Args>(args)...));
    } catch (const std::bad_alloc &) {
      return {Result::kOutOfMemory, {}};
    } catch (...) {
      return {Result::kResourceError, {}};
    }
  }

  std::pair<Result, Handle> insert(
      Kind kind,
      std::unique_ptr<Resource> resource) noexcept {
    try {
      return control_->insert(kind, std::move(resource));
    } catch (const std::bad_alloc &) {
      return {Result::kOutOfMemory, {}};
    } catch (...) {
      return {Result::kResourceError, {}};
    }
  }

  std::pair<Result, Lease> acquire(Handle handle) const noexcept {
    try {
      std::shared_ptr<Control> control = control_;
      std::lock_guard<std::mutex> lock(control->mutex);
      Slot *slot = control->find_live_slot_locked(handle);
      if (!slot) {
        return {Result::kInvalidHandle, Lease{}};
      }
      std::uint32_t leases =
          slot->lease_count.load(std::memory_order_relaxed);
      do {
        if (leases == (std::numeric_limits<std::uint32_t>::max)()) {
          return {Result::kOutOfMemory, Lease{}};
        }
      } while (!slot->lease_count.compare_exchange_weak(
          leases, leases + 1, std::memory_order_acq_rel,
          std::memory_order_relaxed));
      return {Result::kSuccess,
              Lease(std::move(control), slot, slot->resource.get(), handle)};
    } catch (const std::bad_alloc &) {
      return {Result::kOutOfMemory, Lease{}};
    } catch (...) {
      return {Result::kResourceError, Lease{}};
    }
  }

  Result retire(Handle handle) noexcept {
    try {
      return control_->retire(handle);
    } catch (...) {
      return Result::kResourceError;
    }
  }

  void finalize(std::initializer_list<Kind> kind_order = {}) noexcept {
    control_->finalize(kind_order);
  }

  std::optional<State> state(Handle handle) const noexcept {
    try {
      return control_->state(handle);
    } catch (...) {
      return std::nullopt;
    }
  }

  Stats stats() const noexcept {
    try {
      return control_->stats();
    } catch (...) {
      Stats result;
      result.closed = true;
      return result;
    }
  }

 private:
  std::shared_ptr<Control> control_;
};

}  // namespace taichi::lang
