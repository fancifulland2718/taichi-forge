#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "taichi/common/platform_macros.h"

namespace taichi::lang {

enum class PrimitiveWorkspaceBackend : std::uint8_t {
  any = 0,
  cpu = 1,
  cuda = 2,
  vulkan = 3,
};

enum class PrimitiveWorkspaceFamily : std::uint16_t {
  any = 0,
  ordering = 1,
  scan = 2,
  compact = 3,
  reduce = 4,
  check = 5,
  metric = 6,
  indexed = 7,
  scatter_add = 8,
  bucket = 9,
  grouped = 10,
  histogram = 11,
  transform = 12,
  ordering_aux = 13,
  sparse_algebra = 14,
};

struct PrimitiveWorkspaceKey {
  PrimitiveWorkspaceBackend backend{PrimitiveWorkspaceBackend::any};
  PrimitiveWorkspaceFamily family{PrimitiveWorkspaceFamily::any};
  // A backend execution domain. CUDA uses a stream identity; Vulkan may use
  // a queue/replay slot. Zero is the backend's default ordered domain.
  std::uint64_t execution_domain{0};
  // Provider-specific plan/layout discriminator. It must not contain a raw
  // resource address whose lifetime is shorter than the arena entry.
  std::uint64_t variant{0};

  friend bool operator==(const PrimitiveWorkspaceKey &lhs,
                         const PrimitiveWorkspaceKey &rhs) noexcept {
    return lhs.backend == rhs.backend && lhs.family == rhs.family &&
           lhs.execution_domain == rhs.execution_domain &&
           lhs.variant == rhs.variant;
  }
};

struct PrimitiveWorkspaceSnapshot {
  std::uint64_t budget_bytes{0};
  std::uint64_t reserved_bytes{0};
  std::uint64_t in_use_bytes{0};
  std::uint64_t persistent_bytes{0};
  std::uint64_t reclaimable_bytes{0};
  std::uint64_t over_budget_bytes{0};
  std::uint64_t peak_reserved_bytes{0};
  std::uint64_t peak_in_use_bytes{0};
  std::uint64_t entries{0};
  std::uint64_t active_leases{0};
  std::uint64_t acquisitions{0};
  std::uint64_t cache_hits{0};
  std::uint64_t cache_misses{0};
  std::uint64_t growth_events{0};
  std::uint64_t clear_calls{0};
  std::uint64_t cleared_entries{0};
  std::uint64_t trim_calls{0};
  std::uint64_t evictions{0};
  std::uint64_t lock_samples{0};
  std::uint64_t lock_contentions{0};
  std::uint64_t lock_wait_ns{0};
};

// Program-owned metadata and lifetime domain for backend primitive scratch.
//
// The arena deliberately does not allocate backend memory itself. Providers
// keep strongly typed resource objects and report their allocated_bytes().
// The metadata mutex only covers map lookup/retirement. Potentially blocking
// allocation, enqueue, cudaFree/device deallocation and resource destruction
// happen after that mutex has been released.
//
// Reuse across unordered GPU streams is unsafe. Providers must include the
// stream/queue/replay domain in PrimitiveWorkspaceKey::execution_domain, or
// retain a completion-aware lease outside this class. A lease serializes only
// one exact key. It never introduces a device-wide wait.
class TI_DLL_EXPORT PrimitiveWorkspaceArena final {
 private:
  using SizeFunction = std::size_t (*)(const void *);

  struct Entry {
    PrimitiveWorkspaceKey key;
    const void *type_tag{nullptr};
    SizeFunction size_function{nullptr};
    std::shared_ptr<void> resource;
    std::mutex use_mutex;
    std::atomic<bool> retired{false};
    std::atomic<bool> persistent{false};
    std::atomic<std::uint32_t> leases{0};
    std::atomic<std::uint64_t> bytes{0};
    std::atomic<std::uint64_t> last_use{0};
  };

  struct KeyHash {
    std::size_t operator()(const PrimitiveWorkspaceKey &key) const noexcept;
  };

 public:
  template <typename Resource>
  class Lease final {
   public:
    Lease() = default;
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;

    Lease(Lease &&other) noexcept
        : arena_(std::exchange(other.arena_, nullptr)),
          entry_(std::move(other.entry_)),
          resource_(std::exchange(other.resource_, nullptr)),
          lock_(std::move(other.lock_)) {
    }

    Lease &operator=(Lease &&) = delete;

    ~Lease() {
      release();
    }

    Resource *operator->() const noexcept {
      return resource_;
    }

    Resource &operator*() const noexcept {
      return *resource_;
    }

    explicit operator bool() const noexcept {
      return resource_ != nullptr;
    }

    const PrimitiveWorkspaceKey &key() const noexcept {
      return entry_->key;
    }

   private:
    friend class PrimitiveWorkspaceArena;

    Lease(PrimitiveWorkspaceArena *arena,
          std::shared_ptr<Entry> entry,
          Resource *resource,
          std::unique_lock<std::mutex> lock) noexcept
        : arena_(arena),
          entry_(std::move(entry)),
          resource_(resource),
          lock_(std::move(lock)) {
    }

    void release() noexcept {
      if (arena_ == nullptr) {
        return;
      }
      arena_->release_lease_locked(entry_);
      lock_.unlock();
      resource_ = nullptr;
      entry_.reset();
      arena_ = nullptr;
    }

    PrimitiveWorkspaceArena *arena_{nullptr};
    std::shared_ptr<Entry> entry_;
    Resource *resource_{nullptr};
    std::unique_lock<std::mutex> lock_;
  };

  PrimitiveWorkspaceArena() = default;
  PrimitiveWorkspaceArena(const PrimitiveWorkspaceArena &) = delete;
  PrimitiveWorkspaceArena &operator=(const PrimitiveWorkspaceArena &) =
      delete;
  ~PrimitiveWorkspaceArena();

  template <typename Resource, typename Factory>
  Lease<Resource> acquire(const PrimitiveWorkspaceKey &key,
                          Factory factory,
                          bool persistent = false) {
    if (key.backend == PrimitiveWorkspaceBackend::any ||
        key.family == PrimitiveWorkspaceFamily::any) {
      throw std::invalid_argument(
          "Primitive workspace keys require a concrete backend and family");
    }
    const void *expected_type = workspace_type_tag<Resource>();

    for (;;) {
      std::shared_ptr<Entry> entry;
      {
        std::lock_guard<std::mutex> lock(metadata_mutex_);
        const auto found = entries_.find(key);
        if (found != entries_.end()) {
          entry = found->second;
          cache_hits_.fetch_add(1, std::memory_order_relaxed);
        }
      }

      if (!entry) {
        auto created = factory();
        std::shared_ptr<Resource> typed_resource = std::move(created);
        if (!typed_resource) {
          throw std::invalid_argument(
              "Primitive workspace factory returned a null resource");
        }
        auto candidate = std::make_shared<Entry>();
        candidate->key = key;
        candidate->type_tag = expected_type;
        candidate->size_function = [](const void *resource) {
          return static_cast<std::size_t>(
              static_cast<const Resource *>(resource)->allocated_bytes());
        };
        candidate->resource = std::move(typed_resource);
        candidate->persistent.store(persistent, std::memory_order_relaxed);

        std::lock_guard<std::mutex> lock(metadata_mutex_);
        auto [position, inserted] = entries_.emplace(key, candidate);
        if (inserted) {
          entry = std::move(candidate);
          cache_misses_.fetch_add(1, std::memory_order_relaxed);
        } else {
          entry = position->second;
          cache_hits_.fetch_add(1, std::memory_order_relaxed);
        }
      }

      std::unique_lock<std::mutex> use_lock(entry->use_mutex,
                                            std::defer_lock);
      lock_samples_.fetch_add(1, std::memory_order_relaxed);
      const auto wait_started = std::chrono::steady_clock::now();
      if (!use_lock.try_lock()) {
        lock_contentions_.fetch_add(1, std::memory_order_relaxed);
        use_lock.lock();
      }
      const auto wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - wait_started)
                               .count();
      lock_wait_ns_.fetch_add(static_cast<std::uint64_t>(wait_ns),
                              std::memory_order_relaxed);

      if (entry->retired.load(std::memory_order_acquire)) {
        use_lock.unlock();
        continue;
      }
      if (entry->type_tag != expected_type) {
        throw std::logic_error(
            "Primitive workspace key was reused with a different resource "
            "type");
      }
      if (persistent) {
        entry->persistent.store(true, std::memory_order_release);
      }

      const std::uint64_t bytes = entry->bytes.load(std::memory_order_relaxed);
      entry->leases.fetch_add(1, std::memory_order_relaxed);
      entry->last_use.store(
          next_use_.fetch_add(1, std::memory_order_relaxed),
          std::memory_order_relaxed);
      active_leases_.fetch_add(1, std::memory_order_relaxed);
      acquisitions_.fetch_add(1, std::memory_order_relaxed);
      add_and_update_peak(in_use_bytes_, peak_in_use_bytes_, bytes);
      return Lease<Resource>(this, entry,
                             static_cast<Resource *>(entry->resource.get()),
                             std::move(use_lock));
    }
  }

  PrimitiveWorkspaceSnapshot snapshot(
      PrimitiveWorkspaceBackend backend = PrimitiveWorkspaceBackend::any,
      PrimitiveWorkspaceFamily family =
          PrimitiveWorkspaceFamily::any) const noexcept;

  // Callers must establish a backend-safe release boundary before clear/trim
  // when resource destruction may wait for GPU work. Program::finalize() and
  // the Python clear bridge synchronize first. No acquire path invokes these
  // methods implicitly.
  void clear(PrimitiveWorkspaceBackend backend =
                 PrimitiveWorkspaceBackend::any,
             PrimitiveWorkspaceFamily family =
                 PrimitiveWorkspaceFamily::any) noexcept;
  void trim_to_budget() noexcept;

  void set_budget_bytes(std::uint64_t bytes) noexcept {
    budget_bytes_.store(bytes, std::memory_order_release);
  }

  std::uint64_t budget_bytes() const noexcept {
    return budget_bytes_.load(std::memory_order_acquire);
  }

 private:
  template <typename Resource>
  static const void *workspace_type_tag() noexcept {
    static const std::uint8_t tag = 0;
    return &tag;
  }

  static bool matches(const PrimitiveWorkspaceKey &key,
                      PrimitiveWorkspaceBackend backend,
                      PrimitiveWorkspaceFamily family) noexcept;
  static void update_peak(std::atomic<std::uint64_t> &peak,
                          std::uint64_t value) noexcept;
  static void add_and_update_peak(std::atomic<std::uint64_t> &value,
                                  std::atomic<std::uint64_t> &peak,
                                  std::uint64_t amount) noexcept;
  void refresh_entry_bytes_locked(const std::shared_ptr<Entry> &entry) noexcept;
  void release_lease_locked(const std::shared_ptr<Entry> &entry) noexcept;
  void release_retired_entry(const std::shared_ptr<Entry> &entry) noexcept;

  mutable std::mutex metadata_mutex_;
  std::unordered_map<PrimitiveWorkspaceKey, std::shared_ptr<Entry>, KeyHash>
      entries_;
  std::atomic<std::uint64_t> budget_bytes_{0};
  std::atomic<std::uint64_t> reserved_bytes_{0};
  std::atomic<std::uint64_t> in_use_bytes_{0};
  std::atomic<std::uint64_t> peak_reserved_bytes_{0};
  std::atomic<std::uint64_t> peak_in_use_bytes_{0};
  std::atomic<std::uint64_t> active_leases_{0};
  std::atomic<std::uint64_t> acquisitions_{0};
  std::atomic<std::uint64_t> cache_hits_{0};
  std::atomic<std::uint64_t> cache_misses_{0};
  std::atomic<std::uint64_t> growth_events_{0};
  std::atomic<std::uint64_t> clear_calls_{0};
  std::atomic<std::uint64_t> cleared_entries_{0};
  std::atomic<std::uint64_t> trim_calls_{0};
  std::atomic<std::uint64_t> evictions_{0};
  std::atomic<std::uint64_t> lock_samples_{0};
  std::atomic<std::uint64_t> lock_contentions_{0};
  std::atomic<std::uint64_t> lock_wait_ns_{0};
  std::atomic<std::uint64_t> next_use_{1};
};

}  // namespace taichi::lang
