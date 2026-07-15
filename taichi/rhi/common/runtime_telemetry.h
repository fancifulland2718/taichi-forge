#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace taichi::lang {

// Sampled host-lock telemetry keeps the ordinary acquisition path bounded to
// one TLS increment, one branch and the original lock operation. Wait time is
// the measured total for contended samples, not an extrapolation.
template <typename Mutex>
class SampledLockTelemetry final {
 public:
  struct Snapshot {
    std::uint64_t sampled_acquisitions{0};
    std::uint64_t contended_acquisitions{0};
    std::uint64_t sampled_wait_ns{0};
  };

  static constexpr std::uint32_t kSamplingPeriod = 64;

  std::unique_lock<Mutex> acquire(Mutex &mutex) {
    if ((++sampling_tick_ & (kSamplingPeriod - 1)) != 0) {
      return std::unique_lock<Mutex>(mutex);
    }

    if (mutex.try_lock()) {
      sampled_acquisitions_.fetch_add(1, std::memory_order_relaxed);
      return std::unique_lock<Mutex>(mutex, std::adopt_lock);
    }

    const auto started = std::chrono::steady_clock::now();
    mutex.lock();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    sampled_acquisitions_.fetch_add(1, std::memory_order_relaxed);
    contended_acquisitions_.fetch_add(1, std::memory_order_relaxed);
    sampled_wait_ns_.fetch_add(static_cast<std::uint64_t>(elapsed.count()),
                               std::memory_order_relaxed);
    return std::unique_lock<Mutex>(mutex, std::adopt_lock);
  }

  Snapshot snapshot() const noexcept {
    return {
        sampled_acquisitions_.load(std::memory_order_relaxed),
        contended_acquisitions_.load(std::memory_order_relaxed),
        sampled_wait_ns_.load(std::memory_order_relaxed),
    };
  }

 private:
  static_assert((kSamplingPeriod & (kSamplingPeriod - 1)) == 0);
  inline static thread_local std::uint64_t sampling_tick_{0};
  std::atomic<std::uint64_t> sampled_acquisitions_{0};
  std::atomic<std::uint64_t> contended_acquisitions_{0};
  std::atomic<std::uint64_t> sampled_wait_ns_{0};
};

class BackendWaitTelemetry final {
 public:
  struct Snapshot {
    std::uint64_t waits{0};
    std::uint64_t wait_ns{0};
  };

  void record(std::uint64_t wait_ns) noexcept {
    waits_.fetch_add(1, std::memory_order_relaxed);
    wait_ns_.fetch_add(wait_ns, std::memory_order_relaxed);
  }

  Snapshot snapshot() const noexcept {
    return {
        waits_.load(std::memory_order_relaxed),
        wait_ns_.load(std::memory_order_relaxed),
    };
  }

 private:
  std::atomic<std::uint64_t> waits_{0};
  std::atomic<std::uint64_t> wait_ns_{0};
};

class ScopedBackendWaitTelemetry final {
 public:
  explicit ScopedBackendWaitTelemetry(BackendWaitTelemetry *telemetry) noexcept
      : telemetry_(telemetry),
        started_(telemetry ? std::chrono::steady_clock::now()
                           : std::chrono::steady_clock::time_point{}) {
  }

  ~ScopedBackendWaitTelemetry() {
    if (!telemetry_) {
      return;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started_);
    telemetry_->record(static_cast<std::uint64_t>(elapsed.count()));
  }

  ScopedBackendWaitTelemetry(const ScopedBackendWaitTelemetry &) = delete;
  ScopedBackendWaitTelemetry &operator=(const ScopedBackendWaitTelemetry &) =
      delete;

 private:
  BackendWaitTelemetry *telemetry_;
  std::chrono::steady_clock::time_point started_;
};

}  // namespace taichi::lang
