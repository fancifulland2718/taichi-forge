#include "taichi/program/primitive_workspace.h"

#include <algorithm>

namespace taichi::lang {

namespace {

std::uint64_t saturating_subtract(std::uint64_t lhs,
                                  std::uint64_t rhs) noexcept {
  return lhs >= rhs ? lhs - rhs : 0;
}

}  // namespace

std::size_t PrimitiveWorkspaceArena::KeyHash::operator()(
    const PrimitiveWorkspaceKey &key) const noexcept {
  std::uint64_t hash = static_cast<std::uint64_t>(key.backend);
  hash = hash * 0x9e3779b185ebca87ULL +
         static_cast<std::uint64_t>(key.family);
  hash = hash * 0x9e3779b185ebca87ULL + key.execution_domain;
  hash = hash * 0x9e3779b185ebca87ULL + key.variant;
  hash ^= hash >> 33;
  return static_cast<std::size_t>(hash);
}

PrimitiveWorkspaceArena::~PrimitiveWorkspaceArena() {
  clear();
}

bool PrimitiveWorkspaceArena::matches(
    const PrimitiveWorkspaceKey &key,
    PrimitiveWorkspaceBackend backend,
    PrimitiveWorkspaceFamily family) noexcept {
  return (backend == PrimitiveWorkspaceBackend::any ||
          key.backend == backend) &&
         (family == PrimitiveWorkspaceFamily::any || key.family == family);
}

void PrimitiveWorkspaceArena::update_peak(
    std::atomic<std::uint64_t> &peak,
    std::uint64_t value) noexcept {
  std::uint64_t current = peak.load(std::memory_order_relaxed);
  while (current < value &&
         !peak.compare_exchange_weak(current, value,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
  }
}

void PrimitiveWorkspaceArena::add_and_update_peak(
    std::atomic<std::uint64_t> &value,
    std::atomic<std::uint64_t> &peak,
    std::uint64_t amount) noexcept {
  const std::uint64_t updated =
      value.fetch_add(amount, std::memory_order_relaxed) + amount;
  update_peak(peak, updated);
}

void PrimitiveWorkspaceArena::refresh_entry_bytes_locked(
    const std::shared_ptr<Entry> &entry) noexcept {
  if (!entry->resource || !entry->size_function) {
    return;
  }
  std::uint64_t measured = 0;
  try {
    measured = static_cast<std::uint64_t>(
        entry->size_function(entry->resource.get()));
  } catch (...) {
    // Telemetry must never turn a completed backend operation into a failure.
    return;
  }
  const std::uint64_t previous =
      entry->bytes.exchange(measured, std::memory_order_relaxed);
  if (measured > previous) {
    const std::uint64_t growth = measured - previous;
    add_and_update_peak(reserved_bytes_, peak_reserved_bytes_, growth);
    if (entry->leases.load(std::memory_order_relaxed) != 0) {
      add_and_update_peak(in_use_bytes_, peak_in_use_bytes_, growth);
    }
    growth_events_.fetch_add(1, std::memory_order_relaxed);
  } else if (previous > measured) {
    const std::uint64_t shrink = previous - measured;
    reserved_bytes_.fetch_sub(shrink, std::memory_order_relaxed);
    if (entry->leases.load(std::memory_order_relaxed) != 0) {
      in_use_bytes_.fetch_sub(shrink, std::memory_order_relaxed);
    }
  }
}

void PrimitiveWorkspaceArena::release_lease_locked(
    const std::shared_ptr<Entry> &entry) noexcept {
  refresh_entry_bytes_locked(entry);
  const std::uint64_t bytes = entry->bytes.load(std::memory_order_relaxed);
  const std::uint32_t previous =
      entry->leases.fetch_sub(1, std::memory_order_relaxed);
  if (previous == 0) {
    std::terminate();
  }
  active_leases_.fetch_sub(1, std::memory_order_relaxed);
  in_use_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
  entry->last_use.store(next_use_.fetch_add(1, std::memory_order_relaxed),
                        std::memory_order_relaxed);
}

void PrimitiveWorkspaceArena::release_retired_entry(
    const std::shared_ptr<Entry> &entry) noexcept {
  std::unique_lock<std::mutex> lock(entry->use_mutex);
  refresh_entry_bytes_locked(entry);
  const std::uint64_t bytes =
      entry->bytes.exchange(0, std::memory_order_relaxed);
  if (bytes != 0) {
    reserved_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
  }
  entry->resource.reset();
}

PrimitiveWorkspaceSnapshot PrimitiveWorkspaceArena::snapshot(
    PrimitiveWorkspaceBackend backend,
    PrimitiveWorkspaceFamily family) const noexcept {
  PrimitiveWorkspaceSnapshot result;
  result.budget_bytes = budget_bytes_.load(std::memory_order_relaxed);
  result.peak_reserved_bytes =
      peak_reserved_bytes_.load(std::memory_order_relaxed);
  result.peak_in_use_bytes =
      peak_in_use_bytes_.load(std::memory_order_relaxed);
  result.acquisitions = acquisitions_.load(std::memory_order_relaxed);
  result.cache_hits = cache_hits_.load(std::memory_order_relaxed);
  result.cache_misses = cache_misses_.load(std::memory_order_relaxed);
  result.growth_events = growth_events_.load(std::memory_order_relaxed);
  result.clear_calls = clear_calls_.load(std::memory_order_relaxed);
  result.cleared_entries = cleared_entries_.load(std::memory_order_relaxed);
  result.trim_calls = trim_calls_.load(std::memory_order_relaxed);
  result.evictions = evictions_.load(std::memory_order_relaxed);
  result.lock_samples = lock_samples_.load(std::memory_order_relaxed);
  result.lock_contentions =
      lock_contentions_.load(std::memory_order_relaxed);
  result.lock_wait_ns = lock_wait_ns_.load(std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(metadata_mutex_);
  for (const auto &[key, entry] : entries_) {
    if (!matches(key, backend, family) ||
        entry->retired.load(std::memory_order_acquire)) {
      continue;
    }
    const std::uint64_t bytes = entry->bytes.load(std::memory_order_relaxed);
    const std::uint64_t leases =
        entry->leases.load(std::memory_order_relaxed);
    ++result.entries;
    result.active_leases += leases;
    result.reserved_bytes += bytes;
    if (leases != 0) {
      result.in_use_bytes += bytes;
    }
    if (entry->persistent.load(std::memory_order_relaxed)) {
      result.persistent_bytes += bytes;
    } else if (leases == 0) {
      result.reclaimable_bytes += bytes;
    }
  }
  result.over_budget_bytes =
      result.budget_bytes == 0
          ? 0
          : saturating_subtract(result.reserved_bytes, result.budget_bytes);
  return result;
}

void PrimitiveWorkspaceArena::clear(PrimitiveWorkspaceBackend backend,
                                    PrimitiveWorkspaceFamily family) noexcept {
  clear_calls_.fetch_add(1, std::memory_order_relaxed);
  std::vector<std::shared_ptr<Entry>> retired;
  {
    std::lock_guard<std::mutex> lock(metadata_mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
      if (!matches(it->first, backend, family)) {
        ++it;
        continue;
      }
      it->second->retired.store(true, std::memory_order_release);
      retired.push_back(it->second);
      it = entries_.erase(it);
    }
  }
  for (const auto &entry : retired) {
    release_retired_entry(entry);
  }
  cleared_entries_.fetch_add(retired.size(), std::memory_order_relaxed);
}

void PrimitiveWorkspaceArena::trim_to_budget() noexcept {
  trim_calls_.fetch_add(1, std::memory_order_relaxed);
  const std::uint64_t budget = budget_bytes_.load(std::memory_order_acquire);
  if (budget == 0) {
    return;
  }

  for (;;) {
    const auto current = snapshot();
    if (current.reserved_bytes <= budget) {
      return;
    }

    std::shared_ptr<Entry> victim;
    {
      std::lock_guard<std::mutex> lock(metadata_mutex_);
      auto victim_it = entries_.end();
      std::uint64_t oldest = (std::numeric_limits<std::uint64_t>::max)();
      for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        const auto &entry = it->second;
        if (entry->retired.load(std::memory_order_relaxed) ||
            entry->persistent.load(std::memory_order_relaxed) ||
            entry->leases.load(std::memory_order_relaxed) != 0) {
          continue;
        }
        const std::uint64_t last_use =
            entry->last_use.load(std::memory_order_relaxed);
        if (last_use < oldest) {
          oldest = last_use;
          victim_it = it;
        }
      }
      if (victim_it == entries_.end()) {
        return;
      }
      victim = victim_it->second;
      victim->retired.store(true, std::memory_order_release);
      entries_.erase(victim_it);
    }
    release_retired_entry(victim);
    evictions_.fetch_add(1, std::memory_order_relaxed);
  }
}

}  // namespace taichi::lang
