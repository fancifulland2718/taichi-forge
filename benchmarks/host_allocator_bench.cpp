#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "taichi/rhi/common/host_memory_pool.h"
#include "taichi/util/environ_config.h"

namespace {

using Clock = std::chrono::steady_clock;
using taichi::lang::HostMemoryPool;
using taichi::lang::HostMemoryPoolStats;

struct TrialResult {
  double ns_per_operation{0.0};
  HostMemoryPoolStats stats;
};

TrialResult run_sequential(std::size_t iterations, bool mixed) {
  HostMemoryPool pool;
  void *warm = pool.allocate(1, 16);
  pool.release(1, warm);

  std::vector<std::pair<void *, std::size_t>> allocations;
  allocations.reserve(iterations);
  const auto begin = Clock::now();
  for (std::size_t index = 0; index < iterations; ++index) {
    const std::size_t size =
        mixed ? 1 + ((index * 1315423911ULL) % 4096) : 32;
    const std::size_t alignment = mixed ? 1ULL << (index % 7) : 16;
    allocations.emplace_back(pool.allocate(size, alignment), size);
  }
  for (const auto &[ptr, size] : allocations) {
    pool.release(size, ptr);
  }
  const auto elapsed = Clock::now() - begin;
  return {
      std::chrono::duration<double, std::nano>(elapsed).count() /
          (2.0 * iterations),
      pool.get_stats(),
  };
}

TrialResult run_concurrent(std::size_t iterations, std::size_t thread_count) {
  HostMemoryPool pool;
  void *warm = pool.allocate(1, 16);
  pool.release(1, warm);
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(thread_count);

  for (std::size_t thread = 0; thread < thread_count; ++thread) {
    workers.emplace_back([&, thread] {
      std::vector<std::pair<void *, std::size_t>> allocations;
      allocations.reserve(iterations);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (std::size_t index = 0; index < iterations; ++index) {
        const std::size_t size = 1 + ((thread * 17 + index) % 64);
        allocations.emplace_back(pool.allocate(size, 16), size);
      }
      for (const auto &[ptr, size] : allocations) {
        pool.release(size, ptr);
      }
    });
  }

  const auto begin = Clock::now();
  start.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  const auto elapsed = Clock::now() - begin;
  const double operations =
      2.0 * static_cast<double>(iterations) * thread_count;
  return {
      std::chrono::duration<double, std::nano>(elapsed).count() / operations,
      pool.get_stats(),
  };
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
  return values[std::min(index, values.size() - 1)];
}

template <typename Workload>
void report(const char *name,
            std::size_t trials,
            std::size_t operation_count,
            Workload &&workload) {
  std::vector<double> samples;
  samples.reserve(trials);
  HostMemoryPoolStats last;
  for (std::size_t trial = 0; trial < trials; ++trial) {
    auto result = workload();
    samples.push_back(result.ns_per_operation);
    last = result.stats;
  }
  std::cout << std::fixed << std::setprecision(3)
            << "workload=" << name << " trials=" << trials
            << " operations_per_trial=" << operation_count
            << " median_ns_per_op=" << percentile(samples, 0.5)
            << " p95_ns_per_op=" << percentile(samples, 0.95)
            << " reserved_bytes=" << last.reserved_bytes
            << " used_bytes=" << last.used_bytes
            << " wasted_bytes=" << last.wasted_bytes
            << " chunks=" << last.unified_chunks << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
  const std::size_t iterations =
      argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 200000;
  const std::size_t trials =
      argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 7;
  const std::size_t threads =
      argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 8;
  if (iterations == 0 || trials == 0 || threads == 0) {
    std::cerr << "iterations, trials, and threads must be non-zero"
              << std::endl;
    return 2;
  }

  std::cout << "policy="
            << (taichi::lang::get_environ_config(
                    "TI_HOST_ALLOCATOR_ADAPTIVE_CHUNKS", 1)
                        ? "adaptive"
                        : "legacy")
            << std::endl;
  report("tiny", trials, iterations * 2,
         [&] { return run_sequential(iterations, false); });
  report("mixed", trials, iterations * 2,
         [&] { return run_sequential(iterations, true); });
  report("contended", trials, iterations * threads * 2,
         [&] { return run_concurrent(iterations, threads); });
  return 0;
}
