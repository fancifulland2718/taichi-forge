#pragma once
// P1.b — In-process mirror of the offline-cache .tic files.
//
// After ti.reset()+ti.init() (or across multiple Program instances within
// the same process) the on-disk offline cache still makes subsequent runs
// much faster than a cold compile, but every kernel still pays the cost
// of a file open + read. This class keeps the raw serialized bytes of
// previously loaded/written CompiledKernelData in memory, keyed by the
// same `kernel_key` used on disk, so a second Program can deserialize
// directly from memory and skip the disk I/O entirely.
//
// Semantics are identical to loading from disk: deserialization itself is
// still performed per Program, producing a fresh CompiledKernelData whose
// `kernel_launch_handle_` is null. This is intentional — the launch
// handle is per-runtime (see runtime/{cpu,cuda,gfx}/kernel_launcher.cpp)
// and must not be shared across Programs.
//
// Thread-safe. Bounded by a byte cap read once from the environment
// variable `TI_INPROC_DISK_MIRROR_MB` (default 256 MB; set to 0 to
// disable). Eviction is FIFO — when a new entry would exceed the cap,
// the oldest entries are dropped until it fits.

#include <cstddef>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace taichi::lang {

class InprocDiskMirror {
 public:
  // Returns a copy of the cached serialized bytes for `kernel_key`, or
  // std::nullopt if the key is not present (or the mirror is disabled).
  // The internal lock is released before the copy is returned so callers
  // can safely deserialize without blocking other lookups.
  static std::optional<std::string> get(const std::string &kernel_key);

  // Stores `bytes` under `kernel_key`. No-ops if the mirror is disabled
  // (cap == 0) or if `bytes` alone exceeds the cap. Overwrites an existing
  // entry for the same key. Evicts oldest entries FIFO as needed.
  static void put(const std::string &kernel_key, std::string bytes);

  // Drops all entries. Used by tests.
  static void clear();

  // Diagnostics — stable snapshots, may be called concurrently.
  static std::size_t total_bytes();
  static std::size_t hits();
  static std::size_t misses();
};

}  // namespace taichi::lang
