/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "taichi/common/core.h"
#include "taichi/system/timer.h"

namespace taichi {

class ProfilerRecords;

// Captures running time between the construction and destruction of the
// profiler instance
class ScopedProfiler {
 public:
  explicit ScopedProfiler(std::string name, uint64 elements = -1);

  void stop();

  static void enable();

  static void disable();

  ~ScopedProfiler();

 private:
  std::string name_;
  float64 start_time_;
  uint64 elements_;
  bool stopped_;
};

// Phase 0': trace event for Chrome Tracing export.
// Only populated when tracing is enabled (env TI_COMPILE_PROFILE / explicit
// API). Zero cost when disabled.
struct TraceEvent {
  std::string name;
  uint64 tid;
  // Wall-clock timestamps in microseconds (double keeps sub-us precision).
  double ts_us;
  double dur_us;
};

// A profiling system for multithreaded applications
class Profiling {
 public:
  void print_profile_info();
  void clear_profile_info();
  ProfilerRecords *get_this_thread_profiler();

  // Phase 0': flat dumps of the scoped profiler state.
  //   export_csv          — recursive flatten of the per-thread tree.
  //   export_chrome_trace — events captured while tracing was enabled
  //                         (chrome://tracing / Perfetto compatible).
  // Returns false on I/O failure.
  bool export_csv(const std::string &path);
  bool export_chrome_trace(const std::string &path);

  // Low-overhead: guarded by is_tracing_enabled(); use record_trace_event
  // unconditionally from ScopedProfiler::stop().
  void record_trace_event(TraceEvent &&ev);

  // One-shot check of TI_COMPILE_PROFILE env variable. Evaluated lazily.
  // If a runtime override has been set (see set_tracing_runtime_override),
  // that value takes precedence over the env variable.
  static bool is_tracing_enabled();

  // Phase 0' / P-Compile-7: runtime override of the env-driven tracing
  // gate. Used by `ti.compile_profile()` Python context manager so users
  // can scope tracing to a specific code region without having to set
  // TI_COMPILE_PROFILE before importing Taichi. Pass true/false to enable
  // /disable; call clear_tracing_runtime_override() to fall back to env.
  static void set_tracing_runtime_override(bool enabled);
  static void clear_tracing_runtime_override();

  static Profiling &get_instance();

 private:
  std::mutex mut_;
  std::unordered_map<std::thread::id, ProfilerRecords *> profilers_;

  std::mutex trace_mut_;
  std::vector<TraceEvent> trace_events_;
};

#define TI_PROFILER(name) taichi::ScopedProfiler _profiler_##__LINE__(name);

#define TI_AUTO_PROF TI_PROFILER(__FUNCTION__)

}  // namespace taichi
