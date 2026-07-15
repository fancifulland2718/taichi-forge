#pragma once

#include <mutex>
#include <unordered_map>
#include <taichi/rhi/cuda/cuda_capability.h>

#include "taichi/program/kernel_profiler.h"
#include "taichi/rhi/cuda/cuda_driver.h"

namespace taichi::lang {

// Note:
// It would be ideal to create a CUDA context per Taichi program, yet CUDA
// context creation takes time. Therefore we use a shared context to accelerate
// cases such as unit testing where many Taichi programs are created/destroyed.

class CUDADriver;

class CUDAContext {
 private:
  void *device_;
  void *context_;
  int dev_count_;
  // Keep the hardware capability separate from the target accepted by the
  // bundled LLVM NVPTX backend. User code and runtime feature checks need the
  // former, while JIT target selection and cache keys need the latter.
  int device_compute_capability_;
  int codegen_compute_capability_;
  int ptx_version_;
  std::string mcpu_;
  std::string mattrs_;
  std::mutex lock_;
  // One CUDA primary context and its legacy default stream are shared by all
  // Taichi Programs in this process. A submission transaction keeps graph
  // capture/replay, ordinary multi-task kernels, and direct driver kernels
  // from interleaving their host-side setup. It never waits for queued device
  // work on steady-state field/ndarray launches.
  std::recursive_mutex submission_mutex_;
  CUDASampledRecursiveLockTelemetry submission_lock_telemetry_;
  CUDASampledLockTelemetry lock_telemetry_;
  std::mutex graph_capture_mutex_;
  KernelProfilerBase *profiler_;
  CUDADriver &driver_;
  int max_shared_memory_bytes_;
  bool debug_{false};
  bool supports_mem_pool_{false};

 public:
  CUDAContext();

  std::size_t get_total_memory();
  std::size_t get_free_memory();
  std::string get_device_name();

  bool detected() const {
    return dev_count_ != 0;
  }

  void launch(void *func,
              const std::string &task_name,
              std::vector<void *> arg_pointers,
              std::vector<int> arg_sizes,
              unsigned grid_dim,
              unsigned block_dim,
              std::size_t dynamic_shared_mem_bytes,
              void *stream = nullptr);

  void set_profiler(KernelProfilerBase *profiler) {
    profiler_ = profiler;
  }

  void set_debug(bool debug) {
    debug_ = debug;
  }

  std::string get_mcpu() const {
    return mcpu_;
  }

  std::string get_mattrs() const {
    return mattrs_;
  }

  void *get_context() {
    return context_;
  }

  void make_current() {
    driver_.context_set_current(context_);
  }

  int get_compute_capability() const {
    return device_compute_capability_;
  }

  int get_codegen_compute_capability() const {
    return codegen_compute_capability_;
  }

  bool supports_mem_pool() const {
    return supports_mem_pool_;
  }

  ~CUDAContext();

  class ContextGuard {
   private:
    void *old_ctx_;
    void *new_ctx_;

   public:
    explicit ContextGuard(CUDAContext *new_ctx)
        : old_ctx_(nullptr), new_ctx_(new_ctx->context_) {
      CUDADriver::get_instance().context_get_current(&old_ctx_);
      if (old_ctx_ != new_ctx_)
        new_ctx->make_current();
    }

    ~ContextGuard() {
      if (old_ctx_ != new_ctx_) {
        CUDADriver::get_instance().context_set_current(old_ctx_);
      }
    }
  };

  ContextGuard get_guard() {
    return ContextGuard(this);
  }

  std::unique_lock<std::mutex> get_lock_guard() {
    return lock_telemetry_.acquire(lock_);
  }

  std::unique_lock<std::recursive_mutex> get_submission_lock_guard() {
    return submission_lock_telemetry_.acquire(submission_mutex_);
  }

  CUDASampledLockTelemetry::Snapshot get_lock_telemetry_snapshot() const {
    return lock_telemetry_.snapshot();
  }

  CUDASampledRecursiveLockTelemetry::Snapshot
  get_submission_lock_telemetry_snapshot() const {
    return submission_lock_telemetry_.snapshot();
  }

  std::unique_lock<std::mutex> get_graph_capture_lock_guard() {
    return std::unique_lock<std::mutex>(graph_capture_mutex_);
  }

  static CUDAContext &get_instance();

};

}  // namespace taichi::lang
