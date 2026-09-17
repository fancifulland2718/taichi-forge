#pragma once

#include <mutex>
#include <unordered_map>
#include <thread>

#include "taichi/program/kernel_profiler.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"

namespace taichi {
namespace lang {

class AMDGPUDriver;

class AMDGPUContext {
 private:
  int device_{0};
  int dev_count_{0};
  int compute_capability_{0};
  int warp_size_{32};
  std::string mcpu_;
  std::mutex lock_;
  KernelProfilerBase *profiler_{nullptr};
  AMDGPUDriver &driver_;
  bool debug_{false};
  std::vector<void *> kernel_arg_pointer_;

 public:
  AMDGPUContext();

  std::size_t get_total_memory();
  std::size_t get_free_memory();
  std::string get_device_name();

  bool detected() const {
    return dev_count_ != 0;
  }

  void push_back_kernel_arg_pointer(void *ptr) {
    kernel_arg_pointer_.push_back(ptr);
  }

  void free_kernel_arg_pointer() {
    for (auto &i : kernel_arg_pointer_) {
      AMDGPUDriver::get_instance().mem_free(i);
    }
    kernel_arg_pointer_.erase(kernel_arg_pointer_.begin(),
                              kernel_arg_pointer_.end());
  }

  void pack_args(std::vector<void *> arg_pointers,
                 std::vector<int> arg_sizes,
                 char *arg_packed);

  int get_args_byte(std::vector<int> arg_sizes);

  void set_profiler(KernelProfilerBase *profiler) {
    profiler_ = profiler;
  }

  void launch(void *func,
              const std::string &task_name,
              const std::vector<void *> &arg_pointers,
              const std::vector<int> &arg_sizes,
              unsigned grid_dim,
              unsigned block_dim,
              std::size_t dynamic_shared_mem_bytes);

  void set_debug(bool debug) {
    debug_ = debug;
  }

  std::string get_mcpu() const {
    return mcpu_;
  }

  int get_warp_size() const {
    return warp_size_;
  }

  std::string get_target_features() const {
    return warp_size_ == 64 ? "+wavefrontsize64,-wavefrontsize32"
                            : "+wavefrontsize32,-wavefrontsize64";
  }

  void make_current() {
    driver_.device_set_current(device_);
  }

  int get_compute_capability() const {
    return compute_capability_;
  }

  ~AMDGPUContext();

  class ContextGuard {
   private:
    int old_device_;
    int new_device_;

   public:
    explicit ContextGuard(AMDGPUContext *new_ctx)
        : old_device_(0), new_device_(new_ctx->device_) {
      AMDGPUDriver::get_instance().device_get_current(&old_device_);
      if (old_device_ != new_device_)
        new_ctx->make_current();
    }

    ~ContextGuard() {
      if (old_device_ != new_device_) {
        AMDGPUDriver::get_instance().device_set_current(old_device_);
      }
    }
  };

  ContextGuard get_guard() {
    return ContextGuard(this);
  }

  std::unique_lock<std::mutex> get_lock_guard() {
    return std::unique_lock<std::mutex>(lock_);
  }

  static AMDGPUContext &get_instance();
};

}  // namespace lang
}  // namespace taichi
