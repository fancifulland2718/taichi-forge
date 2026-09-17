#define TI_RUNTIME_HOST
#include "amdgpu_context.h"

#include <unordered_map>
#include <mutex>

#include "taichi/util/lang_util.h"
#include "taichi/system/threading.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_profiler.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/util/offline_cache.h"

namespace taichi {
namespace lang {

AMDGPUContext::AMDGPUContext()
    : driver_(AMDGPUDriver::get_instance_without_context()) {
  TI_ERROR_IF(!driver_.detected(),
              "No compatible HIP runtime or AMD GPU available.");
  driver_.device_get_count(&dev_count_);
  TI_ERROR_IF(dev_count_ == 0, "No AMDGPU device detected.");
  // HIP's primary device context is supported on both Linux and Windows.
  // Do not create deprecated, separate hipCtx contexts.
  make_current();

  const auto info = driver_.device_info(device_);
  compute_capability_ = info.major * 100 + info.minor * 10;
  warp_size_ = info.warp_size;
  mcpu_ = info.architecture.substr(0, info.architecture.find(':'));
  TI_ERROR_IF(mcpu_.size() <= 3 || mcpu_.substr(0, 3) != "gfx",
              "HIP returned an invalid AMDGPU target: {}", mcpu_);
  TI_ERROR_IF(warp_size_ != 32 && warp_size_ != 64,
              "Unsupported AMDGPU wavefront size: {}", warp_size_);
  TI_TRACE("Using AMDGPU device [id=0]: {}", get_device_name());
  TI_TRACE("Emitting AMDGPU code for {}", mcpu_);
}

std::size_t AMDGPUContext::get_total_memory() {
  std::size_t ret, _;
  driver_.mem_get_info(&_, &ret);
  return ret;
}

std::size_t AMDGPUContext::get_free_memory() {
  std::size_t ret, _;
  driver_.mem_get_info(&ret, &_);
  return ret;
}

std::string AMDGPUContext::get_device_name() {
  constexpr uint32_t kMaxNameStringLength = 128;
  char name[kMaxNameStringLength];
  driver_.device_get_name(name, kMaxNameStringLength /*=128*/, device_);
  std::string str(name);
  return str;
}

int AMDGPUContext::get_args_byte(std::vector<int> arg_sizes) {
  int byte_cnt = 0;
  int naive_add = 0;
  for (auto &size : arg_sizes) {
    naive_add += size;
    if (size < 32) {
      if ((byte_cnt + size) % 32 > (byte_cnt) % 32 ||
          (byte_cnt + size) % 32 == 0)
        byte_cnt += size;
      else
        byte_cnt += 32 - byte_cnt % 32 + size;
    } else {
      if (byte_cnt % 32 != 0)
        byte_cnt += 32 - byte_cnt % 32 + size;
      else
        byte_cnt += size;
    }
  }
  return byte_cnt;
}

void AMDGPUContext::pack_args(std::vector<void *> arg_pointers,
                              std::vector<int> arg_sizes,
                              char *arg_packed) {
  int byte_cnt = 0;
  for (int ii = 0; ii < arg_pointers.size(); ii++) {
    // The parameter is taken as a vec4
    if (arg_sizes[ii] < 32) {
      if ((byte_cnt + arg_sizes[ii]) % 32 > (byte_cnt % 32) ||
          (byte_cnt + arg_sizes[ii]) % 32 == 0) {
        std::memcpy(arg_packed + byte_cnt, arg_pointers[ii], arg_sizes[ii]);
        byte_cnt += arg_sizes[ii];
      } else {
        int padding_size = 32 - byte_cnt % 32;
        byte_cnt += padding_size;
        std::memcpy(arg_packed + byte_cnt, arg_pointers[ii], arg_sizes[ii]);
        byte_cnt += arg_sizes[ii];
      }
    } else {
      if (byte_cnt % 32 != 0) {
        int padding_size = 32 - byte_cnt % 32;
        byte_cnt += padding_size;
        std::memcpy(arg_packed + byte_cnt, arg_pointers[ii], arg_sizes[ii]);
        byte_cnt += arg_sizes[ii];
      } else {
        std::memcpy(arg_packed + byte_cnt, arg_pointers[ii], arg_sizes[ii]);
        byte_cnt += arg_sizes[ii];
      }
    }
  }
}

void AMDGPUContext::launch(void *func,
                           const std::string &task_name,
                           const std::vector<void *> &arg_pointers,
                           const std::vector<int> &arg_sizes,
                           unsigned grid_dim,
                           unsigned block_dim,
                           std::size_t dynamic_shared_mem_bytes) {
  KernelProfilerBase::TaskHandle task_handle;
  // Kernel launch
  if (profiler_) {
    KernelProfilerAMDGPU *profiler_amdgpu =
        dynamic_cast<KernelProfilerAMDGPU *>(profiler_);
    std::string primal_task_name, key;
    bool valid =
        offline_cache::try_demangle_name(task_name, primal_task_name, key);
    profiler_amdgpu->trace(task_handle, valid ? primal_task_name : task_name,
                           func, grid_dim, block_dim, 0);
  }
  std::size_t pack_size = get_args_byte(arg_sizes);
  std::vector<char> packed_arg(pack_size);
  pack_args(arg_pointers, arg_sizes, packed_arg.data());
  if (grid_dim > 0) {
    std::lock_guard<std::mutex> _(lock_);
    void *config[] = {(void *)0x01, packed_arg.data(), (void *)0x02,
                      (void *)&pack_size, (void *)0x03};
    driver_.launch_kernel(func, grid_dim, 1, 1, block_dim, 1, 1,
                          dynamic_shared_mem_bytes, nullptr, nullptr,
                          reinterpret_cast<void **>(&config));
  }

  if (profiler_)
    profiler_->stop(task_handle);

  if (debug_) {
    driver_.stream_synchronize(nullptr);
  }
}

AMDGPUContext::~AMDGPUContext() {
}

AMDGPUContext &AMDGPUContext::get_instance() {
  static auto context = new AMDGPUContext();
  return *context;
}

}  // namespace lang
}  // namespace taichi
