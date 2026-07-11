#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/impl_support.h"

#include <cstring>

#include "taichi/jit/jit_module.h"

namespace taichi::lang {
namespace cpu {

CpuDevice::AllocationRecord::~AllocationRecord() {
  if (ptr != nullptr && !use_cached && !is_imported) {
    HostMemoryPool::get_instance().release(size, ptr);
  }
}

CpuDevice::AllocationRecord::AllocationRecord(
    AllocationRecord &&other) noexcept
    : ptr(other.ptr),
      size(other.size),
      use_cached(other.use_cached),
      is_imported(other.is_imported) {
  other.ptr = nullptr;
}

CpuDevice::AllocationRecord &CpuDevice::AllocationRecord::operator=(
    AllocationRecord &&other) noexcept {
  if (this != &other) {
    if (ptr != nullptr && !use_cached && !is_imported) {
      HostMemoryPool::get_instance().release(size, ptr);
    }
    ptr = other.ptr;
    size = other.size;
    use_cached = other.use_cached;
    is_imported = other.is_imported;
    other.ptr = nullptr;
  }
  return *this;
}

CpuDevice::AllocInfo CpuDevice::get_alloc_info(const DeviceAllocation handle) {
  if (handle.device != this) {
    TI_ERROR("invalid DeviceAllocation");
  }
  auto [result, lease] = allocations_.acquire(handle.alloc_id);
  if (result != RhiResult::success) {
    TI_ERROR("invalid DeviceAllocation");
  }
  return lease->info();
}

CpuDevice::CpuDevice() {
}

CpuDevice::~CpuDevice() {
  clear();
}

RhiResult CpuDevice::allocate_memory(const AllocParams &params,
                                     DeviceAllocation *out_devalloc) {
  if (out_devalloc == nullptr) {
    return RhiResult::invalid_usage;
  }

  void *ptr = nullptr;
  if (params.size != 0) {
    ptr = HostMemoryPool::get_instance().allocate(
        params.size, HostMemoryPool::page_size, true /*exclusive*/);
    if (ptr == nullptr) {
      return RhiResult::out_of_memory;
    }
  }

  AllocationRecord record(ptr, params.size, false, false);
  auto [result, alloc_id] = allocations_.emplace(std::move(record));
  if (result != RhiResult::success) {
    return result;
  }
  *out_devalloc = {this, alloc_id};
  return RhiResult::success;
}

DeviceAllocation CpuDevice::allocate_memory_runtime(
    const LlvmRuntimeAllocParams &params) {
  DeviceAllocation alloc;
  RhiResult res = allocate_memory(params, &alloc);
  RHI_ASSERT(res == RhiResult::success &&
             "Failed to allocate memory for runtime");
  return alloc;
}

uint64_t *CpuDevice::allocate_llvm_runtime_memory_jit(
    const LlvmRuntimeAllocParams &params) {
  params.runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", params.runtime, params.size,
      taichi_page_size, params.result_buffer);
  return reinterpret_cast<uint64_t *>(params.result_buffer[0]);
}

void CpuDevice::dealloc_memory(DeviceAllocation handle) {
  if (handle.device != this) {
    TI_WARN("invalid DeviceAllocation");
    return;
  }
  std::lock_guard<std::mutex> lock(mapping_lifecycle_mutex_);
  if (mapped_allocations_.find(handle.alloc_id) != mapped_allocations_.end()) {
    TI_WARN("cannot deallocate a mapped CPU allocation");
    return;
  }
  if (allocations_.retire(handle.alloc_id) != RhiResult::success) {
    TI_WARN("invalid DeviceAllocation");
  }
}

RhiResult CpuDevice::upload_data(DevicePtr *device_ptr,
                                 const void **data,
                                 size_t *size,
                                 int num_alloc) noexcept {
  if (!device_ptr || !data || !size || num_alloc < 0) {
    return RhiResult::invalid_usage;
  }

  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }
    auto [result, lease] =
        allocations_.acquire(device_ptr[i].alloc_id, device_ptr[i].offset,
                             size[i]);
    if (result != RhiResult::success ||
        (size[i] != 0 && lease->ptr == nullptr)) {
      return RhiResult::invalid_usage;
    }
    if (size[i] != 0) {
      std::memcpy(static_cast<uint8_t *>(lease->ptr) + device_ptr[i].offset,
                  data[i], size[i]);
    }
  }

  return RhiResult::success;
}

RhiResult CpuDevice::readback_data(
    DevicePtr *device_ptr,
    void **data,
    size_t *size,
    int num_alloc,
    const std::vector<StreamSemaphore> &wait_sema) noexcept {
  if (!device_ptr || !data || !size || num_alloc < 0) {
    return RhiResult::invalid_usage;
  }

  for (int i = 0; i < num_alloc; i++) {
    if (device_ptr[i].device != this || !data[i]) {
      return RhiResult::invalid_usage;
    }
    auto [result, lease] =
        allocations_.acquire(device_ptr[i].alloc_id, device_ptr[i].offset,
                             size[i]);
    if (result != RhiResult::success ||
        (size[i] != 0 && lease->ptr == nullptr)) {
      return RhiResult::invalid_usage;
    }
    if (size[i] != 0) {
      std::memcpy(data[i], static_cast<uint8_t *>(lease->ptr) +
                               device_ptr[i].offset,
                  size[i]);
    }
  }

  return RhiResult::success;
}

RhiResult CpuDevice::map_range(DevicePtr ptr,
                               uint64_t size,
                               void **mapped_ptr) {
  if (mapped_ptr == nullptr || ptr.device != this) {
    return RhiResult::invalid_usage;
  }
  *mapped_ptr = nullptr;
  std::lock_guard<std::mutex> lock(mapping_lifecycle_mutex_);
  auto [result, lease] = allocations_.acquire(ptr.alloc_id, ptr.offset, size);
  if (result != RhiResult::success || lease->ptr == nullptr) {
    return result == RhiResult::success ? RhiResult::error : result;
  }
  if (mapped_allocations_.find(ptr.alloc_id) != mapped_allocations_.end()) {
    return RhiResult::invalid_usage;
  }
  void *host_ptr = static_cast<uint8_t *>(lease->ptr) + ptr.offset;
  try {
    mapped_allocations_.emplace(ptr.alloc_id, std::move(lease));
  } catch (const std::bad_alloc &) {
    return RhiResult::out_of_memory;
  } catch (...) {
    return RhiResult::error;
  }
  *mapped_ptr = host_ptr;
  return RhiResult::success;
}

RhiResult CpuDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  if (mapped_ptr == nullptr || alloc.device != this) {
    return RhiResult::invalid_usage;
  }
  *mapped_ptr = nullptr;
  std::lock_guard<std::mutex> lock(mapping_lifecycle_mutex_);
  auto [result, lease] = allocations_.acquire(alloc.alloc_id);
  if (result != RhiResult::success || lease->ptr == nullptr) {
    return result == RhiResult::success ? RhiResult::error : result;
  }
  if (mapped_allocations_.find(alloc.alloc_id) != mapped_allocations_.end()) {
    return RhiResult::invalid_usage;
  }
  void *host_ptr = lease->ptr;
  try {
    mapped_allocations_.emplace(alloc.alloc_id, std::move(lease));
  } catch (const std::bad_alloc &) {
    return RhiResult::out_of_memory;
  } catch (...) {
    return RhiResult::error;
  }
  *mapped_ptr = host_ptr;
  return RhiResult::success;
}

RhiResult CpuDevice::map_range_for_cpu_native(DevicePtr ptr,
                                              uint64_t size,
                                              void **mapped_ptr) {
  if (mapped_ptr == nullptr || ptr.device != this) {
    return RhiResult::invalid_usage;
  }
  *mapped_ptr = nullptr;
  auto [result, lease] = allocations_.acquire(ptr.alloc_id, ptr.offset, size);
  if (result != RhiResult::success || lease->ptr == nullptr) {
    return result == RhiResult::success ? RhiResult::error : result;
  }
  // The synchronous Program call owns the field allocation until the native
  // primitive returns. No persistent map state or host copy is needed here.
  *mapped_ptr = static_cast<uint8_t *>(lease->ptr) + ptr.offset;
  return RhiResult::success;
}

void CpuDevice::unmap(DevicePtr ptr) {
  if (ptr.device != this) {
    TI_WARN("invalid DevicePtr");
    return;
  }
  std::lock_guard<std::mutex> lock(mapping_lifecycle_mutex_);
  if (mapped_allocations_.erase(ptr.alloc_id) == 0) {
    TI_WARN("unmapping a CPU allocation that is not mapped");
  }
}

void CpuDevice::unmap(DeviceAllocation alloc) {
  unmap(alloc.get_ptr());
}

void CpuDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  if (dst.device != this || src.device != this) {
    TI_WARN("invalid DeviceAllocation");
    return;
  }
  auto [dst_result, dst_lease] =
      allocations_.acquire(dst.alloc_id, dst.offset, size);
  auto [src_result, src_lease] =
      allocations_.acquire(src.alloc_id, src.offset, size);
  if (dst_result != RhiResult::success || src_result != RhiResult::success ||
      (size != 0 && (dst_lease->ptr == nullptr || src_lease->ptr == nullptr))) {
    TI_WARN("invalid DeviceAllocation");
    return;
  }
  if (size != 0) {
    std::memcpy(static_cast<uint8_t *>(dst_lease->ptr) + dst.offset,
                static_cast<uint8_t *>(src_lease->ptr) + src.offset, size);
  }
}

DeviceAllocation CpuDevice::import_memory(void *ptr, size_t size) {
  AllocationRecord record(ptr, size, false, true);
  auto [result, alloc_id] = allocations_.emplace(std::move(record));
  TI_ERROR_IF(result != RhiResult::success,
              "Failed to import CPU DeviceAllocation: {}", result);
  return {this, alloc_id};
}

void CpuDevice::clear() {
  std::lock_guard<std::mutex> lock(mapping_lifecycle_mutex_);
  if (!mapped_allocations_.empty()) {
    TI_WARN("cannot clear CPU allocations while {} allocation(s) are mapped",
            mapped_allocations_.size());
    return;
  }
  allocations_.clear();
}

}  // namespace cpu
}  // namespace taichi::lang
