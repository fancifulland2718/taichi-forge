#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/llvm/device_memory_pool.h"

#include <memory>

#include "taichi/jit/jit_module.h"

namespace taichi::lang {
namespace cuda {
CudaDevice::AllocationRecord::~AllocationRecord() {
  release();
}

CudaDevice::AllocationRecord::AllocationRecord(
    AllocationRecord &&other) noexcept
    : ptr(other.ptr),
      size(other.size),
      is_imported(other.is_imported),
      use_preallocated(other.use_preallocated),
      use_cached(other.use_cached),
      use_memory_pool(other.use_memory_pool),
      stream(other.stream),
      mapping(std::move(other.mapping)) {
  other.ptr = nullptr;
}

CudaDevice::AllocationRecord &CudaDevice::AllocationRecord::operator=(
    AllocationRecord &&other) noexcept {
  if (this != &other) {
    release();
    ptr = other.ptr;
    size = other.size;
    is_imported = other.is_imported;
    use_preallocated = other.use_preallocated;
    use_cached = other.use_cached;
    use_memory_pool = other.use_memory_pool;
    stream = other.stream;
    mapping = std::move(other.mapping);
    other.ptr = nullptr;
  }
  return *this;
}

void CudaDevice::AllocationRecord::release() {
  mapping.reset();
  if (ptr == nullptr || is_imported) {
    return;
  }

  auto context_guard = CUDAContext::get_instance().get_guard();
  if (use_memory_pool) {
    // cuMemAllocAsync/cuMemFreeAsync require the same stream-ordered
    // lifetime contract. Cross-stream access must have an explicit CUDA
    // dependency before the allocation is retired.
    CUDADriver::get_instance().mem_free_async(ptr, stream);
  } else if (use_cached) {
    DeviceMemoryPool::get_instance().release(size,
                                             static_cast<uint64_t *>(ptr),
                                             false);
  } else if (!use_preallocated) {
    DeviceMemoryPool::get_instance().release(size, ptr, true);
  }
  ptr = nullptr;
}

CudaDevice::CudaDevice() {
  // Initialize the device memory pool
  DeviceMemoryPool::get_instance(true /*merge_upon_release*/);
}

CudaDevice::~CudaDevice() {
  clear();
}

CudaDevice::AllocInfo CudaDevice::get_alloc_info(
    const DeviceAllocation handle) {
  if (handle.device != this) {
    TI_ERROR("invalid DeviceAllocation");
  }
  auto [result, lease] = allocations_.acquire(handle.alloc_id);
  if (result != RhiResult::success) {
    TI_ERROR("invalid DeviceAllocation");
  }
  return lease->info();
}

RhiResult CudaDevice::allocate_memory(const AllocParams &params,
                                      DeviceAllocation *out_devalloc) {
  if (out_devalloc == nullptr) {
    return RhiResult::invalid_usage;
  }

  std::unique_ptr<MappingState> mapping;
  try {
    mapping = std::make_unique<MappingState>();
  } catch (const std::bad_alloc &) {
    return RhiResult::out_of_memory;
  }

  auto context_guard = CUDAContext::get_instance().get_guard();
  auto &mem_pool = DeviceMemoryPool::get_instance();
  const bool managed = params.host_read || params.host_write;
  void *ptr = mem_pool.allocate(params.size, DeviceMemoryPool::page_size,
                                managed);
  if (ptr == nullptr) {
    return RhiResult::out_of_memory;
  }
  if (params.size != 0) {
    CUDADriver::get_instance().memset(ptr, 0, params.size);
  }

  AllocationRecord record(ptr, params.size, false, false, false, false,
                          nullptr, std::move(mapping));
  auto [result, alloc_id] = allocations_.emplace(std::move(record));
  if (result != RhiResult::success) {
    return result;
  }
  *out_devalloc = {this, alloc_id};
  return RhiResult::success;
}

DeviceAllocation CudaDevice::allocate_memory_runtime(
    const LlvmRuntimeAllocParams &params) {
  std::unique_ptr<MappingState> mapping;
  try {
    mapping = std::make_unique<MappingState>();
  } catch (const std::bad_alloc &) {
    TI_ERROR("Failed to allocate CUDA allocation metadata");
  }

  auto context_guard = CUDAContext::get_instance().get_guard();
  const size_t size = taichi::iroundup(params.size, taichi_page_size);
  const CUstream stream = nullptr;
  void *ptr = nullptr;
  if (size != 0) {
    if (params.use_memory_pool) {
      CUDADriver::get_instance().malloc_async(&ptr, size, stream);
    } else {
      ptr = DeviceMemoryPool::get_instance().allocate_with_cache(this, params);
    }
    if (ptr != nullptr) {
      CUDADriver::get_instance().memset(ptr, 0, size);
    }
  }

  AllocationRecord record(ptr, size, false, true, true,
                          params.use_memory_pool, stream, std::move(mapping));
  auto [result, alloc_id] = allocations_.emplace(std::move(record));
  TI_ERROR_IF(result != RhiResult::success,
              "Failed to track CUDA runtime allocation: {}", result);
  return {this, alloc_id};
}

uint64_t *CudaDevice::allocate_llvm_runtime_memory_jit(
    const LlvmRuntimeAllocParams &params) {
  auto context_guard = CUDAContext::get_instance().get_guard();
  params.runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", params.runtime, params.size,
      taichi_page_size, params.result_buffer);
  CUDADriver::get_instance().stream_synchronize(nullptr);
  uint64 *ret{nullptr};
  CUDADriver::get_instance().memcpy_device_to_host(&ret, params.result_buffer,
                                                   sizeof(uint64));
  return ret;
}

void CudaDevice::dealloc_memory(DeviceAllocation handle) {
  auto context_guard = CUDAContext::get_instance().get_guard();
  if (handle.device != this ||
      allocations_.retire(handle.alloc_id) != RhiResult::success) {
    TI_WARN("invalid DeviceAllocation");
  }
}

RhiResult CudaDevice::upload_data(DevicePtr *device_ptr,
                                  const void **data,
                                  size_t *size,
                                  int num_alloc) noexcept {
  if (!device_ptr || !data || !size || num_alloc < 0) {
    return RhiResult::invalid_usage;
  }
  auto context_guard = CUDAContext::get_instance().get_guard();

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
      CUDADriver::get_instance().memcpy_host_to_device(
          static_cast<uint8_t *>(lease->ptr) + device_ptr[i].offset,
          const_cast<void *>(data[i]), size[i]);
    }
  }
  return RhiResult::success;
}

RhiResult CudaDevice::readback_data(
    DevicePtr *device_ptr,
    void **data,
    size_t *size,
    int num_alloc,
    const std::vector<StreamSemaphore> &wait_sema) noexcept {
  if (!device_ptr || !data || !size || num_alloc < 0) {
    return RhiResult::invalid_usage;
  }
  auto context_guard = CUDAContext::get_instance().get_guard();

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
      CUDADriver::get_instance().memcpy_device_to_host(
          data[i], static_cast<uint8_t *>(lease->ptr) + device_ptr[i].offset,
          size[i]);
    }
  }
  return RhiResult::success;
}

RhiResult CudaDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  if (mapped_ptr == nullptr || alloc.device != this) {
    return RhiResult::invalid_usage;
  }
  *mapped_ptr = nullptr;
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto [result, lease] = allocations_.acquire(alloc.alloc_id);
  if (result != RhiResult::success ||
      (lease->size != 0 && lease->ptr == nullptr)) {
    return result == RhiResult::success ? RhiResult::invalid_usage : result;
  }

  auto &mapping = *lease->mapping;
  std::lock_guard<std::mutex> lock(mapping.mutex);
  if (mapping.staging) {
    return RhiResult::invalid_usage;
  }
  try {
    if (lease->size != 0) {
      mapping.staging = std::make_unique<char[]>(lease->size);
      CUDADriver::get_instance().memcpy_device_to_host(
          mapping.staging.get(), lease->ptr, lease->size);
    }
  } catch (const std::bad_alloc &) {
    return RhiResult::out_of_memory;
  } catch (...) {
    return RhiResult::error;
  }
  *mapped_ptr = mapping.staging.get();
  return RhiResult::success;
}

void CudaDevice::unmap(DeviceAllocation alloc) {
  if (alloc.device != this) {
    TI_WARN("invalid DeviceAllocation");
    return;
  }
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto [result, lease] = allocations_.acquire(alloc.alloc_id);
  if (result != RhiResult::success) {
    TI_WARN("invalid DeviceAllocation");
    return;
  }

  auto &mapping = *lease->mapping;
  std::lock_guard<std::mutex> lock(mapping.mutex);
  if (!mapping.staging && lease->size != 0) {
    TI_WARN("unmapping a CUDA allocation that is not mapped");
    return;
  }
  if (lease->size != 0) {
    CUDADriver::get_instance().memcpy_host_to_device(lease->ptr,
                                                     mapping.staging.get(),
                                                     lease->size);
  }
  mapping.staging.reset();
}

void CudaDevice::clear() {
  auto context_guard = CUDAContext::get_instance().get_guard();
  allocations_.clear();
}

void CudaDevice::memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) {
  if (dst.device != this || src.device != this) {
    TI_WARN("invalid DeviceAllocation");
    return;
  }
  auto context_guard = CUDAContext::get_instance().get_guard();
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
    CUDADriver::get_instance().memcpy_device_to_device(
        static_cast<uint8_t *>(dst_lease->ptr) + dst.offset,
        static_cast<uint8_t *>(src_lease->ptr) + src.offset, size);
  }
}

RhiResult CudaDevice::copy_to_external(void *external_ptr,
                                       DevicePtr src,
                                       uint64_t size) {
  if (external_ptr == nullptr || src.device != this) {
    return RhiResult::invalid_usage;
  }
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto [result, lease] = allocations_.acquire(src.alloc_id, src.offset, size);
  if (result != RhiResult::success ||
      (size != 0 && lease->ptr == nullptr)) {
    return RhiResult::invalid_usage;
  }
  if (size != 0) {
    CUDADriver::get_instance().memcpy_device_to_device(
        external_ptr, static_cast<uint8_t *>(lease->ptr) + src.offset, size);
  }
  return RhiResult::success;
}

RhiResult CudaDevice::copy_from_external(DevicePtr dst,
                                         void *external_ptr,
                                         uint64_t size) {
  if (external_ptr == nullptr || dst.device != this) {
    return RhiResult::invalid_usage;
  }
  auto context_guard = CUDAContext::get_instance().get_guard();
  auto [result, lease] = allocations_.acquire(dst.alloc_id, dst.offset, size);
  if (result != RhiResult::success ||
      (size != 0 && lease->ptr == nullptr)) {
    return RhiResult::invalid_usage;
  }
  if (size != 0) {
    CUDADriver::get_instance().memcpy_device_to_device(
        static_cast<uint8_t *>(lease->ptr) + dst.offset, external_ptr, size);
  }
  return RhiResult::success;
}

DeviceAllocation CudaDevice::import_memory(void *ptr, size_t size) {
  std::unique_ptr<MappingState> mapping;
  try {
    mapping = std::make_unique<MappingState>();
  } catch (const std::bad_alloc &) {
    TI_ERROR("Failed to allocate CUDA allocation metadata");
  }
  AllocationRecord record(ptr, size, true, true, false, false, nullptr,
                          std::move(mapping));
  auto [result, alloc_id] = allocations_.emplace(std::move(record));
  TI_ERROR_IF(result != RhiResult::success,
              "Failed to import CUDA DeviceAllocation: {}", result);
  return {this, alloc_id};
}

}  // namespace cuda
}  // namespace taichi::lang
