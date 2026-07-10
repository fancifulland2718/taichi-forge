#include "taichi/rhi/interop/vulkan_cuda_interop.h"

#if TI_WITH_VULKAN && TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif  // TI_WITH_VULKAN && TI_WITH_CUDA

#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || \
    defined(_MSC_VER)
#include "taichi/platform/windows/windows.h"
#else
#include <unistd.h>
#endif

namespace taichi::lang {

#if TI_WITH_VULKAN && TI_WITH_CUDA

using namespace taichi::lang::vulkan;
using namespace taichi::lang::cuda;

namespace {

struct VulkanAllocCacheKey {
  DeviceAllocationId alloc_id{0};
  uint64_t generation{0};

  bool operator==(const VulkanAllocCacheKey &other) const {
    return alloc_id == other.alloc_id && generation == other.generation;
  }
};

struct VulkanAllocCacheKeyHasher {
  size_t operator()(const VulkanAllocCacheKey &key) const {
    return std::hash<DeviceAllocationId>{}(key.alloc_id) ^
           (std::hash<uint64_t>{}(key.generation) + 0x9e3779b97f4a7c15ull +
            (std::hash<DeviceAllocationId>{}(key.alloc_id) << 6) +
            (std::hash<DeviceAllocationId>{}(key.alloc_id) >> 2));
  }
};

struct ImportedVulkanMemory {
  CUexternalMemory external_memory{nullptr};
  unsigned char *mapped_buffer{nullptr};
};

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || \
    defined(_MSC_VER)
class ScopedExternalMemoryHandle {
 public:
  explicit ScopedExternalMemoryHandle(HANDLE handle) : handle_(handle) {
  }
  ~ScopedExternalMemoryHandle() {
    if (handle_ != nullptr) {
      CloseHandle(handle_);
    }
  }

 private:
  HANDLE handle_{nullptr};
};
#else
class ScopedExternalMemoryHandle {
 public:
  explicit ScopedExternalMemoryHandle(int fd) : fd_(fd) {
  }
  ~ScopedExternalMemoryHandle() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }
  void release_to_cuda() {
    fd_ = -1;
  }

 private:
  int fd_{-1};
};
#endif

using VulkanCudaAllocationCache =
    std::unordered_map<
        VulkanDevice *,
        std::unordered_map<CudaDevice *,
                           std::unordered_map<VulkanAllocCacheKey,
                                              ImportedVulkanMemory,
                                              VulkanAllocCacheKeyHasher>>>;

struct VulkanCudaInteropCache {
  std::mutex mutex;
  VulkanCudaAllocationCache allocation_base_ptrs;
};

VulkanCudaInteropCache &vulkan_cuda_interop_cache() {
  static VulkanCudaInteropCache cache;
  return cache;
}

void release_imported_vulkan_memory(
    std::vector<ImportedVulkanMemory> entries) {
  if (entries.empty()) {
    return;
  }
  auto context_guard = CUDAContext::get_instance().get_guard();
  // cuMemcpyDtoD does not provide host completion for device-to-device work.
  // This slow interop path must complete before a Vulkan allocation's external
  // mapping can be freed or recycled.
  CUDADriver::get_instance().stream_synchronize(nullptr);
  for (const auto &entry : entries) {
    if (entry.mapped_buffer != nullptr) {
      CUDADriver::get_instance().mem_free(entry.mapped_buffer);
    }
    if (entry.external_memory != nullptr) {
      CUDADriver::get_instance().external_memory_destroy(
          entry.external_memory);
    }
  }
}

void release_vulkan_interop_allocation(VulkanDevice *vk_dev,
                                       DeviceAllocationId alloc_id,
                                       uint64_t generation) {
  std::vector<ImportedVulkanMemory> released;
  auto &cache = vulkan_cuda_interop_cache();
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto vk_it = cache.allocation_base_ptrs.find(vk_dev);
    if (vk_it == cache.allocation_base_ptrs.end()) {
      return;
    }
    const VulkanAllocCacheKey key{alloc_id, generation};
    for (auto cuda_it = vk_it->second.begin();
         cuda_it != vk_it->second.end();) {
      auto entry_it = cuda_it->second.find(key);
      if (entry_it != cuda_it->second.end()) {
        released.push_back(entry_it->second);
        cuda_it->second.erase(entry_it);
      }
      if (cuda_it->second.empty()) {
        cuda_it = vk_it->second.erase(cuda_it);
      } else {
        ++cuda_it;
      }
    }
    if (vk_it->second.empty()) {
      cache.allocation_base_ptrs.erase(vk_it);
    }
  }
  release_imported_vulkan_memory(std::move(released));
}

void release_vulkan_interop_device(VulkanDevice *vk_dev) {
  std::vector<ImportedVulkanMemory> released;
  auto &cache = vulkan_cuda_interop_cache();
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto vk_it = cache.allocation_base_ptrs.find(vk_dev);
    if (vk_it == cache.allocation_base_ptrs.end()) {
      return;
    }
    for (const auto &[cuda_dev, allocations] : vk_it->second) {
      (void)cuda_dev;
      for (const auto &[key, entry] : allocations) {
        (void)key;
        released.push_back(entry);
      }
    }
    cache.allocation_base_ptrs.erase(vk_it);
  }
  release_imported_vulkan_memory(std::move(released));
}

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || \
    defined(_MSC_VER)
HANDLE get_device_mem_handle(VkDeviceMemory &mem, VkDevice device) {
  HANDLE handle;

  VkMemoryGetWin32HandleInfoKHR memory_get_win32_handle_info = {};
  memory_get_win32_handle_info.sType =
      VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  memory_get_win32_handle_info.pNext = nullptr;
  memory_get_win32_handle_info.memory = mem;
  memory_get_win32_handle_info.handleType =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

  auto fpGetMemoryWin32HandleKHR =
      (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
          device, "vkGetMemoryWin32HandleKHR");

  if (fpGetMemoryWin32HandleKHR == nullptr) {
    TI_ERROR("vkGetMemoryWin32HandleKHR is nullptr");
  }

  auto result =
      fpGetMemoryWin32HandleKHR(device, &memory_get_win32_handle_info, &handle);
  if (result != VK_SUCCESS) {
    TI_ERROR("vkGetMemoryWin32HandleKHR failed");
  }

  return handle;
}
#else
int get_device_mem_handle(VkDeviceMemory &mem, VkDevice device) {
  int fd;

  VkMemoryGetFdInfoKHR memory_get_fd_info = {};
  memory_get_fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  memory_get_fd_info.pNext = nullptr;
  memory_get_fd_info.memory = mem;
  memory_get_fd_info.handleType =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

  auto fpGetMemoryFdKHR =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");

  if (fpGetMemoryFdKHR == nullptr) {
    TI_ERROR("vkGetMemoryFdKHR is nullptr");
  }
  auto result = fpGetMemoryFdKHR(device, &memory_get_fd_info, &fd);
  if (result != VK_SUCCESS) {
    TI_ERROR("vkGetMemoryFdKHR failed");
  }

  return fd;
}
#endif

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || \
    defined(_MSC_VER)
CUexternalMemory import_vk_memory_object_from_handle(HANDLE handle,
                                                     unsigned long long size,
                                                     bool is_dedicated) {
  CUexternalMemory ext_mem = nullptr;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc = {};

  memset(&desc, 0, sizeof(desc));

  desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
  desc.handle.win32.handle = handle;
  desc.size = size;
  if (is_dedicated) {
    desc.flags |= CUDA_EXTERNAL_MEMORY_DEDICATED;
  }

  CUDADriver::get_instance().import_external_memory(&ext_mem, &desc);
  return ext_mem;
}
#else
CUexternalMemory import_vk_memory_object_from_handle(int fd,
                                                     unsigned long long size,
                                                     bool is_dedicated) {
  CUexternalMemory ext_mem = nullptr;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc = {};

  memset(&desc, 0, sizeof(desc));

  desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  desc.size = size;
  if (is_dedicated) {
    desc.flags |= CUDA_EXTERNAL_MEMORY_DEDICATED;
  }
  CUDADriver::get_instance().import_external_memory(&ext_mem, &desc);
  return ext_mem;
}
#endif

void *map_buffer_onto_external_memory(CUexternalMemory ext_mem,
                                      unsigned long long offset,
                                      unsigned long long size) {
  void *ptr = nullptr;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC desc = {};

  memset(&desc, 0, sizeof(desc));

  desc.offset = offset;
  desc.size = size;

  CUDADriver::get_instance().external_memory_get_mapped_buffer(
      (CUdeviceptr *)&ptr, ext_mem, &desc);
  return ptr;
}

ImportedVulkanMemory get_cuda_memory_pointer(VkDeviceMemory mem,
                                              VkDeviceSize mem_size,
                                              VkDeviceSize offset,
                                              VkDeviceSize buffer_size,
                                              VkDevice device) {
  auto handle = get_device_mem_handle(mem, device);
  ScopedExternalMemoryHandle handle_guard(handle);
  CUexternalMemory external_memory =
      import_vk_memory_object_from_handle(handle, mem_size, false);
#if !defined(_WIN32) && !defined(_WIN64) && !defined(WIN32) && \
    !defined(_MSC_VER)
  // CUDA owns an OPAQUE_FD after a successful import. Win32 NT handles remain
  // application-owned and are closed by ScopedExternalMemoryHandle instead.
  handle_guard.release_to_cuda();
#endif
  return {external_memory,
          static_cast<unsigned char *>(map_buffer_onto_external_memory(
              external_memory, offset, buffer_size))};
}

unsigned char *get_or_import_vulkan_allocation(
    VulkanDevice *vk_dev,
    CudaDevice *cuda_dev,
    DeviceAllocation allocation,
    VulkanAllocCacheKey key) {
  vk_dev->set_interop_cleanup_callbacks(&release_vulkan_interop_allocation,
                                        &release_vulkan_interop_device);
  auto &cache = vulkan_cuda_interop_cache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  auto &allocation_base_ptrs = cache.allocation_base_ptrs[vk_dev][cuda_dev];
  auto it = allocation_base_ptrs.find(key);
  if (it != allocation_base_ptrs.end()) {
    return it->second.mapped_buffer;
  }

  auto [base_mem, alloc_offset, alloc_size] =
      vk_dev->get_vkmemory_offset_size(allocation);
  TI_ERROR_IF(alloc_offset > std::numeric_limits<size_t>::max() - alloc_size,
              "Vulkan external-memory size overflow");
  // This may be smaller than the complete VkDeviceMemory allocation, but it
  // covers this buffer's mapped region and is enough for this cache entry.
  const size_t mem_size = alloc_offset + alloc_size;
  auto imported_memory = get_cuda_memory_pointer(
      base_mem, mem_size, alloc_offset, alloc_size, vk_dev->vk_device());
  auto [inserted, was_inserted] =
      allocation_base_ptrs.emplace(key, std::move(imported_memory));
  TI_ASSERT(was_inserted);
  return inserted->second.mapped_buffer;
}

void memcpy_cuda_to_vulkan_impl(VulkanDevice *vk_dev,
                                CudaDevice *cuda_dev,
                                DevicePtr dst,
                                DevicePtr src,
                                uint64_t size) {
  auto context_guard = CUDAContext::get_instance().get_guard();
  DeviceAllocation dst_alloc(dst);

  VulkanAllocCacheKey dst_key{
      dst_alloc.alloc_id, vk_dev->allocation_generation(dst_alloc)};
  unsigned char *dst_cuda_ptr =
      get_or_import_vulkan_allocation(vk_dev, cuda_dev, dst_alloc, dst_key) +
      dst.offset;

  TI_ASSERT(cuda_dev->copy_to_external(dst_cuda_ptr, src, size) ==
            RhiResult::success);
  CUDADriver::get_instance().stream_synchronize(nullptr);
}

}  // namespace

bool is_cuda_to_vulkan_copy(Device *dst_device, Device *src_device) {
  return dynamic_cast<VulkanDevice *>(dst_device) &&
         dynamic_cast<CudaDevice *>(src_device);
}

void memcpy_cuda_to_vulkan_fast(DevicePtr dst, DevicePtr src, uint64_t size) {
  VulkanDevice *vk_dev = dynamic_cast<VulkanDevice *>(dst.device);
  CudaDevice *cuda_dev = dynamic_cast<CudaDevice *>(src.device);
  TI_ASSERT(vk_dev && cuda_dev);
  memcpy_cuda_to_vulkan_impl(vk_dev, cuda_dev, dst, src, size);
}

void memcpy_cuda_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size) {
  VulkanDevice *vk_dev = dynamic_cast<VulkanDevice *>(dst.device);
  CudaDevice *cuda_dev = dynamic_cast<CudaDevice *>(src.device);
  memcpy_cuda_to_vulkan_impl(vk_dev, cuda_dev, dst, src, size);
}

void memcpy_vulkan_to_cuda(DevicePtr dst, DevicePtr src, uint64_t size) {
  VulkanDevice *vk_dev = dynamic_cast<VulkanDevice *>(src.device);
  CudaDevice *cuda_dev = dynamic_cast<CudaDevice *>(dst.device);
  auto context_guard = CUDAContext::get_instance().get_guard();

  DeviceAllocation src_alloc(src);

  VulkanAllocCacheKey src_key{
      src_alloc.alloc_id, vk_dev->allocation_generation(src_alloc)};
  unsigned char *src_cuda_ptr =
      get_or_import_vulkan_allocation(vk_dev, cuda_dev, src_alloc, src_key) +
      src.offset;

  TI_ASSERT(cuda_dev->copy_from_external(dst, src_cuda_ptr, size) ==
            RhiResult::success);
  CUDADriver::get_instance().stream_synchronize(nullptr);
}

#else
bool is_cuda_to_vulkan_copy(Device *dst_device, Device *src_device) {
  return false;
}

void memcpy_cuda_to_vulkan_fast(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

void memcpy_cuda_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}

void memcpy_vulkan_to_cuda(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}
#endif  // TI_WITH_VULKAN && TI_WITH_CUDA

}  // namespace taichi::lang
