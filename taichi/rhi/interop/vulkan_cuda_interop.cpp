#include "taichi/rhi/interop/vulkan_cuda_interop.h"

#if TI_WITH_VULKAN && TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif  // TI_WITH_VULKAN && TI_WITH_CUDA

#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <tuple>
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

void destroy_imported_vulkan_memory(ImportedVulkanMemory &entry) {
  if (entry.mapped_buffer != nullptr) {
    CUDADriver::get_instance().mem_free(entry.mapped_buffer);
    entry.mapped_buffer = nullptr;
  }
  if (entry.external_memory != nullptr) {
    CUDADriver::get_instance().external_memory_destroy(entry.external_memory);
    entry.external_memory = nullptr;
  }
}

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || \
    defined(_MSC_VER)
class ScopedExternalHandle {
 public:
  explicit ScopedExternalHandle(HANDLE handle) : handle_(handle) {
  }
  ~ScopedExternalHandle() {
    if (handle_ != nullptr) {
      CloseHandle(handle_);
    }
  }

 private:
  HANDLE handle_{nullptr};
};
#else
class ScopedExternalHandle {
 public:
  explicit ScopedExternalHandle(int fd) : fd_(fd) {
  }
  ~ScopedExternalHandle() {
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
  for (auto &entry : entries) {
    destroy_imported_vulkan_memory(entry);
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

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
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

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
HANDLE get_device_semaphore_handle(VkSemaphore semaphore, VkDevice device) {
  VkSemaphoreGetWin32HandleInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  info.semaphore = semaphore;
  info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  auto get_handle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
      vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
  TI_ERROR_IF(get_handle == nullptr,
              "vkGetSemaphoreWin32HandleKHR is unavailable");
  HANDLE handle = nullptr;
  TI_ERROR_IF(get_handle(device, &info, &handle) != VK_SUCCESS,
              "Unable to export Vulkan semaphore Win32 handle");
  return handle;
}
#else
int get_device_semaphore_handle(VkSemaphore semaphore, VkDevice device) {
  VkSemaphoreGetFdInfoKHR info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  info.semaphore = semaphore;
  info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
  auto get_fd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
      vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
  TI_ERROR_IF(get_fd == nullptr, "vkGetSemaphoreFdKHR is unavailable");
  int fd = -1;
  TI_ERROR_IF(get_fd(device, &info, &fd) != VK_SUCCESS,
              "Unable to export Vulkan semaphore file descriptor");
  return fd;
}
#endif

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
CUexternalSemaphore import_vk_semaphore_from_handle(HANDLE handle) {
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc{};
  desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
  desc.handle.win32.handle = handle;
  CUexternalSemaphore semaphore = nullptr;
  CUDADriver::get_instance().import_external_semaphore(&semaphore, &desc);
  return semaphore;
}
#else
CUexternalSemaphore import_vk_semaphore_from_handle(int fd) {
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc{};
  desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  CUexternalSemaphore semaphore = nullptr;
  CUDADriver::get_instance().import_external_semaphore(&semaphore, &desc);
  return semaphore;
}
#endif

VkExternalMemoryHandleTypeFlagBits external_memory_handle_type() {
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

VkExternalSemaphoreHandleTypeFlagBits external_semaphore_handle_type() {
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

void get_vulkan_device_uuid(VulkanDevice *device,
                            std::array<unsigned char, VK_UUID_SIZE> &uuid) {
  VkPhysicalDeviceIDProperties id_properties{};
  id_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
  VkPhysicalDeviceProperties2 properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties.pNext = &id_properties;
  auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
      vkGetInstanceProcAddr(device->vk_instance(),
                            "vkGetPhysicalDeviceProperties2"));
  if (get_properties == nullptr) {
    get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        vkGetInstanceProcAddr(device->vk_instance(),
                              "vkGetPhysicalDeviceProperties2KHR"));
  }
  TI_ERROR_IF(get_properties == nullptr,
              "Vulkan physical-device UUID query is unavailable");
  get_properties(device->vk_physical_device(), &properties);
  std::memcpy(uuid.data(), id_properties.deviceUUID, uuid.size());
}

void validate_vulkan_cuda_device_identity(VulkanDevice *vulkan_device) {
  std::array<unsigned char, VK_UUID_SIZE> vulkan_uuid{};
  get_vulkan_device_uuid(vulkan_device, vulkan_uuid);
  CUuuid cuda_uuid{};
  auto &cuda_context = CUDAContext::get_instance();
  CUDADriver::get_instance().device_get_uuid(&cuda_uuid,
                                             cuda_context.get_device());
  TI_ERROR_IF(
      std::memcmp(vulkan_uuid.data(), cuda_uuid.bytes, vulkan_uuid.size()) != 0,
      "Vulkan and CUDA devices have different UUIDs");
}

void validate_external_interop_capabilities(VulkanDevice *device) {
  TI_ERROR_IF(!device->vk_caps().external_memory,
              "Vulkan external memory is unavailable");
  TI_ERROR_IF(!device->vk_caps().external_semaphore,
              "Vulkan external semaphore is unavailable");

  VkPhysicalDeviceExternalBufferInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
  buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_info.handleType = external_memory_handle_type();
  VkExternalBufferProperties buffer_properties{};
  buffer_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
  auto get_buffer_properties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceExternalBufferProperties>(
          vkGetInstanceProcAddr(device->vk_instance(),
                                "vkGetPhysicalDeviceExternalBufferProperties"));
  if (get_buffer_properties == nullptr) {
    get_buffer_properties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceExternalBufferProperties>(
            vkGetInstanceProcAddr(
                device->vk_instance(),
                "vkGetPhysicalDeviceExternalBufferPropertiesKHR"));
  }
  TI_ERROR_IF(get_buffer_properties == nullptr,
              "Vulkan external-buffer capability query is unavailable");
  get_buffer_properties(device->vk_physical_device(), &buffer_info,
                        &buffer_properties);
  TI_ERROR_IF(
      !(buffer_properties.externalMemoryProperties.externalMemoryFeatures &
        VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT),
      "Vulkan buffer handle type is not exportable");

  VkPhysicalDeviceExternalSemaphoreInfo semaphore_info{};
  semaphore_info.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
  semaphore_info.handleType = external_semaphore_handle_type();
  VkExternalSemaphoreProperties semaphore_properties{};
  semaphore_properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
  auto get_semaphore_properties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceExternalSemaphoreProperties>(
          vkGetInstanceProcAddr(
              device->vk_instance(),
              "vkGetPhysicalDeviceExternalSemaphoreProperties"));
  if (get_semaphore_properties == nullptr) {
    get_semaphore_properties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceExternalSemaphoreProperties>(
            vkGetInstanceProcAddr(
                device->vk_instance(),
                "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR"));
  }
  TI_ERROR_IF(get_semaphore_properties == nullptr,
              "Vulkan external-semaphore capability query is unavailable");
  get_semaphore_properties(device->vk_physical_device(), &semaphore_info,
                           &semaphore_properties);
  TI_ERROR_IF(!(semaphore_properties.externalSemaphoreFeatures &
                VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT),
              "Vulkan semaphore handle type is not exportable");
}

struct ImportedExternalSemaphore {
  vkapi::IVkSemaphore vulkan;
  StreamSemaphore stream;
  CUexternalSemaphore cuda{nullptr};
};

ImportedExternalSemaphore create_external_semaphore(VulkanDevice *device) {
  VkExportSemaphoreCreateInfo export_info{};
  export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
  export_info.handleTypes = external_semaphore_handle_type();
  auto vulkan_semaphore =
      vkapi::create_semaphore(device->vk_device(), 0, &export_info);
  auto handle = get_device_semaphore_handle(vulkan_semaphore->semaphore,
                                            device->vk_device());
  ScopedExternalHandle handle_guard(handle);
  auto cuda_semaphore = import_vk_semaphore_from_handle(handle);
#if !defined(_WIN32) && !defined(_WIN64) && !defined(WIN32) && \
    !defined(_MSC_VER)
  handle_guard.release_to_cuda();
#endif
  auto stream_semaphore = std::make_shared<VulkanStreamSemaphoreObject>(
      device->backend_fault_reporter(), vulkan_semaphore);
  return {std::move(vulkan_semaphore), std::move(stream_semaphore),
          cuda_semaphore};
}

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
                                             VkDevice device,
                                             bool is_dedicated = false) {
  auto handle = get_device_mem_handle(mem, device);
  ScopedExternalHandle handle_guard(handle);
  CUexternalMemory external_memory =
      import_vk_memory_object_from_handle(handle, mem_size, is_dedicated);
#if !defined(_WIN32) && !defined(_WIN64) && !defined(WIN32) && \
    !defined(_MSC_VER)
  // CUDA owns an OPAQUE_FD after a successful import. Win32 NT handles remain
  // application-owned and are closed by ScopedExternalHandle instead.
  handle_guard.release_to_cuda();
#endif
  ImportedVulkanMemory imported{external_memory, nullptr};
  try {
    imported.mapped_buffer = static_cast<unsigned char *>(
        map_buffer_onto_external_memory(external_memory, offset, buffer_size));
  } catch (...) {
    destroy_imported_vulkan_memory(imported);
    throw;
  }
  return imported;
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
  decltype(allocation_base_ptrs.begin()) inserted;
  bool was_inserted = false;
  try {
    std::tie(inserted, was_inserted) =
        allocation_base_ptrs.emplace(key, imported_memory);
  } catch (...) {
    destroy_imported_vulkan_memory(imported_memory);
    throw;
  }
  TI_ASSERT(was_inserted);
  return inserted->second.mapped_buffer;
}

void memcpy_cuda_to_vulkan_impl(VulkanDevice *vk_dev,
                                CudaDevice *cuda_dev,
                                DevicePtr dst,
                                DevicePtr src,
                                uint64_t size) {
  if (size == 0) {
    return;
  }
  if (!vk_dev->vk_caps().external_memory) {
    TI_ERROR_IF(size > std::numeric_limits<size_t>::max(),
                "CUDA-Vulkan host staging size overflow");
    std::vector<uint8_t> host_staging(static_cast<size_t>(size));
    DevicePtr source = src;
    void *host_ptr = host_staging.data();
    size_t copy_size = host_staging.size();
    TI_ASSERT(cuda_dev->readback_data(&source, &host_ptr, &copy_size) ==
              RhiResult::success);
    DevicePtr destination = dst;
    const void *input_ptr = host_staging.data();
    TI_ASSERT(vk_dev->upload_data(&destination, &input_ptr, &copy_size) ==
              RhiResult::success);
    return;
  }

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

class VulkanCudaExternalAllocation::Impl {
 public:
  Impl(VulkanDevice *vulkan_device,
       CudaDevice *cuda_device,
       DeviceAllocation vulkan_allocation)
      : vulkan_device_(vulkan_device), cuda_device_(cuda_device) {
    TI_ERROR_IF(vulkan_device_ == nullptr || cuda_device_ == nullptr,
                "Vulkan-CUDA interop requires both devices");
    TI_ERROR_IF(vulkan_allocation == kDeviceNullAllocation ||
                    vulkan_allocation.device != vulkan_device_,
                "Vulkan-CUDA interop received an invalid Vulkan allocation");
    static std::atomic<std::uint64_t> next_identity{1};
    identity_ = next_identity.fetch_add(1, std::memory_order_relaxed);
    TI_ERROR_IF(identity_ == 0, "External synchronization domain exhausted");

    validate_external_interop_capabilities(vulkan_device_);
    validate_vulkan_cuda_device_identity(vulkan_device_);
    auto [memory, offset, size] =
        vulkan_device_->get_vkmemory_offset_size(vulkan_allocation);
    TI_ERROR_IF(size == 0, "Cannot import an empty Vulkan allocation");
    TI_ERROR_IF(offset != 0,
                "Vulkan-CUDA sharing requires a dedicated Vulkan allocation");
    allocation_size_ = size;

    auto context_guard = CUDAContext::get_instance().get_guard();
    try {
      imported_memory_ = get_cuda_memory_pointer(
          memory, size, 0, size, vulkan_device_->vk_device(), true);
      cuda_allocation_ = cuda_device_->import_memory(
          imported_memory_.mapped_buffer, allocation_size_);
      vulkan_to_cuda_ = create_external_semaphore(vulkan_device_);
      cuda_to_vulkan_ = create_external_semaphore(vulkan_device_);
    } catch (...) {
      destroy_resources();
      throw;
    }
  }

  ~Impl() {
    try {
      close();
    } catch (const std::exception &error) {
      TI_WARN("Vulkan-CUDA interop close failed during destruction: {}",
              error.what());
    } catch (...) {
      TI_WARN("Vulkan-CUDA interop close failed during destruction");
    }
  }

  std::uint64_t identity() const noexcept {
    return identity_;
  }

  DeviceAllocation cuda_allocation() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return cuda_allocation_;
  }

  std::size_t allocation_size() const noexcept {
    return allocation_size_;
  }

  AccessState access_state() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
  }

  PrepareCudaAccessResult prepare_cuda_access(VulkanStream &stream,
                                              CommandList *cmdlist) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    if (state_ == AccessState::kAwaitingCudaAcquire) {
      return PrepareCudaAccessResult::kAlreadyReleased;
    }
    TI_ERROR_IF(
        state_ != AccessState::kVulkanOwned &&
            state_ != AccessState::kAwaitingVulkanAcquire,
        "Vulkan-CUDA allocation cannot prepare CUDA access from state {}",
        static_cast<int>(state_));
    TI_ERROR_IF(cmdlist == nullptr,
                "Vulkan-CUDA allocation received a null handoff command");
    bind_vulkan_stream(stream);

    const bool rearm_discarded_cuda_producer =
        state_ == AccessState::kAwaitingVulkanAcquire;
    StreamSemaphore completion;
    if (rearm_discarded_cuda_producer) {
      completion = stream.submit_with_semaphores(
          cmdlist, {cuda_to_vulkan_.stream}, {vulkan_to_cuda_.stream});
    } else {
      completion = stream.submit_with_semaphores(
          cmdlist, {}, {vulkan_to_cuda_.stream});
    }
    TI_ERROR_IF(!completion, "Vulkan-to-CUDA handoff submission failed");
    last_vulkan_completion_ = completion;
    active_cuda_stream_ = {};
    state_ = AccessState::kAwaitingCudaAcquire;
    return rearm_discarded_cuda_producer
               ? PrepareCudaAccessResult::kRearmedAfterCudaRelease
               : PrepareCudaAccessResult::kReleasedFromVulkan;
  }

  StreamSemaphore release_vulkan_to_cuda(VulkanStream &stream,
                                         CommandList *cmdlist) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    TI_ERROR_IF(state_ != AccessState::kVulkanOwned,
                "Vulkan-CUDA allocation is not owned by Vulkan");
    bind_vulkan_stream(stream);
    auto completion =
        stream.submit_with_semaphores(cmdlist, {}, {vulkan_to_cuda_.stream});
    TI_ERROR_IF(!completion, "Vulkan producer submission failed");
    last_vulkan_completion_ = completion;
    state_ = AccessState::kAwaitingCudaAcquire;
    return completion;
  }

  void acquire_for_consumer(const ExternalStreamDomain &stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    validate_cuda_stream(stream);
    TI_ERROR_IF(state_ != AccessState::kAwaitingCudaAcquire,
                "CUDA acquire does not follow a Vulkan release");
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().wait_external_semaphore_async(
        &vulkan_to_cuda_.cuda, &params, 1,
        static_cast<CUstream>(stream.native_stream));
    active_cuda_stream_ = stream;
    state_ = AccessState::kCudaOwned;
  }

  void release_from_consumer(const ExternalStreamDomain &stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    validate_cuda_stream(stream);
    TI_ERROR_IF(state_ != AccessState::kCudaOwned,
                "CUDA release does not follow a CUDA acquire");
    TI_ERROR_IF(!active_cuda_stream_.same_stream(stream),
                "CUDA acquire and release must use the same stream domain");
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{};
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().signal_external_semaphore_async(
        &cuda_to_vulkan_.cuda, &params, 1,
        static_cast<CUstream>(stream.native_stream));
    state_ = AccessState::kAwaitingVulkanAcquire;
  }

  StreamSemaphore acquire_vulkan_from_cuda(VulkanStream &stream,
                                           CommandList *cmdlist) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    TI_ERROR_IF(state_ != AccessState::kAwaitingVulkanAcquire,
                "Vulkan acquire does not follow a CUDA release");
    bind_vulkan_stream(stream);
    auto completion = stream.submit(cmdlist, {cuda_to_vulkan_.stream});
    TI_ERROR_IF(!completion, "Vulkan consumer submission failed");
    last_vulkan_completion_ = completion;
    active_cuda_stream_ = {};
    state_ = AccessState::kVulkanOwned;
    return completion;
  }

  StreamSemaphore cycle_vulkan_to_cuda(
      VulkanStream &stream,
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &additional_waits) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    TI_ERROR_IF(state_ != AccessState::kAwaitingVulkanAcquire,
                "Vulkan frame cycle does not follow a CUDA release");
    bind_vulkan_stream(stream);
    std::vector<StreamSemaphore> waits = additional_waits;
    waits.push_back(cuda_to_vulkan_.stream);
    auto completion = stream.submit_with_semaphores(
        cmdlist, waits, {vulkan_to_cuda_.stream});
    TI_ERROR_IF(!completion, "Vulkan frame cycle submission failed");
    last_vulkan_completion_ = completion;
    active_cuda_stream_ = {};
    state_ = AccessState::kAwaitingCudaAcquire;
    return completion;
  }

  bool closed() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ == AccessState::kClosed;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == AccessState::kClosed) {
      return;
    }

    if ((state_ == AccessState::kAwaitingCudaAcquire ||
         state_ == AccessState::kVulkanOwned) &&
        last_vulkan_completion_) {
      TI_ERROR_IF(!last_vulkan_completion_->wait(),
                  "Failed to wait for the last Vulkan interop submission");
    }
    if (state_ == AccessState::kCudaOwned ||
        state_ == AccessState::kAwaitingVulkanAcquire) {
      auto context_guard = CUDAContext::get_instance().get_guard();
      CUDADriver::get_instance().stream_synchronize(
          static_cast<CUstream>(active_cuda_stream_.native_stream));
    }
    destroy_resources();
    state_ = AccessState::kClosed;
  }

 private:
  void require_open() const {
    TI_ERROR_IF(state_ == AccessState::kClosed,
                "Vulkan-CUDA interop allocation is closed");
  }

  void validate_cuda_stream(const ExternalStreamDomain &stream) const {
    TI_ERROR_IF(!stream.valid() || stream.api != ExternalExecutionApi::kCuda,
                "Vulkan-CUDA synchronization requires an explicit CUDA "
                "stream domain");
  }

  void bind_vulkan_stream(VulkanStream &stream) {
    TI_ERROR_IF(
        bound_vulkan_stream_ != nullptr && bound_vulkan_stream_ != &stream,
        "A Vulkan-CUDA synchronization domain cannot switch Vulkan "
        "streams");
    bound_vulkan_stream_ = &stream;
  }

  void destroy_resources() {
    auto context_guard = CUDAContext::get_instance().get_guard();
    if (cuda_allocation_ != kDeviceNullAllocation && cuda_device_ != nullptr) {
      cuda_device_->dealloc_memory(cuda_allocation_);
      cuda_allocation_ = kDeviceNullAllocation;
    }
    if (vulkan_to_cuda_.cuda != nullptr) {
      CUDADriver::get_instance().external_semaphore_destroy(
          vulkan_to_cuda_.cuda);
      vulkan_to_cuda_.cuda = nullptr;
    }
    if (cuda_to_vulkan_.cuda != nullptr) {
      CUDADriver::get_instance().external_semaphore_destroy(
          cuda_to_vulkan_.cuda);
      cuda_to_vulkan_.cuda = nullptr;
    }
    vulkan_to_cuda_.stream.reset();
    cuda_to_vulkan_.stream.reset();
    vulkan_to_cuda_.vulkan.reset();
    cuda_to_vulkan_.vulkan.reset();
    destroy_imported_vulkan_memory(imported_memory_);
    last_vulkan_completion_.reset();
  }

  mutable std::mutex mutex_;
  VulkanDevice *vulkan_device_{nullptr};
  CudaDevice *cuda_device_{nullptr};
  VulkanStream *bound_vulkan_stream_{nullptr};
  std::uint64_t identity_{0};
  std::size_t allocation_size_{0};
  DeviceAllocation cuda_allocation_{kDeviceNullAllocation};
  ImportedVulkanMemory imported_memory_;
  ImportedExternalSemaphore vulkan_to_cuda_;
  ImportedExternalSemaphore cuda_to_vulkan_;
  ExternalStreamDomain active_cuda_stream_;
  StreamSemaphore last_vulkan_completion_;
  AccessState state_{AccessState::kVulkanOwned};
};

std::shared_ptr<VulkanCudaExternalAllocation>
VulkanCudaExternalAllocation::create(VulkanDevice *vulkan_device,
                                     CudaDevice *cuda_device,
                                     DeviceAllocation vulkan_allocation) {
  return std::shared_ptr<VulkanCudaExternalAllocation>(
      new VulkanCudaExternalAllocation(std::make_unique<Impl>(
          vulkan_device, cuda_device, vulkan_allocation)));
}

VulkanCudaExternalAllocation::VulkanCudaExternalAllocation(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}

VulkanCudaExternalAllocation::~VulkanCudaExternalAllocation() = default;

std::uint64_t VulkanCudaExternalAllocation::identity() const noexcept {
  return impl_->identity();
}

DeviceAllocation VulkanCudaExternalAllocation::cuda_allocation()
    const noexcept {
  return impl_->cuda_allocation();
}

std::size_t VulkanCudaExternalAllocation::allocation_size() const noexcept {
  return impl_->allocation_size();
}

VulkanCudaExternalAllocation::AccessState
VulkanCudaExternalAllocation::access_state() const noexcept {
  return impl_->access_state();
}

VulkanCudaExternalAllocation::PrepareCudaAccessResult
VulkanCudaExternalAllocation::prepare_cuda_access(
    VulkanStream &stream,
    CommandList *cmdlist) {
  return impl_->prepare_cuda_access(stream, cmdlist);
}

bool VulkanCudaExternalAllocation::closed() const noexcept {
  return impl_->closed();
}

StreamSemaphore VulkanCudaExternalAllocation::release_vulkan_to_cuda(
    VulkanStream &stream,
    CommandList *cmdlist) {
  return impl_->release_vulkan_to_cuda(stream, cmdlist);
}

void VulkanCudaExternalAllocation::acquire_for_consumer(
    const ExternalStreamDomain &stream) {
  impl_->acquire_for_consumer(stream);
}

void VulkanCudaExternalAllocation::release_from_consumer(
    const ExternalStreamDomain &stream) {
  impl_->release_from_consumer(stream);
}

StreamSemaphore VulkanCudaExternalAllocation::acquire_vulkan_from_cuda(
    VulkanStream &stream,
    CommandList *cmdlist) {
  return impl_->acquire_vulkan_from_cuda(stream, cmdlist);
}

StreamSemaphore VulkanCudaExternalAllocation::cycle_vulkan_to_cuda(
    VulkanStream &stream,
    CommandList *cmdlist,
    const std::vector<StreamSemaphore> &additional_waits) {
  return impl_->cycle_vulkan_to_cuda(stream, cmdlist, additional_waits);
}

void VulkanCudaExternalAllocation::close() {
  impl_->close();
}

bool is_cuda_to_vulkan_copy(Device *dst_device, Device *src_device) {
  auto *vk_dev = dynamic_cast<VulkanDevice *>(dst_device);
  return vk_dev != nullptr &&
         dynamic_cast<CudaDevice *>(src_device) != nullptr &&
         vk_dev->vk_caps().external_memory;
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
  if (size == 0) {
    return;
  }
  if (!vk_dev->vk_caps().external_memory) {
    TI_ERROR_IF(size > std::numeric_limits<size_t>::max(),
                "Vulkan-CUDA host staging size overflow");
    std::vector<uint8_t> host_staging(static_cast<size_t>(size));
    DevicePtr source = src;
    void *host_ptr = host_staging.data();
    size_t copy_size = host_staging.size();
    TI_ASSERT(vk_dev->readback_data(&source, &host_ptr, &copy_size) ==
              RhiResult::success);
    DevicePtr destination = dst;
    const void *input_ptr = host_staging.data();
    TI_ASSERT(cuda_dev->upload_data(&destination, &input_ptr, &copy_size) ==
              RhiResult::success);
    return;
  }
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
class VulkanCudaExternalAllocation::Impl {};

std::shared_ptr<VulkanCudaExternalAllocation>
VulkanCudaExternalAllocation::create(vulkan::VulkanDevice *,
                                     cuda::CudaDevice *,
                                     DeviceAllocation) {
  TI_ERROR("Vulkan-CUDA external allocation requires both backends");
  return nullptr;
}

VulkanCudaExternalAllocation::VulkanCudaExternalAllocation(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}
VulkanCudaExternalAllocation::~VulkanCudaExternalAllocation() = default;
std::uint64_t VulkanCudaExternalAllocation::identity() const noexcept {
  return 0;
}
DeviceAllocation VulkanCudaExternalAllocation::cuda_allocation()
    const noexcept {
  return kDeviceNullAllocation;
}
std::size_t VulkanCudaExternalAllocation::allocation_size() const noexcept {
  return 0;
}
VulkanCudaExternalAllocation::AccessState
VulkanCudaExternalAllocation::access_state() const noexcept {
  return AccessState::kClosed;
}
VulkanCudaExternalAllocation::PrepareCudaAccessResult
VulkanCudaExternalAllocation::prepare_cuda_access(vulkan::VulkanStream &,
                                                  CommandList *) {
  TI_NOT_IMPLEMENTED;
  return PrepareCudaAccessResult::kAlreadyReleased;
}
bool VulkanCudaExternalAllocation::closed() const noexcept {
  return true;
}
StreamSemaphore VulkanCudaExternalAllocation::release_vulkan_to_cuda(
    vulkan::VulkanStream &,
    CommandList *) {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}
void VulkanCudaExternalAllocation::acquire_for_consumer(
    const ExternalStreamDomain &) {
  TI_NOT_IMPLEMENTED;
}
void VulkanCudaExternalAllocation::release_from_consumer(
    const ExternalStreamDomain &) {
  TI_NOT_IMPLEMENTED;
}
StreamSemaphore VulkanCudaExternalAllocation::acquire_vulkan_from_cuda(
    vulkan::VulkanStream &,
    CommandList *) {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}
StreamSemaphore VulkanCudaExternalAllocation::cycle_vulkan_to_cuda(
    vulkan::VulkanStream &,
    CommandList *,
    const std::vector<StreamSemaphore> &) {
  TI_NOT_IMPLEMENTED;
  return nullptr;
}
void VulkanCudaExternalAllocation::close() {
  TI_NOT_IMPLEMENTED;
}

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
