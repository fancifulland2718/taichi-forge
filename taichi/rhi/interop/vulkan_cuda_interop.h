#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "taichi/rhi/device.h"
#include "taichi/rhi/interop/external_sync.h"

namespace taichi::lang {

namespace cuda {
class CudaDevice;
}
namespace vulkan {
class VulkanDevice;
class VulkanStream;
}  // namespace vulkan

// Shared Vulkan allocation imported into CUDA without a staging copy. The
// object owns the CUDA external-memory mapping, the imported CudaDevice
// allocation, and a reusable pair of external binary semaphores. It borrows
// the Vulkan allocation; callers must keep that allocation and both devices
// alive until close() returns.
class RHI_DLL_EXPORT VulkanCudaExternalAllocation final
    : public ExternalSynchronizationDomain {
 public:
  enum class AccessState : std::uint8_t {
    kVulkanOwned,
    kAwaitingCudaAcquire,
    kCudaOwned,
    kAwaitingVulkanAcquire,
    kClosed,
  };

  static std::shared_ptr<VulkanCudaExternalAllocation> create(
      vulkan::VulkanDevice *vulkan_device,
      cuda::CudaDevice *cuda_device,
      DeviceAllocation vulkan_allocation);

  ~VulkanCudaExternalAllocation() override;
  VulkanCudaExternalAllocation(const VulkanCudaExternalAllocation &) = delete;
  VulkanCudaExternalAllocation &operator=(
      const VulkanCudaExternalAllocation &) = delete;

  std::uint64_t identity() const noexcept override;
  DeviceAllocation cuda_allocation() const noexcept;
  std::size_t allocation_size() const noexcept;
  AccessState access_state() const noexcept;
  bool closed() const noexcept;

  // Submit Vulkan work that produces the shared allocation and signals CUDA.
  // No host/device wait is introduced.
  StreamSemaphore release_vulkan_to_cuda(vulkan::VulkanStream &stream,
                                         CommandList *cmdlist);

  // Enqueue CUDA's wait/signal on the exact caller-provided stream. Both calls
  // in an epoch must use the same stream domain; any number of CUDA kernels may
  // be submitted between them.
  void acquire_for_consumer(const ExternalStreamDomain &stream) override;
  void release_from_consumer(const ExternalStreamDomain &stream) override;

  // Submit Vulkan work that waits for CUDA and consumes the shared allocation.
  // The returned completion token covers the wait and consumer command list.
  StreamSemaphore acquire_vulkan_from_cuda(vulkan::VulkanStream &stream,
                                           CommandList *cmdlist);

  // Idempotent. Close waits only for the last participating CUDA stream or
  // Vulkan fence when an epoch is still in flight; it never waits device-wide.
  void close();

 private:
  class Impl;
  explicit VulkanCudaExternalAllocation(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

bool is_cuda_to_vulkan_copy(Device *dst_device, Device *src_device);

void memcpy_cuda_to_vulkan_fast(DevicePtr dst, DevicePtr src, uint64_t size);

void memcpy_cuda_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size);

void memcpy_vulkan_to_cuda(DevicePtr dst, DevicePtr src, uint64_t size);

}  // namespace taichi::lang
