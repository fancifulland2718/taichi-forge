#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "taichi/rhi/device.h"
#include "taichi/rhi/interop/external_sync.h"

namespace taichi::lang {

namespace cuda {
class CudaDevice;
}

enum class ExternalOpaqueHandleType : std::uint8_t {
  kOpaqueWin32,
  kOpaqueFd,
};

// Ownership of a valid handle transfers to the import call. Win32 NT handles
// are closed after CUDA has imported them. CUDA consumes OPAQUE_FD handles
// after a successful import; failed imports close the descriptor locally.
struct RHI_DLL_EXPORT ExternalOpaqueHandle {
  ExternalOpaqueHandleType type{ExternalOpaqueHandleType::kOpaqueWin32};
  std::uintptr_t value{0};

  bool valid() const noexcept {
    return value != 0;
  }
};

struct RHI_DLL_EXPORT CudaExternalMemoryImport {
  ExternalOpaqueHandle handle;
  std::size_t allocation_size{0};
  bool dedicated{true};
  std::array<std::uint8_t, 16> device_uuid{};
};

struct RHI_DLL_EXPORT CudaExternalSemaphorePairImport {
  // The external producer signals ready_for_cuda; CUDA waits before accessing
  // the allocation. CUDA signals ready_for_external after the access epoch.
  ExternalOpaqueHandle ready_for_cuda;
  ExternalOpaqueHandle ready_for_external;
};

// CUDA mapping of an allocation exported by an external API such as Vulkan.
// It is also an ExternalSynchronizationDomain when a semaphore pair was
// supplied. The allocation can be registered by multiple external-storage
// owners; every view shares this object's identity and access epoch.
class RHI_DLL_EXPORT CudaExternalAllocation final
    : public ExternalSynchronizationDomain {
 public:
  static std::shared_ptr<CudaExternalAllocation> create(
      cuda::CudaDevice *cuda_device,
      CudaExternalMemoryImport memory,
      std::optional<CudaExternalSemaphorePairImport> semaphores = std::nullopt);

  ~CudaExternalAllocation() override;
  CudaExternalAllocation(const CudaExternalAllocation &) = delete;
  CudaExternalAllocation &operator=(const CudaExternalAllocation &) = delete;

  std::uint64_t identity() const noexcept override;
  bool retirement_waits_for_consumer() const noexcept override {
    return synchronized();
  }
  void acquire_for_consumer(const ExternalStreamDomain &stream) override;
  void release_from_consumer(const ExternalStreamDomain &stream) override;

  DeviceAllocation cuda_allocation() const noexcept;
  std::size_t allocation_size() const noexcept;
  const std::array<std::uint8_t, 16> &device_uuid() const noexcept;
  int device_ordinal() const noexcept;
  bool synchronized() const noexcept;
  bool closed() const noexcept;

  // Idempotent. It waits only for this allocation's last participating CUDA
  // stream before releasing the imported mapping and CUDA-side semaphores.
  void close();

 private:
  class Impl;
  explicit CudaExternalAllocation(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

RHI_DLL_EXPORT std::array<std::uint8_t, 16> current_cuda_external_device_uuid();

}  // namespace taichi::lang
