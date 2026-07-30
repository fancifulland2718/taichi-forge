#include "taichi/rhi/interop/cuda_external_interop.h"

#include <atomic>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <utility>

#include "taichi/common/logging.h"

#if TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#endif

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
#include "taichi/platform/windows/windows.h"
#else
#include <unistd.h>
#endif

namespace taichi::lang {

#if TI_WITH_CUDA

namespace {

using cuda::CudaDevice;

bool platform_accepts(ExternalOpaqueHandleType type) {
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
  return type == ExternalOpaqueHandleType::kOpaqueWin32;
#else
  return type == ExternalOpaqueHandleType::kOpaqueFd;
#endif
}

class ConsumedExternalHandle {
 public:
  explicit ConsumedExternalHandle(ExternalOpaqueHandle handle,
                                  bool owns_handle = true)
      : handle_(handle), owns_handle_(owns_handle) {
  }

  void validate() const {
    TI_ERROR_IF(!platform_accepts(handle_.type),
                "External interop handle type is unavailable on this "
                "platform");
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
    TI_ERROR_IF(handle_.value == 0 ||
                    reinterpret_cast<HANDLE>(handle_.value) ==
                        INVALID_HANDLE_VALUE,
                "External interop Win32 handle is invalid");
#else
    TI_ERROR_IF(handle_.value > static_cast<std::uintptr_t>(
                                    (std::numeric_limits<int>::max)()),
                "External interop file descriptor exceeds the native range");
#endif
  }

  ~ConsumedExternalHandle() {
    close_locally();
  }

  ConsumedExternalHandle(const ConsumedExternalHandle &) = delete;
  ConsumedExternalHandle &operator=(const ConsumedExternalHandle &) = delete;

  void *win32() const noexcept {
    return reinterpret_cast<void *>(handle_.value);
  }

  int fd() const noexcept {
    return static_cast<int>(handle_.value);
  }

  bool owns_distinct_handle() const noexcept {
    return owns_handle_;
  }

  void release_to_cuda() noexcept {
#if !defined(_WIN32) && !defined(_WIN64) && !defined(WIN32) && \
    !defined(_MSC_VER)
    owns_handle_ = false;
    handle_.value = (std::numeric_limits<std::uintptr_t>::max)();
#endif
  }

 private:
  void close_locally() noexcept {
    if (!owns_handle_) {
      return;
    }
    if (!platform_accepts(handle_.type)) {
      owns_handle_ = false;
      return;
    }
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
    if (handle_.value != 0 &&
        reinterpret_cast<HANDLE>(handle_.value) != INVALID_HANDLE_VALUE) {
      CloseHandle(reinterpret_cast<HANDLE>(handle_.value));
    }
#else
    if (handle_.value <= static_cast<std::uintptr_t>(
                             (std::numeric_limits<int>::max)())) {
      close(static_cast<int>(handle_.value));
    }
#endif
    owns_handle_ = false;
  }

  ExternalOpaqueHandle handle_;
  bool owns_handle_{true};
};

std::array<std::uint8_t, 16> query_cuda_device_uuid() {
  CUuuid uuid{};
  auto context_guard = CUDAContext::get_instance().get_guard();
  CUDADriver::get_instance().device_get_uuid(
      &uuid, CUDAContext::get_instance().get_device());
  std::array<std::uint8_t, 16> result{};
  static_assert(sizeof(uuid.bytes) == result.size());
  std::memcpy(result.data(), uuid.bytes, result.size());
  return result;
}

CUexternalMemory import_external_memory(const CudaExternalMemoryImport &memory,
                                        ConsumedExternalHandle &handle) {
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC descriptor{};
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
  descriptor.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
  descriptor.handle.win32.handle = handle.win32();
#else
  descriptor.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  descriptor.handle.fd = handle.fd();
#endif
  descriptor.size = memory.allocation_size;
  if (memory.dedicated) {
    descriptor.flags |= CUDA_EXTERNAL_MEMORY_DEDICATED;
  }

  CUexternalMemory external_memory = nullptr;
  CUDADriver::get_instance().import_external_memory(&external_memory,
                                                    &descriptor);
  handle.release_to_cuda();
  return external_memory;
}

CUexternalSemaphore import_external_semaphore(ConsumedExternalHandle &handle) {
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC descriptor{};
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(_MSC_VER)
  descriptor.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
  descriptor.handle.win32.handle = handle.win32();
#else
  descriptor.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
  descriptor.handle.fd = handle.fd();
#endif

  CUexternalSemaphore semaphore = nullptr;
  CUDADriver::get_instance().import_external_semaphore(&semaphore, &descriptor);
  handle.release_to_cuda();
  return semaphore;
}

}  // namespace

class CudaExternalAllocation::Impl {
 public:
  Impl(CudaDevice *cuda_device,
       CudaExternalMemoryImport memory,
       std::optional<CudaExternalSemaphorePairImport> semaphores)
      : cuda_device_(cuda_device),
        allocation_size_(memory.allocation_size),
        device_uuid_(memory.device_uuid),
        synchronized_(semaphores.has_value()) {
    // Take ownership before validation so every failure path releases all
    // handles supplied to create().
    ConsumedExternalHandle memory_handle(memory.handle);
    std::optional<ConsumedExternalHandle> ready_for_cuda_handle;
    std::optional<ConsumedExternalHandle> ready_for_external_handle;
    if (semaphores) {
      const bool cuda_handle_is_unique =
          semaphores->ready_for_cuda.type != memory.handle.type ||
          semaphores->ready_for_cuda.value != memory.handle.value;
      const bool external_handle_is_unique =
          (semaphores->ready_for_external.type != memory.handle.type ||
           semaphores->ready_for_external.value != memory.handle.value) &&
          (semaphores->ready_for_external.type !=
               semaphores->ready_for_cuda.type ||
           semaphores->ready_for_external.value !=
               semaphores->ready_for_cuda.value);
      ready_for_cuda_handle.emplace(semaphores->ready_for_cuda,
                                    cuda_handle_is_unique);
      ready_for_external_handle.emplace(semaphores->ready_for_external,
                                        external_handle_is_unique);
    }

    memory_handle.validate();
    if (semaphores) {
      ready_for_cuda_handle->validate();
      ready_for_external_handle->validate();
      TI_ERROR_IF(
          !ready_for_cuda_handle->owns_distinct_handle() ||
              !ready_for_external_handle->owns_distinct_handle(),
          "External memory and semaphore imports require distinct handles");
    }
    TI_ERROR_IF(cuda_device_ == nullptr,
                "External CUDA allocation requires a CUDA device");
    TI_ERROR_IF(allocation_size_ == 0,
                "External CUDA allocation size must be positive");
    TI_ERROR_IF(device_uuid_ != query_cuda_device_uuid(),
                "External allocation and CUDA devices have different UUIDs");

    static std::atomic<std::uint64_t> next_identity{1};
    identity_ = next_identity.fetch_add(1, std::memory_order_relaxed);
    TI_ERROR_IF(identity_ == 0, "External synchronization domain exhausted");

    auto context_guard = CUDAContext::get_instance().get_guard();
    try {
      external_memory_ = import_external_memory(memory, memory_handle);

      CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer_descriptor{};
      buffer_descriptor.size = allocation_size_;
      CUDADriver::get_instance().external_memory_get_mapped_buffer(
          reinterpret_cast<CUdeviceptr *>(&mapped_buffer_), external_memory_,
          &buffer_descriptor);
      TI_ERROR_IF(mapped_buffer_ == nullptr,
                  "CUDA external-memory import returned a null mapping");
      cuda_allocation_ =
          cuda_device_->import_memory(mapped_buffer_, allocation_size_);

      if (semaphores) {
        ready_for_cuda_ = import_external_semaphore(*ready_for_cuda_handle);
        ready_for_external_ =
            import_external_semaphore(*ready_for_external_handle);
      }
    } catch (...) {
      destroy_resources_noexcept();
      throw;
    }
  }

  ~Impl() {
    try {
      close();
    } catch (const std::exception &error) {
      TI_WARN("External CUDA allocation close failed during destruction: {}",
              error.what());
    } catch (...) {
      TI_WARN("External CUDA allocation close failed during destruction");
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

  const std::array<std::uint8_t, 16> &device_uuid() const noexcept {
    return device_uuid_;
  }

  int device_ordinal() const noexcept {
    return device_ordinal_;
  }

  bool synchronized() const noexcept {
    return synchronized_;
  }

  bool closed() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

  void acquire_for_consumer(const ExternalStreamDomain &stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    TI_ERROR_IF(!synchronized_,
                "External CUDA allocation has no synchronization domain");
    validate_cuda_stream(stream);
    TI_ERROR_IF(cuda_owned_, "External CUDA allocation was acquired twice");

    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params{};
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().wait_external_semaphore_async(
        &ready_for_cuda_, &params, 1,
        static_cast<CUstream>(stream.native_stream));
    active_cuda_stream_ = stream;
    cuda_owned_ = true;
  }

  void release_from_consumer(const ExternalStreamDomain &stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    TI_ERROR_IF(!synchronized_,
                "External CUDA allocation has no synchronization domain");
    validate_cuda_stream(stream);
    TI_ERROR_IF(!cuda_owned_,
                "External CUDA allocation release does not follow acquire");
    TI_ERROR_IF(!active_cuda_stream_.same_stream(stream),
                "External CUDA allocation changed streams during an access "
                "epoch");

    signal_external(stream);
    cuda_owned_ = false;
    last_cuda_stream_ = stream;
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }

    std::exception_ptr failure;
    const auto attempt = [&failure](auto &&action) {
      try {
        action();
      } catch (...) {
        if (!failure) {
          failure = std::current_exception();
        }
      }
    };

    if (cuda_owned_) {
      attempt([this] { signal_external(active_cuda_stream_); });
      cuda_owned_ = false;
      last_cuda_stream_ = active_cuda_stream_;
    }
    if (last_cuda_stream_.valid()) {
      attempt([this] {
        auto context_guard = CUDAContext::get_instance().get_guard();
        CUDADriver::get_instance().stream_synchronize(
            static_cast<CUstream>(last_cuda_stream_.native_stream));
      });
    }
    attempt([this] {
      auto context_guard = CUDAContext::get_instance().get_guard();
      destroy_resources();
    });
    closed_ = true;
    if (failure) {
      std::rethrow_exception(failure);
    }
  }

 private:
  void require_open() const {
    TI_ERROR_IF(closed_, "External CUDA allocation is closed");
  }

  void validate_cuda_stream(const ExternalStreamDomain &stream) {
    TI_ERROR_IF(!stream.valid() || stream.api != ExternalExecutionApi::kCuda,
                "External CUDA synchronization requires a CUDA stream "
                "domain");
    if (owner_domain_ == 0) {
      owner_domain_ = stream.owner_domain;
    }
    TI_ERROR_IF(owner_domain_ != stream.owner_domain,
                "External CUDA allocation belongs to another runtime");
  }

  void signal_external(const ExternalStreamDomain &stream) {
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params{};
    auto context_guard = CUDAContext::get_instance().get_guard();
    CUDADriver::get_instance().signal_external_semaphore_async(
        &ready_for_external_, &params, 1,
        static_cast<CUstream>(stream.native_stream));
  }

  void destroy_resources() {
    std::exception_ptr failure;
    const auto attempt = [&failure](auto &&action) {
      try {
        action();
      } catch (...) {
        if (!failure) {
          failure = std::current_exception();
        }
      }
    };

    attempt([this] {
      if (cuda_allocation_ != kDeviceNullAllocation &&
          cuda_device_ != nullptr) {
        cuda_device_->dealloc_memory(cuda_allocation_);
        cuda_allocation_ = kDeviceNullAllocation;
      }
    });
    attempt([this] {
      if (ready_for_cuda_ != nullptr) {
        CUDADriver::get_instance().external_semaphore_destroy(ready_for_cuda_);
        ready_for_cuda_ = nullptr;
      }
    });
    attempt([this] {
      if (ready_for_external_ != nullptr) {
        CUDADriver::get_instance().external_semaphore_destroy(
            ready_for_external_);
        ready_for_external_ = nullptr;
      }
    });
    attempt([this] {
      if (mapped_buffer_ != nullptr) {
        CUDADriver::get_instance().mem_free(mapped_buffer_);
        mapped_buffer_ = nullptr;
      }
    });
    attempt([this] {
      if (external_memory_ != nullptr) {
        CUDADriver::get_instance().external_memory_destroy(external_memory_);
        external_memory_ = nullptr;
      }
    });
    if (failure) {
      std::rethrow_exception(failure);
    }
  }

  void destroy_resources_noexcept() noexcept {
    try {
      destroy_resources();
    } catch (const std::exception &error) {
      TI_WARN("External CUDA allocation cleanup failed: {}", error.what());
    } catch (...) {
      TI_WARN("External CUDA allocation cleanup failed");
    }
  }

  mutable std::mutex mutex_;
  CudaDevice *cuda_device_{nullptr};
  std::uint64_t identity_{0};
  std::uint64_t owner_domain_{0};
  std::size_t allocation_size_{0};
  std::array<std::uint8_t, 16> device_uuid_{};
  int device_ordinal_{0};
  bool synchronized_{false};
  bool cuda_owned_{false};
  bool closed_{false};
  DeviceAllocation cuda_allocation_{kDeviceNullAllocation};
  CUexternalMemory external_memory_{nullptr};
  void *mapped_buffer_{nullptr};
  CUexternalSemaphore ready_for_cuda_{nullptr};
  CUexternalSemaphore ready_for_external_{nullptr};
  ExternalStreamDomain active_cuda_stream_;
  ExternalStreamDomain last_cuda_stream_;
};

std::shared_ptr<CudaExternalAllocation> CudaExternalAllocation::create(
    CudaDevice *cuda_device,
    CudaExternalMemoryImport memory,
    std::optional<CudaExternalSemaphorePairImport> semaphores) {
  return std::shared_ptr<CudaExternalAllocation>(
      new CudaExternalAllocation(std::make_unique<Impl>(
          cuda_device, std::move(memory), std::move(semaphores))));
}

CudaExternalAllocation::CudaExternalAllocation(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}

CudaExternalAllocation::~CudaExternalAllocation() = default;

std::uint64_t CudaExternalAllocation::identity() const noexcept {
  return impl_->identity();
}

DeviceAllocation CudaExternalAllocation::cuda_allocation() const noexcept {
  return impl_->cuda_allocation();
}

std::size_t CudaExternalAllocation::allocation_size() const noexcept {
  return impl_->allocation_size();
}

const std::array<std::uint8_t, 16> &CudaExternalAllocation::device_uuid()
    const noexcept {
  return impl_->device_uuid();
}

int CudaExternalAllocation::device_ordinal() const noexcept {
  return impl_->device_ordinal();
}

bool CudaExternalAllocation::synchronized() const noexcept {
  return impl_->synchronized();
}

bool CudaExternalAllocation::closed() const noexcept {
  return impl_->closed();
}

void CudaExternalAllocation::acquire_for_consumer(
    const ExternalStreamDomain &stream) {
  impl_->acquire_for_consumer(stream);
}

void CudaExternalAllocation::release_from_consumer(
    const ExternalStreamDomain &stream) {
  impl_->release_from_consumer(stream);
}

void CudaExternalAllocation::close() {
  impl_->close();
}

std::array<std::uint8_t, 16> current_cuda_external_device_uuid() {
  return query_cuda_device_uuid();
}

#else

class CudaExternalAllocation::Impl {};

std::shared_ptr<CudaExternalAllocation> CudaExternalAllocation::create(
    cuda::CudaDevice *,
    CudaExternalMemoryImport,
    std::optional<CudaExternalSemaphorePairImport>) {
  TI_ERROR("External CUDA allocation requires CUDA support");
  return nullptr;
}

CudaExternalAllocation::CudaExternalAllocation(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}

CudaExternalAllocation::~CudaExternalAllocation() = default;

std::uint64_t CudaExternalAllocation::identity() const noexcept {
  return 0;
}

DeviceAllocation CudaExternalAllocation::cuda_allocation() const noexcept {
  return kDeviceNullAllocation;
}

std::size_t CudaExternalAllocation::allocation_size() const noexcept {
  return 0;
}

const std::array<std::uint8_t, 16> &CudaExternalAllocation::device_uuid()
    const noexcept {
  static const std::array<std::uint8_t, 16> empty{};
  return empty;
}

int CudaExternalAllocation::device_ordinal() const noexcept {
  return 0;
}

bool CudaExternalAllocation::synchronized() const noexcept {
  return false;
}

bool CudaExternalAllocation::closed() const noexcept {
  return true;
}

void CudaExternalAllocation::acquire_for_consumer(
    const ExternalStreamDomain &) {
  TI_NOT_IMPLEMENTED;
}

void CudaExternalAllocation::release_from_consumer(
    const ExternalStreamDomain &) {
  TI_NOT_IMPLEMENTED;
}

void CudaExternalAllocation::close() {
  TI_NOT_IMPLEMENTED;
}

std::array<std::uint8_t, 16> current_cuda_external_device_uuid() {
  TI_ERROR("External CUDA allocation requires CUDA support");
  return {};
}

#endif

}  // namespace taichi::lang
