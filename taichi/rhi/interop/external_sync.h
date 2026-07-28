#pragma once

#include <cstdint>

#include "taichi/rhi/public_device.h"

namespace taichi::lang {

// Native stream handles are deliberately submission-scoped. A stable domain
// identifies the runtime that owns a stream, while stream_identity prevents a
// wait on one stream from being paired with a signal on another by accident.
enum class ExternalExecutionApi : std::uint8_t {
  kCuda,
  kVulkan,
};

struct RHI_DLL_EXPORT ExternalStreamDomain {
  ExternalExecutionApi api{ExternalExecutionApi::kCuda};
  std::uint64_t owner_domain{0};
  std::uint64_t stream_identity{0};
  void *native_stream{nullptr};

  static ExternalStreamDomain cuda(std::uint64_t owner_domain,
                                   std::uint64_t stream_identity,
                                   void *native_stream = nullptr) noexcept {
    return {ExternalExecutionApi::kCuda, owner_domain, stream_identity,
            native_stream};
  }

  bool valid() const noexcept {
    return owner_domain != 0 && stream_identity != 0;
  }

  bool same_stream(const ExternalStreamDomain &other) const noexcept {
    return api == other.api && owner_domain == other.owner_domain &&
           stream_identity == other.stream_identity &&
           native_stream == other.native_stream;
  }
};

// A synchronization domain brackets one coarse access epoch. Consumers may
// enqueue any number of kernels between acquire_for_consumer() and
// release_from_consumer(); implementations must not allocate one stream,
// event, semaphore, or worker thread per kernel/view.
class RHI_DLL_EXPORT ExternalSynchronizationDomain {
 public:
  virtual ~ExternalSynchronizationDomain() = default;

  virtual std::uint64_t identity() const noexcept = 0;
  virtual void acquire_for_consumer(const ExternalStreamDomain &stream) = 0;
  virtual void release_from_consumer(const ExternalStreamDomain &stream) = 0;
};

}  // namespace taichi::lang
