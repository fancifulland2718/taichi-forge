#pragma once

#include <cstdint>

namespace taichi::lang {

// Storage created by one CUDA capture, shared by JIT and fixed AOT execution.
// Finalize after capture ends and before graph instantiation, never on replay.
// The executable retains it until all launches retire. release() is idempotent;
// false abandons backend handles without calling a failed/destroyed backend.
// This runtime contract does not serialize handles or change a provider ABI.
class CudaGraphCaptureResources {
 public:
  virtual ~CudaGraphCaptureResources() = default;
  virtual void finalize() = 0;
  virtual std::uint64_t requested_device_bytes() const = 0;
  virtual void release(bool backend_safe) noexcept = 0;
};

}  // namespace taichi::lang
