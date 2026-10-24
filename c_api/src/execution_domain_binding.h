#pragma once

#include <functional>
#include <cstdint>
#include <memory>

#include "taichi/rhi/arch.h"
#include "taichi/runtime/runtime_client.h"
#include "taichi/runtime/native_data_upload.h"
#include "taichi/runtime/native_operation_description.h"
#include "taichi/program/runtime_completion.h"

namespace taichi::lang {
class RuntimeFaultDomain;
class NativeOperation;
namespace gfx {
class GfxRuntime;
}
namespace cuda {
class ExecutionDomain;
class ExecutionInstance;
}
namespace vulkan {
class ExecutionDomain;
class ExecutionInstance;
}
}

namespace capi {

struct PreparedExecutionInstance {
  std::shared_ptr<taichi::lang::cuda::ExecutionInstance> cuda;
  std::shared_ptr<taichi::lang::vulkan::ExecutionInstance> vulkan;
};

// Cold C API selection and logical-instance lifetime. The native recordings,
// queues, parameter pools and submission paths remain backend-owned.
class PreparedExecutionDomain {
 public:
  explicit PreparedExecutionDomain(
      std::shared_ptr<taichi::lang::cuda::ExecutionDomain> domain);
  explicit PreparedExecutionDomain(
      std::shared_ptr<taichi::lang::vulkan::ExecutionDomain> domain);
  const std::shared_ptr<taichi::lang::cuda::ExecutionDomain> cuda;
  const std::shared_ptr<taichi::lang::vulkan::ExecutionDomain> vulkan;
  taichi::Arch backend() const;
  struct Snapshot {
    std::uint64_t owner;
    bool accepting;
    bool closed;
  };
  Snapshot snapshot() const;
  bool belongs_to(const std::shared_ptr<taichi::lang::RuntimeFaultDomain> &owner) const;
  void close();
  // Cold, synchronous construction transfer on this selected domain. No
  // ordinary stream, extra queue lease, or public/live allocation mutation.
  void upload_owned_data(
      taichi::lang::Device &device,
      const std::vector<taichi::lang::NativeDataUpload> &uploads);
  // Caller certifies completed read inputs and newly owned outputs. Returns
  // false, without submitting, if this queue cannot record the operation.
  bool complete_owned_initialization(
      taichi::lang::Device &device,
      taichi::lang::gfx::GfxRuntime *gfx_runtime,
      std::shared_ptr<taichi::lang::NativeOperation> operation,
      const std::vector<taichi::lang::NativeArrayBinding> &inputs);
  // Join generation-specific CUDA allocation initialization on this domain,
  // retaining leases until completion. No ordinary queue prefix is recorded.
  taichi::lang::RuntimeCompletion join_allocation_initialization(
      taichi::lang::Device &device,
      const std::vector<taichi::lang::DeviceAllocation> &allocations);
  void retire_instance(const PreparedExecutionInstance &instance);
  bool instance_retired(const PreparedExecutionInstance &instance) const;
  PreparedExecutionInstance prepare_owned_instance(
      const std::shared_ptr<taichi::lang::RuntimeClient> &client,
      const std::function<void(const PreparedExecutionInstance &)> &prepare);
};

}  // namespace capi
