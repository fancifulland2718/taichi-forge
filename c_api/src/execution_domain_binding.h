#pragma once

#include <functional>
#include <cstdint>
#include <memory>

#include "taichi/rhi/arch.h"
#include "taichi/runtime/runtime_client.h"

namespace taichi::lang {
class RuntimeFaultDomain;
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
  void retire_instance(const PreparedExecutionInstance &instance);
  bool instance_retired(const PreparedExecutionInstance &instance) const;
  PreparedExecutionInstance prepare_owned_instance(
      const std::shared_ptr<taichi::lang::RuntimeClient> &client,
      const std::function<void(const PreparedExecutionInstance &)> &prepare);
};

}  // namespace capi
