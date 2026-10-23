#include "execution_domain_binding.h"

#include <stdexcept>
#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/execution_domain.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/runtime/vulkan_execution_domain.h"
#endif

namespace capi {

PreparedExecutionDomain::PreparedExecutionDomain(
    std::shared_ptr<taichi::lang::cuda::ExecutionDomain> domain)
    : cuda(std::move(domain)) {
  if (!cuda)
    throw std::invalid_argument("Missing CUDA execution domain");
}
PreparedExecutionDomain::PreparedExecutionDomain(
    std::shared_ptr<taichi::lang::vulkan::ExecutionDomain> domain)
    : vulkan(std::move(domain)) {
  if (!vulkan)
    throw std::invalid_argument("Missing Vulkan execution domain");
}

taichi::Arch PreparedExecutionDomain::backend() const {
  return cuda ? taichi::Arch::cuda : taichi::Arch::vulkan;
}

PreparedExecutionDomain::Snapshot PreparedExecutionDomain::snapshot() const {
#ifdef TI_WITH_CUDA
  if (cuda) {
    const auto value = cuda->snapshot();
    return {value.owner, value.accepting, value.closed};
  }
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan) {
    const auto value = vulkan->snapshot();
    return {value.owner, value.accepting, value.closed};
  }
#endif
  throw std::runtime_error("Execution domain backend is unavailable");
}

bool PreparedExecutionDomain::belongs_to(
    const std::shared_ptr<taichi::lang::RuntimeFaultDomain> &owner) const {
#ifdef TI_WITH_CUDA
  if (cuda)
    return cuda->belongs_to(owner);
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan)
    return vulkan->belongs_to(owner);
#endif
  return false;
}

void PreparedExecutionDomain::close() {
#ifdef TI_WITH_CUDA
  if (cuda)
    return cuda->close();
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan)
    return vulkan->close();
#endif
  throw std::runtime_error("Execution domain backend is unavailable");
}

bool PreparedExecutionDomain::instance_retired(
    const PreparedExecutionInstance &instance) const {
#ifdef TI_WITH_CUDA
  if (cuda)
    return cuda->instance_retired(instance.cuda);
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan)
    return vulkan->instance_retired(instance.vulkan);
#endif
  return false;
}

PreparedExecutionInstance PreparedExecutionDomain::prepare_owned_instance(
    const std::shared_ptr<taichi::lang::RuntimeClient> &client,
    const std::function<void(const PreparedExecutionInstance &)> &prepare) {
  PreparedExecutionInstance result;
#ifdef TI_WITH_CUDA
  if (cuda) {
    result.cuda = cuda->prepare_owned_instance(client, [&](const auto &instance) {
      prepare({instance, {}});
    });
    return result;
  }
#endif
#ifdef TI_WITH_VULKAN
  if (vulkan) {
    result.vulkan = vulkan->prepare_owned_instance(client, [&](const auto &instance) {
      prepare({{}, instance});
    });
    return result;
  }
#endif
  throw std::runtime_error("Execution domain backend is unavailable");
}

}  // namespace capi
