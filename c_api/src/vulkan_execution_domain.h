#pragma once

#include "taichi_core_impl.h"
#include "completion.h"

#ifdef TI_WITH_VULKAN
#include "taichi/taichi_vulkan.h"
namespace taichi::lang::vulkan {
class ExecutionDomain;
class ExecutionInstance;
}
namespace capi {
std::shared_ptr<taichi::lang::vulkan::ExecutionDomain> get_execution_domain(
    TiVulkanExecutionDomain handle);
TiVulkanExecutionInstance publish_execution_instance(
    std::shared_ptr<taichi::lang::vulkan::ExecutionDomain> domain,
    std::shared_ptr<taichi::lang::vulkan::ExecutionInstance> instance);
}
#endif
