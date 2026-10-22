#pragma once

#include "taichi_core_impl.h"
#include "taichi/taichi_cuda.h"
#include "completion.h"

#ifdef TI_WITH_CUDA
namespace taichi::lang::cuda {
class ExecutionDomain;
class ExecutionInstance;
}
namespace capi {

std::shared_ptr<taichi::lang::cuda::ExecutionDomain> get_execution_domain(
    TiCudaExecutionDomain handle);
TiCudaExecutionInstance publish_execution_instance(
    std::shared_ptr<taichi::lang::cuda::ExecutionDomain> domain,
    std::shared_ptr<taichi::lang::cuda::ExecutionInstance> instance);

}  // namespace capi
#endif
