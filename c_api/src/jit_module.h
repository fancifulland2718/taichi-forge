#pragma once

#include "taichi/runtime/aot_runtime_client.h"

class Runtime;

namespace taichi::lang {
class LaunchContextBuilder;
namespace aot {
class Kernel;
}
}  // namespace taichi::lang

namespace capi {
// Caller holds the Program tree/resource and Runtime submission guards.
std::uintptr_t import_jit_module(
    taichi::lang::Program &program,
    ::Runtime &runtime,
    const std::vector<taichi::lang::AotRuntimeClient::JitKernelImport>
        &kernels,
    const taichi::lang::AotRuntimeClient::JitDataPayloads &data);
std::unique_ptr<taichi::lang::LaunchContextBuilder>
make_borrowed_launch_context(taichi::lang::aot::Kernel *kernel,
                             bool cpu_bounded_range);
}  // namespace capi
