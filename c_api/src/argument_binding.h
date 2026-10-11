#pragma once

#include "taichi_core_impl.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/program/matrix.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"

namespace capi {
// Validate primitive/tensor storage and return its declared logical byte size.
uint64_t validated_ndarray_bytes(Runtime &runtime, const TiNdArray &array);

// Cold ABI marshalling shared by ordinary launch and prepared execution. These
// own the host descriptors referenced by LaunchContextBuilder, not device data.
struct BoundKernelArguments {
  AotModule *owner{nullptr};
  taichi::lang::aot::Kernel *kernel{nullptr};
  std::vector<TiMemory> memories;
  std::vector<std::unique_ptr<taichi::lang::DeviceAllocation>> allocations;
  std::unique_ptr<taichi::lang::LaunchContextBuilder> context;
};

struct BoundGraphArguments {
  AotModule *owner{nullptr};
  taichi::lang::aot::CompiledGraph *graph{nullptr};
  std::vector<TiMemory> memories;
  std::vector<taichi::lang::Ndarray> arrays;
  std::vector<taichi::lang::Ndarray> gradient_arrays;
  std::vector<std::unique_ptr<taichi::lang::Texture>> textures;
  std::vector<taichi::lang::Matrix> matrices;
  std::unordered_map<std::string, taichi::lang::aot::IValue> arguments;
  std::unordered_map<std::string, const taichi::lang::Ndarray *> gradients;
};

std::unique_ptr<BoundKernelArguments> bind_kernel_arguments(
    Runtime &runtime,
    TiKernel kernel,
    uint32_t arg_count,
    const TiArgument *args,
    uint32_t binding_count,
    const TiNdArrayBinding *bindings);
std::unique_ptr<BoundGraphArguments> bind_graph_arguments(
    Runtime &runtime,
    TiComputeGraph graph,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings);

}  // namespace capi
