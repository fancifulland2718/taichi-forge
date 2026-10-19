#pragma once

#include <map>

#include "taichi_core_impl.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/program/matrix.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/native_operation_description.h"

namespace taichi::lang {
class NativeSampledImage;
class NativeKernelResourceBinding;
}  // namespace taichi::lang

namespace capi {
struct NdArrayStorageLayout {
  uint64_t bytes;
  uint64_t element_bytes;
  uint32_t alignment;
};
NdArrayStorageLayout ndarray_storage_layout(const TiNdArray &array);
taichi::lang::DataType ndarray_scalar_type(TiDataType type);

// Validate primitive/tensor storage and return its declared logical byte size.
uint64_t validated_ndarray_bytes(Runtime &runtime,
                                 const TiNdArray &array,
                                 uint64_t *element_bytes = nullptr,
                                 bool prepared_external_access = false);
taichi::lang::NativeArrayBinding bind_native_array(
    Runtime &runtime,
    const TiNdArray &array,
    bool prepared_external_access = false);

// Cold ABI marshalling shared by ordinary launch and prepared execution. These
// own the host descriptors referenced by LaunchContextBuilder, not device data.
struct BoundKernelArguments {
  AotModule *owner{nullptr};
  taichi::lang::aot::Kernel *kernel{nullptr};
  std::vector<TiMemory> memories;
  std::vector<TiNativeResource> native_resources;
  std::vector<std::unique_ptr<taichi::lang::DeviceAllocation>> allocations;
  std::vector<std::unique_ptr<taichi::lang::storage::DenseStorageDescriptor>>
      views;
  std::unique_ptr<taichi::lang::LaunchContextBuilder> context;
  std::vector<std::shared_ptr<taichi::lang::NativeKernelResourceBinding>>
      kernel_resources;
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
    const TiNdArrayBinding *bindings,
    bool cpu_bounded_range = false,
    const std::map<uint32_t, std::shared_ptr<taichi::lang::NativeResource>>
        &resources = {},
    bool prepared_external_access = false);
std::unique_ptr<BoundGraphArguments> bind_graph_arguments(
    Runtime &runtime,
    TiComputeGraph graph,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings);

}  // namespace capi
