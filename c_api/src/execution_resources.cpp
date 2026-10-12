#include "execution_resources.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace capi {
ExecutionResourceStorage::~ExecutionResourceStorage() {
  for (auto i = memories_.rbegin(); i != memories_.rend(); ++i) {
    try {
      runtime_.free_memory(*i);
    } catch (const std::exception &error) {
      TI_WARN("Unable to release execution resource: {}", error.what());
    }
  }
}

TiMemory ExecutionResourceStorage::allocate(uint64_t bytes) {
  // Empty logical arrays still need a live, aligned binding identity.
  const auto physical = std::max<uint64_t>(1, bytes);
  if (physical > std::numeric_limits<std::size_t>::max() ||
      physical > std::numeric_limits<uint64_t>::max() - bytes_)
    throw MemoryBindingError("Execution resources overflow address space");
  taichi::lang::Device::AllocParams params{};
  params.size = physical;
  params.usage = taichi::lang::AllocUsage::Storage;
  auto memory = runtime_.allocate_memory(params);
  try {
    memories_.push_back(memory);
  } catch (...) {
    runtime_.free_memory(memory);
    throw;
  }
  bytes_ += physical;
  return memory;
}

void ExecutionResourceStorage::remove_owned_dependencies(
    std::vector<TiMemory> &dependencies) const {
  const std::unordered_set<TiMemory> owned(memories_.begin(), memories_.end());
  dependencies.erase(
      std::remove_if(dependencies.begin(), dependencies.end(),
                     [&](TiMemory memory) { return owned.count(memory); }),
      dependencies.end());
}

namespace {
TiDataType scalar_type(const std::string &name) {
#define RESOURCE_TYPE(c, native) \
  if (name == #native)           \
  return TI_DATA_TYPE_##c
  RESOURCE_TYPE(F16, f16);
  RESOURCE_TYPE(F32, f32);
  RESOURCE_TYPE(F64, f64);
  RESOURCE_TYPE(I8, i8);
  RESOURCE_TYPE(I16, i16);
  RESOURCE_TYPE(I32, i32);
  RESOURCE_TYPE(I64, i64);
  RESOURCE_TYPE(U8, u8);
  RESOURCE_TYPE(U16, u16);
  RESOURCE_TYPE(U32, u32);
  RESOURCE_TYPE(U64, u64);
#undef RESOURCE_TYPE
  throw MemoryBindingError("Unsupported execution resource dtype: " + name);
}
}  // namespace

ExecutionResourceBindings bind_execution_resources(
    Runtime &runtime,
    const taichi::lang::aot::ExecutionResources &resources,
    std::unordered_map<std::string, const TiArgument *> &arguments,
    std::unordered_map<std::string, const TiNamedNdArrayBinding *> &layouts) {
  using Recipes = taichi::lang::aot::ExecutionResources;
  ExecutionResourceBindings result;
  result.arguments.resize(resources.order.size());
  std::vector<uint64_t> sizes(resources.order.size());
  std::unordered_set<const TiArgument *> local, validated_sources;
  auto resolve = [&](const Recipes::Extent &ref) {
    const auto found = arguments.find(ref.array);
    if (found == arguments.end() ||
        found->second->type != TI_ARGUMENT_TYPE_NDARRAY)
      throw MemoryBindingError("Execution shape source is not an ndarray: " +
                               ref.array);
    const auto &array = found->second->value.ndarray;
    if (!local.count(found->second) &&
        validated_sources.insert(found->second).second)
      validated_ndarray_bytes(runtime, array);
    if (ref.axis >= array.shape.dim_count)
      throw MemoryBindingError("Execution shape axis exceeds array rank: " +
                               ref.array);
    return array.shape.dims[ref.axis];
  };
  // Resolve the entire metadata closure before allocating any private buffer.
  for (std::size_t pos = 0; pos < resources.order.size(); ++pos) {
    const auto &recipe = resources.declarations[resources.order[pos]];
    if (std::holds_alternative<taichi::lang::NativeOperationDescription>(
            recipe.value))
      continue;
    auto &argument = result.arguments[pos];
    const auto dtype = scalar_type(recipe.dtype);
    if (const auto *desc = std::get_if<Recipes::Array>(&recipe.value)) {
      argument.type = TI_ARGUMENT_TYPE_NDARRAY;
      auto &array = argument.value.ndarray;
      constexpr auto rank_capacity =
          sizeof(array.shape.dims) / sizeof(uint32_t);
      if (desc->shape.size() > rank_capacity ||
          desc->element_shape.size() > rank_capacity)
        throw MemoryBindingError(
            "Execution resource rank exceeds the C API shape capacity");
      array.elem_type = dtype;
      array.shape.dim_count = static_cast<uint32_t>(desc->shape.size());
      array.elem_shape.dim_count =
          static_cast<uint32_t>(desc->element_shape.size());
      for (std::size_t axis = 0; axis < desc->shape.size(); ++axis) {
        const auto &dimension = desc->shape[axis];
        const auto *literal = std::get_if<uint32_t>(&dimension);
        array.shape.dims[axis] =
            literal ? *literal : resolve(std::get<Recipes::Extent>(dimension));
      }
      std::copy(desc->element_shape.begin(), desc->element_shape.end(),
                array.elem_shape.dims);
      sizes[pos] = ndarray_storage_layout(array).bytes;
    } else {
      const auto extent = resolve(std::get<Recipes::Extent>(recipe.value));
      argument.type = TI_ARGUMENT_TYPE_SCALAR;
      argument.value.scalar.type = dtype;
      if (dtype == TI_DATA_TYPE_I32 &&
          extent > std::numeric_limits<int32_t>::max())
        throw MemoryBindingError("Execution extent exceeds i32");
      if (dtype == TI_DATA_TYPE_I64 || dtype == TI_DATA_TYPE_U64)
        argument.value.scalar.value.x64 = extent;
      else
        argument.value.scalar.value.x32 = extent;
    }
    if (!arguments.emplace(recipe.name, &argument).second)
      throw MemoryBindingError(
          "Execution resource shadows an external binding: " + recipe.name);
    local.insert(&argument);
  }

  result.gradients.reserve(resources.gradients.size());
  for (const auto &[primal, gradient] : resources.gradients) {
    const auto &value = *arguments.at(primal);
    const auto &derivative = *arguments.at(gradient);
    if (value.type != TI_ARGUMENT_TYPE_NDARRAY ||
        derivative.type != TI_ARGUMENT_TYPE_NDARRAY)
      throw MemoryBindingError("Execution gradient names must bind ndarrays");
    TiNamedNdArrayBinding binding{};
    if (auto previous = layouts.find(primal); previous != layouts.end()) {
      binding = *previous->second;
      if (binding.gradient)
        throw MemoryBindingError(
            "Execution gradient is specified by both plan and caller: " +
            primal);
    }
    binding.name = primal.c_str();
    binding.gradient = &derivative.value.ndarray;
    result.gradients.push_back(binding);
    layouts[primal] = &result.gradients.back();
  }

  for (std::size_t pos = 0; pos < resources.order.size(); ++pos) {
    auto &argument = result.arguments[pos];
    if (std::holds_alternative<Recipes::Array>(
            resources.declarations[resources.order[pos]].value) &&
        resources.materialize[resources.order[pos]] &&
        argument.type == TI_ARGUMENT_TYPE_NDARRAY) {
      if (!result.storage)
        result.storage = std::make_unique<ExecutionResourceStorage>(runtime);
      argument.value.ndarray.memory = result.storage->allocate(sizes[pos]);
    }
  }
  return result;
}
}  // namespace capi
