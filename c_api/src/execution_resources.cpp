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
    sizes_.emplace(memory, physical);
    memories_.push_back(memory);
  } catch (...) {
    sizes_.erase(memory);
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

TiMemory ExecutionResourceStorage::view(TiMemory parent,
                                        uint64_t offset,
                                        uint64_t bytes) {
  const auto source = runtime_.get_memory(parent);
  if (offset > source.size || bytes > source.size - offset ||
      offset > std::numeric_limits<uint64_t>::max() - source.base_offset ||
      offset > std::numeric_limits<uint64_t>::max() - source.byte_offset)
    throw MemoryBindingError("Execution view exceeds its parent storage");
  // Borrowed JIT storage keeps its real lease. The execution's transitive
  // closure retains ordinary parents; module aliases share the module owner.
  auto memory = runtime_.register_memory(
      source.allocation, bytes, false, source.base_offset + offset,
      source.storage_owner, source.byte_offset + offset, false);
  try {
    sizes_.emplace(memory, 0);
    memories_.push_back(memory);
  } catch (...) {
    sizes_.erase(memory);
    runtime_.free_memory(memory);
    throw;
  }
  return memory;
}

TiMemory ExecutionResourceStorage::module_view(taichi::lang::DevicePtr pointer,
                                               uint64_t bytes) {
  auto memory = runtime_.register_memory(pointer, bytes, false, pointer.offset,
                                         {}, pointer.offset, false);
  try {
    sizes_.emplace(memory, 0);
    memories_.push_back(memory);
  } catch (...) {
    sizes_.erase(memory);
    runtime_.free_memory(memory);
    throw;
  }
  return memory;
}

void ExecutionResourceStorage::retain(
    const std::unordered_set<TiMemory> &memories) {
  for (std::size_t i = memories_.size(); i-- > 0;) {
    const auto memory = memories_[i];
    if (memories.count(memory))
      continue;
    runtime_.free_memory(memory);
    bytes_ -= sizes_.at(memory);
    sizes_.erase(memory);
    memories_.erase(memories_.begin() + i);
  }
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
    taichi::lang::aot::Module &module,
    const taichi::lang::aot::ExecutionResources &resources,
    std::unordered_map<std::string, const TiArgument *> &arguments,
    std::unordered_map<std::string, const TiNamedNdArrayBinding *> &layouts,
    SharedExecutionResources *shared,
    const std::unordered_set<std::string> &frozen_metadata) {
  using Recipes = taichi::lang::aot::ExecutionResources;
  ExecutionResourceBindings result;
  result.arguments.resize(resources.order.size());
  std::vector<uint64_t> sizes(resources.order.size());
  std::vector<uint64_t> view_offsets(resources.order.size());
  std::vector<taichi::lang::DevicePtr> module_pointers(resources.order.size());
  std::map<std::string, std::vector<std::uint8_t>> payloads;
  std::unordered_set<const TiArgument *> local, validated_sources;
  auto array_source = [&](const std::string &name) -> const TiNdArray & {
    const auto found = arguments.find(name);
    if (found == arguments.end() ||
        found->second->type != TI_ARGUMENT_TYPE_NDARRAY)
      throw MemoryBindingError("Execution shape source is not an ndarray: " +
                               name);
    const auto &array = found->second->value.ndarray;
    if (!local.count(found->second) && !frozen_metadata.count(name) &&
        validated_sources.insert(found->second).second)
      validated_ndarray_bytes(runtime, array);
    return array;
  };
  auto resolve = [&](const Recipes::Extent &ref) {
    const auto &array = array_source(ref.array);
    if (ref.axis >= array.shape.dim_count)
      throw MemoryBindingError("Execution shape axis exceeds array rank: " +
                               ref.array);
    return array.shape.dims[ref.axis];
  };
  // Resolve the entire metadata closure before allocating any private buffer.
  for (std::size_t pos = 0; pos < resources.order.size(); ++pos) {
    const auto &recipe = resources.declarations[resources.order[pos]];
    if (std::holds_alternative<taichi::lang::NativeOperationDescription>(
            recipe.value) ||
        std::holds_alternative<Recipes::HostBlob>(recipe.value))
      continue;
    auto &argument = result.arguments[pos];
    const auto *view = std::get_if<Recipes::View>(&recipe.value);
    auto dtype_name = recipe.dtype;
    Recipes::Array module_array;
    const auto *array_description = std::get_if<Recipes::Array>(&recipe.value);
    if (const auto *source = std::get_if<Recipes::ModuleArray>(&recipe.value)) {
      const auto bound = module.get_root_storage_view(source->view);
      dtype_name = bound.layout.dtype;
      for (const auto extent : bound.layout.shape)
        module_array.shape.emplace_back(extent);
      module_array.element_shape = bound.layout.element_shape;
      array_description = &module_array;
      module_pointers[pos] = bound.pointer;
      if (bound.pointer.device != &runtime.get())
        throw MemoryBindingError("Module storage belongs to another device");
    }
    const auto dtype =
        view ? array_source(view->array).elem_type : scalar_type(dtype_name);
    if (view) {
      argument.type = TI_ARGUMENT_TYPE_NDARRAY;
      auto &array = argument.value.ndarray;
      array = array_source(view->array);
      const auto parent = ndarray_storage_layout(array);
      constexpr auto rank_capacity =
          sizeof(array.shape.dims) / sizeof(uint32_t);
      if (view->shape.size() > rank_capacity)
        throw MemoryBindingError(
            "Execution view rank exceeds the C API shape capacity");
      array.memory = TI_NULL_HANDLE;
      array.shape.dim_count = static_cast<uint32_t>(view->shape.size());
      for (std::size_t axis = 0; axis < view->shape.size(); ++axis) {
        const auto &dimension = view->shape[axis];
        const auto *literal = std::get_if<uint32_t>(&dimension);
        array.shape.dims[axis] =
            literal ? *literal : resolve(std::get<Recipes::Extent>(dimension));
      }
      sizes[pos] = ndarray_storage_layout(array).bytes;
      if (view->offset > parent.bytes / parent.element_bytes)
        throw MemoryBindingError("Execution view offset exceeds parent shape");
      view_offsets[pos] = view->offset * parent.element_bytes;
      if (sizes[pos] > parent.bytes - view_offsets[pos])
        throw MemoryBindingError("Execution view shape exceeds parent shape");
    } else if (const auto *desc = array_description) {
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
      if (!desc->initial_data.empty() &&
          !(shared && shared->arrays.count(recipe.name)) &&
          resources.materialize[resources.order[pos]]) {
        const auto found = payloads.find(desc->initial_data);
        if (found == payloads.end())
          payloads.emplace(desc->initial_data,
                           module.load_data(desc->initial_data, sizes[pos]));
        else if (found->second.size() != sizes[pos])
          throw MemoryBindingError(
              "Execution initial data size differs between resource "
              "declarations");
      }
      if (desc->zero_initialize ||
          (!desc->initial_data.empty() && sizes[pos])) {
        // Only private allocations may expose padding to fills and uploads.
        // Keep the logical ndarray extent unchanged, including empty arrays.
        const auto bytes = std::max<uint64_t>(1, sizes[pos]);
        if (bytes > std::numeric_limits<uint64_t>::max() - 3)
          throw MemoryBindingError(
              "Execution resource initialization size overflow");
        sizes[pos] = (bytes + 3) & ~uint64_t(3);
      }
    } else if (const auto *constant =
                   std::get_if<Recipes::Scalar>(&recipe.value)) {
      argument.type = TI_ARGUMENT_TYPE_SCALAR;
      argument.value.scalar.type = dtype;
      auto &value = argument.value.scalar.value;
      switch (dtype) {
        case TI_DATA_TYPE_I8:
        case TI_DATA_TYPE_U8:
          value.x8 = static_cast<std::uint8_t>(constant->bits);
          break;
        case TI_DATA_TYPE_I16:
        case TI_DATA_TYPE_U16:
          value.x16 = static_cast<std::uint16_t>(constant->bits);
          break;
        case TI_DATA_TYPE_I64:
        case TI_DATA_TYPE_U64:
        case TI_DATA_TYPE_F64:
          value.x64 = constant->bits;
          break;
        default:
          value.x32 = static_cast<std::uint32_t>(constant->bits);
          break;
      }
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

  // Vulkan buffer transfers require four-byte extents. Keep artifact hashes
  // and size checks over logical bytes; initialize only owned tail padding.
  for (auto &[key, bytes] : payloads) {
    if (bytes.size() > std::numeric_limits<std::size_t>::max() - 3)
      throw MemoryBindingError("Execution initial data size overflow");
    bytes.resize((bytes.size() + 3) & ~std::size_t(3), 0);
  }
  std::vector<taichi::lang::DevicePtr> destinations;
  std::vector<const void *> sources;
  std::vector<std::size_t> upload_sizes;
  auto shared_array = [&](const std::string &name, TiNdArray &next) {
    if (!shared)
      return false;
    if (const auto previous = shared->arrays.find(name);
        previous != shared->arrays.end()) {
      const auto &old = previous->second;
      if (old.elem_type != next.elem_type ||
          old.shape.dim_count != next.shape.dim_count ||
          old.elem_shape.dim_count != next.elem_shape.dim_count ||
          !std::equal(old.shape.dims, old.shape.dims + old.shape.dim_count,
                      next.shape.dims) ||
          !std::equal(old.elem_shape.dims,
                      old.elem_shape.dims + old.elem_shape.dim_count,
                      next.elem_shape.dims))
        throw MemoryBindingError(
            "Shared private resource layout changed; prepare a new binding "
            "family: " +
            name);
      next.memory = old.memory;
      return true;
    }
    if (shared->sealed)
      throw MemoryBindingError("Shared private resource is missing: " + name);
    if (!shared->storage)
      shared->storage = std::make_shared<ExecutionResourceStorage>(runtime);
    return false;
  };
  for (std::size_t pos = 0; pos < resources.order.size(); ++pos) {
    auto &argument = result.arguments[pos];
    const auto &name = resources.declarations[resources.order[pos]].name;
    if (module_pointers[pos].device) {
      if (resources.materialize[resources.order[pos]]) {
        if (shared && shared->stable_names.count(name)) {
          if (!shared_array(name, argument.value.ndarray)) {
            argument.value.ndarray.memory =
                shared->storage->module_view(module_pointers[pos], sizes[pos]);
            shared->arrays.emplace(name, argument.value.ndarray);
          }
          continue;
        }
        if (!result.storage)
          result.storage = std::make_unique<ExecutionResourceStorage>(runtime);
        argument.value.ndarray.memory =
            result.storage->module_view(module_pointers[pos], sizes[pos]);
      }
      continue;
    }
    if (const auto *view = std::get_if<Recipes::View>(
            &resources.declarations[resources.order[pos]].value)) {
      if (resources.materialize[resources.order[pos]]) {
        if (shared && shared->stable_names.count(name)) {
          if (!shared_array(name, argument.value.ndarray)) {
            argument.value.ndarray.memory = shared->storage->view(
                arguments.at(view->array)->value.ndarray.memory,
                view_offsets[pos], sizes[pos]);
            shared->arrays.emplace(name, argument.value.ndarray);
          }
          continue;
        }
        if (!result.storage)
          result.storage = std::make_unique<ExecutionResourceStorage>(runtime);
        argument.value.ndarray.memory = result.storage->view(
            arguments.at(view->array)->value.ndarray.memory, view_offsets[pos],
            sizes[pos]);
      }
      continue;
    }
    if (std::holds_alternative<Recipes::Array>(
            resources.declarations[resources.order[pos]].value) &&
        resources.materialize[resources.order[pos]] &&
        argument.type == TI_ARGUMENT_TYPE_NDARRAY) {
      if (shared) {
        if (shared_array(name, argument.value.ndarray))
          continue;
        argument.value.ndarray.memory = shared->storage->allocate(sizes[pos]);
        shared->arrays.emplace(name, argument.value.ndarray);
      } else {
        if (!result.storage)
          result.storage = std::make_unique<ExecutionResourceStorage>(runtime);
        argument.value.ndarray.memory = result.storage->allocate(sizes[pos]);
      }
      if (std::get<Recipes::Array>(
              resources.declarations[resources.order[pos]].value)
              .zero_initialize)
        result.zero_initializations.push_back(
            {argument.value.ndarray.memory, sizes[pos]});
      const auto &key = std::get<Recipes::Array>(
                            resources.declarations[resources.order[pos]].value)
                            .initial_data;
      if (!key.empty() && sizes[pos]) {
        destinations.push_back(runtime.get_memory(argument.value.ndarray.memory)
                                   .allocation.get_ptr());
        sources.push_back(payloads.at(key).data());
        upload_sizes.push_back(static_cast<std::size_t>(sizes[pos]));
      }
    }
  }
  // Upload only owned initial state, before any provider can inspect it. The
  // RHI transfer owns/synchronizes staging, so host payloads retire here. Batch
  // Vulkan copies in one submission; chunk only at the RHI count ABI limit.
  for (std::size_t offset = 0; offset < destinations.size();) {
    const auto count = static_cast<int>(std::min<std::size_t>(
        destinations.size() - offset, std::numeric_limits<int>::max()));
    if (runtime.get().upload_data(destinations.data() + offset,
                                  sources.data() + offset,
                                  upload_sizes.data() + offset,
                                  count) != taichi::lang::RhiResult::success)
      throw MemoryBindingError("Unable to upload execution initial data");
    offset += count;
  }
  return result;
}
}  // namespace capi
