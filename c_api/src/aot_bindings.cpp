#include "taichi_core_impl.h"

#include <atomic>
#include <limits>

#include "taichi/aot/value_layout.h"
#include "taichi/aot/value_parameters.h"
#include "taichi/common/json_serde.h"

namespace {
taichi::lang::DataType ndarray_payload(taichi::lang::aot::Kernel &kernel,
                                       const std::vector<int> &index) {
  using namespace taichi::lang;
  const auto *array =
      kernel.args_type->get_element_type(index)->as<StructType>();
  return DataType(
             array->get_element_type({TypeFactory::DATA_PTR_POS_IN_NDARRAY}))
      .ptr_removed();
}

TiDataType scalar_type(taichi::lang::DataType type) {
  using taichi::lang::PrimitiveType;
#define SCALAR(c, native)            \
  if (type == PrimitiveType::native) \
  return TI_DATA_TYPE_##c
  SCALAR(F16, f16);
  SCALAR(F32, f32);
  SCALAR(F64, f64);
  SCALAR(I8, i8);
  SCALAR(I16, i16);
  SCALAR(I32, i32);
  SCALAR(I64, i64);
  SCALAR(U1, u1);
  SCALAR(U8, u8);
  SCALAR(U16, u16);
  SCALAR(U32, u32);
  SCALAR(U64, u64);
#undef SCALAR
  throw MemoryBindingError("Unsupported structure scalar type");
}

liong::json::JsonValue value_tree(
    taichi::lang::DataType type,
    const taichi::lang::aot::ValueLayout &layout,
    std::size_t &leaf) {
  using namespace liong::json;
  if (const auto *structure = type->cast<taichi::lang::StructType>()) {
    JsonArray members;
    std::unordered_set<std::string> names;
    for (const auto &member : structure->elements()) {
      if (member.name.empty() || !names.insert(member.name).second)
        throw MemoryBindingError("Invalid value structure member name");
      members.inner.emplace_back(JsonObject{
          {"name", JsonValue(member.name)},
          {"value", value_tree(member.type, layout, leaf)}});
    }
    return JsonValue(JsonObject{{"kind", JsonValue("struct")},
                                {"members", JsonValue(std::move(members))}});
  }
  const auto &member = layout.members.at(leaf++);
  JsonObject result{
      {"kind", JsonValue(member.shape.empty() ? "scalar" : "tensor")},
      {"dtype", JsonValue(member.scalar_type->to_string())},
      {"shape", serialize(member.shape)},
      {"offset", JsonValue(member.offset)},
      {"scalar_size", JsonValue(member.scalar_size)}};
  if (!member.component_offsets.empty())
    result.inner.emplace("component_offsets", serialize(member.component_offsets));
  return JsonValue(std::move(result));
}

std::string return_layout_json(taichi::lang::aot::Kernel &kernel,
                                taichi::Arch arch) {
  using namespace liong::json;
  using namespace taichi::lang::aot;
  JsonArray results;
  if (kernel.ret_type) {
    ValueLayout layout(kernel.ret_type, kernel.ret_size,
                       taichi::arch_uses_llvm(arch) ? ValueStorage::Host
                                                  : ValueStorage::Gfx);
    std::size_t leaf = 0;
    for (const auto &member : kernel.ret_type->elements())
      results.inner.push_back(value_tree(member.type, layout, leaf));
    if (leaf != layout.members.size())
      throw MemoryBindingError("Inconsistent return value layout");
  } else if (kernel.ret_size) {
    throw MemoryBindingError("Missing compiled return type");
  }
  return print(JsonValue(JsonObject{{"version", JsonValue(1)},
                                    {"buffer_size", JsonValue(kernel.ret_size)},
                                    {"results", JsonValue(std::move(results))}}));
}
}  // namespace

std::uintptr_t Runtime::new_handle_token() {
  static std::atomic<std::uintptr_t> next{1};
  auto token = next.load(std::memory_order_relaxed);
  do {
    if (token == std::numeric_limits<std::uintptr_t>::max())
      throw std::bad_alloc();
  } while (
      !next.compare_exchange_weak(token, token + 1, std::memory_order_relaxed));
  return token;
}

TiKernel Runtime::register_kernel(KernelBinding binding) {
  const auto handle = reinterpret_cast<TiKernel>(new_handle_token());
  kernels_.emplace(handle, std::move(binding));
  return handle;
}
TiComputeGraph Runtime::register_graph(GraphBinding binding) {
  const auto handle = reinterpret_cast<TiComputeGraph>(new_handle_token());
  graphs_.emplace(handle, std::move(binding));
  return handle;
}
TiNdArrayLayout Runtime::register_layout(AotModule *owner,
                                         taichi::lang::DataType type) {
  const auto handle = reinterpret_cast<TiNdArrayLayout>(new_handle_token());
  layouts_.emplace(handle,
                   LayoutBinding{owner, taichi::lang::aot::StructLayout(type)});
  return handle;
}
const Runtime::KernelBinding &Runtime::get_kernel(TiKernel kernel) const {
  const auto found = kernels_.find(kernel);
  if (found == kernels_.end())
    throw MemoryBindingError("Kernel is not live in this runtime");
  return found->second;
}

const Runtime::KernelBinding &Runtime::get_kernel_value_arguments(
    TiKernel handle) const {
  using namespace liong::json;
  using namespace taichi::lang::aot;
  const auto &binding = get_kernel(handle);
  if (!binding.argument_layout.empty())
    return binding;
  const auto &kernel = *binding.kernel;
  JsonArray values;
  std::map<uint32_t, KernelBinding::ValueWindow> windows;
  const auto &parameters = kernel.args_type->elements();
  if (kernel.value_parameter_types) {
    for (const auto &[indices, logical] : *kernel.value_parameter_types) {
      // ArgPack members use their resource owner's binding contract.
      if (indices.size() != 1)
        continue;
      const auto index = indices.front();
      if (index < 0 || index >= parameters.size())
        throw MemoryBindingError("Value argument exceeds compiled parameters");
      const auto offset = kernel.args_type->get_element_offset({index});
      const auto end = index + 1 < parameters.size()
                           ? kernel.args_type->get_element_offset({index + 1})
                           : kernel.args_size;
      if (offset > end || end > kernel.args_size)
        throw MemoryBindingError("Invalid compiled value argument window");
      ValueLayout layout(logical, compiled_value_parameter_type(kernel, indices),
                          end - offset,
                          taichi::arch_uses_llvm(arch) ? ValueStorage::Host
                                                       : ValueStorage::Gfx);
      std::size_t leaf = 0;
      auto tree = value_tree(logical, layout, leaf);
      if (leaf != layout.members.size())
        throw MemoryBindingError("Inconsistent value argument layout");
      values.inner.emplace_back(JsonObject{
          {"index", JsonValue(index)},
          {"buffer_size", JsonValue(end - offset)},
          {"value", std::move(tree)}});
      windows.emplace(index, KernelBinding::ValueWindow{offset, end - offset});
    }
  }
  auto json = print(JsonValue(JsonObject{
      {"version", JsonValue(1)},
      {"argument_count", JsonValue(parameters.size())},
      {"values", JsonValue(std::move(values))}}));
  binding.value_arguments = std::move(windows);
  binding.argument_layout = std::move(json);
  return binding;
}

const Runtime::GraphBinding &Runtime::get_graph(TiComputeGraph graph) const {
  const auto found = graphs_.find(graph);
  if (found == graphs_.end())
    throw MemoryBindingError("Graph is not live in this runtime");
  return found->second;
}
const taichi::lang::aot::StructLayout &Runtime::get_layout(
    TiNdArrayLayout layout) const {
  const auto found = layouts_.find(layout);
  if (found == layouts_.end())
    throw MemoryBindingError("Layout is not live in this runtime");
  return found->second.layout;
}
void Runtime::retire_module(AotModule *owner) {
  modules_.erase(owner);
  invalidate_module_entries(owner);
}

void Runtime::wait_for_kernel_returns(AotModule *owner) {
  for (const auto &[handle, binding] : kernels_) {
    if (binding.owner == owner && binding.pending_return) {
      // A failed wait must leave the module and its readback buffers alive.
      wait();
      for (const auto &[other_handle, other] : kernels_)
        other.pending_return.reset();
      return;
    }
  }
}

void Runtime::invalidate_module_entries(AotModule *owner) {
  auto retire = [owner](auto &items) {
    for (auto it = items.begin(); it != items.end();) {
      if (it->second.owner == owner)
        it = items.erase(it);
      else
        ++it;
    }
  };
  retire(kernels_);
  retire(graphs_);
  retire(layouts_);
}

void Runtime::register_module(AotModule *owner) {
  modules_.insert(owner);
}

void Runtime::retire_resources(bool synchronize) {
  std::lock_guard<std::recursive_mutex> lock(submission_mutex());
  std::exception_ptr failure;
  auto cleanup = [&](auto &&fn) {
    try {
      fn();
    } catch (...) {
      if (!failure)
        failure = std::current_exception();
    }
  };
  // Neither failed synchronization nor failed domain retirement proves that
  // code/storage may be freed. Keep the entire owner closure on that failure.
  if (synchronize)
    wait();
  clear_prepared(false);
  cleanup([&] { clear_native_resources(false); });
  while (!modules_.empty()) {
    auto *module = *modules_.begin();
    modules_.erase(module);
    delete module;
  }
  decltype(memories_) memories;
  {
    std::lock_guard<std::mutex> memory_lock(memory_mutex_);
    memories.swap(memories_);
    for (const auto &[handle, memory] : memories)
      if (memory.borrowed)
        memory.borrowed->live = false;
  }
  for (const auto &[handle, memory] : memories) {
    // Disarm Program revocation callbacks even if a cold metadata snapshot
    // still holds BorrowedMemory after this Runtime has been destroyed.
    if (memory.borrowed) {
      memory.borrowed->external_access.reset();
      memory.borrowed->owner.reset();
    }
    if (memory.owns_allocation)
      cleanup([&] { get().dealloc_memory(memory.allocation); });
  }
  if (failure)
    std::rethrow_exception(failure);
}

AotModule::~AotModule() {
  runtime_->retire_module(this);
}

TiNdArrayLayout AotModule::layout_handle(taichi::lang::DataType type) {
  auto found =
      layout_handles_.find(static_cast<const taichi::lang::Type *>(type));
  if (found != layout_handles_.end())
    return found->second;
  auto handle = runtime_->register_layout(this, type);
  layout_handles_.emplace(static_cast<const taichi::lang::Type *>(type),
                          handle);
  return handle;
}

TiKernel AotModule::get_kernel_handle(const std::string &name) {
  check_code_live();
  auto found = kernel_handles_.find(name);
  if (found != kernel_handles_.end())
    return found->second;
  auto *kernel = get_kernel(name);
  if (!kernel)
    return TI_NULL_HANDLE;
  Runtime::KernelBinding binding{this, kernel, {}};
  for (const auto &[index, parameter] : kernel->nested_parameters) {
    if (parameter.ptype != ParameterType::kNdarray)
      continue;
    const auto payload = ndarray_payload(*kernel, index);
    if (!payload->is<taichi::lang::StructType>())
      continue;
    if (index.size() != 1 || parameter.needs_grad) {
      throw taichi::lang::aot::ArtifactLoadError(
          "Unsupported structured ndarray ABI");
    }
    binding.layouts.emplace(index[0], layout_handle(payload));
  }
  auto handle = runtime_->register_kernel(std::move(binding));
  kernel_handles_.emplace(name, handle);
  return handle;
}

TiNdArrayLayout AotModule::get_layout_handle(const std::string &name) {
  check_code_live();
  return layout_handle(aot_module_->get_ndarray_layout(name));
}

TiComputeGraph AotModule::get_graph_handle(const std::string &name) {
  check_code_live();
  auto found = graph_handles_.find(name);
  if (found != graph_handles_.end())
    return found->second;
  auto *graph = get_cgraph(name);
  if (!graph)
    return TI_NULL_HANDLE;
  Runtime::GraphBinding binding{this, graph, {}};
  for (const auto &dispatch : graph->dispatches) {
    if (!dispatch.compiled_kernel) {
      throw taichi::lang::aot::ArtifactLoadError("Unresolved Graph kernel");
    }
    for (uint32_t i = 0; i < dispatch.symbolic_args.size(); ++i) {
      const auto &arg = dispatch.symbolic_args[i];
      if (arg.tag != taichi::lang::aot::ArgKind::kNdarray)
        continue;
      const auto payload = ndarray_payload(*dispatch.compiled_kernel, {int(i)});
      if (arg.struct_dtype || payload->is<taichi::lang::StructType>()) {
        if (!arg.struct_dtype ||
            payload != taichi::lang::DataType(arg.struct_dtype)) {
          throw taichi::lang::aot::ArtifactLoadError(
              "Graph layout differs from kernel ABI");
        }
        const auto parameter =
            dispatch.compiled_kernel->nested_parameters.find({int(i)});
        if (parameter == dispatch.compiled_kernel->nested_parameters.end() ||
            parameter->second.needs_grad ||
            parameter->second.total_dim != arg.field_dim ||
            !arg.element_shape.empty()) {
          throw taichi::lang::aot::ArtifactLoadError(
              "Graph structure rank differs from kernel ABI");
        }
        binding.layouts.emplace(arg.name, layout_handle(payload));
      }
    }
  }
  auto handle = runtime_->register_graph(std::move(binding));
  graph_handles_.emplace(name, handle);
  return handle;
}

void ti_get_kernel_return_info_ext(TiRuntime runtime,
                                    TiKernel kernel,
                                    TiKernelReturnInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(info);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &binding = rt.get_kernel(kernel);
  if (binding.return_layout.empty())
    binding.return_layout = return_layout_json(*binding.kernel, rt.arch);
  *info = {binding.kernel->ret_size, binding.return_layout.c_str()};
  TI_CAPI_TRY_CATCH_END();
}

void ti_get_kernel_argument_info_ext(TiRuntime runtime,
                                      TiKernel kernel,
                                      TiKernelArgumentInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(info);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  const auto &binding = rt.get_kernel_value_arguments(kernel);
  *info = {static_cast<uint32_t>(binding.kernel->args_type->elements().size()),
           binding.argument_layout.c_str()};
  TI_CAPI_TRY_CATCH_END();
}

TiNdArrayLayout ti_get_kernel_ndarray_layout_ext(TiRuntime runtime,
                                                 TiKernel kernel,
                                                 uint32_t argument_index) {
  TiNdArrayLayout result = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  const auto &binding = ((Runtime *)runtime)->get_kernel(kernel);
  const auto found = binding.layouts.find(argument_index);
  if (found == binding.layouts.end())
    throw MemoryBindingError("Argument has no structure layout");
  result = found->second;
  TI_CAPI_TRY_CATCH_END();
  return result;
}

TiNdArrayLayout ti_get_aot_module_ndarray_layout_ext(TiRuntime runtime,
                                                  TiAotModule module,
                                                  const char *name) {
  TiNdArrayLayout result = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(module);
  TI_CAPI_ARGUMENT_NULL_RV(name);
  auto &rt = *reinterpret_cast<Runtime *>(runtime);
  std::lock_guard<std::recursive_mutex> lock(rt.submission_mutex());
  auto *owner = reinterpret_cast<AotModule *>(module);
  if (!rt.owns_module(owner))
    throw MemoryBindingError("Module is not live in this runtime");
  result = owner->get_layout_handle(name);
  TI_CAPI_TRY_CATCH_END();
  return result;
}
TiNdArrayLayout ti_get_compute_graph_ndarray_layout_ext(TiRuntime runtime,
                                                        TiComputeGraph graph,
                                                        const char *name) {
  TiNdArrayLayout result = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(name);
  const auto &binding = ((Runtime *)runtime)->get_graph(graph);
  const auto found = binding.layouts.find(name);
  if (found == binding.layouts.end())
    throw MemoryBindingError("Argument has no structure layout");
  result = found->second;
  TI_CAPI_TRY_CATCH_END();
  return result;
}
void ti_get_ndarray_layout_info_ext(TiRuntime runtime,
                                    TiNdArrayLayout layout,
                                    TiNdArrayLayoutInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(info);
  const auto &value = ((Runtime *)runtime)->get_layout(layout);
  *info = {value.kVersion, static_cast<uint32_t>(value.members.size()),
           value.stride, value.alignment};
  TI_CAPI_TRY_CATCH_END();
}
void ti_get_ndarray_layout_member_ext(TiRuntime runtime,
                                      TiNdArrayLayout layout,
                                      uint32_t index,
                                      TiNdArrayLayoutMember *member) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(member);
  const auto &value = ((Runtime *)runtime)->get_layout(layout);
  if (index >= value.members.size())
    throw MemoryBindingError("Layout member index out of range");
  const auto &source = value.members[index];
  *member = {};
  member->path = source.path.c_str();
  member->scalar_type = scalar_type(source.scalar_type);
  member->element_shape.dim_count = source.shape.size();
  std::copy(source.shape.begin(), source.shape.end(),
            member->element_shape.dims);
  member->offset = source.offset;
  member->size = source.size;
  member->alignment = source.alignment;
  TI_CAPI_TRY_CATCH_END();
}
