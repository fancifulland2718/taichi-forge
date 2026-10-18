#include "taichi_core_impl.h"

#include <atomic>
#include <limits>

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
  }
  for (const auto &[handle, memory] : memories)
    if (memory.owns_allocation)
      cleanup([&] { get().dealloc_memory(memory.allocation); });
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

TiComputeGraph AotModule::get_graph_handle(const std::string &name) {
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
