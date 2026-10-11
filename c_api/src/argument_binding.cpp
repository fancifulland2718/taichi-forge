#include "argument_binding.h"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include "fp16.h"

namespace {
constexpr uint32_t kTiNdShapeDimensionCapacity =
    sizeof(((TiNdShape *)nullptr)->dims) / sizeof(uint32_t);

// Kernel and Graph entrypoints must decode the same public scalar ABI. Copy
// bytes to preserve signed integers and avoid aliasing-dependent union reads.
template <typename F>
bool with_scalar_value(const TiScalar &scalar, F &&consume) {
  switch (scalar.type) {
#define SCALAR_CASE(id, type, field)                         \
  case TI_DATA_TYPE_##id: {                                  \
    type value;                                              \
    std::memcpy(&value, &scalar.value.field, sizeof(value)); \
    consume(value);                                          \
    return true;                                             \
  }
    SCALAR_CASE(I8, int8_t, x8);
    SCALAR_CASE(I16, int16_t, x16);
    SCALAR_CASE(I32, int32_t, x32);
    SCALAR_CASE(I64, int64_t, x64);
    SCALAR_CASE(U8, uint8_t, x8);
    SCALAR_CASE(U16, uint16_t, x16);
    SCALAR_CASE(U32, uint32_t, x32);
    SCALAR_CASE(U64, uint64_t, x64);
    // Preserve the legacy C API convention: f16 scalars use float32 storage.
    SCALAR_CASE(F16, float, x32);
    SCALAR_CASE(F32, float, x32);
    SCALAR_CASE(F64, double, x64);
#undef SCALAR_CASE
    default:
      return false;
  }
}

uint32_t ti_tensor_value_capacity(TiDataType type) {
  switch (type) {
    case TI_DATA_TYPE_I8:
    case TI_DATA_TYPE_U8:
      return sizeof(((TiTensorValue *)nullptr)->x8) / sizeof(uint8_t);
    case TI_DATA_TYPE_I16:
    case TI_DATA_TYPE_U16:
    case TI_DATA_TYPE_F16:
      return sizeof(((TiTensorValue *)nullptr)->x16) / sizeof(uint16_t);
    case TI_DATA_TYPE_I32:
    case TI_DATA_TYPE_U32:
    case TI_DATA_TYPE_F32:
      return sizeof(((TiTensorValue *)nullptr)->x32) / sizeof(uint32_t);
    case TI_DATA_TYPE_I64:
    case TI_DATA_TYPE_U64:
    case TI_DATA_TYPE_F64:
      return sizeof(((TiTensorValue *)nullptr)->x64) / sizeof(uint64_t);
    default:
      return 0;
  }
}

}  // namespace

namespace {
bool gradient_error(const std::string &message) {
  ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, message.c_str());
  return false;
}

void validate_struct_argument(Runtime &runtime,
                              const TiArgument &argument,
                              TiNdArrayLayout provided,
                              TiNdArrayLayout expected,
                              uint32_t version,
                              size_t ndim) {
  const auto &layout = runtime.get_layout(provided);
  const auto &required = runtime.get_layout(expected);
  if (version != taichi::lang::aot::StructLayout::kVersion ||
      layout.type != required.type || layout.stride != required.stride) {
    throw MemoryBindingError(
        "Structure layout/version does not match exported ABI");
  }
  if (argument.type != TI_ARGUMENT_TYPE_NDARRAY) {
    throw MemoryBindingError("Structure binding requires an ndarray argument");
  }
  const auto &array = argument.value.ndarray;
  if (array.elem_type != TI_DATA_TYPE_UNKNOWN ||
      array.elem_shape.dim_count != 0 || array.shape.dim_count != ndim ||
      array.shape.dim_count > kTiNdShapeDimensionCapacity) {
    throw MemoryBindingError(
        "Structure ndarray type or rank does not match exported ABI");
  }
  uint64_t bytes = layout.stride;
  for (uint32_t i = 0; i < array.shape.dim_count; ++i) {
    const auto extent = array.shape.dims[i];
    if (extent > std::numeric_limits<int>::max() ||
        (extent != 0 &&
         bytes > std::numeric_limits<uint64_t>::max() / extent)) {
      throw MemoryBindingError(
          "Structure ndarray shape overflows native indexing");
    }
    bytes *= extent;
  }
  const auto memory = runtime.get_memory(array.memory);
  if (memory.size < bytes) {
    throw MemoryBindingError(
        "Structure ndarray buffer is smaller than its declared shape");
  }
  if (memory.base_offset % layout.alignment != 0) {
    throw MemoryBindingError(
        "Imported memory does not satisfy structure alignment");
  }
}

bool same_ndarray_shape(const TiNdShape &a, const TiNdShape &b) {
  return a.dim_count <= kTiNdShapeDimensionCapacity &&
         b.dim_count == a.dim_count &&
         std::equal(a.dims, a.dims + a.dim_count, b.dims);
}

taichi::lang::DataType primitive_type(TiDataType type) {
  using taichi::lang::PrimitiveType;
  switch (type) {
#define ARRAY_TYPE(c, native) \
  case TI_DATA_TYPE_##c:      \
    return PrimitiveType::native
    ARRAY_TYPE(F16, f16);
    ARRAY_TYPE(F32, f32);
    ARRAY_TYPE(F64, f64);
    ARRAY_TYPE(I8, i8);
    ARRAY_TYPE(I16, i16);
    ARRAY_TYPE(I32, i32);
    ARRAY_TYPE(I64, i64);
    ARRAY_TYPE(U8, u8);
    ARRAY_TYPE(U16, u16);
    ARRAY_TYPE(U32, u32);
    ARRAY_TYPE(U64, u64);
#undef ARRAY_TYPE
    default:
      throw MemoryBindingError("Unsupported ndarray scalar type");
  }
}

void validate_array_storage(Runtime &runtime, const TiNdArray &array) {
  if (array.shape.dim_count > kTiNdShapeDimensionCapacity ||
      array.elem_shape.dim_count > kTiNdShapeDimensionCapacity) {
    throw MemoryBindingError("Ndarray rank exceeds the C API shape capacity");
  }
  const auto type = primitive_type(array.elem_type);
  const auto scalar_bytes = taichi::lang::data_type_size(type);
  uint64_t bytes = scalar_bytes;
  // Check element shape even for an empty logical array. Zero element extents
  // are not valid tensor types; zero array extents are valid empty ndarrays.
  for (const auto *shape : {&array.elem_shape, &array.shape}) {
    for (uint32_t i = 0; i < shape->dim_count; ++i) {
      const auto extent = shape->dims[i];
      if ((shape == &array.elem_shape && extent == 0) ||
          extent > std::numeric_limits<int>::max() ||
          (extent && bytes > std::numeric_limits<uint64_t>::max() / extent)) {
        throw MemoryBindingError("Ndarray shape overflows native indexing");
      }
      bytes *= extent;
    }
  }
  const auto memory = runtime.get_memory(array.memory);
  if (memory.size < bytes)
    throw MemoryBindingError(
        "Ndarray buffer is smaller than its declared shape");
  if (memory.base_offset % scalar_bytes)
    throw MemoryBindingError("Imported ndarray memory is not scalar aligned");
}

void validate_kernel_array(Runtime &runtime,
                           taichi::lang::aot::Kernel &kernel,
                           uint32_t index,
                           const TiNdArray &array) {
  using namespace taichi::lang;
  auto parameter = kernel.nested_parameters.find({int(index)});
  if (parameter == kernel.nested_parameters.end() ||
      parameter->second.ptype != ParameterType::kNdarray) {
    throw MemoryBindingError(
        "Ndarray argument does not match the exported ABI");
  }
  validate_array_storage(runtime, array);
  const auto &param = parameter->second;
  const auto *abi =
      kernel.args_type->get_element_type({int(index)})->as<StructType>();
  const auto payload =
      DataType(abi->get_element_type({TypeFactory::DATA_PTR_POS_IN_NDARRAY}))
          .ptr_removed()
          .get_element_type();
  if (payload != primitive_type(array.elem_type) ||
      param.element_shape.size() != array.elem_shape.dim_count ||
      !std::equal(param.element_shape.begin(), param.element_shape.end(),
                  array.elem_shape.dims) ||
      array.shape.dim_count + array.elem_shape.dim_count != param.total_dim) {
    throw MemoryBindingError(
        "Ndarray dtype or shape does not match the exported kernel ABI");
  }
}

bool validate_gradient_pair(const TiArgument &argument,
                            const TiNdArray &gradient) {
  if (argument.type != TI_ARGUMENT_TYPE_NDARRAY)
    return gradient_error("Gradient target must be an ndarray");
  const auto &primal = argument.value.ndarray;
  if (!primal.memory || !gradient.memory)
    return gradient_error("Primal and gradient memory must be non-null");
  if (primal.elem_type != gradient.elem_type ||
      !same_ndarray_shape(primal.shape, gradient.shape) ||
      !same_ndarray_shape(primal.elem_shape, gradient.elem_shape))
    return gradient_error("Gradient dtype and shape must match the primal");
  return true;
}

bool validate_gradient_parameter(
    const taichi::lang::CallableBase::Parameter &param,
    const TiArgument *argument,
    const TiNdArray *gradient) {
  if (!param.needs_grad)
    return true;
  if (!argument || !gradient)
    return gradient_error(
        "Missing ndarray gradient binding for needs_grad argument");
  if (!validate_gradient_pair(*argument, *gradient))
    return false;
  const auto &array = argument->value.ndarray;
  auto type = param.get_element_type();
  // LLVM keeps the full ndarray ABI struct; GFX reconstructs scalar/tensor
  // parameter metadata from the serialized argument attributes.
  if (auto *array_type = type->cast<taichi::lang::StructType>()) {
    type = taichi::lang::DataType(
               array_type->get_element_type(
                   {taichi::lang::TypeFactory::DATA_PTR_POS_IN_NDARRAY}))
               .ptr_removed();
  }
  using taichi::lang::PrimitiveType;
  bool dtype_matches =
      (array.elem_type == TI_DATA_TYPE_F16 && type == PrimitiveType::f16) ||
      (array.elem_type == TI_DATA_TYPE_F32 && type == PrimitiveType::f32) ||
      (array.elem_type == TI_DATA_TYPE_F64 && type == PrimitiveType::f64);
  const auto &element_shape = param.element_shape;
  if (!dtype_matches || element_shape.size() != array.elem_shape.dim_count ||
      !std::equal(element_shape.begin(), element_shape.end(),
                  array.elem_shape.dims) ||
      array.shape.dim_count + element_shape.size() != param.total_dim)
    return gradient_error(
        "Primal ndarray does not match the exported gradient ABI");
  return true;
}
}  // namespace

namespace capi {

std::unique_ptr<BoundKernelArguments> bind_kernel_arguments(
    Runtime &runtime2,
    TiKernel kernel,
    uint32_t arg_count,
    const TiArgument *args,
    uint32_t binding_count,
    const TiNdArrayBinding *bindings) {
  TI_CAPI_ARGUMENT_NULL_RV(kernel);
  if (arg_count > 0) {
    TI_CAPI_ARGUMENT_NULL_RV(args);
  }
  if (arg_count > taichi_max_num_args_total) {
    ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "arg_count");
    return nullptr;
  }

  const auto &kernel_binding = runtime2.get_kernel(kernel);
  auto ti_kernel = kernel_binding.kernel;
  if (arg_count != ti_kernel->args_type->elements().size()) {
    throw MemoryBindingError(
        "Argument count does not match the exported kernel ABI");
  }

  if (binding_count)
    TI_CAPI_ARGUMENT_NULL_RV(bindings);
  if (binding_count > arg_count) {
    gradient_error("Too many ndarray bindings");
    return nullptr;
  }
  std::vector<const TiNdArrayBinding *> bound_args(arg_count, nullptr);
  std::vector<TiNdArrayGradientBinding> gradient_bindings;
  gradient_bindings.reserve(binding_count);
  for (uint32_t i = 0; i < binding_count; ++i) {
    const auto &binding = bindings[i];
    const auto index = binding.argument_index;
    if (index >= arg_count || bound_args[index]) {
      throw MemoryBindingError("Duplicate or out-of-range ndarray binding");
    }
    bound_args[index] = &binding;
    auto expected = kernel_binding.layouts.find(index);
    if (expected == kernel_binding.layouts.end()) {
      if (binding.layout || binding.layout_version != 0 || !binding.gradient) {
        throw MemoryBindingError("Unexpected structure layout binding");
      }
    } else if (binding.gradient) {
      throw MemoryBindingError("Structure ndarray gradients are not supported");
    }
    if (binding.gradient)
      gradient_bindings.push_back({index, *binding.gradient});
  }
  for (const auto &[index, expected] : kernel_binding.layouts) {
    if (index >= arg_count || !bound_args[index]) {
      throw MemoryBindingError("Missing structure ndarray layout binding");
    }
    const auto &binding = *bound_args[index];
    validate_struct_argument(
        runtime2, args[index], binding.layout, expected, binding.layout_version,
        ti_kernel->nested_parameters.at({int(index)}).total_dim);
  }
  const auto gradient_count = gradient_bindings.size();
  const auto *gradients = gradient_bindings.data();

  if (gradient_count > 0)
    TI_CAPI_ARGUMENT_NULL_RV(gradients);
  std::vector<const TiNdArray *> gradient_args(arg_count, nullptr);
  for (uint32_t i = 0; i < gradient_count; ++i) {
    const auto &binding = gradients[i];
    auto index = binding.argument_index;
    if (index >= arg_count || gradient_args[index] != nullptr) {
      gradient_error("Duplicate or out-of-range gradient argument index");
      return nullptr;
    }
    auto param = ti_kernel->nested_parameters.find({static_cast<int>(index)});
    if (param == ti_kernel->nested_parameters.end() ||
        !param->second.needs_grad) {
      gradient_error("Gradient target was not exported with needs_grad");
      return nullptr;
    }
    if (!validate_gradient_pair(args[index], binding.gradient))
      return nullptr;
    gradient_args[index] = &binding.gradient;
  }
  for (const auto &entry : ti_kernel->nested_parameters) {
    if (!entry.second.needs_grad)
      continue;
    if (entry.first.size() != 1) {
      gradient_error(
          "Nested gradient arguments are not supported by this extension");
      return nullptr;
    }
    auto index = static_cast<uint32_t>(entry.first[0]);
    if (!validate_gradient_parameter(
            entry.second, index < arg_count ? &args[index] : nullptr,
            index < arg_count ? gradient_args[index] : nullptr))
      return nullptr;
  }

  auto result = std::make_unique<BoundKernelArguments>();
  result->owner = kernel_binding.owner;
  result->kernel = ti_kernel;
  result->context =
      std::make_unique<taichi::lang::LaunchContextBuilder>(ti_kernel);
  auto &builder = *result->context;
  auto &devallocs = result->allocations;

  for (uint32_t i = 0; i < arg_count; ++i) {
    const auto &arg = args[i];
    if (arg.type == TI_ARGUMENT_TYPE_NDARRAY) {
      if (!kernel_binding.layouts.count(i))
        validate_kernel_array(runtime2, *ti_kernel, i, arg.value.ndarray);
      result->memories.push_back(arg.value.ndarray.memory);
      if (gradient_args[i]) {
        validate_array_storage(runtime2, *gradient_args[i]);
        result->memories.push_back(gradient_args[i]->memory);
      }
    }
    switch (arg.type) {
      case TI_ARGUMENT_TYPE_SCALAR: {
        if (!with_scalar_value(arg.value.scalar, [&](auto value) {
              const auto type =
                  arg.value.scalar.type == TI_DATA_TYPE_F16
                      ? taichi::lang::PrimitiveType::f16
                      : taichi::lang::get_data_type<decltype(value)>();
              if (type !=
                  taichi::lang::DataType(
                      ti_kernel->args_type->get_element_type({int(i)}))) {
                throw MemoryBindingError(
                    "Scalar type does not match the exported kernel ABI");
              }
              if (arg.value.scalar.type == TI_DATA_TYPE_F16)
                builder.set_arg_float({(int)i}, value);
              else
                builder.set_arg({(int)i}, value);
            })) {
          ti_set_last_error(
              TI_ERROR_ARGUMENT_OUT_OF_RANGE,
              ("args[" + std::to_string(i) + "].value.scalar.type").c_str());
          return nullptr;
        }
        break;
      }

      case TI_ARGUMENT_TYPE_I32: {
        builder.set_arg({(int)i}, arg.value.i32);
        break;
      }
      case TI_ARGUMENT_TYPE_F32: {
        builder.set_arg({(int)i}, arg.value.f32);
        break;
      }
      case TI_ARGUMENT_TYPE_NDARRAY: {
        TI_CAPI_ARGUMENT_NULL_RV(args[i].value.ndarray.memory);

        // Don't allocate it on stack. `DeviceAllocation` is referred to by
        // `GfxRuntime::launch_kernel`.
        std::unique_ptr<taichi::lang::DeviceAllocation> devalloc =
            std::make_unique<taichi::lang::DeviceAllocation>(
                devmem2devalloc(runtime2, arg.value.ndarray.memory));
        const TiNdArray &ndarray = arg.value.ndarray;

        if (ndarray.shape.dim_count > kTiNdShapeDimensionCapacity) {
          ti_set_last_error(
              TI_ERROR_ARGUMENT_OUT_OF_RANGE,
              ("args[" + std::to_string(i) + "].value.ndarray.shape.dim_count")
                  .c_str());
          return nullptr;
        }
        if (ndarray.elem_shape.dim_count > kTiNdShapeDimensionCapacity) {
          ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                            ("args[" + std::to_string(i) +
                             "].value.ndarray.elem_shape.dim_count")
                                .c_str());
          return nullptr;
        }

        std::vector<int> shape(ndarray.shape.dims,
                               ndarray.shape.dims + ndarray.shape.dim_count);

        std::unique_ptr<taichi::lang::DeviceAllocation> gradalloc;
        if (gradient_args[i]) {
          gradalloc = std::make_unique<taichi::lang::DeviceAllocation>(
              devmem2devalloc(runtime2, gradient_args[i]->memory));
        }
        builder.set_arg_ndarray_impl({(int)i}, (intptr_t)devalloc.get(), shape,
                                     (intptr_t)gradalloc.get());
        if (gradalloc)
          devallocs.emplace_back(std::move(gradalloc));

        devallocs.emplace_back(std::move(devalloc));
        break;
      }
      case TI_ARGUMENT_TYPE_TEXTURE: {
        TI_CAPI_ARGUMENT_NULL_RV(args[i].value.texture.image);
        std::unique_ptr<taichi::lang::DeviceAllocation> devalloc =
            std::make_unique<taichi::lang::DeviceAllocation>(
                devimg2devalloc(runtime2, arg.value.texture.image));
        int width = arg.value.texture.extent.width;
        int height = arg.value.texture.extent.height;
        int depth = arg.value.texture.extent.depth;
        builder.set_arg_rw_texture_impl({(int)i}, (intptr_t)devalloc.get(),
                                        {width, height, depth});
        devallocs.emplace_back(std::move(devalloc));
        break;
      }
      case TI_ARGUMENT_TYPE_TENSOR: {
        auto &tensor = arg.value.tensor;
        const uint32_t capacity = ti_tensor_value_capacity(tensor.type);
        if (capacity != 0 && tensor.contents.length > capacity) {
          ti_set_last_error(
              TI_ERROR_ARGUMENT_OUT_OF_RANGE,
              ("args[" + std::to_string(i) + "].value.tensor.contents.length")
                  .c_str());
          return nullptr;
        }
        if (tensor.type == TI_DATA_TYPE_I16 ||
            tensor.type == TI_DATA_TYPE_U16 ||
            tensor.type == TI_DATA_TYPE_F16) {
          for (int j = 0; j < tensor.contents.length; j++) {
            builder.set_struct_arg_impl({(int)i, j},
                                        tensor.contents.data.x16[j]);
          }
        } else if (tensor.type == TI_DATA_TYPE_I32 ||
                   tensor.type == TI_DATA_TYPE_U32 ||
                   tensor.type == TI_DATA_TYPE_F32) {
          for (int j = 0; j < tensor.contents.length; j++) {
            builder.set_struct_arg_impl({(int)i, j},
                                        tensor.contents.data.x32[j]);
          }
        } else {
          ti_set_last_error(TI_ERROR_NOT_SUPPORTED,
                            ("args[" + std::to_string(i) + "].type").c_str());
          return nullptr;
        }
        break;
      }
      default: {
        ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                          ("args[" + std::to_string(i) + "].type").c_str());
        return nullptr;
      }
    }
  }
  return result;
}

std::unique_ptr<BoundGraphArguments> bind_graph_arguments(
    Runtime &runtime2,
    TiComputeGraph compute_graph,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings) {
  TI_CAPI_ARGUMENT_NULL_RV(compute_graph);
  if (arg_count > 0) {
    TI_CAPI_ARGUMENT_NULL_RV(args);
  }
  if (arg_count > taichi_max_num_args_total) {
    ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "arg_count");
    return nullptr;
  }

  const auto &graph_binding = runtime2.get_graph(compute_graph);
  auto &graph = *graph_binding.graph;
  std::unordered_map<std::string, const TiArgument *> named_args;
  for (uint32_t i = 0; i < arg_count; ++i) {
    TI_CAPI_ARGUMENT_NULL_RV(args[i].name);
    if (!named_args.emplace(args[i].name, &args[i].argument).second) {
      gradient_error("Duplicate Graph argument name");
      return nullptr;
    }
  }
  if (binding_count)
    TI_CAPI_ARGUMENT_NULL_RV(bindings);
  if (binding_count > arg_count) {
    gradient_error("Too many Graph ndarray bindings");
    return nullptr;
  }
  std::unordered_map<std::string, const TiNamedNdArrayBinding *> bound_args;
  std::vector<TiNamedNdArrayGradientBinding> gradient_bindings;
  gradient_bindings.reserve(binding_count);
  for (uint32_t i = 0; i < binding_count; ++i) {
    const auto &binding = bindings[i];
    TI_CAPI_ARGUMENT_NULL_RV(binding.name);
    auto arg = named_args.find(binding.name);
    if (arg == named_args.end() ||
        !bound_args.emplace(binding.name, &binding).second) {
      throw MemoryBindingError("Unknown or duplicate Graph ndarray binding");
    }
    auto expected = graph_binding.layouts.find(binding.name);
    if (expected == graph_binding.layouts.end()) {
      if (binding.layout || binding.layout_version != 0 || !binding.gradient) {
        throw MemoryBindingError("Unexpected Graph structure layout binding");
      }
    } else if (binding.gradient) {
      throw MemoryBindingError("Structure ndarray gradients are not supported");
    }
    if (binding.gradient)
      gradient_bindings.push_back({binding.name, *binding.gradient});
  }
  for (const auto &[name, expected] : graph_binding.layouts) {
    auto arg = named_args.find(name);
    auto bound = bound_args.find(name);
    if (arg == named_args.end() || bound == bound_args.end()) {
      throw MemoryBindingError(
          "Missing Graph structure ndarray layout binding");
    }
    const auto &binding = *bound->second;
    validate_struct_argument(runtime2, *arg->second, binding.layout, expected,
                             binding.layout_version,
                             graph.args.at(name).field_dim);
  }
  const auto gradient_count = gradient_bindings.size();
  const auto *gradients = gradient_bindings.data();
  std::unordered_map<std::string, const TiNdArray *> gradient_args;
  for (uint32_t i = 0; i < gradient_count; ++i) {
    TI_CAPI_ARGUMENT_NULL_RV(gradients[i].name);
    const auto &binding = gradients[i];
    auto arg = named_args.find(binding.name);
    if (arg == named_args.end() ||
        !gradient_args.emplace(binding.name, &binding.gradient).second) {
      gradient_error("Unknown or duplicate Graph gradient name");
      return nullptr;
    }
    if (!validate_gradient_pair(*arg->second, binding.gradient))
      return nullptr;
  }
  std::unordered_set<std::string> required_gradients;
  // Validate every dispatch before executing any of them, including a reverse
  // dispatch that appears after otherwise valid forward work.
  for (const auto &dispatch : graph.dispatches) {
    if (dispatch.compiled_kernel == nullptr) {
      ti_set_last_error(TI_ERROR_CORRUPTED_DATA,
                        "AOT Graph contains an unresolved kernel");
      return nullptr;
    }
    for (const auto &entry : dispatch.compiled_kernel->nested_parameters) {
      if (!entry.second.needs_grad)
        continue;
      if (entry.first.size() != 1 || entry.first[0] < 0 ||
          entry.first[0] >= dispatch.symbolic_args.size()) {
        gradient_error("Unsupported Graph gradient argument metadata");
        return nullptr;
      }
      const auto &name = dispatch.symbolic_args[entry.first[0]].name;
      required_gradients.insert(name);
      auto arg = named_args.find(name);
      auto grad = gradient_args.find(name);
      if (!validate_gradient_parameter(
              entry.second, arg == named_args.end() ? nullptr : arg->second,
              grad == gradient_args.end() ? nullptr : grad->second))
        return nullptr;
    }
  }
  for (const auto &entry : gradient_args) {
    if (!required_gradients.count(entry.first)) {
      gradient_error("Graph gradient target was not exported with needs_grad");
      return nullptr;
    }
  }
  auto result = std::make_unique<BoundGraphArguments>();
  result->owner = graph_binding.owner;
  result->graph = &graph;
  auto &gradient_arrays = result->gradient_arrays;
  auto &gradient_views = result->gradients;
  auto &arg_map = result->arguments;
  auto &ndarrays = result->arrays;
  auto &textures = result->textures;
  auto &matrices = result->matrices;
  gradient_arrays.reserve(gradient_count);
  ndarrays.reserve(arg_count);
  textures.reserve(arg_count);
  matrices.reserve(arg_count);

  for (uint32_t i = 0; i < arg_count; ++i) {
    TI_CAPI_ARGUMENT_NULL_RV(args[i].name);

    const auto &arg = args[i];
    if (arg.argument.type == TI_ARGUMENT_TYPE_NDARRAY) {
      const auto &array = arg.argument.value.ndarray;
      if (!graph_binding.layouts.count(arg.name)) {
        validate_array_storage(runtime2, array);
        const auto found = graph.args.find(arg.name);
        if (found == graph.args.end() ||
            found->second.tag != taichi::lang::aot::ArgKind::kNdarray ||
            found->second.dtype().get_element_type() !=
                primitive_type(array.elem_type) ||
            found->second.field_dim != array.shape.dim_count ||
            found->second.element_shape.size() != array.elem_shape.dim_count ||
            !std::equal(found->second.element_shape.begin(),
                        found->second.element_shape.end(),
                        array.elem_shape.dims)) {
          throw MemoryBindingError(
              "Ndarray dtype or shape does not match the exported Graph ABI");
        }
      }
      result->memories.push_back(array.memory);
      const auto grad = gradient_args.find(arg.name);
      if (grad != gradient_args.end()) {
        validate_array_storage(runtime2, *grad->second);
        result->memories.push_back(grad->second->memory);
      }
    }
    switch (arg.argument.type) {
      case TI_ARGUMENT_TYPE_SCALAR: {
        if (!with_scalar_value(arg.argument.value.scalar, [&](auto value) {
              const bool half =
                  arg.argument.value.scalar.type == TI_DATA_TYPE_F16;
              const auto type =
                  half ? taichi::lang::PrimitiveType::f16
                       : taichi::lang::get_data_type<decltype(value)>();
              const auto expected = graph.args.find(arg.name);
              if (expected == graph.args.end() ||
                  expected->second.tag != taichi::lang::aot::ArgKind::kScalar ||
                  expected->second.dtype() != type) {
                throw MemoryBindingError(
                    "Scalar type or name does not match the exported Graph "
                    "ABI");
              }
              // Graph IValues carry raw bits, unlike the C API's legacy f16
              // scalar representation. Convert before the shared Graph binder.
              if (half)
                arg_map.emplace(
                    arg.name,
                    taichi::lang::aot::IValue::create(
                        fp16_ieee_from_fp32_value(static_cast<float>(value))));
              else
                arg_map.emplace(arg.name,
                                taichi::lang::aot::IValue::create(value));
            })) {
          ti_set_last_error(
              TI_ERROR_ARGUMENT_OUT_OF_RANGE,
              ("args[" + std::to_string(i) + "].value.scalar.type").c_str());
          return nullptr;
        }
        break;
      }
      case TI_ARGUMENT_TYPE_I32: {
        arg_map.emplace(
            std::make_pair(arg.name, taichi::lang::aot::IValue::create<int32_t>(
                                         arg.argument.value.i32)));
        break;
      }
      case TI_ARGUMENT_TYPE_F32: {
        arg_map.emplace(std::make_pair(
            arg.name,
            taichi::lang::aot::IValue::create<float>(arg.argument.value.f32)));
        break;
      }
      case TI_ARGUMENT_TYPE_NDARRAY: {
        TI_CAPI_ARGUMENT_NULL_RV(args[i].argument.value.ndarray.memory);

        taichi::lang::DeviceAllocation devalloc =
            devmem2devalloc(runtime2, arg.argument.value.ndarray.memory);
        const TiNdArray &ndarray = arg.argument.value.ndarray;

        if (ndarray.shape.dim_count > kTiNdShapeDimensionCapacity) {
          ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                            ("args[" + std::to_string(i) +
                             "].argument.value.ndarray.shape.dim_count")
                                .c_str());
          return nullptr;
        }
        if (ndarray.elem_shape.dim_count > kTiNdShapeDimensionCapacity) {
          ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                            ("args[" + std::to_string(i) +
                             "].argument.value.ndarray.elem_shape.dim_count")
                                .c_str());
          return nullptr;
        }

        std::vector<int> shape(ndarray.shape.dims,
                               ndarray.shape.dims + ndarray.shape.dim_count);

        std::vector<int> elem_shape(
            ndarray.elem_shape.dims,
            ndarray.elem_shape.dims + ndarray.elem_shape.dim_count);

        auto structure = graph_binding.layouts.find(arg.name);
        if (structure != graph_binding.layouts.end()) {
          ndarrays.emplace_back(devalloc,
                                runtime2.get_layout(structure->second).type,
                                shape, ExternalArrayLayout::kAOS);
          arg_map.emplace(arg.name,
                          taichi::lang::aot::IValue::create(ndarrays.back()));
          break;
        }

        const taichi::lang::DataType *prim_ty;
        switch (ndarray.elem_type) {
          case TI_DATA_TYPE_F16:
            prim_ty = &taichi::lang::PrimitiveType::f16;
            break;
          case TI_DATA_TYPE_F32:
            prim_ty = &taichi::lang::PrimitiveType::f32;
            break;
          case TI_DATA_TYPE_F64:
            prim_ty = &taichi::lang::PrimitiveType::f64;
            break;
          case TI_DATA_TYPE_I8:
            prim_ty = &taichi::lang::PrimitiveType::i8;
            break;
          case TI_DATA_TYPE_I16:
            prim_ty = &taichi::lang::PrimitiveType::i16;
            break;
          case TI_DATA_TYPE_I32:
            prim_ty = &taichi::lang::PrimitiveType::i32;
            break;
          case TI_DATA_TYPE_I64:
            prim_ty = &taichi::lang::PrimitiveType::i64;
            break;
          case TI_DATA_TYPE_U8:
            prim_ty = &taichi::lang::PrimitiveType::u8;
            break;
          case TI_DATA_TYPE_U16:
            prim_ty = &taichi::lang::PrimitiveType::u16;
            break;
          case TI_DATA_TYPE_U32:
            prim_ty = &taichi::lang::PrimitiveType::u32;
            break;
          case TI_DATA_TYPE_U64:
            prim_ty = &taichi::lang::PrimitiveType::u64;
            break;
          case TI_DATA_TYPE_GEN:
            prim_ty = &taichi::lang::PrimitiveType::gen;
            break;
          default: {
            ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                              ("args[" + std::to_string(i) +
                               "].argument.value.ndarray.elem_type")
                                  .c_str());
            return nullptr;
          }
        }

        taichi::lang::DataType dtype = *prim_ty;
        if (elem_shape.size() > 0) {
          dtype = taichi::lang::TypeFactory::get_instance().get_tensor_type(
              elem_shape, dtype);
        }
        ndarrays.emplace_back(taichi::lang::Ndarray(devalloc, dtype, shape));
        auto grad = gradient_args.find(arg.name);
        if (grad != gradient_args.end()) {
          gradient_arrays.emplace_back(
              devmem2devalloc(runtime2, grad->second->memory), dtype, shape);
          gradient_views.emplace(arg.name, &gradient_arrays.back());
        }
        arg_map.emplace(std::make_pair(
            arg.name, taichi::lang::aot::IValue::create(ndarrays.back())));
        break;
      }
      case TI_ARGUMENT_TYPE_TEXTURE: {
        TI_CAPI_ARGUMENT_NULL_RV(args[i].argument.value.texture.image);

        taichi::lang::DeviceAllocation devalloc =
            devimg2devalloc(runtime2, arg.argument.value.texture.image);
        taichi::lang::BufferFormat format =
            (taichi::lang::BufferFormat)arg.argument.value.texture.format;
        uint32_t width = arg.argument.value.texture.extent.width;
        uint32_t height = arg.argument.value.texture.extent.height;
        uint32_t depth = arg.argument.value.texture.extent.depth;

        textures.push_back(std::make_unique<taichi::lang::Texture>(
            devalloc, format, width, height, depth));
        arg_map.emplace(std::make_pair(
            arg.name, taichi::lang::aot::IValue::create(*textures.back())));
        break;
      }
      case TI_ARGUMENT_TYPE_TENSOR: {
        TI_CAPI_ARGUMENT_NULL_RV(
            args[i].argument.value.tensor.contents.data.x8);

        uint32_t length = arg.argument.value.tensor.contents.length;
        const uint32_t capacity =
            ti_tensor_value_capacity(arg.argument.value.tensor.type);
        if (capacity != 0 && length > capacity) {
          ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                            ("args[" + std::to_string(i) +
                             "].argument.value.tensor.contents.length")
                                .c_str());
          return nullptr;
        }
        const taichi::lang::DataType *prim_ty;
        switch (arg.argument.value.tensor.type) {
          case TI_DATA_TYPE_F16:
            prim_ty = &taichi::lang::PrimitiveType::f16;
            break;
          case TI_DATA_TYPE_F32:
            prim_ty = &taichi::lang::PrimitiveType::f32;
            break;
          case TI_DATA_TYPE_F64:
            prim_ty = &taichi::lang::PrimitiveType::f64;
            break;
          case TI_DATA_TYPE_I8:
            prim_ty = &taichi::lang::PrimitiveType::i8;
            break;
          case TI_DATA_TYPE_I16:
            prim_ty = &taichi::lang::PrimitiveType::i16;
            break;
          case TI_DATA_TYPE_I32:
            prim_ty = &taichi::lang::PrimitiveType::i32;
            break;
          case TI_DATA_TYPE_I64:
            prim_ty = &taichi::lang::PrimitiveType::i64;
            break;
          case TI_DATA_TYPE_U8:
            prim_ty = &taichi::lang::PrimitiveType::u8;
            break;
          case TI_DATA_TYPE_U16:
            prim_ty = &taichi::lang::PrimitiveType::u16;
            break;
          case TI_DATA_TYPE_U32:
            prim_ty = &taichi::lang::PrimitiveType::u32;
            break;
          case TI_DATA_TYPE_U64:
            prim_ty = &taichi::lang::PrimitiveType::u64;
            break;
          default: {
            ti_set_last_error(
                TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                ("args[" + std::to_string(i) + "].argument.value.tensor.type")
                    .c_str());
            return nullptr;
          }
        }
        intptr_t data = reinterpret_cast<intptr_t>(
            arg.argument.value.tensor.contents.data.x8);

        taichi::lang::DataType dtype = *prim_ty;
        matrices.emplace_back(taichi::lang::Matrix(length, dtype, data));
        arg_map.emplace(std::make_pair(
            arg.name, taichi::lang::aot::IValue::create(matrices.back())));
        break;
      }
      default: {
        ti_set_last_error(
            TI_ERROR_ARGUMENT_OUT_OF_RANGE,
            ("args[" + std::to_string(i) + "].argument.type").c_str());
        return nullptr;
      }
    }
  }
  return result;
}

}  // namespace capi
