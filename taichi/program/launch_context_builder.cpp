#include <algorithm>
#include <limits>

#include <taichi/program/launch_context_builder.h>
#include "taichi/program/program.h"
#define TI_RUNTIME_HOST
#include <taichi/program/context.h>
#undef TI_RUNTIME_HOST
#include "taichi/system/profiler.h"
#include "taichi/system/profiler_annotation.h"
#include "fp16.h"

namespace taichi::lang {

namespace {
template <typename T>
inline std::vector<T> concatenate_vector(const std::vector<T> &lhs,
                                         const std::vector<T> &rhs) {
  std::vector<T> result;
  result.assign(lhs.begin(), lhs.end());
  result.insert(result.end(), rhs.begin(), rhs.end());
  return result;
}
}  // namespace

LaunchContextBuilder::LaunchContextBuilder(CallableBase *kernel,
                                           bool cpu_bounded_range)
    : kernel_(kernel),
      owned_ctx_(std::make_unique<RuntimeContext>()),
      ctx_(owned_ctx_.get()),
      arg_buffer_(std::make_unique<char[]>(
          (cpu_bounded_range ? sizeof(CpuBoundedRangeBinding) : 0) +
          kernel->args_size)),
      result_buffer_(std::make_unique<char[]>(kernel->ret_size)),
      ret_type_(kernel->ret_type),
      has_cpu_bounded_range_binding_(cpu_bounded_range),
      arg_buffer_size(kernel->args_size),
      args_type(kernel->args_type),
      result_buffer_size(kernel->ret_size) {
  TI_COMPILE_PROFILER("launch_context_builder_ctor");
  ctx_->result_buffer = (uint64 *)result_buffer_.get();
  ctx_->arg_buffer = arg_buffer_.get() +
                     (cpu_bounded_range ? sizeof(CpuBoundedRangeBinding) : 0);
  if (const auto *label = current_dispatch_label()) {
    dispatch_label_ = *label;
  }
}

void LaunchContextBuilder::set_cpu_bounded_range(void *extent,
                                                 std::int32_t capacity) {
  TI_ASSERT(has_cpu_bounded_range_binding_);
  TI_ASSERT(extent != nullptr);
  TI_ASSERT(capacity > 0);
  auto *binding = reinterpret_cast<CpuBoundedRangeBinding *>(
      ctx_->arg_buffer - sizeof(CpuBoundedRangeBinding));
  binding->extent = reinterpret_cast<std::uintptr_t>(extent);
  binding->capacity = capacity;
  binding->reserved = 0;
}

void LaunchContextBuilder::append_dispatch_label(const std::string &label) {
  if (label.empty()) {
    return;
  }
  validate_dispatch_label(label);
  if (!dispatch_label_.empty()) {
    dispatch_label_.push_back('/');
  }
  dispatch_label_ += label;
}

void LaunchContextBuilder::set_arg_float(const std::vector<int> &arg_id,
                                         float64 d) {
  auto dt = kernel_->args_type->get_element_type(arg_id);
  TI_ASSERT_INFO(dt->is<PrimitiveType>(),
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  PrimitiveTypeID typeId = dt->as<PrimitiveType>()->type;

  switch (typeId) {
#define PER_C_TYPE(tp, ctype)  \
  case PrimitiveTypeID::tp:    \
    set_arg(arg_id, (ctype)d); \
    break;
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
    case PrimitiveTypeID::f16: {
      uint16 half = fp16_ieee_from_fp32_value((float32)d);
      set_arg(arg_id, half);
      break;
    }
    default:
      TI_NOT_IMPLEMENTED
  }
}

template <typename T>
void LaunchContextBuilder::set_struct_arg(std::vector<int> arg_indices, T d) {
  auto dt = kernel_->args_type->get_element_type(arg_indices);

  TI_ASSERT(dt->is<PrimitiveType>() || dt->is<PointerType>());
  if (dt->is<PointerType>()) {
    set_struct_arg_impl(arg_indices, (uint64)d);
    return;
  }
  PrimitiveTypeID typeId = dt->as<PrimitiveType>()->type;

  switch (typeId) {
#define PER_C_TYPE(tp, ctype)                   \
  case PrimitiveTypeID::tp:                     \
    set_struct_arg_impl(arg_indices, (ctype)d); \
    break;
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
    case PrimitiveTypeID::f16: {
      uint16 half = fp16_ieee_from_fp32_value((float32)d);
      set_struct_arg_impl(arg_indices, half);
      break;
    }
    default:
      TI_NOT_IMPLEMENTED
  }
}

void LaunchContextBuilder::set_ndarray_ptrs(const std::vector<int> &arg_id,
                                            uint64 data_ptr,
                                            uint64 grad_ptr) {
  auto param_indices = arg_id;
  param_indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
  set_struct_arg(param_indices, data_ptr);
  if (kernel_->nested_parameters[arg_id].needs_grad) {
    param_indices.back() = TypeFactory::GRAD_PTR_POS_IN_NDARRAY;
    set_struct_arg(param_indices, grad_ptr);
  }
}

void LaunchContextBuilder::set_argpack_ptr(const std::vector<int> &arg_id,
                                           uint64 data_ptr) {
  auto param_indices = arg_id;
  param_indices.push_back(TypeFactory::DATA_PTR_POS_IN_ARGPACK);
  set_struct_arg(param_indices, data_ptr);
}

template void LaunchContextBuilder::set_struct_arg(std::vector<int> arg_indices,
                                                   uint64 v);
template void LaunchContextBuilder::set_struct_arg(std::vector<int> arg_indices,
                                                   int64 v);
template void LaunchContextBuilder::set_struct_arg(std::vector<int> arg_indices,
                                                   float64 v);

void LaunchContextBuilder::set_arg_int(const std::vector<int> &arg_id,
                                       int64 d) {
  auto dt = kernel_->args_type->get_element_type(arg_id);

  TI_ASSERT_INFO(dt->is<PrimitiveType>(),
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    set_arg(arg_id, (int32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    set_arg(arg_id, (int64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    set_arg(arg_id, (int8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    set_arg(arg_id, (int16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
    set_arg(arg_id, (uint1)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    set_arg(arg_id, (uint8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    set_arg(arg_id, (uint16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    set_arg(arg_id, (uint32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    set_arg(arg_id, (uint64)d);
  } else {
    TI_INFO(dt->to_string());
    TI_NOT_IMPLEMENTED
  }
}

void LaunchContextBuilder::set_arg_uint(const std::vector<int> &arg_id,
                                        uint64 d) {
  set_arg_int(arg_id, d);
}

template <>
void LaunchContextBuilder::set_arg<TypedConstant>(const std::vector<int> &i,
                                                  TypedConstant d) {
  if (is_real(d.dt)) {
    set_arg_float(i, d.val_float());
  } else {
    if (is_signed(d.dt)) {
      set_arg_int(i, d.val_int());
    } else {
      set_arg_uint(i, d.val_uint());
    }
  }
}

template <typename T>
void LaunchContextBuilder::set_struct_arg_impl(std::vector<int> arg_indices,
                                               T v) {
  int offset = args_type->get_element_offset(arg_indices);
  TI_ASSERT(offset + sizeof(T) <= arg_buffer_size);
  *(T *)(ctx_->arg_buffer + offset) = v;
}

template <typename T>
T LaunchContextBuilder::get_arg(const std::vector<int> &i) {
  return get_struct_arg<T>(i);
}

template <typename T>
T LaunchContextBuilder::get_struct_arg(std::vector<int> arg_indices) {
  int offset = args_type->get_element_offset(arg_indices);
  TI_ASSERT(offset + sizeof(T) <= arg_buffer_size);
  return *(T *)(ctx_->arg_buffer + offset);
}

template <typename T>
void LaunchContextBuilder::set_arg(const std::vector<int> &i, T v) {
  set_struct_arg_impl(i, v);
  set_array_device_allocation_type(i, DevAllocType::kNone);
}

template <typename T>
T LaunchContextBuilder::get_ret(int i) {
  return taichi_union_cast_with_different_sizes<T>(ctx_->result_buffer[i]);
}

#define PER_C_TYPE(type, ctype)                                            \
  template void LaunchContextBuilder::set_struct_arg_impl(                 \
      std::vector<int> arg_indices, ctype v);                              \
  template ctype LaunchContextBuilder::get_arg(const std::vector<int> &i); \
  template ctype LaunchContextBuilder::get_struct_arg(                     \
      std::vector<int> arg_indices);                                       \
  template void LaunchContextBuilder::set_arg(const std::vector<int> &i,   \
                                              ctype v);                    \
  template ctype LaunchContextBuilder::get_ret(int i);
#include "taichi/inc/data_type_with_c_type.inc.h"
PER_C_TYPE(gen, void *)  // Register void* as a valid type
#undef PER_C_TYPE

void LaunchContextBuilder::set_array_runtime_size(const std::vector<int> &i,
                                                  uint64 size) {
  array_runtime_sizes[i] = size;
}

void LaunchContextBuilder::set_array_device_allocation_type(
    const std::vector<int> &i,
    DevAllocType usage) {
  device_allocation_type[i] = usage;
}

void LaunchContextBuilder::set_array_shape_and_strides(
    const std::vector<int> &arg_id,
    const std::vector<std::int64_t> &shape,
    const std::vector<std::int64_t> *strides_bytes,
    bool affine_mode) {
  const auto parameter = kernel_->nested_parameters.find(arg_id);
  TI_ASSERT(parameter != kernel_->nested_parameters.end());
  const std::size_t index_rank =
      parameter->second.total_dim - parameter->second.element_shape.size();
  TI_ERROR_IF(shape.size() != index_rank,
              "Array shape rank does not match the kernel argument");
  TI_ERROR_IF(strides_bytes != nullptr && strides_bytes->size() != index_rank,
              "Array stride rank does not match the kernel argument");

  std::vector<std::int64_t> canonical_strides(index_rank);
  if (strides_bytes == nullptr) {
    const auto *array_type =
        args_type->get_element_type(arg_id)->as<StructType>();
    const auto scalar_type =
        array_type->get_element_type({TypeFactory::DATA_PTR_POS_IN_NDARRAY})
            ->as<PointerType>()
            ->get_pointee_type();
    std::int64_t stride = data_type_size(scalar_type);
    for (int extent : parameter->second.element_shape) {
      TI_ERROR_IF(extent < 0 ||
                      (extent > 0 &&
                       stride > (std::numeric_limits<std::int32_t>::max)() /
                                    extent),
                  "Array element shape exceeds the int32 indexing ABI");
      stride *= extent;
    }
    for (std::size_t reverse = 0; reverse < index_rank; ++reverse) {
      const std::size_t axis = index_rank - reverse - 1;
      canonical_strides[axis] = stride;
      TI_ERROR_IF(shape[axis] < 0 ||
                      (shape[axis] > 0 &&
                       stride > (std::numeric_limits<std::int32_t>::max)() /
                                    shape[axis]),
                  "Array shape exceeds the int32 indexing ABI");
      stride *= shape[axis];
    }
    strides_bytes = &canonical_strides;
  }

  for (std::size_t axis = 0; axis < index_rank; ++axis) {
    TI_ERROR_IF(shape[axis] < 0 ||
                    shape[axis] > (std::numeric_limits<std::int32_t>::max)(),
                "Array shape exceeds the int32 indexing ABI");
    const std::int64_t stride = (*strides_bytes)[axis];
    TI_ERROR_IF(stride < 0 ||
                    stride > (std::numeric_limits<std::int32_t>::max)(),
                "Array byte stride exceeds the positive int32 affine ABI");
    set_struct_arg(concatenate_vector<int>(arg_id, {0, (int32)axis}),
                   static_cast<int32>(shape[axis]));
    set_struct_arg(
        concatenate_vector<int>(
            arg_id,
            {0, TypeFactory::stride_pos_in_ndarray(
                    static_cast<int>(index_rank), static_cast<int>(axis))}),
        static_cast<int32>(stride));
  }
  set_struct_arg(
      concatenate_vector<int>(
          arg_id,
          {0, TypeFactory::affine_mode_pos_in_ndarray(
                  static_cast<int>(index_rank))}),
      static_cast<int32>(affine_mode));
}
void LaunchContextBuilder::set_arg_external_array_with_shape(
    const std::vector<int> &arg_id,
    uintptr_t ptr,
    uint64 size,
    const std::vector<int64> &shape,
    uintptr_t grad_ptr) {
  TI_ASSERT_INFO(
      kernel_->nested_parameters[arg_id].is_array,
      "Assigning external (numpy) array to scalar argument is not allowed.");

  TI_ASSERT_INFO(shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  array_ptrs[concatenate_vector<int>(
      arg_id, {TypeFactory::DATA_PTR_POS_IN_NDARRAY})] = (void *)ptr;
  array_ptrs[concatenate_vector<int>(
      arg_id, {TypeFactory::GRAD_PTR_POS_IN_NDARRAY})] = (void *)grad_ptr;
  set_array_runtime_size(arg_id, size);
  set_array_device_allocation_type(arg_id, DevAllocType::kNone);
  set_array_shape_and_strides(arg_id, shape, nullptr, false);
}

void LaunchContextBuilder::set_arg_ndarray(const std::vector<int> &arg_id,
                                           const Ndarray &arr) {
  Program *owner = arr.owning_program();
  const bool registry_owned = owner != nullptr;
  if (registry_owned) {
    NdarrayResourceRef ref;
    ref.arg_offset = args_type->get_element_offset(arg_id);
    ref.owner = owner;
    ref.data = &arr;
    ref.data_handle = arr.runtime_resource_handle();
    TI_ERROR_IF(!ref.data_handle,
                "Cannot bind an unregistered Ndarray runtime resource");
    bind_ndarray_resource_ref(std::move(ref));
  }
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  // Store the address as an opaque context value, but never dereference it
  // before Program validates the captured generation and owns the view. This
  // avoids rebuilding and rehashing the same array_ptrs keys at submission.
  set_arg_ndarray_impl(arg_id, arr.get_device_allocation_ptr_as_int(), arr.shape);
}

void LaunchContextBuilder::set_arg_argpack(const std::vector<int> &arg_id,
                                           const ArgPack &argpack) {
  Program *owner = argpack.owning_program();
  TI_ERROR_IF(owner == nullptr,
              "Program-owned ArgPack is missing its owning Program");
  auto submission_guard = owner->acquire_runtime_resource_submission_guard();
  argpack_ptrs[arg_id] = &argpack;
  argpack_resource_handles[arg_id] =
      owner->capture_argpack_resource_handle(&argpack);
  if (arg_id.size() == 1) {
    // Program resolves and leases this non-owning view immediately before the
    // backend first dereferences it. Keep only a placeholder here so a retire
    // racing the gap between context construction and launch cannot turn this
    // method into a use-after-free.
    set_argpack_ptr(arg_id, 0);
  }
  // TODO: Consider renaming this method to `set_device_allocation_type`
  set_array_device_allocation_type(arg_id, DevAllocType::kArgPack);
}

void LaunchContextBuilder::debug_set_argpack_resource_handle(
    const std::vector<int> &arg_id,
    RuntimeResourceHandle handle) {
  const auto found = argpack_resource_handles.find(arg_id);
  TI_ERROR_IF(found == argpack_resource_handles.end(),
              "Cannot override an unbound ArgPack resource handle");
  found->second = handle;
}

void LaunchContextBuilder::set_arg_ndarray_with_grad(
    const std::vector<int> &arg_id,
    const Ndarray &arr,
    const Ndarray &arr_grad) {
  Program *owner = arr.owning_program();
  Program *grad_owner = arr_grad.owning_program();
  const bool registry_owned = owner != nullptr;
  const bool grad_registry_owned = grad_owner != nullptr;
  TI_ERROR_IF(registry_owned != grad_registry_owned,
              "Ndarray primal and grad must share the same ownership model");
  TI_ERROR_IF(registry_owned && owner != grad_owner,
              "Ndarray primal and grad must belong to the same Program");
  if (registry_owned) {
    NdarrayResourceRef ref;
    ref.arg_offset = args_type->get_element_offset(arg_id);
    ref.owner = owner;
    ref.data = &arr;
    ref.grad = &arr_grad;
    ref.data_handle = arr.runtime_resource_handle();
    ref.grad_handle = arr_grad.runtime_resource_handle();
    TI_ERROR_IF(!ref.data_handle || !ref.grad_handle,
                "Cannot bind an unregistered Ndarray runtime resource");
    bind_ndarray_resource_ref(std::move(ref));
  }
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  set_arg_ndarray_impl(arg_id, arr.get_device_allocation_ptr_as_int(), arr.shape,
                       arr_grad.get_device_allocation_ptr_as_int());
}

void LaunchContextBuilder::set_arg_dense_storage(
    const std::vector<int> &arg_id,
    const storage::DenseStorageDescriptor &descriptor) {
  TI_ASSERT_INFO(
      kernel_->nested_parameters[arg_id].is_array,
      "Assigning a dense storage view to a scalar argument is not allowed.");
  const auto &properties = descriptor.properties();
  const bool safe_positive_affine =
      !properties.has_negative_stride && properties.element_contiguous &&
      properties.uniqueness ==
          storage::StorageMappingUniqueness::kProvenUnique;
  TI_ERROR_IF(!properties.ndarray_abi_compatible && !safe_positive_affine,
              "Dense storage kernel binding requires a unique, positive, "
              "element-contiguous affine layout");
  TI_ERROR_IF(descriptor.access() != storage::StorageAccess::kReadWrite,
              "Dense storage kernel binding currently requires read-write "
              "access");

  const int arg_offset = args_type->get_element_offset(arg_id);
  const auto found = std::find_if(
      dense_storage_ptrs.begin(), dense_storage_ptrs.end(),
      [arg_offset](const auto &ref) { return ref.arg_offset == arg_offset; });
  if (found == dense_storage_ptrs.end()) {
    dense_storage_ptrs.push_back(
        DenseStorageResourceRef{arg_offset, &descriptor, nullptr, {}});
  } else {
    auto &ref = *found;
    ref.descriptor = &descriptor;
    ref.runtime_argument = nullptr;
    ref.resolved = {};
  }

  const std::size_t index_rank = descriptor.index_rank();
  TI_ASSERT_INFO(index_rank <= taichi_max_num_indices,
                 "Dense storage view cannot have too many indices");
  const auto shape = descriptor.index_shape();
  const auto strides = descriptor.index_strides_bytes();
  set_array_shape_and_strides(arg_id, shape, &strides,
                              !properties.ndarray_abi_compatible);
  set_array_runtime_size(arg_id, properties.item_count);
  set_array_device_allocation_type(arg_id, DevAllocType::kDenseStorage);
}

void LaunchContextBuilder::set_arg_runtime_storage(
    const std::vector<int> &arg_id,
    const storage::RuntimeStorageArgument &argument) {
  const auto &qualification = argument.qualification();
  TI_ERROR_IF(!qualification.capabilities.bindable ||
                  !qualification.capabilities.zero_copy_qualified,
              "Runtime storage argument is not directly bindable: {}",
              storage::to_string(qualification.reason));
  set_arg_dense_storage(arg_id, argument.descriptor());
  const int arg_offset = args_type->get_element_offset(arg_id);
  const auto found = std::find_if(
      dense_storage_ptrs.begin(), dense_storage_ptrs.end(),
      [arg_offset](const auto &ref) { return ref.arg_offset == arg_offset; });
  TI_ASSERT(found != dense_storage_ptrs.end());
  found->runtime_argument = &argument;
}

void LaunchContextBuilder::set_resolved_dense_storage(
    std::size_t resource_index,
    const storage::ResolvedDenseBinding &binding) {
  TI_ERROR_IF(resource_index >= dense_storage_ptrs.size() || !binding.valid,
              "Cannot install an invalid resolved dense storage binding");
  dense_storage_ptrs[resource_index].resolved = binding;
}

void LaunchContextBuilder::set_arg_resolved_dense_storage(
    const std::vector<int> &arg_id,
    const storage::DenseStorageDescriptor &descriptor,
    const storage::ResolvedDenseBinding &binding) {
  set_arg_dense_storage(arg_id, descriptor);
  const int arg_offset = args_type->get_element_offset(arg_id);
  const auto found = std::find_if(
      dense_storage_ptrs.begin(), dense_storage_ptrs.end(),
      [arg_offset](const auto &ref) { return ref.arg_offset == arg_offset; });
  TI_ASSERT(found != dense_storage_ptrs.end());
  set_resolved_dense_storage(
      static_cast<std::size_t>(found - dense_storage_ptrs.begin()), binding);
}

const storage::ResolvedDenseBinding &
LaunchContextBuilder::get_resolved_dense_storage(
    const std::vector<int> &arg_id) const {
  const int arg_offset = args_type->get_element_offset(arg_id);
  const auto found = std::find_if(
      dense_storage_ptrs.begin(), dense_storage_ptrs.end(),
      [arg_offset](const auto &ref) { return ref.arg_offset == arg_offset; });
  TI_ERROR_IF(found == dense_storage_ptrs.end() || !found->resolved.valid,
              "Dense storage binding was not resolved for this submission");
  return found->resolved;
}

void LaunchContextBuilder::clear_resolved_dense_storage() noexcept {
  for (auto &ref : dense_storage_ptrs) {
    ref.resolved = {};
  }
}

void LaunchContextBuilder::bind_ndarray_resource_ref(
    NdarrayResourceRef ref) {
  const auto found = std::find_if(
      ndarray_ptrs.begin(), ndarray_ptrs.end(), [&](const auto &current) {
        return current.arg_offset == ref.arg_offset;
      });
  if (found == ndarray_ptrs.end()) {
    ndarray_ptrs.push_back(std::move(ref));
  } else {
    *found = std::move(ref);
  }
}

void LaunchContextBuilder::debug_set_ndarray_resource_handle(
    const std::vector<int> &arg_id,
    RuntimeResourceHandle handle) {
  const int arg_offset = args_type->get_element_offset(arg_id);
  const auto found = std::find_if(
      ndarray_ptrs.begin(), ndarray_ptrs.end(), [&](const auto &ref) {
        return ref.arg_offset == arg_offset;
      });
  TI_ERROR_IF(found == ndarray_ptrs.end(),
              "Cannot override an unbound Ndarray resource handle");
  found->data_handle = handle;
}

void LaunchContextBuilder::set_arg_texture(const std::vector<int> &arg_id,
                                           const Texture &tex) {
  if (Program *owner = tex.owning_program()) {
    TextureResourceRef ref;
    ref.arg_offset = args_type->get_element_offset(arg_id);
    ref.owner = owner;
    ref.texture = &tex;
    ref.handle = tex.runtime_resource_handle();
    TI_ERROR_IF(!ref.handle,
                "Cannot bind an unregistered Texture runtime resource");
    texture_ptrs.push_back(std::move(ref));
  }
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  set_arg_texture_impl(arg_id, ptr);
}

void LaunchContextBuilder::set_arg_rw_texture(const std::vector<int> &arg_id,
                                              const Texture &tex) {
  if (Program *owner = tex.owning_program()) {
    TextureResourceRef ref;
    ref.arg_offset = args_type->get_element_offset(arg_id);
    ref.owner = owner;
    ref.texture = &tex;
    ref.handle = tex.runtime_resource_handle();
    TI_ERROR_IF(!ref.handle,
                "Cannot bind an unregistered Texture runtime resource");
    texture_ptrs.push_back(std::move(ref));
  }
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  set_arg_rw_texture_impl(arg_id, ptr, tex.get_size());
}

void LaunchContextBuilder::debug_set_texture_resource_handle(
    const std::vector<int> &arg_id,
    RuntimeResourceHandle handle) {
  const int arg_offset = args_type->get_element_offset(arg_id);
  const auto found = std::find_if(
      texture_ptrs.begin(), texture_ptrs.end(), [&](const auto &ref) {
        return ref.arg_offset == arg_offset;
      });
  TI_ERROR_IF(found == texture_ptrs.end(),
              "Cannot override an unbound Texture resource handle");
  found->handle = handle;
}

RuntimeContext &LaunchContextBuilder::get_context() {
  return *ctx_;
}

void LaunchContextBuilder::set_arg_texture_impl(const std::vector<int> &arg_id,
                                                intptr_t alloc_ptr) {
  array_ptrs[arg_id] = (void *)alloc_ptr;
  set_array_device_allocation_type(arg_id, DevAllocType::kTexture);
}

void LaunchContextBuilder::set_arg_rw_texture_impl(
    const std::vector<int> &arg_id,
    intptr_t alloc_ptr,
    const std::array<int, 3> &shape) {
  array_ptrs[arg_id] = (void *)alloc_ptr;
  set_array_device_allocation_type(arg_id, DevAllocType::kRWTexture);
  TI_ASSERT(shape.size() <= taichi_max_num_indices);
  for (int i = 0; i < shape.size(); ++i) {
    set_struct_arg(concatenate_vector<int>(arg_id, {0, i}), shape[i]);
  }
}

void LaunchContextBuilder::set_arg_ndarray_impl(const std::vector<int> &arg_id,
                                                intptr_t devalloc_ptr,
                                                const std::vector<int> &shape,
                                                intptr_t devalloc_ptr_grad) {
  // Set array ptr
  array_ptrs[concatenate_vector<int>(
      arg_id, {TypeFactory::DATA_PTR_POS_IN_NDARRAY})] = (void *)devalloc_ptr;
  if (devalloc_ptr != 0) {
    array_ptrs[concatenate_vector<int>(
        arg_id, {TypeFactory::GRAD_PTR_POS_IN_NDARRAY})] =
        (void *)devalloc_ptr_grad;
  }
  // Set device allocation type and runtime size
  set_array_device_allocation_type(arg_id, DevAllocType::kNdarray);
  TI_ASSERT(shape.size() <= taichi_max_num_indices);
  size_t total_size = 1;
  std::vector<std::int64_t> runtime_shape;
  runtime_shape.reserve(shape.size());
  for (int extent : shape) {
    runtime_shape.push_back(extent);
    total_size *= extent;
  }
  set_array_shape_and_strides(arg_id, runtime_shape, nullptr, false);
  set_array_runtime_size(arg_id, total_size);
}

void LaunchContextBuilder::set_arg_matrix(int arg_id, const Matrix &matrix) {
  int type_size = data_type_size(matrix.dtype());
  for (uint32_t i = 0; i < matrix.length(); i++) {
    switch (type_size) {
      case 1:
        set_struct_arg_impl({arg_id, (int32)i},
                            taichi_union_cast_with_different_sizes<int8>(
                                reinterpret_cast<uint8_t *>(matrix.data())[i]));
        break;
      case 2:
        set_struct_arg_impl(
            {arg_id, (int32)i},
            taichi_union_cast_with_different_sizes<int16>(
                reinterpret_cast<uint16_t *>(matrix.data())[i]));
        break;
      case 4:
        set_struct_arg_impl(
            {arg_id, (int32)i},
            taichi_union_cast_with_different_sizes<int32>(
                reinterpret_cast<uint32_t *>(matrix.data())[i]));
        break;
      case 8:
        set_struct_arg_impl(
            {arg_id, (int32)i},
            taichi_union_cast_with_different_sizes<int64>(
                reinterpret_cast<uint64_t *>(matrix.data())[i]));
        break;
      default:
        TI_ERROR("Unsupported type size {}", type_size);
    }
  }
}

TypedConstant LaunchContextBuilder::fetch_ret(const std::vector<int> &index) {
  const Type *dt = ret_type_->get_element_type(index);
  int offset = ret_type_->get_element_offset(index);
  return fetch_ret_impl(offset, dt);
}

float64 LaunchContextBuilder::get_struct_ret_float(
    const std::vector<int> &index) {
  return fetch_ret(index).val_float();
}

int64 LaunchContextBuilder::get_struct_ret_int(const std::vector<int> &index) {
  return fetch_ret(index).val_int();
}

uint64 LaunchContextBuilder::get_struct_ret_uint(
    const std::vector<int> &index) {
  return fetch_ret(index).val_uint();
}

TypedConstant LaunchContextBuilder::fetch_ret_impl(int offset, const Type *dt) {
  TI_ASSERT(dt->is<PrimitiveType>());
  auto primitive_type = dt->as<PrimitiveType>();
  char *ptr = result_buffer_.get() + offset;
  switch (primitive_type->type) {
#define PER_C_TYPE(type, ctype) \
  case PrimitiveTypeID::type:   \
    return TypedConstant(*(ctype *)ptr);
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
    case PrimitiveTypeID::f16: {
      // first fetch the data as u16, and then convert it to f32
      uint16 half = *(uint16 *)ptr;
      return TypedConstant(fp16_ieee_to_fp32_value(half));
    }
    default:
      TI_NOT_IMPLEMENTED
  }
}

}  // namespace taichi::lang
