#pragma once
#include <taichi/program/callable.h>
#include "taichi/program/ndarray.h"
#include "taichi/program/argpack.h"
#include "taichi/program/texture.h"
#include "taichi/program/matrix.h"
#include "taichi/program/runtime_resource_registry.h"
#include "taichi/program/storage_view.h"

namespace taichi::lang {

struct RuntimeContext;

class LaunchContextBuilder {
 public:
  enum class DevAllocType : int8_t {
    kNone = 0,
    kNdarray = 1,
    kTexture = 2,
    kRWTexture = 3,
    kArgPack = 4,
    kDenseStorage = 5,
    kAccelerationStructure = 6,
  };

  struct NdarrayResourceRef {
    // Byte offset in the immutable callable argument layout. Unlike copying
    // arg_id, this is allocation-free on every launch and remains an exact
    // identity for the debug generation-fault hook.
    int arg_offset{-1};
    Program *owner{nullptr};
    const Ndarray *data{nullptr};
    RuntimeResourceHandle data_handle;
    const Ndarray *grad{nullptr};
    RuntimeResourceHandle grad_handle;
  };

  struct TextureResourceRef {
    int arg_offset{-1};
    Program *owner{nullptr};
    const Texture *texture{nullptr};
    RuntimeResourceHandle handle;
  };

  struct DenseStorageResourceRef {
    int arg_offset{-1};
    // Non-owning immutable metadata. Ordinary launch and Graph callers keep
    // the source object alive through the complete submission transaction.
    const storage::DenseStorageDescriptor *descriptor{nullptr};
    const storage::RuntimeStorageArgument *runtime_argument{nullptr};
    storage::ResolvedDenseBinding resolved;
  };

  struct AccelerationStructureResourceRef {
    int arg_offset{-1};
    Program *owner{nullptr};
    std::uint64_t handle{0};
  };

  explicit LaunchContextBuilder(CallableBase *kernel,
                                bool cpu_bounded_range = false);

  void set_cpu_bounded_range(void *extent, std::int32_t capacity);
  void set_cuda_bounded_range(const std::vector<int> &extent_arg_id,
                              std::int32_t capacity);

  bool has_cuda_bounded_range() const noexcept {
    return has_cuda_bounded_range_binding_;
  }

  const std::vector<int> &cuda_bounded_extent_arg_id() const noexcept {
    return cuda_bounded_extent_arg_id_;
  }

  std::int32_t cuda_bounded_capacity() const noexcept {
    return cuda_bounded_capacity_;
  }

  LaunchContextBuilder(LaunchContextBuilder &&) = default;
  LaunchContextBuilder &operator=(LaunchContextBuilder &&) = default;
  LaunchContextBuilder(const LaunchContextBuilder &) = delete;
  LaunchContextBuilder &operator=(const LaunchContextBuilder &) = delete;

  void set_arg_float(const std::vector<int> &arg_id, float64 d);

  // Created signed and unsigned version for argument range check of pybind
  void set_arg_int(const std::vector<int> &arg_id, int64 d);
  void set_arg_uint(const std::vector<int> &arg_id, uint64 d);

  void set_array_runtime_size(const std::vector<int> &i, uint64 size);

  void set_array_device_allocation_type(const std::vector<int> &i,
                                        DevAllocType usage);

  template <typename T>
  void set_arg(const std::vector<int> &i, T v);

  // The following two functions can be used to set struct args and primitive
  // args. The first element of `arg_indices` is the index of the argument. The
  // rest of the elements are the index of the field in each depth of the nested
  // struct.

  template <typename T>
  void set_struct_arg_impl(std::vector<int> arg_indices, T v);

  template <typename T>
  void set_struct_arg(std::vector<int> arg_indices, T v);

  void set_ndarray_ptrs(const std::vector<int> &arg_id,
                        uint64 data_ptr,
                        uint64 grad_ptr);
  void set_argpack_ptr(const std::vector<int> &arg_id, uint64 data_ptr);
  void debug_set_argpack_resource_handle(
      const std::vector<int> &arg_id,
      RuntimeResourceHandle handle);

  template <typename T>
  T get_arg(const std::vector<int> &i);

  template <typename T>
  T get_struct_arg(std::vector<int> arg_indices);

  template <typename T>
  T get_ret(int i);

  void set_arg_external_array_with_shape(const std::vector<int> &arg_id,
                                         uintptr_t ptr,
                                         uint64 size,
                                         const std::vector<int64> &shape,
                                         uintptr_t grad_ptr = 0);

  void set_arg_ndarray_impl(const std::vector<int> &arg_id,
                            intptr_t devalloc_ptr,
                            const std::vector<int> &shape,
                            intptr_t devalloc_ptr_grad = 0);
  // Rebinding the same argument slot replaces its generation-qualified
  // resource ref, allowing a launch context to be reused without retaining a
  // stale view from an earlier submission.
  void set_arg_ndarray(const std::vector<int> &arg_id, const Ndarray &arr);
  void set_arg_argpack(const std::vector<int> &arg_id, const ArgPack &argpack);
  void set_arg_ndarray_with_grad(const std::vector<int> &arg_id,
                                 const Ndarray &arr,
                                 const Ndarray &arr_grad);
  void set_arg_dense_storage(
      const std::vector<int> &arg_id,
      const storage::DenseStorageDescriptor &descriptor);
  void set_arg_runtime_storage(const std::vector<int> &arg_id,
                               const storage::RuntimeStorageArgument &argument);
  void set_arg_resolved_dense_storage(
      const std::vector<int> &arg_id,
      const storage::DenseStorageDescriptor &descriptor,
      const storage::ResolvedDenseBinding &binding);
  void set_resolved_dense_storage(
      std::size_t resource_index,
      const storage::ResolvedDenseBinding &binding);
  const storage::ResolvedDenseBinding &get_resolved_dense_storage(
      const std::vector<int> &arg_id) const;
  void clear_resolved_dense_storage() noexcept;
  void debug_set_ndarray_resource_handle(
      const std::vector<int> &arg_id,
      RuntimeResourceHandle handle);
  void debug_set_texture_resource_handle(
      const std::vector<int> &arg_id,
      RuntimeResourceHandle handle);

  void set_arg_texture_impl(const std::vector<int> &arg_id, intptr_t alloc_ptr);
  void set_arg_texture(const std::vector<int> &arg_id, const Texture &tex);
  void set_arg_rw_texture_impl(const std::vector<int> &arg_id,
                               intptr_t alloc_ptr,
                               const std::array<int, 3> &shape);
  void set_arg_rw_texture(const std::vector<int> &arg_id, const Texture &tex);
  void set_arg_acceleration_structure(const std::vector<int> &arg_id,
                                      Program *owner,
                                      std::uint64_t handle);

  void set_arg_matrix(int arg_id, const Matrix &matrix);

  TypedConstant fetch_ret(const std::vector<int> &index);
  float64 get_struct_ret_float(const std::vector<int> &index);
  int64 get_struct_ret_int(const std::vector<int> &index);
  uint64 get_struct_ret_uint(const std::vector<int> &index);

  RuntimeContext &get_context();

  void append_dispatch_label(const std::string &label);

  const std::string &dispatch_label() const noexcept {
    return dispatch_label_;
  }

 private:
  void set_array_shape_and_strides(
      const std::vector<int> &arg_id,
      const std::vector<std::int64_t> &shape,
      const std::vector<std::int64_t> *strides_bytes,
      bool affine_mode);
  void bind_ndarray_resource_ref(NdarrayResourceRef ref);
  TypedConstant fetch_ret_impl(int offset, const Type *dt);
  CallableBase *kernel_;
  std::unique_ptr<RuntimeContext> owned_ctx_;
  // |ctx_| *almost* always points to |owned_ctx_|. However, it is possible
  // that the caller passes a RuntimeContext pointer externally. In that case,
  // |owned_ctx_| will be nullptr.
  // Invariant: |ctx_| will never be nullptr.
  RuntimeContext *ctx_;
  std::unique_ptr<char[]> arg_buffer_;
  std::unique_ptr<char[]> result_buffer_;
  const StructType *ret_type_;
  bool has_cpu_bounded_range_binding_{false};
  bool has_cuda_bounded_range_binding_{false};
  std::vector<int> cuda_bounded_extent_arg_id_;
  std::int32_t cuda_bounded_capacity_{0};
  std::string dispatch_label_;

 public:
  size_t arg_buffer_size{0};
  const StructType *args_type{nullptr};
  size_t result_buffer_size{0};

  // Note that I've tried to group `array_runtime_size` and
  // `is_device_allocations` into a small struct. However, it caused some test
  // cases to stuck.

  // `array_runtime_size` records the runtime size of the
  // corresponding array arguments.
  std::
      unordered_map<std::vector<int>, uint64, hashing::Hasher<std::vector<int>>>
          array_runtime_sizes;
  // `device_allocation_type` is set iff i-th arg is a `DeviceAllocation*`,
  // otherwise it is set to DevAllocType::kNone
  std::unordered_map<std::vector<int>,
                     DevAllocType,
                     hashing::Hasher<std::vector<int>>>
      device_allocation_type;

  std::
      unordered_map<std::vector<int>, void *, hashing::Hasher<std::vector<int>>>
          array_ptrs;
  std::unordered_map<std::vector<int>,
                     const ArgPack *,
                     hashing::Hasher<std::vector<int>>>
      argpack_ptrs;
  std::unordered_map<std::vector<int>,
                     RuntimeResourceHandle,
                     hashing::Hasher<std::vector<int>>>
      argpack_resource_handles;
  // Non-owning high-level views. Program resolves these through its
  // generation-qualified runtime-resource registry immediately before the
  // backend first dereferences the DeviceAllocation placeholders.
  // Runtime contexts overwhelmingly bind one or a handful of ndarrays. A
  // flat vector avoids unordered_map node/bucket/hash work on every launch;
  // arg_id is retained only for the generation-fault injection debug hook.
  std::vector<NdarrayResourceRef> ndarray_ptrs;
  std::vector<TextureResourceRef> texture_ptrs;
  std::vector<DenseStorageResourceRef> dense_storage_ptrs;
  std::vector<AccelerationStructureResourceRef> acceleration_structure_ptrs;
};

}  // namespace taichi::lang
