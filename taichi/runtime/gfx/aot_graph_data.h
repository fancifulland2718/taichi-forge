#pragma once
#include <exception>
#include <functional>
#include <mutex>

#include "taichi/runtime/gfx/runtime.h"

namespace taichi::lang {
namespace gfx {
// Public kernel aliases share one physical registration. The module can
// retire a used handle without materializing an unused one on destruction.
struct AotKernelRegistration {
  // Only split, owned Vulkan artifacts need assembly. Alias lookups and
  // construction of a LaunchContext can use the callable ABI immediately.
  std::function<void()> load_code;
  // Portable LLVM has a callable ABI before any target tasks exist. Its
  // materializer supplies the complete target registration on demand.
  std::unique_ptr<GfxRuntime::RegisterParams> compiled_params;

  void materialize_code() {
    if (load_code)
      std::call_once(code_once, load_code);
  }

  GfxRuntime::KernelHandle get(GfxRuntime &runtime,
                               const GfxRuntime::RegisterParams &params) {
    std::call_once(once, [&] {
      try {
        materialize_code();
        handle = runtime.register_taichi_kernel(compiled_params ? *compiled_params : params);
      } catch (...) {
        failure = std::current_exception();
      }
    });
    if (failure)
      std::rethrow_exception(failure);
    return handle;
  }

  GfxRuntime::KernelHandle handle;

 private:
  std::once_flag code_once;
  std::once_flag once;
  std::exception_ptr failure;
};

class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(GfxRuntime *runtime, GfxRuntime::RegisterParams &&params,
                      const GfxRuntime::KernelHandle *shared_handle = nullptr,
                      std::vector<uint32_t> device_range_tasks = {},
                      std::shared_ptr<AotKernelRegistration> registration = {})
      : runtime_(runtime), params_(std::move(params)),
        device_range_tasks_(std::move(device_range_tasks)),
        registration_(std::move(registration)) {
    if (!registration_)
      handle_ = shared_handle ? *shared_handle
                              : runtime_->register_taichi_kernel(params_);
    ret_type = params_.kernel_attribs.ctx_attribs.rets_type();
    ret_size = params_.kernel_attribs.ctx_attribs.rets_bytes();
    args_type = params_.kernel_attribs.ctx_attribs.args_type();
    args_size = params_.kernel_attribs.ctx_attribs.args_bytes();
    runtime_affine_arrays = params_.kernel_attribs.runtime_affine_arrays;
    for (const auto &[indices, attrib] :
         params_.kernel_attribs.ctx_attribs.args()) {
      CallableBase::Parameter parameter(
          PrimitiveType::get(attrib.dtype), attrib.is_array,
          attrib.is_argpack, /*size_unused=*/0,
          static_cast<int>(attrib.field_dim + attrib.element_shape.size()),
          attrib.element_shape, attrib.format);
      parameter.name = attrib.name;
      parameter.ptype = attrib.ptype;
      parameter.texture_lod = attrib.texture_lod;
      if (attrib.ptype == ParameterType::kNdarray) {
        // The serialized ABI already records the optional grad_ptr member.
        // Recover it without changing the artifact schema or guessing from
        // gradient accesses (forward kernels may not access grad_ptr at all).
        auto *array_type = args_type->get_element_type(indices)
                               ->cast<taichi::lang::StructType>();
        TI_ASSERT(array_type != nullptr);
        parameter.needs_grad = array_type->elements().size() >
                               TypeFactory::GRAD_PTR_POS_IN_NDARRAY;
        auto payload = DataType(array_type->get_element_type(
            {TypeFactory::DATA_PTR_POS_IN_NDARRAY})).ptr_removed();
        if (payload->is<taichi::lang::StructType>()) {
          parameter = CallableBase::Parameter(
              array_type, attrib.is_array, attrib.is_argpack, 0,
              static_cast<int>(attrib.field_dim), {}, attrib.format,
              parameter.needs_grad);
          parameter.name = attrib.name;
          parameter.ptype = attrib.ptype;
        }
      }
      nested_parameters.emplace(indices, std::move(parameter));
    }
    for (const auto &argument : runtime_affine_arrays) {
      const auto found = nested_parameters.find(argument);
      TI_ERROR_IF(found == nested_parameters.end() ||
                      found->second.ptype != ParameterType::kNdarray,
                  "Invalid GFX runtime affine argument declaration");
    }
    for (const auto &[indices, type] :
         params_.kernel_attribs.ctx_attribs.argpack_types()) {
      TI_ASSERT(type != nullptr && type->is<taichi::lang::StructType>());
      argpack_types.emplace(indices,
                            type->as<taichi::lang::StructType>());
    }
    name = params_.kernel_attribs.name;
    arch = Arch::vulkan;  // Only for letting the launch context builder know
                          // the arch does not use LLVM.
                          // TODO: remove arch after the refactoring of
                          //  SPIR-V based backends completes.
  }

  void launch(LaunchContextBuilder &ctx) override {
    TI_ERROR_IF(!device_range_tasks_.empty(), "Device range dispatch requires Graph recording");
    runtime_->launch_kernel(handle(), ctx);
  }

  const GfxRuntime::RegisterParams &params() {
    if (registration_) {
      registration_->materialize_code();
      if (registration_->compiled_params)
        return *registration_->compiled_params;
    }
    return params_;
  }

  const std::vector<uint32_t> &device_range_tasks() const { return device_range_tasks_; }

  GfxRuntime::KernelHandle handle() const {
    return registration_ ? registration_->get(*runtime_, params_) : handle_;
  }

  GfxRuntime &runtime() const {
    return *runtime_;
  }

 private:
  GfxRuntime *const runtime_;
  GfxRuntime::KernelHandle handle_;
  const GfxRuntime::RegisterParams params_;
  const std::vector<uint32_t> device_range_tasks_;
  const std::shared_ptr<AotKernelRegistration> registration_;
};
}  // namespace gfx
}  // namespace taichi::lang
