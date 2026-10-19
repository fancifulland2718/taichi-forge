#ifdef TI_WITH_LLVM

#include "taichi_core_impl.h"
#include "taichi_llvm_impl.h"

#include "taichi/taichi_cpu.h"
#include "taichi/taichi_cuda.h"

#include "taichi/program/compile_config.h"
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/runtime/cpu/kernel_launcher.h"
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/runtime/native_operation_registry.h"

#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif

namespace capi {

LlvmRuntime::LlvmRuntime(taichi::Arch arch) : Runtime(arch) {
  cfg_ = std::make_unique<taichi::lang::CompileConfig>();
  cfg_->arch = arch;

  owned_executor_ =
      std::make_unique<taichi::lang::LlvmRuntimeExecutor>(*cfg_.get(), nullptr);
  executor_ = owned_executor_.get();

  // materialize_runtime() takes in a uint64_t** (pointer object's address) and
  // modifies the address it points to.
  //
  // Therefore we can't use host_result_buffer_.data() here,
  // since it returns a temporary copy of the internal data pointer,
  // thus we won't be able to modify the address where the std::array's data
  // pointer is pointing to.
  executor_->materialize_runtime(nullptr /*kNoProfiler*/, &result_buffer);
}

LlvmRuntime::LlvmRuntime(taichi::lang::LlvmRuntimeExecutor &executor,
                         taichi::uint64 *buffer)
    : Runtime(executor.get_config().arch),
      executor_(&executor),
      result_buffer(buffer) {
}

LlvmRuntime::~LlvmRuntime() {
  try {
    retire_execution_domains();
  } catch (...) {
    // Failed drain cannot prove that domains stopped accessing the executor.
    // Preserve its finite allocation and referenced configuration on failure.
    (void)owned_executor_.release();
    (void)cfg_.release();
    return;
  }
  owned_executor_.reset();
  cfg_.reset();
}

void LlvmRuntime::check_runtime_error() {
  executor_->check_runtime_error(this->result_buffer);
}

taichi::lang::Device &LlvmRuntime::get() {
  taichi::lang::Device *device = executor_->get_compute_device();
  return *device;
}

void LlvmRuntime::prepare_cpu_scheduler(
    taichi::lang::NativePreparationContext &context) {
  if (!taichi::arch_is_cpu(arch))
    return;
  context.cpu_thread_pool = executor_->get_cpu_thread_pool();
  context.cpu_max_threads =
      std::max(1, executor_->get_config().cpu_max_num_threads);
}

TiMemory LlvmRuntime::allocate_memory(
    const taichi::lang::Device::AllocParams &params) {
  taichi::lang::LLVMRuntime *llvm_runtime = executor_->get_llvm_runtime();
  taichi::lang::LlvmDevice *llvm_device = executor_->llvm_device();
  taichi::lang::DeviceAllocation devalloc =
      llvm_device->allocate_memory_runtime(
          {params, executor_->get_runtime_jit_module(), llvm_runtime,
           result_buffer, executor_->use_device_memory_pool()});

  return devalloc2devmem(*this, devalloc, params.size);
}

void LlvmRuntime::free_memory(TiMemory devmem) {
  Runtime::free_memory(devmem);
}

TiAotModule LlvmRuntime::load_aot_module(const char *module_path) {
  const auto &config = executor_->get_config();
  std::unique_ptr<taichi::lang::aot::Module> aot_module{nullptr};

  if (taichi::arch_is_cpu(config.arch)) {
    taichi::lang::cpu::KernelLauncher::Config cfg;
    cfg.executor = executor_;
    taichi::lang::LLVM::AotModuleParams aot_params;
    aot_params.executor_ = executor_;
    aot_params.kernel_launcher =
        std::make_unique<taichi::lang::cpu::KernelLauncher>(std::move(cfg));
    aot_params.module_path = module_path;
    aot_module = taichi::lang::LLVM::make_aot_module(std::move(aot_params));
  } else {
#ifdef TI_WITH_CUDA
    TI_ASSERT(config.arch == taichi::Arch::cuda);
    taichi::lang::cuda::KernelLauncher::Config cfg;
    cfg.executor = executor_;
    taichi::lang::LLVM::AotModuleParams aot_params;
    aot_params.executor_ = executor_;
    aot_params.kernel_launcher =
        std::make_unique<taichi::lang::cuda::KernelLauncher>(std::move(cfg));
    aot_params.module_path = module_path;
    aot_module = taichi::lang::LLVM::make_aot_module(std::move(aot_params));
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  /* TODO(zhanlue): expose allocate/deallocate_snode_tree_type() to C-API
     Let's initialize SNodeTrees automatically for now since SNodeTreeType isn't
     ready yet.
  */
  auto *llvm_aot_module =
      dynamic_cast<taichi::lang::LLVM::LlvmAotModule *>(aot_module.get());
  TI_ASSERT(llvm_aot_module != nullptr);
  const auto tree_ids = llvm_aot_module->get_snode_tree_ids();
  // The module has already validated and reserved its resource mapping.
  for (int tree_id : tree_ids) {
    auto *snode_tree = aot_module->get_snode_tree(std::to_string(tree_id));
    taichi::lang::LLVM::allocate_aot_snode_tree_type(
        aot_module.get(), snode_tree, this->result_buffer);
  }

  return (TiAotModule)(new AotModule(*this, std::move(aot_module)));
}

void LlvmRuntime::buffer_copy(const taichi::lang::DevicePtr &dst,
                              const taichi::lang::DevicePtr &src,
                              size_t size) {
  get().memcpy_internal(dst, src, size);
}

void LlvmRuntime::flush() {
  // (penguinliong) Flush in LLVM backends is a nop atm.
  // TI_NOT_IMPLEMENTED;
}

void LlvmRuntime::wait() {
  try {
    synchronize_execution_domains();
    check_runtime_error();
  } catch (const taichi::lang::BackendRuntimeError &error) {
    runtime_fault_domain()->report_backend_error(error, 0);
    throw;
  }
}

}  // namespace capi

// function.export_cpu_runtime
void ti_export_cpu_memory(TiRuntime runtime,
                          TiMemory memory,
                          TiCpuMemoryInteropInfo *interop_info) {
  TI_CAPI_TRY_CATCH_BEGIN();
#ifdef TI_WITH_LLVM
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(memory);
  TI_CAPI_ARGUMENT_NULL(interop_info);
  if (((Runtime *)runtime)->arch != taichi::Arch::x64 &&
      ((Runtime *)runtime)->arch != taichi::Arch::arm64) {
    ti_set_last_error(TI_ERROR_INVALID_INTEROP, "arch!= cpu");
    return;
  }

  capi::LlvmRuntime *llvm_runtime =
      static_cast<capi::LlvmRuntime *>((Runtime *)runtime);
  const auto range = llvm_runtime->get_memory(memory);

  auto &device = llvm_runtime->get();
  auto &cpu_device = static_cast<taichi::lang::cpu::CpuDevice &>(device);

  auto cpu_info = cpu_device.get_alloc_info(range.allocation);

  interop_info->ptr = static_cast<char *>(cpu_info.ptr) + range.byte_offset;
  interop_info->size = range.size;
#else
  TI_NOT_IMPLEMENTED;
#endif  // TI_WITH_LLVM
  TI_CAPI_TRY_CATCH_END();
}

// function.import_cpu_runtime
TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_cpu_memory(TiRuntime runtime,
                                                        void *ptr,
                                                        size_t memory_size) {
  TiMemory out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
#ifdef TI_WITH_LLVM
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(ptr);
  if (((Runtime *)runtime)->arch != taichi::Arch::x64 &&
      ((Runtime *)runtime)->arch != taichi::Arch::arm64) {
    ti_set_last_error(TI_ERROR_INVALID_INTEROP, "arch!= cpu");
    return TI_NULL_HANDLE;
  }
  capi::LlvmRuntime *llvm_runtime =
      static_cast<capi::LlvmRuntime *>((Runtime *)runtime);

  auto &device = llvm_runtime->get();
  auto &cpu_device = static_cast<taichi::lang::cpu::CpuDevice &>(device);

  taichi::lang::DeviceAllocation device_alloc =
      cpu_device.import_memory(ptr, memory_size);

  // prepare memory object
  out = devalloc2devmem(*llvm_runtime, device_alloc, memory_size, true,
                        reinterpret_cast<uintptr_t>(ptr));
#else
  TI_NOT_IMPLEMENTED;
#endif  // TI_WITH_LLVM
  TI_CAPI_TRY_CATCH_END();
  return out;
}

// function.export_cuda_runtime
void ti_export_cuda_memory(TiRuntime runtime,
                           TiMemory memory,
                           TiCudaMemoryInteropInfo *interop_info) {
  TI_CAPI_TRY_CATCH_BEGIN();
#ifdef TI_WITH_CUDA
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(memory);
  TI_CAPI_ARGUMENT_NULL(interop_info);
  if (((Runtime *)runtime)->arch != taichi::Arch::cuda) {
    ti_set_last_error(TI_ERROR_INVALID_INTEROP, "arch!= cuda");
    return;
  }

  auto *llvm_runtime = static_cast<capi::LlvmRuntime *>((Runtime *)runtime);
  const auto range = llvm_runtime->get_memory(memory);

  auto &device = llvm_runtime->get();
  auto &cuda_device = static_cast<taichi::lang::cuda::CudaDevice &>(device);

  auto cuda_info = cuda_device.get_alloc_info(range.allocation);

  interop_info->ptr = static_cast<char *>(cuda_info.ptr) + range.byte_offset;
  interop_info->size = range.size;
#else
  TI_NOT_IMPLEMENTED;
#endif
  TI_CAPI_TRY_CATCH_END();
}

// function.import_cuda_runtime
TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_cuda_memory(TiRuntime runtime,
                                                         void *ptr,
                                                         size_t memory_size) {
  TiMemory out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
#ifdef TI_WITH_CUDA
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(ptr);
  TI_CAPI_INVALID_INTEROP_ARCH_RV(((Runtime *)runtime)->arch, cuda);
  capi::LlvmRuntime *llvm_runtime =
      static_cast<capi::LlvmRuntime *>((Runtime *)runtime);

  auto &device = llvm_runtime->get();
  auto &cuda_device = static_cast<taichi::lang::cuda::CudaDevice &>(device);

  taichi::lang::DeviceAllocation device_alloc =
      cuda_device.import_memory(ptr, memory_size);

  // prepare memory object
  out = devalloc2devmem(*llvm_runtime, device_alloc, memory_size, true,
                        reinterpret_cast<uintptr_t>(ptr));
#else
  TI_NOT_IMPLEMENTED;
#endif
  TI_CAPI_TRY_CATCH_END();
  return out;
}

// function.set_cuda_stream
TI_DLL_EXPORT void TI_API_CALL ti_set_cuda_stream(void *stream) {
#ifdef TI_WITH_CUDA
  // Ordinary kernels and Graph replay share the legacy default stream.
  // Accepting another stream here would falsely promise interop ordering.
  if (stream != nullptr) {
    ti_set_last_error(TI_ERROR_NOT_SUPPORTED,
                      "CUDA runtime uses the legacy default stream");
  }

#else
  TI_NOT_IMPLEMENTED;
#endif
}

// function.get_cuda_stream
TI_DLL_EXPORT void TI_API_CALL ti_get_cuda_stream(void **stream) {
#ifdef TI_WITH_CUDA
  TI_CAPI_ARGUMENT_NULL(stream);
  *stream = nullptr;
#else
  TI_NOT_IMPLEMENTED;

#endif
}

#endif  // TI_WITH_LLVM
