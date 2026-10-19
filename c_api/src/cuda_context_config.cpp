#include "taichi_core_impl.h"
#include "taichi/taichi_cuda.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_retained_memory.h"

#include <cstring>
#include <stdexcept>

void ti_configure_cuda_context_ext(const TiCudaContextConfig *config) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(config);
  using namespace taichi::lang;
  if (config->struct_size != sizeof(TiCudaContextConfig) ||
      config->schema_version != 1 || config->graphics_api != 1 ||
      config->mode < TI_CUDA_CONTEXT_DISABLED ||
      config->mode > TI_CUDA_CONTEXT_AUTO ||
      config->queue_data_size > 1024 * 1024)
    throw std::invalid_argument("Unsupported CUDA context binding descriptor");
  const bool bound = config->queue_data_size != 0;
  if (bound != bool(config->queue_data) || bound != bool(config->owner) ||
      bound != bool(config->retain_owner) ||
      bound != bool(config->release_owner) ||
      (config->mode == TI_CUDA_CONTEXT_REQUIRED && !bound) ||
      (config->mode == TI_CUDA_CONTEXT_DISABLED && bound))
    throw std::invalid_argument(
        "CUDA Vulkan binding requires bytes and a complete native owner lease");
  CudaContextConfig value;
  value.mode = static_cast<CudaContextMode>(config->mode);
  std::memcpy(value.device_uuid.data(), config->device_uuid, 16);
  if (bound) {
    const auto *bytes = static_cast<const std::uint8_t *>(config->queue_data);
    value.queue_data.assign(bytes, bytes + config->queue_data_size);
    if (config->retain_owner(config->owner) != TI_ERROR_SUCCESS)
      throw std::runtime_error("Vulkan external queue owner refused its lease");
    value.owner = std::shared_ptr<void>(config->owner, config->release_owner);
  }
  CUDAContext::configure(std::move(value));
  TI_CAPI_TRY_CATCH_END();
}

void ti_get_cuda_context_info_ext(TiCudaContextInfo *info) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(info);
  if (info->struct_size != sizeof(TiCudaContextInfo) ||
      info->schema_version != 1)
    throw std::invalid_argument("Unsupported CUDA context info descriptor");
  const auto current = taichi::lang::CUDAContext::configuration();
  *info = {};
  info->struct_size = sizeof(*info);
  info->schema_version = 1;
  info->initialized = current.initialized;
  info->vulkan_cig = current.vulkan_cig;
  info->driver_entry = current.driver_entry;
  info->device_capability = current.device_capability;
  info->generation = current.generation;
  std::memcpy(info->device_uuid, current.device_uuid.data(), 16);
  info->allocation_checks = current.allocation_checks;
  info->process_lifetime = true;
  if (current.initialized && current.vulkan_cig) {
    info->allocation_route = 1;
    auto stats =
        taichi::lang::cuda::RetainedMemoryPool::process_pool().snapshot();
    info->native_allocation_calls = stats.at("native_allocation_calls");
    info->native_free_calls = stats.at("native_free_calls");
    info->reuse_calls = stats.at("reuse_calls");
    info->retirement_calls = stats.at("retirement_calls");
    info->reserved_current_bytes = stats.at("reserved_current_bytes");
    info->reserved_high_bytes = stats.at("reserved_high_bytes");
    info->used_current_bytes = stats.at("used_current_bytes");
    info->used_high_bytes = stats.at("used_high_bytes");
  }
  const auto length = std::min(current.fallback_reason.size(),
                               sizeof(info->fallback_reason) - 1);
  std::memcpy(info->fallback_reason, current.fallback_reason.data(), length);
  TI_CAPI_TRY_CATCH_END();
}
