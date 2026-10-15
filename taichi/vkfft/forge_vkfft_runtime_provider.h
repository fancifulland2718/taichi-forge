#pragma once

#include "taichi/external_runtime/forge_runtime_provider.h"
#include "taichi/vkfft/forge_vkfft_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Additive entry point; the original VkFFT ABI1 remains independently usable.
#define TI_FORGE_VKFFT_RUNTIME_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_vkfft_runtime_provider_query"

typedef struct TiForgeVkfftExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  TiForgeVkfftApi provider;
  TiForgeVkfftRecipeApi recipe;
  TiForgeVkfftRecordInlineFn record_inline;
} TiForgeVkfftExecutionApi;

// The session is device-independent. Plan creation receives the caller's
// Vulkan device/queue explicitly, and each plan must retain the session until
// all recorded commands retire. VkFFT and its shader compiler are embedded in
// the optional adapter, so library_path must be null or empty.
#if defined(TI_FORGE_VKFFT_PROVIDER_BUILD)
TI_FORGE_VKFFT_EXPORT TiForgeRuntimeProviderResult
taichi_forge_vkfft_runtime_provider_query(uint32_t requested_abi_version,
                                          size_t api_size,
                                          TiForgeRuntimeProviderApi *out_api);
#endif

#ifdef __cplusplus
}
#endif
