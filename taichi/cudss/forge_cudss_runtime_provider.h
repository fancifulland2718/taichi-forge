#pragma once

#include "taichi/cudss/forge_cudss_provider.h"
#include "taichi/external_runtime/forge_runtime_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Additive common-provider entry point. The published cuDSS ABI1 entry points
// and vendor status codes remain unchanged.
#define TI_FORGE_CUDSS_RUNTIME_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_cudss_runtime_provider_query"

typedef struct TiForgeCudssExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  // These functions accept the same opaque runtime as the common API. Its
  // owner alone calls create_runtime/destroy_runtime; domain plans retain that
  // owner until all contexts, descriptors and captured work have retired.
  TiForgeCudssProviderApi provider;
  // Missing optional vendor symbols leave the respective extension empty.
  TiForgeCudssConfigurationApi configuration;
  TiForgeCudssAllocatorApi allocator;
  TiForgeCudssFactorStatisticsApi factor_statistics;
} TiForgeCudssExecutionApi;

TI_FORGE_CUDSS_EXPORT TiForgeRuntimeProviderResult
taichi_forge_cudss_runtime_provider_query(uint32_t requested_abi_version,
                                          size_t api_size,
                                          TiForgeRuntimeProviderApi *out_api);

#ifdef __cplusplus
}
#endif
