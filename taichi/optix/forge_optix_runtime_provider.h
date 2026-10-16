#pragma once

// SDK components install the public ABI headers together. Source builds keep
// the common ABI in external_runtime; neither layout needs vendor SDK headers.
#if __has_include("forge_runtime_provider.h")
#include "forge_runtime_provider.h"
#else
#include "../external_runtime/forge_runtime_provider.h"
#endif
#include "forge_optix_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_OPTIX_RUNTIME_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_optix_runtime_provider_query"
#define TI_FORGE_OPTIX_RUNTIME_VERSION_FAMILY "8.1,9.0,9.1"

// Common sessions retain the existing adapter's vendor-library lease. They do
// not create CUDA/OptiX contexts, pipelines or scenes. Each domain resource must
// retain its session until its final device work and destruction have completed.
// The embedded table keeps the original OptiX ABI and result codes; it must not
// be checked as a TiForgeRuntimeProviderResult.
typedef struct TiForgeOptixExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  TiForgeOptixProviderApi provider;
} TiForgeOptixExecutionApi;

// The family describes the bundled adapter set. provider.info identifies the
// specific SDK/OptiX ABI of this adapter. Runtime version fields are zero
// (unknown), not the SDK version: the driver negotiates its function table but
// does not expose a runtime release through this interface. library_path is the
// requested path, with an empty string denoting platform-default discovery.
// Probe info strings last until the next probe on this thread; creation info
// strings last until runtime destruction. Copy facts when retaining them longer.
TI_FORGE_OPTIX_EXPORT TiForgeRuntimeProviderResult
taichi_forge_optix_runtime_provider_query(uint32_t requested_abi_version,
                                        size_t api_size,
                                        TiForgeRuntimeProviderApi *out_api);

#ifdef __cplusplus
}
#endif
