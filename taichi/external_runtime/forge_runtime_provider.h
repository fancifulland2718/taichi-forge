#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(TI_FORGE_RUNTIME_PROVIDER_BUILD)
#define TI_FORGE_RUNTIME_PROVIDER_EXPORT __declspec(dllexport)
#else
#define TI_FORGE_RUNTIME_PROVIDER_EXPORT __declspec(dllimport)
#endif
#else
#define TI_FORGE_RUNTIME_PROVIDER_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_RUNTIME_PROVIDER_ABI_VERSION 1u

#define TI_FORGE_CUSPARSELT_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_cusparselt_provider_query"
#define TI_FORGE_CUTENSOR_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_cutensor_provider_query"
#define TI_FORGE_AMGX_PROVIDER_QUERY_SYMBOL "taichi_forge_amgx_provider_query"

typedef enum TiForgeRuntimeProviderResult {
  TI_FORGE_RUNTIME_PROVIDER_SUCCESS = 0,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_INVALID_ARGUMENT = 1,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_ABI_MISMATCH = 2,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_UNAVAILABLE = 3,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_RUNTIME_INCOMPATIBLE = 4,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_OUT_OF_MEMORY = 5,
  TI_FORGE_RUNTIME_PROVIDER_ERROR_INTERNAL = 6,
} TiForgeRuntimeProviderResult;

typedef enum TiForgeRuntimeProviderFeature {
  TI_FORGE_RUNTIME_PROVIDER_FEATURE_VERSION_QUERY = 1ull << 0,
  TI_FORGE_RUNTIME_PROVIDER_FEATURE_REQUIRED_SYMBOL_AUDIT = 1ull << 1,
  TI_FORGE_RUNTIME_PROVIDER_FEATURE_TRANSIENT_PROBE = 1ull << 2,
  TI_FORGE_RUNTIME_PROVIDER_FEATURE_EXECUTION_API = 1ull << 3,
} TiForgeRuntimeProviderFeature;

typedef struct TiForgeRuntimeProviderInfo {
  uint32_t struct_size;
  uint32_t provider_abi_version;
  uint64_t features;
  uint32_t required_symbol_count;
  uint32_t reserved;
  const char *provider_id;
  const char *provider_name;
  const char *supported_version_family;
  const char *build_identity;
} TiForgeRuntimeProviderInfo;

typedef struct TiForgeRuntimeInfo {
  uint32_t struct_size;
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t version_patch;
  uint32_t cuda_runtime_version;
  uint32_t reserved;
  const char *library_path;
  const char *build_version;
} TiForgeRuntimeInfo;

typedef void *TiForgeRuntimeProviderRuntime;

typedef TiForgeRuntimeProviderResult (*TiForgeRuntimeProviderProbeRuntimeFn)(
    const char *library_path,
    TiForgeRuntimeInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeRuntimeProviderCreateRuntimeFn)(
    const char *library_path,
    TiForgeRuntimeProviderRuntime *out_runtime,
    TiForgeRuntimeInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeRuntimeProviderDestroyRuntimeFn)(
    TiForgeRuntimeProviderRuntime runtime);
typedef size_t (*TiForgeRuntimeProviderGetLastErrorFn)(char *destination,
                                                       size_t destination_size);

typedef struct TiForgeRuntimeProviderApi {
  uint32_t struct_size;
  uint32_t provider_abi_version;
  TiForgeRuntimeProviderInfo info;
  TiForgeRuntimeProviderProbeRuntimeFn probe_runtime;
  TiForgeRuntimeProviderCreateRuntimeFn create_runtime;
  TiForgeRuntimeProviderDestroyRuntimeFn destroy_runtime;
  TiForgeRuntimeProviderGetLastErrorFn get_last_error;
} TiForgeRuntimeProviderApi;

typedef TiForgeRuntimeProviderResult (*TiForgeRuntimeProviderQueryFn)(
    uint32_t requested_abi_version,
    size_t api_size,
    TiForgeRuntimeProviderApi *out_api);

#ifdef __cplusplus
}
#endif
