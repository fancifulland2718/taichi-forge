#pragma once

#include "forge_runtime_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_CUB_SOURCE_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_cub_source_provider_query"

typedef void *TiForgeCubSourcePlan;

// Operation values match the independently built CUB addon C ABI 1.
typedef enum TiForgeCubSourceOperation {
  TI_FORGE_CUB_RADIX_SORT_PAIRS_U32 = 1,
  TI_FORGE_CUB_RADIX_SORT_PAIRS_U64 = 2,
  TI_FORGE_CUB_EXCLUSIVE_SCAN_U32 = 3,
  TI_FORGE_CUB_SELECT_FLAGGED_U32 = 4,
  TI_FORGE_CUB_SEGMENTED_INCLUSIVE_SCAN_U32 = 5,
  TI_FORGE_CUB_SEGMENTED_EXCLUSIVE_SCAN_U32 = 6,
} TiForgeCubSourceOperation;

typedef struct TiForgeCubSourcePlanDesc {
  uint32_t struct_size;
  uint32_t operation;
  uint64_t num_items;
} TiForgeCubSourcePlanDesc;

typedef struct TiForgeCubSourcePlanInfo {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t workspace_bytes;
} TiForgeCubSourcePlanInfo;

// Sort: keys/values -> keys/values; scan: input0 -> output0;
// select: input0/flags -> output0/count; segmented scan: input0/head bitset
// -> output0. Operands and workspace remain caller-owned. The caller validates
// storage extents, types and disjointness before recording any device work.
typedef struct TiForgeCubSourceExecDesc {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t input0, input1, output0, output1;
  uint64_t workspace, workspace_bytes;
  uint64_t cuda_stream;
} TiForgeCubSourceExecDesc;

typedef TiForgeRuntimeProviderResult (*TiForgeCubSourceCreatePlanFn)(
    TiForgeRuntimeProviderRuntime runtime,
    const TiForgeCubSourcePlanDesc *desc,
    TiForgeCubSourcePlan *out_plan,
    TiForgeCubSourcePlanInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeCubSourceExecuteFn)(
    TiForgeCubSourcePlan plan,
    const TiForgeCubSourceExecDesc *desc);
typedef TiForgeRuntimeProviderResult (*TiForgeCubSourceDestroyPlanFn)(
    TiForgeCubSourcePlan plan);

typedef struct TiForgeCubSourceExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  TiForgeCubSourceCreatePlanFn create_plan;
  TiForgeCubSourceExecuteFn execute;
  TiForgeCubSourceDestroyPlanFn destroy_plan;
} TiForgeCubSourceExecutionApi;

// The common runtime leases the external addon. Retire captured/submitted
// uses before destroying plans, then destroy the runtime. Runtime info reports
// CUB's version, compiled CUDART version and sha256:<hex> of the loaded addon.
TI_FORGE_RUNTIME_PROVIDER_EXPORT TiForgeRuntimeProviderResult
taichi_forge_cub_source_provider_query(uint32_t requested_abi_version,
                                       size_t api_size,
                                       TiForgeRuntimeProviderApi *out_api);

#ifdef __cplusplus
}
#endif
