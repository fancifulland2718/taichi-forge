#pragma once

#include "forge_runtime_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_CUTLASS_SOURCE_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_cutlass_source_provider_query"

typedef void *TiForgeCutlassSourcePlan;

// Frozen choices from the existing independently built CUTLASS addon C ABI 1.
typedef enum TiForgeCutlassSourceStrategy {
  TI_FORGE_CUTLASS_DIRECT_FUSED = 0,
  TI_FORGE_CUTLASS_SPLIT_K_REDUCE_FUSED = 1,
  TI_FORGE_CUTLASS_SPLIT_K_WIDE_REDUCE_FUSED = 2,
} TiForgeCutlassSourceStrategy;

typedef struct TiForgeCutlassSourcePlanDesc {
  uint32_t struct_size, strategy, transpose_a, transpose_b;
  int32_t m, n, k;
  int32_t activation;  // 0 = identity, 1 = ReLU.
  float alpha, beta;
} TiForgeCutlassSourcePlanDesc;

typedef struct TiForgeCutlassSourcePlanInfo {
  uint32_t struct_size, reserved;
  uint64_t workspace_bytes;
} TiForgeCutlassSourcePlanInfo;

// Compact f32 operands: output = activation(alpha * op(a) * op(b) +
// beta * output). All arrays/workspace remain caller-owned; validate their
// extents, types and physical disjointness before recording device work.
typedef struct TiForgeCutlassSourceExecDesc {
  uint32_t struct_size, reserved;
  uint64_t a, b, output, workspace, workspace_bytes, cuda_stream;
} TiForgeCutlassSourceExecDesc;

typedef TiForgeRuntimeProviderResult (*TiForgeCutlassSourceCreatePlanFn)(
    TiForgeRuntimeProviderRuntime runtime,
    const TiForgeCutlassSourcePlanDesc *desc,
    TiForgeCutlassSourcePlan *out_plan,
    TiForgeCutlassSourcePlanInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeCutlassSourceExecuteFn)(
    TiForgeCutlassSourcePlan plan,
    const TiForgeCutlassSourceExecDesc *desc);
typedef TiForgeRuntimeProviderResult (*TiForgeCutlassSourceDestroyPlanFn)(
    TiForgeCutlassSourcePlan plan);

typedef struct TiForgeCutlassSourceExecutionApi {
  uint32_t struct_size, execution_abi_version;
  TiForgeCutlassSourceCreatePlanFn create_plan;
  TiForgeCutlassSourceExecuteFn execute;
  TiForgeCutlassSourceDestroyPlanFn destroy_plan;
} TiForgeCutlassSourceExecutionApi;

// Retire captured/submitted uses before destroying plans, then the runtime.
// Runtime info reports CUTLASS version and sha256:<hex> of the loaded addon.
// The old addon ABI has no CUDART version query: cuda_runtime_version is zero.
// Addon target-code/runtime dependencies remain its build manifest's contract.
TI_FORGE_RUNTIME_PROVIDER_EXPORT TiForgeRuntimeProviderResult
taichi_forge_cutlass_source_provider_query(uint32_t requested_abi_version,
                                           size_t api_size,
                                           TiForgeRuntimeProviderApi *out_api);

#ifdef __cplusplus
}
#endif
