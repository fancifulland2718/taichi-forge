#pragma once

#include "forge_runtime_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_CUBLASLT_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_cublaslt_provider_query"

typedef void *TiForgeCublasLtPlan;

// Documented AlgoConfig attributes: id, tile, split_k, reduction, swizzle,
// custom, stages, inner_shape, cluster_shape. The final two are optional u16;
// the others are i32 (id/split_k) or u32. No opaque vendor object is persisted.
typedef struct TiForgeCublasLtAlgorithm {
  uint32_t struct_size;
  uint32_t present_mask;
  int64_t values[9];
  uint64_t workspace_bytes;
} TiForgeCublasLtAlgorithm;

// Compact logical f32 A/B/output, with in-place C == D feedback. layout is
// 0 (row major) or 1 (equivalent transposed column major); epilogue is 0
// (identity) or 1 (ReLU). The latter layout swaps physical A/B internally.
// Callers activate the owning CUDA context and serialize access to each plan.
typedef struct TiForgeCublasLtPlanDesc {
  uint32_t struct_size;
  uint32_t layout;
  int64_t m, n, k;
  uint32_t batch_count;
  uint32_t transpose_a;
  uint32_t transpose_b;
  uint32_t epilogue;
  float alpha, beta;
  uint64_t workspace_limit_bytes;
} TiForgeCublasLtPlanDesc;

typedef struct TiForgeCublasLtPlanInfo {
  uint32_t struct_size;
  uint32_t reserved;
  TiForgeCublasLtAlgorithm algorithm;
  // Physical A/B/C/D requirements reported by the selected algorithm.
  uint32_t minimum_alignment[4];
  uint64_t heuristic_calls;
  uint64_t restoration_calls;
} TiForgeCublasLtPlanInfo;

typedef struct TiForgeCublasLtExecDesc {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t a, b, output;
  uint64_t workspace, workspace_bytes;
  uint64_t cuda_stream;
} TiForgeCublasLtExecDesc;

// A null algorithm requests one cold heuristic result. A frozen algorithm is
// restored and checked exactly, without heuristic search or mathematical work.
typedef TiForgeRuntimeProviderResult (*TiForgeCublasLtCreatePlanFn)(
    TiForgeRuntimeProviderRuntime runtime,
    const TiForgeCublasLtPlanDesc *desc,
    const TiForgeCublasLtAlgorithm *algorithm,
    TiForgeCublasLtPlan *out_plan,
    TiForgeCublasLtPlanInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeCublasLtExecuteFn)(
    TiForgeCublasLtPlan plan,
    const TiForgeCublasLtExecDesc *desc);
// Retire all submitted/captured uses before destroy, and destroy plans before
// their common runtime. Workspace and operand storage remain caller-owned.
typedef TiForgeRuntimeProviderResult (*TiForgeCublasLtDestroyPlanFn)(
    TiForgeCublasLtPlan plan);

typedef struct TiForgeCublasLtExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  TiForgeCublasLtCreatePlanFn create_plan;
  TiForgeCublasLtExecuteFn execute;
  TiForgeCublasLtDestroyPlanFn destroy_plan;
} TiForgeCublasLtExecutionApi;

#ifdef __cplusplus
}
#endif
