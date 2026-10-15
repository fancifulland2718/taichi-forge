#pragma once

#include "forge_runtime_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TI_FORGE_CUSOLVERDN_PROVIDER_QUERY_SYMBOL \
  "taichi_forge_cusolverdn_provider_query"

typedef void *TiForgeCusolverDnPlan;

// Scalar 0 is f32, 1 is f64. Factor is row-major lower (seen by cuSOLVER
// as column-major upper). Status is device i32[2], factor then solve.
// Callers activate the owning CUDA context, serialize plan access, and retain
// device storage until submitted work retires. Destroy plans only after their
// captured graphs and in-flight work retire, before closing the runtime.
typedef struct TiForgeCusolverDnPlanDesc {
  uint32_t struct_size;
  uint32_t scalar;
  int64_t rows;
  int64_t rhs_count;
  uint64_t factor;
  uint64_t status;
} TiForgeCusolverDnPlanDesc;

typedef struct TiForgeCusolverDnPlanInfo {
  uint32_t struct_size;
  uint32_t reserved;
  uint64_t device_workspace_bytes;
  uint64_t host_workspace_bytes;
} TiForgeCusolverDnPlanInfo;

// Phase 0 factors in place; phase 1 solves into solution, initially containing
// the RHS, shaped (rhs_count, rows). The enclosing recipe performs input/RHS
// copies and status initialization. Creation does not execute numerical work.
typedef struct TiForgeCusolverDnExecDesc {
  uint32_t struct_size;
  uint32_t phase;
  uint64_t workspace;
  uint64_t workspace_bytes;
  uint64_t solution;
  uint64_t cuda_stream;
} TiForgeCusolverDnExecDesc;

typedef TiForgeRuntimeProviderResult (*TiForgeCusolverDnCreatePlanFn)(
    TiForgeRuntimeProviderRuntime runtime,
    const TiForgeCusolverDnPlanDesc *desc,
    TiForgeCusolverDnPlan *out_plan,
    TiForgeCusolverDnPlanInfo *out_info);
typedef TiForgeRuntimeProviderResult (*TiForgeCusolverDnExecuteFn)(
    TiForgeCusolverDnPlan plan,
    const TiForgeCusolverDnExecDesc *desc);
typedef TiForgeRuntimeProviderResult (*TiForgeCusolverDnDestroyPlanFn)(
    TiForgeCusolverDnPlan plan);

typedef struct TiForgeCusolverDnExecutionApi {
  uint32_t struct_size;
  uint32_t execution_abi_version;
  TiForgeCusolverDnCreatePlanFn create_plan;
  TiForgeCusolverDnExecuteFn execute;
  TiForgeCusolverDnDestroyPlanFn destroy_plan;
} TiForgeCusolverDnExecutionApi;

#ifdef __cplusplus
}
#endif
