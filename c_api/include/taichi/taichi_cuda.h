#pragma once

#ifndef TAICHI_H
#include "taichi.h"
#endif  // TAICHI_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Structure `TiCudaMemoryInteropInfo`
typedef struct TiCudaMemoryInteropInfo {
  void *ptr;
  uint64_t size;
} TiCudaMemoryInteropInfo;

// Function `ti_export_cuda_memory`
TI_DLL_EXPORT void TI_API_CALL
ti_export_cuda_memory(TiRuntime runtime,
                      TiMemory memory,
                      TiCudaMemoryInteropInfo *interop_info);

// Function `ti_import_cuda_memory`
TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_cuda_memory(TiRuntime runtime,
                                                         void *ptr,
                                                         size_t memory_size);

// Function `ti_set_cuda_stream`
// Forge uses the legacy default stream. nullptr is accepted; non-null streams
// report TI_ERROR_NOT_SUPPORTED without changing the runtime's stream.
TI_DLL_EXPORT void TI_API_CALL ti_set_cuda_stream(void *stream);

// Function `ti_get_cuda_stream`
// Writes nullptr (the legacy default stream). `stream` must not be null.
TI_DLL_EXPORT void TI_API_CALL ti_get_cuda_stream(void **stream);

// Runtime-owned nonblocking streams. Domain identity and completion handles
// are runtime objects and must never be serialized into AOT artifacts.
typedef struct TiCudaExecutionDomain_t *TiCudaExecutionDomain;
typedef struct TiCompletion_t *TiCompletion;

typedef struct TiCudaExecutionDomainCreateInfo {
  uint32_t capacity;
  uint32_t instance_capacity;
  int32_t priority;
} TiCudaExecutionDomainCreateInfo;

typedef struct TiCudaExecutionDomainInfo {
  uint64_t identity;
  uint64_t owner;
  uint64_t context;
  uint64_t stream;
  int32_t requested_priority;
  int32_t effective_priority;
  int32_t least_priority;
  int32_t greatest_priority;
  uint64_t capacity;
  uint64_t pending;
  TiBool accepting;
  TiBool closed;
} TiCudaExecutionDomainInfo;

typedef struct TiCompletionInfo {
  uint64_t owner;
  uint64_t sequence;
  TiBool has_backend_work;
} TiCompletionInfo;

// Cold construction, with explicit positive queue and instance capacities.
// Priority is a scheduling hint; query the effective value on this device.
TI_DLL_EXPORT TiCudaExecutionDomain TI_API_CALL
ti_create_cuda_execution_domain_ext(
    TiRuntime runtime,
    const TiCudaExecutionDomainCreateInfo *create_info);
TI_DLL_EXPORT void TI_API_CALL
ti_get_cuda_execution_domain_info_ext(TiCudaExecutionDomain domain,
                                      TiCudaExecutionDomainInfo *info);
// Closes admission permanently without waiting for device work.
TI_DLL_EXPORT void TI_API_CALL
ti_stop_cuda_execution_domain_ext(TiCudaExecutionDomain domain);
// Cold destructive close. Driver/provider teardown may synchronize the context.
// An external producer must signal any previously submitted device waits.
TI_DLL_EXPORT void TI_API_CALL
ti_close_cuda_execution_domain_ext(TiCudaExecutionDomain domain);
// Host-only handle release. The runtime retains the bounded domain until close
// or runtime retirement; release never cancels submitted work.
TI_DLL_EXPORT void TI_API_CALL
ti_release_cuda_execution_domain_ext(TiCudaExecutionDomain domain);
TI_DLL_EXPORT uint32_t TI_API_CALL
ti_collect_cuda_execution_domain_ext(TiCudaExecutionDomain domain,
                                     uint32_t budget);
// Snapshot the already accepted prefix, including the empty prefix. Subsequent
// submissions cannot extend this ticket. The caller owns the returned handle.
TI_DLL_EXPORT TiCompletion TI_API_CALL
ti_get_cuda_execution_domain_completion_ext(TiCudaExecutionDomain domain);

// Cold binding of an existing complete AOT plan. Runtime handles and stream
// identity are not part of the artifact. Call ti_initialize_prepared_ext before
// submitting, even if the artifact has no explicit initialization operations.
// Only compiler-qualified independent CUDA routes are admitted.
TI_DLL_EXPORT TiPreparedExecution TI_API_CALL
ti_prepare_execution_plan_on_domain_ext(TiRuntime runtime,
                                        TiAotModule module,
                                        TiCudaExecutionDomain domain,
                                        uint64_t description_size,
                                        const char *description,
                                        uint32_t arg_count,
                                        const TiNamedArgument *args,
                                        uint32_t binding_count,
                                        const TiNamedNdArrayBinding *bindings);
// True publishes an owned exact completion. False with no error means full;
// no work/dependency wait is queued and *completion is null. All dependencies
// must belong to the same runtime generation. Does not implicitly initialize.
// Releasing a domain-bound prepared handle retains its finite recording and
// dependency closure until domain close; explicit module/memory destruction is
// a cold boundary and closes each affected domain before freeing storage.
TI_DLL_EXPORT TiBool TI_API_CALL
ti_try_submit_prepared_ext(TiRuntime runtime,
                           TiPreparedExecution execution,
                           uint32_t dependency_count,
                           const TiCompletion *dependencies,
                           TiCompletion *completion);

// These calls do not access or lock the originating Runtime. Query, wait and
// independent submission may proceed concurrently. Reset/fault remains visible
// through the ticket's owner state; the caller must release each handle.
TI_DLL_EXPORT TiBool TI_API_CALL
ti_query_completion_ext(TiCompletion completion);
TI_DLL_EXPORT void TI_API_CALL ti_wait_completion_ext(TiCompletion completion);
TI_DLL_EXPORT void TI_API_CALL
ti_get_completion_info_ext(TiCompletion completion, TiCompletionInfo *info);
TI_DLL_EXPORT void TI_API_CALL
ti_release_completion_ext(TiCompletion completion);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
