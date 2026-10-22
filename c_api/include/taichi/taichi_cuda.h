#pragma once

#ifndef TAICHI_H
#include "taichi.h"
#endif  // TAICHI_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Initialization-only CUDA/Vulkan context association. This is NOT an AOT
// artifact field. Configure before any CUDA Program, runtime, probe or stream.
typedef enum TiCudaContextMode {
  TI_CUDA_CONTEXT_DISABLED = 0,
  TI_CUDA_CONTEXT_REQUIRED = 1,
  TI_CUDA_CONTEXT_AUTO = 2,
} TiCudaContextMode;

// Native, noexcept callbacks. retain returns TI_ERROR_SUCCESS on success;
// release drops exactly that reference. They must not depend on a Python
// interpreter. The owner retains the Vulkan external queue AND its device.
typedef TiError(TI_API_CALL *TiCudaContextOwnerRetain)(void *owner);
typedef void(TI_API_CALL *TiCudaContextOwnerRelease)(void *owner);

typedef struct TiCudaContextConfig {
  uint32_t struct_size;
  uint32_t schema_version;  // 1
  TiCudaContextMode mode;
  uint32_t graphics_api;  // 1 = Vulkan, including disabled/auto-without-binding
  uint8_t device_uuid[16];
  const void *queue_data;
  uint64_t queue_data_size;
  void *owner;
  TiCudaContextOwnerRetain retain_owner;
  TiCudaContextOwnerRelease release_owner;
} TiCudaContextConfig;

typedef struct TiCudaContextInfo {
  uint32_t struct_size;
  uint32_t schema_version;  // 1
  TiBool initialized;
  TiBool vulkan_cig;
  TiBool driver_entry;
  TiBool device_capability;
  uint64_t generation;
  uint8_t device_uuid[16];
  // Cold checks: 1=device, 2=managed, 4=async, 8=owned pool, 16=capture.
  uint32_t allocation_checks;
  // First version: context and native owner are process-lived. Closing a
  // Program/runtime retires its resources, but does NOT release this lease.
  TiBool process_lifetime;
  // 0=ordinary CUDA, 1=CiG retained ordinary device blocks. Counts below are
  // the context's parameter/scratch pool, not Graph-owned pool statistics.
  uint32_t allocation_route;
  uint64_t native_allocation_calls;
  uint64_t native_free_calls;
  uint64_t reuse_calls;
  uint64_t retirement_calls;
  uint64_t reserved_current_bytes;
  uint64_t reserved_high_bytes;
  uint64_t used_current_bytes;
  uint64_t used_high_bytes;
  char fallback_reason[256];
} TiCudaContextInfo;

TI_DLL_EXPORT void TI_API_CALL
ti_configure_cuda_context_ext(const TiCudaContextConfig *config);
// Does not create a context or probe a device.
TI_DLL_EXPORT void TI_API_CALL
ti_get_cuda_context_info_ext(TiCudaContextInfo *info);

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

// Opaque external allocation plus one binary ready/consumed semaphore pair.
// These are runtime import facts, never AOT artifact fields. Values are native
// NT handles on Windows or file descriptors on Linux, not CUDA pointers.
typedef enum TiCudaExternalHandleType {
  TI_CUDA_EXTERNAL_HANDLE_OPAQUE_WIN32 = 1,
  TI_CUDA_EXTERNAL_HANDLE_OPAQUE_FD = 2,
} TiCudaExternalHandleType;

typedef struct TiCudaExternalMemoryImportInfo {
  TiCudaExternalHandleType handle_type;
  TiBool dedicated;
  uint64_t memory_handle;
  uint64_t ready_for_cuda;
  uint64_t ready_for_external;
  uint64_t allocation_size;
  uint64_t byte_offset;
  uint64_t byte_size;
  uint8_t device_uuid[16];
} TiCudaExternalMemoryImportInfo;

// Cold import of a finite linear buffer range on the same physical GPU.
// All original OS handles remain caller-owned on success and failure. Internal
// duplicates are imported and released by this runtime. Keep the external
// allocation, producer API device and semaphore objects alive until
// ti_free_memory (or runtime destruction) returns. Each complete
// selected-domain recipe consumes one ready signal and publishes one consumed
// signal. The producer must enqueue the ready signal before CUDA submission;
// it may still be pending on device. Enqueue the external consumer wait after
// CUDA submission returns. Ordinary/default entrypoints reject this memory. Free is a cold drain
// of its dependent execution domains. The external producer remains responsible
// for draining its own queued work. Import each physical semaphore pair once;
// derive aliases through plan views.
TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_cuda_external_memory_ext(
    TiRuntime runtime,
    const TiCudaExternalMemoryImportInfo *import_info);

// Nonblocking native query for the next ready signal on one imported pair.
// Epochs start at 1 and advance only after an accepted access enqueues its real
// CUDA semaphore wait. Busy attempts consume neither epochs nor signals.
// Return TI_ERROR_SUCCESS and set ready to TI_TRUE only after that exact
// producer signal has completed. Preserve it until the consumer acquires it.
// Queries must not wait, submit GPU work, reenter Forge, or depend on Python.
typedef TiError(TI_API_CALL *TiCudaExternalReadyQuery)(void *owner,
                                                       uint64_t epoch,
                                                       TiBool *ready);
typedef struct TiCudaExternalReadyGate {
  uint32_t struct_size;
  uint32_t schema_version;  // 1
  void *owner;
  TiCudaContextOwnerRetain retain_owner;
  TiCudaContextOwnerRelease release_owner;
  TiCudaExternalReadyQuery query;
} TiCudaExternalReadyGate;

// Optional cold import with ready-only admission. The descriptor is copied;
// its native owner must retain the query state and producer device through
// CUDA retirement. The original allocation/semaphore lifetime rules still
// apply. Calls are serialized per runtime.
// Prepared submit returns false/no error before any device effect if any
// input is unready. A ready query never replaces acquire/release semaphores.
// Initialization also checks readiness and must be performed with ready inputs.
// Pending explicit completion dependencies retain their usual device waits.
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_import_cuda_external_memory_with_ready_gate_ext(
    TiRuntime runtime,
    const TiCudaExternalMemoryImportInfo *import_info,
    const TiCudaExternalReadyGate *ready_gate);

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
// True publishes an owned exact completion. False with no error means full
// or an optional external ready gate is not ready;
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

// Ready-only variant: additionally returns false/no error if any validated
// completion dependency is still pending. It does not wait or enqueue partial
// dependencies. Completed dependencies retain the ordinary ownership checks.
TI_DLL_EXPORT TiBool TI_API_CALL
ti_try_submit_prepared_ready_ext(TiRuntime runtime,
                                 TiPreparedExecution execution,
                                 uint32_t dependency_count,
                                 const TiCompletion *dependencies,
                                 TiCompletion *completion);

// Cold declaration of replay-only u64 scalar parameters in positional order.
// Each independent instance owns preallocated pinned/device argument storage.
// A parameterized instance accepts one in-flight submission; prepare separate
// instances for overlap. No new kernel compilation or artifact format is used.
TI_DLL_EXPORT TiPreparedExecution TI_API_CALL
ti_prepare_execution_plan_parameters_on_domain_ext(
    TiRuntime runtime,
    TiAotModule module,
    TiCudaExecutionDomain domain,
    uint64_t description_size,
    const char *description,
    uint32_t arg_count,
    const TiNamedArgument *args,
    uint32_t binding_count,
    const TiNamedNdArrayBinding *bindings,
    uint32_t parameter_count,
    const char *const *parameter_names);

// Values are copied before return; all declared parameters must be supplied.
// Busy consumes no sequence, device upload, dependency or external signal.
// Accepted completion covers both uploads and execution on the selected stream.
TI_DLL_EXPORT TiBool TI_API_CALL
ti_try_submit_prepared_parameters_ext(TiRuntime runtime,
                                      TiPreparedExecution execution,
                                      uint32_t dependency_count,
                                      const TiCompletion *dependencies,
                                      TiCompletion *completion,
                                      TiBool ready_only,
                                      uint32_t parameter_count,
                                      const uint64_t *parameters);

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
