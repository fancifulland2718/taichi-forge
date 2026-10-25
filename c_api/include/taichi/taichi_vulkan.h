// # Vulkan Backend Features
//
// Taichi's Vulkan API gives you further control over the Vulkan version and
// extension requirements and allows you to interop with external Vulkan
// applications with shared resources.
//
#pragma once

#ifndef TAICHI_H
#include "taichi.h"
#endif  // TAICHI_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Structure `TiVulkanRuntimeInteropInfo` (1.4.0)
//
// Necessary detail to share the same Vulkan runtime between Taichi and external
// procedures.
//
//
// **NOTE** `compute_queue` and `graphics_queue` can be the same if the queue
// family have `VK_QUEUE_COMPUTE_BIT` and `VK_QUEUE_GRAPHICS_BIT` set at the
// same tiem.
typedef struct TiVulkanRuntimeInteropInfo {
  // Pointer to Vulkan loader function `vkGetInstanceProcAddr`.
  PFN_vkGetInstanceProcAddr get_instance_proc_addr;
  // Target Vulkan API version.
  uint32_t api_version;
  // Vulkan instance handle.
  VkInstance instance;
  // Vulkan physical device handle.
  VkPhysicalDevice physical_device;
  // Vulkan logical device handle.
  VkDevice device;
  // Vulkan queue handle created in the queue family at
  // `structure.vulkan_runtime_interop_info.compute_queue_family_index`.
  VkQueue compute_queue;
  // Index of a Vulkan queue family with the `VK_QUEUE_COMPUTE_BIT` set.
  uint32_t compute_queue_family_index;
  // Vulkan queue handle created in the queue family at
  // `structure.vulkan_runtime_interop_info.graphics_queue_family_index`.
  VkQueue graphics_queue;
  // Index of a Vulkan queue family with the `VK_QUEUE_GRAPHICS_BIT` set.
  uint32_t graphics_queue_family_index;
} TiVulkanRuntimeInteropInfo;

// Structure `TiVulkanMemoryInteropInfo` (1.4.0)
//
// Necessary detail to share the same piece of Vulkan buffer between Taichi and
// external procedures.
typedef struct TiVulkanMemoryInteropInfo {
  // Vulkan buffer.
  VkBuffer buffer;
  // Size of the piece of memory in bytes.
  uint64_t size;
  // Vulkan buffer usage. In most of the cases, Taichi requires the
  // `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`.
  VkBufferUsageFlags usage;
  // Device memory binded to the Vulkan buffer.
  VkDeviceMemory memory;
  // Offset in `VkDeviceMemory` object to the beginning of this allocation, in
  // bytes.
  uint64_t offset;
} TiVulkanMemoryInteropInfo;

// Structure `TiVulkanImageInteropInfo` (1.4.0)
//
// Necessary detail to share the same piece of Vulkan image between Taichi and
// external procedures.
typedef struct TiVulkanImageInteropInfo {
  // Vulkan image.
  VkImage image;
  // Vulkan image allocation type.
  VkImageType image_type;
  // Pixel format.
  VkFormat format;
  // Image extent.
  VkExtent3D extent;
  // Number of mip-levels of the image.
  uint32_t mip_level_count;
  // Number of array layers.
  uint32_t array_layer_count;
  // Number of samples per pixel.
  VkSampleCountFlagBits sample_count;
  // Image tiling.
  VkImageTiling tiling;
  // Vulkan image usage. In most cases, Taichi requires the
  // `VK_IMAGE_USAGE_STORAGE_BIT` and the `VK_IMAGE_USAGE_SAMPLED_BIT`.
  VkImageUsageFlags usage;
} TiVulkanImageInteropInfo;

// Function `ti_create_vulkan_runtime_ext` (1.4.0)
//
// Creates a Vulkan Taichi runtime with user-controlled capability settings.
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_create_vulkan_runtime_ext(uint32_t api_version,
                             uint32_t instance_extension_count,
                             const char **instance_extensions,
                             uint32_t device_extension_count,
                             const char **device_extensions);

// Function `ti_import_vulkan_runtime` (1.4.0)
//
// Imports the Vulkan runtime owned by Taichi to external procedures.
TI_DLL_EXPORT TiRuntime TI_API_CALL
ti_import_vulkan_runtime(const TiVulkanRuntimeInteropInfo *interop_info);

// Function `ti_export_vulkan_runtime` (1.4.0)
//
// Exports a Vulkan runtime from external procedures to Taichi.
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_runtime(TiRuntime runtime,
                         TiVulkanRuntimeInteropInfo *interop_info);

// Function `ti_import_vulkan_memory` (1.4.0)
//
// Imports the Vulkan buffer owned by Taichi to external procedures.
TI_DLL_EXPORT TiMemory TI_API_CALL
ti_import_vulkan_memory(TiRuntime runtime,
                        const TiVulkanMemoryInteropInfo *interop_info);

// Function `ti_export_vulkan_memory` (1.4.0)
//
// Exports a Vulkan buffer from external procedures to Taichi.
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_memory(TiRuntime runtime,
                        TiMemory memory,
                        TiVulkanMemoryInteropInfo *interop_info);

// Actual producer allocation/buffer facts for an OPAQUE_WIN32 import. These
// are runtime facts, never AOT artifact fields. All three HANDLE values remain
// caller-owned on success and failure; Forge imports private duplicates.
typedef struct TiVulkanExternalBufferImportInfo {
  uint32_t struct_size;
  uint32_t schema_version;  // 1
  uint64_t memory_handle;
  uint64_t ready_for_consumer;
  uint64_t ready_for_external;
  uint64_t buffer_size;
  uint64_t allocation_size;
  uint64_t bind_offset;
  uint32_t memory_type_index;
  uint32_t buffer_flags;
  uint32_t buffer_usage;
  uint32_t allocation_flags;
  uint32_t sharing_mode;
  uint32_t dedicated;
  uint8_t device_uuid[16];
  uint8_t driver_uuid[16];
  // Finite ndarray view within buffer_size. Ownership always covers the whole
  // buffer; a view does not create a separate semaphore pair or access epoch.
  uint64_t byte_offset;
  uint64_t byte_size;
} TiVulkanExternalBufferImportInfo;

// Native, noexcept callbacks, independent of Python. A successful retain must
// keep the query state, producer device, allocation and both semaphores alive;
// release drops exactly that reference after the last Forge import retires.
typedef TiError(TI_API_CALL *TiVulkanExternalOwnerRetain)(void *owner);
typedef void(TI_API_CALL *TiVulkanExternalOwnerRelease)(void *owner);
typedef TiError(TI_API_CALL *TiVulkanExternalReadyQuery)(void *owner,
                                                      uint64_t epoch,
                                                      TiBool *ready);
typedef struct TiVulkanExternalReadyGate {
  uint32_t struct_size;
  uint32_t schema_version;  // 1
  void *owner;
  TiVulkanExternalOwnerRetain retain_owner;
  TiVulkanExternalOwnerRelease release_owner;
  TiVulkanExternalReadyQuery query;
} TiVulkanExternalReadyGate;

// Cold import for independent Vulkan prepared executions only. Requires
// Windows x64, the same physical GPU and driver, a dedicated EXCLUSIVE buffer
// bound at offset zero, and its exact creation/allocation facts. Storage usage
// is required; device-address usage/allocation flags must also be present when
// the runtime's ndarray ABI requires them. No ownership transfer occurs here.
// Import each physical buffer/semaphore pair once; derive aliases through plan
// views. The producer releases the whole buffer to VK_QUEUE_FAMILY_EXTERNAL
// before signaling ready_for_consumer, and waits ready_for_external before its
// next acquire. Every accepted complete recipe waits/acquires and releases/
// signals exactly once on its selected queue.
//
// ready_gate is mandatory and copied. Epochs start at 1. Query returns
// TI_ERROR_SUCCESS and TI_TRUE only once that epoch's matching producer signal
// is actually enqueued (it may still be pending). Preserve that signal until
// Forge consumes it. Query must be nonblocking and side-effect-free: no GPU
// work, waiting, Forge reentry or Python. Busy attempts consume neither epochs
// nor signals. Ordinary launch/copy/map/export entrypoints reject this memory.
// ti_free_memory retires dependent Forge consumption and imports; it does not
// drain the producer's queued work. The producer remains responsible for its
// own device/queue lifetime and for draining work before destroying resources.
TI_DLL_EXPORT TiMemory TI_API_CALL ti_import_vulkan_external_buffer_ext(
    TiRuntime runtime,
    const TiVulkanExternalBufferImportInfo *import_info,
    const TiVulkanExternalReadyGate *ready_gate);

// Function `ti_import_vulkan_image` (1.4.0)
//
// Imports an externally owned Vulkan image into Taichi. The owner must keep the
// image alive until imported-image use completes and the imported handle is freed.
// `layout` is the image's current VkImageLayout; importing does not transition it.
// Supported layouts are UNDEFINED, GENERAL, COLOR_ATTACHMENT_OPTIMAL,
// DEPTH_STENCIL_ATTACHMENT_OPTIMAL, DEPTH_STENCIL_READ_ONLY_OPTIMAL,
// SHADER_READ_ONLY_OPTIMAL, TRANSFER_SRC_OPTIMAL, TRANSFER_DST_OPTIMAL and
// PRESENT_SRC_KHR. Other layouts return null with TI_ERROR_ARGUMENT_OUT_OF_RANGE
// before creating an image view. Queue ownership and producer completion remain
// the caller's responsibility.
TI_DLL_EXPORT TiImage TI_API_CALL
ti_import_vulkan_image(TiRuntime runtime,
                       const TiVulkanImageInteropInfo *interop_info,
                       VkImageViewType view_type,
                       VkImageLayout layout);

// Function `ti_export_vulkan_image` (1.4.0)
//
// Exports a Taichi image's Vulkan handles to external procedures.
TI_DLL_EXPORT void TI_API_CALL
ti_export_vulkan_image(TiRuntime runtime,
                       TiImage image,
                       TiVulkanImageInteropInfo *interop_info);

// Independent compute queues use only queues actually created on this device.
// Creation requires enabled timeline semaphores and an unused additional queue
// from the ordinary compute family. There is no default-queue fallback.
typedef struct TiVulkanExecutionDomain_t *TiVulkanExecutionDomain;
typedef struct TiVulkanExecutionInstance_t *TiVulkanExecutionInstance;
typedef enum TiVulkanExecutionQueueKind {
  TI_VULKAN_EXECUTION_QUEUE_COMPUTE = 0,
  TI_VULKAN_EXECUTION_QUEUE_GRAPHICS = 1
} TiVulkanExecutionQueueKind;

typedef struct TiVulkanExecutionDomainCreateInfo {
  uint32_t capacity;
  uint32_t instance_capacity;
  TiBool gpu_timing;
  TiVulkanExecutionQueueKind queue_kind;
} TiVulkanExecutionDomainCreateInfo;
typedef struct TiVulkanExecutionDomainInfo {
  uint64_t identity;
  uint64_t owner;
  uint32_t queue_index;
  uint64_t capacity;
  uint64_t pending;
  TiBool accepting;
  TiBool closed;
  uint32_t queue_family_index;
} TiVulkanExecutionDomainInfo;

TI_DLL_EXPORT TiVulkanExecutionDomain TI_API_CALL
ti_create_vulkan_execution_domain_ext(TiRuntime runtime,
                                      const TiVulkanExecutionDomainCreateInfo *info);
TI_DLL_EXPORT void TI_API_CALL
ti_get_vulkan_execution_domain_info_ext(TiVulkanExecutionDomain domain,
                                        TiVulkanExecutionDomainInfo *info);
TI_DLL_EXPORT void TI_API_CALL ti_stop_vulkan_execution_domain_ext(TiVulkanExecutionDomain domain);
TI_DLL_EXPORT void TI_API_CALL ti_close_vulkan_execution_domain_ext(TiVulkanExecutionDomain domain);
// Host-only release; runtime ownership remains until close/retirement.
TI_DLL_EXPORT void TI_API_CALL ti_release_vulkan_execution_domain_ext(TiVulkanExecutionDomain domain);
TI_DLL_EXPORT uint32_t TI_API_CALL
ti_collect_vulkan_execution_domain_ext(TiVulkanExecutionDomain domain, uint32_t budget);
TI_DLL_EXPORT TiCompletion TI_API_CALL
ti_get_vulkan_execution_domain_completion_ext(TiVulkanExecutionDomain domain);

TI_DLL_EXPORT TiVulkanExecutionInstance TI_API_CALL
ti_get_prepared_vulkan_instance_ext(TiRuntime runtime, TiPreparedExecution execution);
// Retires only this instance's exact prefix; no runtime pointer or lock needed.
TI_DLL_EXPORT void TI_API_CALL
ti_retire_vulkan_execution_instance_ext(TiVulkanExecutionInstance instance);
TI_DLL_EXPORT void TI_API_CALL
ti_release_vulkan_execution_instance_ext(TiVulkanExecutionInstance instance);

// Cold preparation of a qualified fixed plan. Call ti_initialize_prepared_ext
// before submission. Mutable parameters are replay-only i32/u32/i64/u64
// scalars; count=0 freezes every scalar. Commands, parameter pools and sampled
// timestamps stay on the chosen queue. Private offscreen graphics requires a
// graphics-capable queue family; live writable image, AS and shared sparse
// contracts remain separate.
TI_DLL_EXPORT TiPreparedExecution TI_API_CALL
ti_prepare_execution_plan_on_vulkan_domain_ext(
    TiRuntime runtime, TiAotModule module, TiVulkanExecutionDomain domain,
    uint64_t description_size, const char *description,
    uint32_t arg_count, const TiNamedArgument *args,
    uint32_t binding_count, const TiNamedNdArrayBinding *bindings,
    uint32_t parameter_count, const char *const *parameter_names);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
