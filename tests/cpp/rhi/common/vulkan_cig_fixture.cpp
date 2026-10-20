// Windows native owner fixture for the public, renderer-independent CiG ABI.
// No Flora, NVIGI, CUDA toolkit, or Forge context initialization is used here.
#define WIN32_LEAN_AND_MEAN
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#undef TI_WITH_OPENGL
#include <windows.h>
#include <vulkan/vulkan.h>
#include "taichi/taichi_cuda.h"
#include "taichi/taichi_vulkan.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
HMODULE vulkan = LoadLibraryW(L"vulkan-1.dll");
#define VK_LOAD(name) \
  const auto name = reinterpret_cast<PFN_##name>(GetProcAddress(vulkan, #name))
VK_LOAD(vkGetInstanceProcAddr);
VK_LOAD(vkGetDeviceProcAddr);
VK_LOAD(vkCreateInstance);
VK_LOAD(vkDestroyInstance);
VK_LOAD(vkEnumeratePhysicalDevices);
VK_LOAD(vkGetPhysicalDeviceProperties2);
VK_LOAD(vkGetPhysicalDeviceQueueFamilyProperties);
VK_LOAD(vkCreateDevice);
VK_LOAD(vkDestroyDevice);
VK_LOAD(vkGetDeviceQueue);
VK_LOAD(vkDeviceWaitIdle);
#undef VK_LOAD
std::string error;

void check(VkResult result, const char *operation) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": " +
                             std::to_string(result));
}

struct Owner {
  std::atomic<unsigned> references{1};
  TiCudaContextConfig binding{};
  TiVulkanRuntimeInteropInfo runtime{};
  VkExternalComputeQueueNV external{};
  PFN_vkDestroyExternalComputeQueueNV destroy_external{};
  std::vector<std::uint8_t> bytes;

  ~Owner() {
    if (runtime.device) {
      vkDeviceWaitIdle(runtime.device);
      if (external)
        destroy_external(runtime.device, external, nullptr);
      vkDestroyDevice(runtime.device, nullptr);
    }
    if (runtime.instance)
      vkDestroyInstance(runtime.instance, nullptr);
  }

  static TiError TI_API_CALL retain(void *ptr) noexcept {
    static_cast<Owner *>(ptr)->references.fetch_add(1);
    return TI_ERROR_SUCCESS;
  }
  static void TI_API_CALL release(void *ptr) noexcept {
    auto *owner = static_cast<Owner *>(ptr);
    if (owner->references.fetch_sub(1) == 1)
      delete owner;
  }

  void initialize() {
    if (!vkCreateInstance || !vkGetPhysicalDeviceProperties2)
      throw std::runtime_error("Vulkan 1.2 loader is unavailable");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    check(vkCreateInstance(&info, nullptr, &runtime.instance),
          "create instance");
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(runtime.instance, &count, nullptr),
          "device count");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(runtime.instance, &count, devices.data()),
          "devices");
    VkPhysicalDeviceIDProperties id{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceExternalComputeQueuePropertiesNV caps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_COMPUTE_QUEUE_PROPERTIES_NV};
    id.pNext = &caps;
    VkPhysicalDeviceProperties2 properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &id;
    for (auto physical : devices) {
      vkGetPhysicalDeviceProperties2(physical, &properties);
      if (properties.properties.vendorID == 0x10de && caps.externalDataSize &&
          caps.maxExternalQueues) {
        runtime.physical_device = physical;
        break;
      }
    }
    if (!runtime.physical_device)
      throw std::runtime_error("No Vulkan external-compute-queue device");
    std::memcpy(binding.device_uuid, id.deviceUUID, 16);
    bytes.resize(caps.externalDataSize);
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(runtime.physical_device, &families,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> queues(families);
    vkGetPhysicalDeviceQueueFamilyProperties(runtime.physical_device, &families,
                                             queues.data());
    uint32_t family = 0;
    while (family < families &&
           (queues[family].queueFlags &
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) !=
               (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
      ++family;
    if (family == families)
      throw std::runtime_error("No shared graphics/compute queue family");
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    const char *extensions[] = {
        VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
    };
    VkExternalComputeQueueDeviceCreateInfoNV reserve{
        VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO_NV};
    reserve.reservedExternalQueues = 1;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timeline.timelineSemaphore = VK_TRUE;
    reserve.pNext = &timeline;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &reserve;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 3;
    device_info.ppEnabledExtensionNames = extensions;
    check(vkCreateDevice(runtime.physical_device, &device_info, nullptr,
                         &runtime.device),
          "create device");
    vkGetDeviceQueue(runtime.device, family, 0, &runtime.compute_queue);
    runtime.graphics_queue = runtime.compute_queue;
    runtime.compute_queue_family_index = runtime.graphics_queue_family_index =
        family;
    runtime.get_instance_proc_addr = vkGetInstanceProcAddr;
    runtime.api_version = VK_API_VERSION_1_2;
    const auto create = reinterpret_cast<PFN_vkCreateExternalComputeQueueNV>(
        vkGetDeviceProcAddr(runtime.device, "vkCreateExternalComputeQueueNV"));
    const auto data = reinterpret_cast<PFN_vkGetExternalComputeQueueDataNV>(
        vkGetDeviceProcAddr(runtime.device, "vkGetExternalComputeQueueDataNV"));
    destroy_external = reinterpret_cast<PFN_vkDestroyExternalComputeQueueNV>(
        vkGetDeviceProcAddr(runtime.device, "vkDestroyExternalComputeQueueNV"));
    if (!create || !data || !destroy_external)
      throw std::runtime_error("External queue entry points unavailable");
    VkExternalComputeQueueCreateInfoNV queue{
        VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_CREATE_INFO_NV};
    queue.preferredQueue = runtime.graphics_queue;
    check(create(runtime.device, &queue, nullptr, &external),
          "create external queue");
    VkExternalComputeQueueDataParamsNV params{
        VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DATA_PARAMS_NV};
    data(external, &params, bytes.data());
    binding.struct_size = sizeof(binding);
    binding.schema_version = 1;
    binding.graphics_api = 1;
    binding.mode = TI_CUDA_CONTEXT_REQUIRED;
    binding.queue_data = bytes.data();
    binding.queue_data_size = bytes.size();
    binding.owner = this;
    binding.retain_owner = retain;
    binding.release_owner = release;
  }
};

// Native producer readiness fixture. The gate owns its query fence/timeline
// and retains the device. Tests keep their external buffers/semaphores alive
// until the corresponding CUDA imports are retired, as for ordinary imports.
struct ReadyOwner {
  std::atomic<unsigned> references{1};
  Owner *parent;
  TiCudaExternalReadyGate descriptor{};
  std::mutex mutex;
  VkFence fence{};
  VkSemaphore hold{};
  PFN_vkGetFenceStatus get_fence_status{};
  uint64_t epoch{0}, hold_epoch{0}, queries{0};
  bool submitted{false};
  TiError injected_error{TI_ERROR_SUCCESS};

  template <typename T>
  T function(const char *name) {
    return reinterpret_cast<T>(
        vkGetDeviceProcAddr(parent->runtime.device, name));
  }

  explicit ReadyOwner(Owner *parent) : parent(parent) {
    Owner::retain(parent);
    try {
      get_fence_status = function<PFN_vkGetFenceStatus>("vkGetFenceStatus");
      VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      check(function<PFN_vkCreateFence>("vkCreateFence")(parent->runtime.device,
                                                         &fi, nullptr, &fence),
            "ready fence");
      VkSemaphoreTypeCreateInfo type{
          VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
      type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
      VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
      si.pNext = &type;
      check(function<PFN_vkCreateSemaphore>("vkCreateSemaphore")(
                parent->runtime.device, &si, nullptr, &hold),
            "ready hold");
    } catch (...) {
      if (fence)
        function<PFN_vkDestroyFence>("vkDestroyFence")(parent->runtime.device,
                                                       fence, nullptr);
      Owner::release(parent);
      throw;
    }
    descriptor = {sizeof(descriptor), 1, this, retain, release, query};
  }
  ~ReadyOwner() {
    // A test may abort with its artificial hold active. Release before drain.
    if (hold_epoch)
      release_hold();
    if (submitted)
      function<PFN_vkWaitForFences>("vkWaitForFences")(
          parent->runtime.device, 1, &fence, VK_TRUE, UINT64_MAX);
    function<PFN_vkDestroyFence>("vkDestroyFence")(parent->runtime.device,
                                                   fence, nullptr);
    function<PFN_vkDestroySemaphore>("vkDestroySemaphore")(
        parent->runtime.device, hold, nullptr);
    Owner::release(parent);
  }
  static TiError TI_API_CALL retain(void *ptr) noexcept {
    static_cast<ReadyOwner *>(ptr)->references.fetch_add(1);
    return TI_ERROR_SUCCESS;
  }
  static void TI_API_CALL release(void *ptr) noexcept {
    auto *owner = static_cast<ReadyOwner *>(ptr);
    if (owner->references.fetch_sub(1) == 1)
      delete owner;
  }
  static TiError TI_API_CALL query(void *ptr,
                                   uint64_t epoch,
                                   TiBool *ready) noexcept {
    auto &owner = *static_cast<ReadyOwner *>(ptr);
    std::lock_guard<std::mutex> lock(owner.mutex);
    ++owner.queries;
    *ready = TI_FALSE;
    if (owner.injected_error != TI_ERROR_SUCCESS)
      return owner.injected_error;
    if (epoch != owner.epoch || !owner.submitted)
      return TI_ERROR_SUCCESS;
    const auto result =
        owner.get_fence_status(owner.parent->runtime.device, owner.fence);
    if (result != VK_SUCCESS && result != VK_NOT_READY)
      return TI_ERROR_INVALID_STATE;
    *ready = result == VK_SUCCESS ? TI_TRUE : TI_FALSE;
    return TI_ERROR_SUCCESS;
  }
  void arm() {
    std::lock_guard<std::mutex> lock(mutex);
    if (submitted)
      check(get_fence_status(parent->runtime.device, fence),
            "ready reuse pending");
    check(function<PFN_vkResetFences>("vkResetFences")(parent->runtime.device,
                                                       1, &fence),
          "ready reset");
    ++epoch;
    submitted = true;
  }
  void enqueue_hold() {
    ++hold_epoch;
    VkTimelineSemaphoreSubmitInfo timeline{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline.waitSemaphoreValueCount = 1;
    timeline.pWaitSemaphoreValues = &hold_epoch;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.pNext = &timeline;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &hold;
    submit.pWaitDstStageMask = &stage;
    check(function<PFN_vkQueueSubmit>("vkQueueSubmit")(
              parent->runtime.compute_queue, 1, &submit, VK_NULL_HANDLE),
          "enqueue hold");
  }
  void release_hold() {
    uint64_t current = 0;
    check(
        function<PFN_vkGetSemaphoreCounterValue>("vkGetSemaphoreCounterValue")(
            parent->runtime.device, hold, &current),
        "hold counter");
    if (current >= hold_epoch)
      return;
    VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    info.semaphore = hold;
    info.value = hold_epoch;
    check(function<PFN_vkSignalSemaphore>("vkSignalSemaphore")(
              parent->runtime.device, &info),
          "release hold");
  }
};
}  // namespace

extern "C" __declspec(dllexport) const TiCudaExternalReadyGate *
cig_fixture_ready_create(const TiCudaContextConfig *binding) {
  try {
    return &(new ReadyOwner(static_cast<Owner *>(binding->owner)))->descriptor;
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
extern "C" __declspec(dllexport) uint64_t
cig_fixture_ready_value(const TiCudaExternalReadyGate *gate, unsigned field) {
  auto &owner = *static_cast<ReadyOwner *>(gate->owner);
  std::lock_guard<std::mutex> lock(owner.mutex);
  switch (field) {
    case 0:
      return uint64_t(owner.fence);
    case 1:
      return owner.references.load();
    case 2:
      return owner.epoch;
    default:
      return owner.queries;
  }
}
extern "C" __declspec(dllexport) int cig_fixture_ready_action(
    const TiCudaExternalReadyGate *gate,
    unsigned action) {
  try {
    auto &owner = *static_cast<ReadyOwner *>(gate->owner);
    switch (action) {
      case 0:
        owner.arm();
        break;
      case 1:
        owner.enqueue_hold();
        break;
      case 2:
        owner.release_hold();
        break;
      case 3:
        owner.injected_error = TI_ERROR_INVALID_STATE;
        break;
      case 4:
        owner.injected_error = TI_ERROR_SUCCESS;
        break;
      case 5:
        ReadyOwner::release(&owner);
        break;
      default:
        throw std::runtime_error("unknown ready action");
    }
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}

extern "C" __declspec(dllexport) const TiCudaContextConfig *
cig_fixture_create() {
  auto *owner = new Owner;
  try {
    owner->initialize();
    return &owner->binding;
  } catch (const std::exception &exception) {
    error = exception.what();
    delete owner;
    return nullptr;
  }
}
extern "C" __declspec(dllexport) const char *cig_fixture_error() {
  return error.c_str();
}
extern "C" __declspec(dllexport) unsigned cig_fixture_references(
    const TiCudaContextConfig *binding) {
  return static_cast<Owner *>(binding->owner)->references.load();
}
extern "C" __declspec(dllexport) void cig_fixture_release(
    const TiCudaContextConfig *binding) {
  Owner::release(binding->owner);
}
extern "C" __declspec(dllexport) const TiVulkanRuntimeInteropInfo *
cig_fixture_vulkan(const TiCudaContextConfig *binding) {
  return &static_cast<Owner *>(binding->owner)->runtime;
}
