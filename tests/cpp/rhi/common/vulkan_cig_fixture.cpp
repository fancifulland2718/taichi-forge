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
}  // namespace

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
