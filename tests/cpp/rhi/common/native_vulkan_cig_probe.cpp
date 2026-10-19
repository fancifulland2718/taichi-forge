#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include "taichi/rhi/cuda/cuda_types.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <cstdlib>

// Standalone Windows qualification probe. No Forge runtime, Flora or NVIGI.
// argv: allocation kind (0=device, 1=managed, 2=async), optional graph,
// optional ordinary. Driver API 13.3 ABI is shared with the toolkit-free Forge
// declarations.
struct CigParams {
  int sharedDataType;
  void *sharedData;
};
struct ContextParams {
  void *execAffinityParams;
  int numExecAffinityParams;
  CigParams *cigParams;
};
static_assert(sizeof(CigParams) == 16 && sizeof(ContextParams) == 24);
HMODULE cuda;
template <class... Args>
int cu(const char *name, Args... args) {
  auto f =
      reinterpret_cast<int(__stdcall *)(Args...)>(GetProcAddress(cuda, name));
  if (!f)
    throw std::runtime_error(name);
  return f(args...);
}
void checked(int code, const char *name) {
  std::printf("%s=%d\n", name, code);
  std::fflush(stdout);
  if (code)
    throw std::runtime_error(name);
}
int main(int argc, char **argv) {
  try {
    cuda = LoadLibraryW(L"nvcuda.dll");
    checked(cu("cuInit", 0u), "cuInit");
    int version = 0;
    checked(cu("cuDriverGetVersion", &version), "cuDriverGetVersion");
    std::printf("driver=%d\n", version);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance{};
    checked(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");
    uint32_t n = 0;
    checked(vkEnumeratePhysicalDevices(instance, &n, nullptr), "enumerate");
    std::vector<VkPhysicalDevice> devices(n);
    vkEnumeratePhysicalDevices(instance, &n, devices.data());
    VkPhysicalDevice physical{};
    int ordinal = -1;
    VkPhysicalDeviceIDProperties ids{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceExternalComputeQueuePropertiesNV external{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_COMPUTE_QUEUE_PROPERTIES_NV};
    ids.pNext = &external;
    VkPhysicalDeviceProperties2 props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &ids;
    int count = 0;
    checked(cu("cuDeviceGetCount", &count), "cuDeviceGetCount");
    for (auto dev : devices) {
      vkGetPhysicalDeviceProperties2(dev, &props);
      std::printf("vk_name=%s uuid=", props.properties.deviceName);
      for (auto b : ids.deviceUUID)
        std::printf("%02x", b);
      std::puts("");
      if (props.properties.vendorID != 0x10de)
        continue;
      for (int i = 0; i < count; i++) {
        unsigned char uuid[16]{};
        checked(cu("cuDeviceGetUuid", uuid, i), "cuDeviceGetUuid");
        std::printf("cuda_v1=");
        for (auto b : uuid)
          std::printf("%02x", b);
        std::puts("");
        checked(cu("cuDeviceGetUuid_v2", uuid, i), "cuDeviceGetUuid_v2");
        std::printf("cuda_v2=");
        for (auto b : uuid)
          std::printf("%02x", b);
        std::puts("");
        if (!std::memcmp(uuid, ids.deviceUUID, 16)) {
          physical = dev;
          ordinal = i;
          break;
        }
      }
      if (physical)
        break;
    }
    if (!physical)
      throw std::runtime_error("no matching GPU UUID");
    std::printf("device=%s ordinal=%d queue_data_bytes=%u max_queues=%u\n",
                props.properties.deviceName, ordinal, external.externalDataSize,
                external.maxExternalQueues);
    int supported = 0;
    checked(cu("cuDeviceGetAttribute", &supported, 138, ordinal),
            "vulkan_cig_attribute");
    if (!supported || !external.externalDataSize)
      throw std::runtime_error("CiG unavailable");
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qn, qs.data());
    uint32_t family = 0;
    while (family < qn && !(qs[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
      ++family;
    float priority = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkExternalComputeQueueDeviceCreateInfoNV reserve{
        VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DEVICE_CREATE_INFO_NV};
    reserve.reservedExternalQueues = 1;
    const char *extensions[] = {VK_NV_EXTERNAL_COMPUTE_QUEUE_EXTENSION_NAME,
                                VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timeline_features.timelineSemaphore = VK_TRUE;
    reserve.pNext = &timeline_features;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &reserve;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = extensions;
    VkDevice device{};
    checked(vkCreateDevice(physical, &dci, nullptr, &device), "vkCreateDevice");
    VkQueue queue{};
    vkGetDeviceQueue(device, family, 0, &queue);
    auto create = reinterpret_cast<PFN_vkCreateExternalComputeQueueNV>(
        vkGetDeviceProcAddr(device, "vkCreateExternalComputeQueueNV"));
    auto data = reinterpret_cast<PFN_vkGetExternalComputeQueueDataNV>(
        vkGetDeviceProcAddr(device, "vkGetExternalComputeQueueDataNV"));
    auto destroy = reinterpret_cast<PFN_vkDestroyExternalComputeQueueNV>(
        vkGetDeviceProcAddr(device, "vkDestroyExternalComputeQueueNV"));
    if (!create || !data || !destroy)
      throw std::runtime_error("missing Vulkan entry");
    VkExternalComputeQueueCreateInfoNV eci{
        VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_CREATE_INFO_NV};
    eci.preferredQueue = queue;
    VkExternalComputeQueueNV eq{};
    checked(create(device, &eci, nullptr, &eq),
            "vkCreateExternalComputeQueueNV");
    std::vector<unsigned char> blob(external.externalDataSize);
    VkExternalComputeQueueDataParamsNV params{
        VK_STRUCTURE_TYPE_EXTERNAL_COMPUTE_QUEUE_DATA_PARAMS_NV};
    std::puts("before_queue_data");
    std::fflush(stdout);
    data(eq, &params, blob.data());
    std::puts("after_queue_data");
    std::fflush(stdout);
    CigParams cig{2, blob.data()};
    ContextParams context_params{nullptr, 0, &cig};
    void *context{};
    checked(
        cu("cuCtxCreate_v4", &context,
           argc > 3 ? static_cast<ContextParams *>(nullptr) : &context_params,
           0u, ordinal),
        "cuCtxCreate_v4");
    void *stream{};
    checked(cu("cuStreamCreate", &stream, 1u), "cuStreamCreate");
    uint64_t pointer = 0;
    int kind = argc > 1 ? std::atoi(argv[1]) : 0;
    int ordinary =
        kind == 2   ? cu("cuMemAllocAsync", &pointer, size_t(4096), stream)
        : kind == 1 ? cu("cuMemAllocManaged", &pointer, size_t(4096), 1u)
                    : cu("cuMemAlloc_v2", &pointer, size_t(4096));
    std::printf("allocation_kind=%d\n", kind);
    checked(ordinary, "device_allocation");
    // Nonempty CUDA kernel, read back an oracle, and real Vulkan transfer work.
    const char *ptx =
        ".version 7.0\n.target sm_80\n.address_size 64\n.visible .entry "
        "fill(.param .u64 p) { .reg .b64 a; .reg .b32 v; ld.param.u64 a,[p]; "
        "mov.u32 v,42; st.global.u32 [a],v; ret; }\n";
    void *module{}, *function{};
    checked(cu("cuModuleLoadData", &module, ptx), "cuModuleLoadData");
    checked(cu("cuModuleGetFunction", &function, module, "fill"),
            "cuModuleGetFunction");
    void *graph = nullptr, *exec = nullptr;
    const bool capture = argc > 2;
    if (capture)
      checked(cu("cuStreamBeginCapture_v2", stream, 1), "begin_capture");
    void *args[] = {&pointer};
    checked(cu("cuLaunchKernel", function, 1u, 1u, 1u, 1u, 1u, 1u, 0u, stream,
               args, static_cast<void **>(nullptr)),
            "kernel");
    if (capture) {
      checked(cu("cuStreamEndCapture", stream, &graph), "end_capture");
      checked(cu("cuGraphInstantiateWithFlags", &exec, graph, 0ull),
              "instantiate");
      checked(cu("cuGraphLaunch", exec, stream), "launch_graph");
    }
    checked(cu("cuStreamSynchronize", stream), "kernel_completion");
    if (capture) {
      checked(cu("cuGraphExecDestroy", exec), "exec_destroy");
      checked(cu("cuGraphDestroy", graph), "graph_destroy");
    }
    uint32_t value = 0;
    checked(cu("cuMemcpyDtoH_v2", &value, pointer, size_t(4)), "readback");
    if (value != 42)
      throw std::runtime_error("wrong kernel oracle");

    // All native state is prepared before enqueueing the deliberately pending
    // wait.
    void *other{}, *done{}, *held{};
    checked(cu("cuStreamCreate", &other, 1u), "other_stream");
    checked(cu("cuEventCreate", &done, 2u), "done_event");
    checked(cu("cuEventCreate", &held, 2u), "held_event");
    if (capture) {
      checked(cu("cuStreamBeginCapture_v2", other, 1), "other_capture");
      checked(cu("cuLaunchKernel", function, 1u, 1u, 1u, 1u, 1u, 1u, 0u, other,
                 args, static_cast<void **>(nullptr)),
              "captured_kernel");
      checked(cu("cuStreamEndCapture", other, &graph), "other_capture_end");
      checked(cu("cuGraphInstantiateWithFlags", &exec, graph, 0ull),
              "other_instantiate");
    }
    VkSemaphoreTypeCreateInfo timeline_type{
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &timeline_type;
    VkSemaphore timeline{}, binary{};
    checked(vkCreateSemaphore(device, &sci, nullptr, &timeline), "timeline");
    VkExportSemaphoreCreateInfo external_sem{
        VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    external_sem.handleTypes =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    sci.pNext = &external_sem;
    checked(vkCreateSemaphore(device, &sci, nullptr, &binary), "binary");
    auto get_handle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
    VkSemaphoreGetWin32HandleInfoKHR hi{
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    hi.semaphore = binary;
    hi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE exported{};
    checked(get_handle(device, &hi, &exported), "export_semaphore");
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc{};
    desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
    desc.handle.win32.handle = exported;
    void *imported{};
    checked(cu("cuImportExternalSemaphore", &imported, &desc),
            "import_semaphore");
    CloseHandle(exported);
    const uint64_t one = 1, zero = 0;
    VkTimelineSemaphoreSubmitInfo values{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    values.waitSemaphoreValueCount = values.signalSemaphoreValueCount = 1;
    values.pWaitSemaphoreValues = &one;
    values.pSignalSemaphoreValues = &zero;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo pending{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    pending.pNext = &values;
    pending.waitSemaphoreCount = pending.signalSemaphoreCount = 1;
    pending.pWaitSemaphores = &timeline;
    pending.pWaitDstStageMask = &stage;
    pending.pSignalSemaphores = &binary;
    checked(vkQueueSubmit(queue, 1, &pending, VK_NULL_HANDLE),
            "pending_vulkan_signal");
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait{};
    checked(cu("cuWaitExternalSemaphoresAsync", &imported, &wait, 1u, stream),
            "pending_cuda_wait");
    checked(cu("cuEventRecord", held, stream), "held_record");
    if (capture)
      checked(cu("cuGraphLaunch", exec, other), "independent_graph");
    else
      checked(cu("cuLaunchKernel", function, 1u, 1u, 1u, 1u, 1u, 1u, 0u, other,
                 args, static_cast<void **>(nullptr)),
              "independent_kernel");
    checked(cu("cuEventRecord", done, other), "done_record");
    auto start = std::chrono::steady_clock::now();
    int progress = 600;
    while (std::chrono::steady_clock::now() - start <
           std::chrono::milliseconds(500)) {
      progress = cu("cuEventQuery", done);
      if (progress != 600)
        break;
      Sleep(1);
    }
    int held_status = cu("cuEventQuery", held);
    if (held_status != 600 || (progress != 0 && progress != 600))
      throw std::runtime_error("Invalid pending dependency observation");
    std::printf(
        "INDEPENDENT_PROGRESS mode=%s graph=%d result=%d held=%d "
        "elapsed_ms=%.3f\n",
        argc > 3 ? "ordinary" : "cig", capture, progress, held_status,
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count());
    std::fflush(stdout);
    VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    signal.semaphore = timeline;
    signal.value = 1;
    checked(vkSignalSemaphore(device, &signal), "release_timeline");
    checked(cu("cuEventSynchronize", held), "held_completed");
    checked(cu("cuEventSynchronize", done), "other_completed");
    checked(cu("cuEventDestroy_v2", held), "held_destroy");
    checked(cu("cuEventDestroy_v2", done), "done_destroy");
    checked(cu("cuStreamDestroy_v2", other), "other_destroy");
    checked(cu("cuDestroyExternalSemaphore", imported), "import_destroy");
    if (capture) {
      checked(cu("cuGraphExecDestroy", exec), "other_exec_destroy");
      checked(cu("cuGraphDestroy", graph), "other_graph_destroy");
    }
    vkDestroySemaphore(device, binary, nullptr);
    vkDestroySemaphore(device, timeline, nullptr);

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 4096;
    bci.usage =
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkBuffer buffer{};
    checked(vkCreateBuffer(device, &bci, nullptr, &buffer), "vk_buffer");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, buffer, &req);
    VkPhysicalDeviceMemoryProperties memprops{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memprops);
    uint32_t memtype = 0;
    for (; memtype < memprops.memoryTypeCount; ++memtype)
      if ((req.memoryTypeBits & (1u << memtype)) &&
          (memprops.memoryTypes[memtype].propertyFlags &
           (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
              (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
        break;
    if (memtype == memprops.memoryTypeCount)
      throw std::runtime_error("no Vulkan host buffer type");
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memtype;
    VkDeviceMemory memory{};
    checked(vkAllocateMemory(device, &mai, nullptr, &memory), "vk_memory");
    checked(vkBindBufferMemory(device, buffer, memory, 0), "vk_bind");
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = family;
    VkCommandPool pool{};
    checked(vkCreateCommandPool(device, &pci, nullptr, &pool), "vk_pool");
    VkCommandBufferAllocateInfo cai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer command{};
    checked(vkAllocateCommandBuffers(device, &cai, &command), "vk_command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    checked(vkBeginCommandBuffer(command, &begin), "vk_begin");
    vkCmdFillBuffer(command, buffer, 0, 4096, 44);
    checked(vkEndCommandBuffer(command), "vk_end");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence{};
    checked(vkCreateFence(device, &fci, nullptr, &fence), "vk_fence");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    checked(vkQueueSubmit(queue, 1, &submit, fence), "vk_submit");
    checked(vkWaitForFences(device, 1, &fence, VK_TRUE, 2000000000ull),
            "vk_completion");
    void *mapped{};
    checked(vkMapMemory(device, memory, 0, 4096, 0, &mapped), "vk_map");
    if (*static_cast<uint32_t *>(mapped) != 44)
      throw std::runtime_error("Vulkan fill oracle");
    vkUnmapMemory(device, memory);
    checked(cu("cuLaunchKernel", function, 1u, 1u, 1u, 1u, 1u, 1u, 0u, stream,
               args, static_cast<void **>(nullptr)),
            "kernel_after_vulkan");
    checked(cu("cuStreamSynchronize", stream), "cuda_after_vulkan");
    value = 0;
    checked(cu("cuMemcpyDtoH_v2", &value, pointer, size_t(4)),
            "readback_after_vulkan");
    if (value != 42)
      throw std::runtime_error("CUDA after Vulkan oracle");
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    checked(cu("cuMemFree_v2", pointer), "device_free");
    checked(cu("cuModuleUnload", module), "cuModuleUnload");
    checked(cu("cuStreamDestroy_v2", stream), "cuStreamDestroy_v2");
    checked(cu("cuCtxDestroy_v2", context), "cuCtxDestroy_v2");
    destroy(device, eq, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::puts("PASS native CUDA and Vulkan work, same associated device");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
