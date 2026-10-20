// Independent Vulkan-only numerical witness for a CiG owner's work queue.
// Run each control in a fresh process; device loss poisons that VkDevice.
#define WIN32_LEAN_AND_MEAN
#define VK_NO_PROTOTYPES
#undef TI_WITH_OPENGL
#include <windows.h>
#include <vulkan/vulkan.h>
#include "taichi/taichi_vulkan.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {
HMODULE loader = LoadLibraryW(L"vulkan-1.dll");
#define LOAD_VK(name) \
  const auto name = reinterpret_cast<PFN_##name>(GetProcAddress(loader, #name))
LOAD_VK(vkGetPhysicalDeviceMemoryProperties);
LOAD_VK(vkCreateBuffer);
LOAD_VK(vkDestroyBuffer);
LOAD_VK(vkGetBufferMemoryRequirements);
LOAD_VK(vkAllocateMemory);
LOAD_VK(vkFreeMemory);
LOAD_VK(vkBindBufferMemory);
LOAD_VK(vkCreateCommandPool);
LOAD_VK(vkDestroyCommandPool);
LOAD_VK(vkAllocateCommandBuffers);
LOAD_VK(vkBeginCommandBuffer);
LOAD_VK(vkEndCommandBuffer);
LOAD_VK(vkCmdFillBuffer);
LOAD_VK(vkCmdCopyBuffer);
LOAD_VK(vkCmdPipelineBarrier);
LOAD_VK(vkQueueSubmit);
LOAD_VK(vkQueueWaitIdle);
LOAD_VK(vkMapMemory);
LOAD_VK(vkUnmapMemory);
#undef LOAD_VK

void check(VkResult result, const char *operation) {
  if (result != VK_SUCCESS)
    throw std::runtime_error(std::string(operation) + "=" +
                             std::to_string(result));
}

struct Probe {
  HMODULE fixture{};
  const void *owner{};
  void (*release)(const void *){};
  TiVulkanRuntimeInteropInfo info{};
  VkBuffer buffers[2]{};
  VkDeviceMemory memory[2]{};
  VkCommandPool pool{};
  void *mapped{};
  bool submitted = false;

  ~Probe() {
    if (submitted)
      vkQueueWaitIdle(info.compute_queue);
    if (mapped)
      vkUnmapMemory(info.device, memory[1]);
    if (pool)
      vkDestroyCommandPool(info.device, pool, nullptr);
    for (int i = 0; i < 2; ++i) {
      if (buffers[i])
        vkDestroyBuffer(info.device, buffers[i], nullptr);
      if (memory[i])
        vkFreeMemory(info.device, memory[i], nullptr);
    }
    if (owner)
      release(owner);
    if (fixture)
      FreeLibrary(fixture);
  }

  void run(const char *path, VkDeviceSize bytes) {
    if (!loader || !vkCreateBuffer)
      throw std::runtime_error("Vulkan loader unavailable");
    fixture = LoadLibraryA(path);
    if (!fixture)
      throw std::runtime_error("Cannot load CiG fixture");
    const auto create = reinterpret_cast<const void *(*)()>(
        GetProcAddress(fixture, "cig_fixture_create"));
    const auto get =
        reinterpret_cast<const TiVulkanRuntimeInteropInfo *(*)(const void *)>(
            GetProcAddress(fixture, "cig_fixture_vulkan"));
    const auto error = reinterpret_cast<const char *(*)()>(
        GetProcAddress(fixture, "cig_fixture_error"));
    release = reinterpret_cast<void (*)(const void *)>(
        GetProcAddress(fixture, "cig_fixture_release"));
    if (!create || !get || !release || !error)
      throw std::runtime_error("Invalid CiG fixture exports");
    owner = create();
    if (!owner)
      throw std::runtime_error(error());
    info = *get(owner);
    std::printf("graphics_family=%u compute_family=%u bytes=%llu\n",
                info.graphics_queue_family_index,
                info.compute_queue_family_index,
                static_cast<unsigned long long>(bytes));
    std::fflush(stdout);
    VkPhysicalDeviceMemoryProperties types{};
    vkGetPhysicalDeviceMemoryProperties(info.physical_device, &types);
    for (int i = 0; i < 2; ++i) {
      VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      buffer.size = bytes;
      buffer.usage =
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      check(vkCreateBuffer(info.device, &buffer, nullptr, &buffers[i]),
            "buffer");
      VkMemoryRequirements requirements{};
      vkGetBufferMemoryRequirements(info.device, buffers[i], &requirements);
      const VkMemoryPropertyFlags flags =
          i == 0 ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                 : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      uint32_t type = 0;
      while (type < types.memoryTypeCount &&
             (!(requirements.memoryTypeBits & (1u << type)) ||
              (types.memoryTypes[type].propertyFlags & flags) != flags))
        ++type;
      if (type == types.memoryTypeCount)
        throw std::runtime_error("No compatible memory type");
      VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      allocation.allocationSize = requirements.size;
      allocation.memoryTypeIndex = type;
      check(vkAllocateMemory(info.device, &allocation, nullptr, &memory[i]),
            "memory");
      check(vkBindBufferMemory(info.device, buffers[i], memory[i], 0), "bind");
    }
    VkCommandPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = info.compute_queue_family_index;
    check(vkCreateCommandPool(info.device, &pool_info, nullptr, &pool), "pool");
    VkCommandBufferAllocateInfo allocation{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer command{};
    check(vkAllocateCommandBuffers(info.device, &allocation, &command),
          "command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin), "begin");
    // Deliberately one full-sized operation each: chunking hides the failure.
    vkCmdFillBuffer(command, buffers[0], 0, bytes, 37);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
    VkBufferCopy copy{0, 0, bytes};
    vkCmdCopyBuffer(command, buffers[0], buffers[1], 1, &copy);
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr,
                         0, nullptr);
    check(vkEndCommandBuffer(command), "end");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(info.compute_queue, 1, &submit, VK_NULL_HANDLE),
          "submit");
    submitted = true;
    check(vkQueueWaitIdle(info.compute_queue), "wait");
    submitted = false;
    check(vkMapMemory(info.device, memory[1], 0, bytes, 0, &mapped), "map");
    for (VkDeviceSize i = 0; i < bytes / sizeof(uint32_t); ++i) {
      if (static_cast<const uint32_t *>(mapped)[i] != 37)
        throw std::runtime_error(
            "Full-buffer numerical oracle failed at word " + std::to_string(i));
    }
    std::printf("NATIVE_VULKAN_FILL_COPY_OK %llu words=%llu\n",
                static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(bytes / sizeof(uint32_t)));
  }
};
}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("usage: probe <CiG fixture DLL> <bytes>");
    char *end = nullptr;
    const auto bytes = std::strtoull(argv[2], &end, 10);
    if (!end || *end || bytes == 0 || bytes % 4 || bytes > 256ull * 1024 * 1024)
      throw std::runtime_error(
          "bytes must be a nonzero multiple of four up to 256 MiB");
    Probe probe;
    probe.run(argv[1], bytes);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}
