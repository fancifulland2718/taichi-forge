#include "taichi/program/vulkan_micromap.h"

#if defined(TI_WITH_VULKAN)
#include <algorithm>
#include <cstring>
#include <map>

namespace taichi::lang {
namespace {
template <typename T>
T load(VkDevice device, const char *name) {
  auto fn = reinterpret_cast<T>(vkGetDeviceProcAddr(device, name));
  TI_ERROR_IF(!fn, "Vulkan opacity micromap requires '{}'.", name);
  return fn;
}
VkDeviceAddress aligned(VkDeviceAddress address, VkDeviceSize alignment) {
  return (address + alignment - 1) & ~(alignment - 1);
}
using Counts = std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t>;
std::vector<VkMicromapUsageEXT> counts(const Counts &values) {
  std::vector<VkMicromapUsageEXT> result;
  for (const auto &value : values) {
    result.push_back({value.second, value.first.first, value.first.second});
  }
  return result;
}
}  // namespace

VulkanOpacityMicromap::Handle::~Handle() {
  if (micromap) {
    destroy(device, micromap, nullptr);
  }
}

VulkanOpacityMicromap::VulkanOpacityMicromap(vulkan::VulkanDevice *device)
    : device_(device) {
  TI_ERROR_IF(!device_ || !device_->vk_caps().opacity_micromap,
              "Vulkan opacity micromap is unavailable on the active device.");
}

DeviceAllocationUnique VulkanOpacityMicromap::allocate(std::size_t bytes,
                                                       AllocUsage usage,
                                                       bool host_write) {
  Device::AllocParams params;
  params.size = bytes;
  params.host_write = host_write;
  params.usage = usage;
  auto [allocation, status] = device_->allocate_memory_unique(params);
  TI_ERROR_IF(status != RhiResult::success,
              "Vulkan opacity micromap allocation failed ({} bytes): {}.",
              bytes, status);
  return std::move(allocation);
}

void VulkanOpacityMicromap::build(const std::string &data,
                                  const std::string &descriptors,
                                  const std::string &indices,
                                  bool indexed,
                                  std::size_t triangle_count) {
  static_assert(sizeof(VkMicromapTriangleEXT) == 8);
  TI_ERROR_IF(descriptors.size() % 8 || triangle_count == 0 ||
                  triangle_count > UINT32_MAX ||
                  descriptors.size() / 8 > UINT32_MAX ||
                  (indexed ? indices.size() != triangle_count * 4
                           : (!indices.empty() ||
                              descriptors.size() / 8 != triangle_count)),
              "Invalid Vulkan micromap descriptor or triangle-index count.");
  VkPhysicalDeviceOpacityMicromapPropertiesEXT limits{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR as_limits{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  limits.pNext = &as_limits;
  VkPhysicalDeviceProperties2 properties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &limits;
  vkGetPhysicalDeviceProperties2(device_->vk_physical_device(), &properties);
  Counts build_counts, used_counts;
  std::vector<VkMicromapTriangleEXT> entries(descriptors.size() / 8);
  if (!entries.empty()) {
    std::memcpy(entries.data(), descriptors.data(), descriptors.size());
  }
  for (const auto &entry : entries) {
    const auto level_limit =
        entry.format == VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT
            ? limits.maxOpacity2StateSubdivisionLevel
            : limits.maxOpacity4StateSubdivisionLevel;
    TI_ERROR_IF((entry.format != VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT &&
                 entry.format != VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT) ||
                    entry.subdivisionLevel > level_limit ||
                    entry.subdivisionLevel > 31,
                "Unsupported Vulkan micromap format or subdivision level.");
    const auto bits =
        (std::uint64_t(1) << (2 * entry.subdivisionLevel)) *
        (entry.format == VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT ? 2 : 1);
    TI_ERROR_IF(entry.dataOffset > data.size() ||
                    (bits + 7) / 8 > data.size() - entry.dataOffset,
                "Vulkan micromap descriptor exceeds baked data bounds.");
    ++build_counts[{entry.subdivisionLevel, entry.format}];
  }
  if (indexed) {
    for (std::size_t i = 0; i < triangle_count; ++i) {
      std::int32_t index;
      std::memcpy(&index, indices.data() + 4 * i, 4);
      TI_ERROR_IF(index < -4 || (index >= 0 && static_cast<std::size_t>(
                                                   index) >= entries.size()),
                  "Vulkan micromap triangle index is out of range.");
      if (index >= 0) {
        const auto &entry = entries[index];
        ++used_counts[{entry.subdivisionLevel, entry.format}];
      }
    }
  } else {
    used_counts = build_counts;
  }
  TI_ERROR_IF(entries.empty() && !data.empty(),
              "Empty Vulkan micromap descriptors require empty data.");
  usage_ = counts(used_counts);
  auto build_usage = counts(build_counts);
  attachment.indexType =
      indexed ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_NONE_KHR;
  attachment.indexStride = indexed ? 4 : 0;
  attachment.usageCountsCount = static_cast<std::uint32_t>(usage_.size());
  attachment.pUsageCounts = usage_.data();

  auto *stream = device_->get_compute_stream();
  auto [commands, command_status] = stream->new_command_list_unique();
  TI_ERROR_IF(command_status != RhiResult::success,
              "Cannot record Vulkan micromap import.");
  auto command_buffer = static_cast<vulkan::VulkanCommandList *>(commands.get())
                            ->vk_command_buffer();
  const auto device = device_->vk_device();
  auto barrier2 = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(
      vkGetDeviceProcAddr(device, "vkCmdPipelineBarrier2KHR"));
  if (!barrier2) {
    barrier2 = load<PFN_vkCmdPipelineBarrier2>(device, "vkCmdPipelineBarrier2");
  }
  auto barrier = [&](VkPipelineStageFlags2 source, VkAccessFlags2 source_access,
                     VkPipelineStageFlags2 destination,
                     VkAccessFlags2 destination_access) {
    VkMemoryBarrier2 memory_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory_barrier.srcStageMask = source;
    memory_barrier.srcAccessMask = source_access;
    memory_barrier.dstStageMask = destination;
    memory_barrier.dstAccessMask = destination_access;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory_barrier;
    barrier2(command_buffer->buffer, &dependency);
  };

  // Inputs are host-written once, submitted once and freed after the cold
  // build.
  std::vector<DeviceAllocationUnique> temporary;
  auto upload = [&](const std::string &bytes, AllocUsage usage,
                    std::size_t alignment) {
    auto allocation = allocate(bytes.size() + alignment,
                               usage | AllocUsage::DeviceAddress, true);
    const auto base = device_->get_buffer_device_address(*allocation);
    const auto address = aligned(base, alignment);
    void *mapped = nullptr;
    const auto status = device_->map(*allocation, &mapped);
    TI_ERROR_IF(status != RhiResult::success,
                "Cannot map Vulkan micromap input.");
    std::memcpy(static_cast<char *>(mapped) + (address - base), bytes.data(),
                bytes.size());
    device_->unmap(*allocation);
    memory[2] += bytes.size() + alignment;
    temporary.push_back(std::move(allocation));
    return address;
  };
  if (indexed) {
    indices_ =
        allocate(indices.size(), AllocUsage::AccelerationStructureBuildInput |
                                     AllocUsage::DeviceAddress);
    const auto address = upload(indices, AllocUsage::Upload, 4);
    const auto &staging = temporary.back();
    commands->buffer_copy(
        indices_->get_ptr(),
        staging->get_ptr(address -
                         device_->get_buffer_device_address(*staging)),
        indices.size());
    memory[1] = indices.size();
    attachment.indexBuffer.deviceAddress =
        device_->get_buffer_device_address(*indices_);
  }
  if (!entries.empty()) {
    VkMicromapBuildInfoEXT build{VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
    build.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
    build.flags = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
    build.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
    build.usageCountsCount = static_cast<std::uint32_t>(build_usage.size());
    build.pUsageCounts = build_usage.data();
    build.triangleArrayStride = 8;
    VkMicromapBuildSizesInfoEXT sizes{
        VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT};
    load<PFN_vkGetMicromapBuildSizesEXT>(device, "vkGetMicromapBuildSizesEXT")(
        device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
        &sizes);
    memory[0] = sizes.micromapSize;
    storage_ = allocate(memory[0], AllocUsage::MicromapStorage);
    handle_ = std::make_shared<Handle>();
    handle_->device = device;
    handle_->destroy =
        load<PFN_vkDestroyMicromapEXT>(device, "vkDestroyMicromapEXT");
    handle_->storage = device_->get_vkbuffer(storage_->get_ptr());
    VkMicromapCreateInfoEXT create{VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT};
    create.buffer = handle_->storage->buffer;
    create.size = memory[0];
    create.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
    const auto result =
        load<PFN_vkCreateMicromapEXT>(device, "vkCreateMicromapEXT")(
            device, &create, nullptr, &handle_->micromap);
    TI_ERROR_IF(result != VK_SUCCESS, "Cannot create Vulkan micromap: {}.",
                static_cast<int>(result));
    attachment.micromap = handle_->micromap;
    build.dstMicromap = handle_->micromap;
    build.data.deviceAddress =
        upload(data, AllocUsage::MicromapBuildInput, 256);
    build.triangleArray.deviceAddress =
        upload(descriptors, AllocUsage::MicromapBuildInput, 256);
    if (sizes.buildScratchSize) {
      const auto bytes =
          sizes.buildScratchSize +
          as_limits.minAccelerationStructureScratchOffsetAlignment;
      auto scratch =
          allocate(bytes, AllocUsage::Storage | AllocUsage::DeviceAddress);
      build.scratchData.deviceAddress =
          aligned(device_->get_buffer_device_address(*scratch),
                  as_limits.minAccelerationStructureScratchOffsetAlignment);
      memory[2] += bytes;
      temporary.push_back(std::move(scratch));
    }
    barrier(VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_WRITE_BIT,
            VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT,
            VK_ACCESS_2_SHADER_READ_BIT);
    load<PFN_vkCmdBuildMicromapsEXT>(device, "vkCmdBuildMicromapsEXT")(
        command_buffer->buffer, 1, &build);
  }
  barrier(VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT |
              VK_PIPELINE_STAGE_2_TRANSFER_BIT,
          VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
          VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
          VK_ACCESS_2_MICROMAP_READ_BIT_EXT |
              VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
  stream->submit_synced(commands.get());
}

void VulkanOpacityMicromap::retain(
    const vkapi::IVkCommandBuffer &commands) const {
  if (handle_)
    commands->refs.push_back(handle_);
  if (indices_)
    commands->refs.push_back(device_->get_vkbuffer(indices_->get_ptr()));
}
}  // namespace taichi::lang
#endif
