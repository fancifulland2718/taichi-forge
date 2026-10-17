#include "taichi/runtime/vulkan_ray_blas.h"

#if defined(TI_WITH_VULKAN)
#include "taichi/runtime/vulkan_ray_build.h"
#include "taichi/runtime/vulkan_micromap.h"
#include <algorithm>
#include <array>
#include <limits>

namespace taichi::lang {
using namespace vulkan_ray_detail;
struct VulkanTriangleBlas::State {
 public:
  State(vulkan::VulkanDevice *device,
        std::size_t vertex_count,
        std::size_t triangle_count,
        std::shared_ptr<VulkanOpacityMicromap> micromap = {},
        bool opaque = true)
      : device_(device),
        vertex_count_(vertex_count),
        triangle_count_(triangle_count),
        micromap_(std::move(micromap)) {
    opaque_ = !micromap_ && opaque;
    TI_ERROR_IF(device_ == nullptr ||
                    !device_->vk_caps().acceleration_structure ||
                    !device_->vk_caps().buffer_device_address,
                "Vulkan triangle BLAS requires acceleration-structure and "
                "device-address support.");
    TI_ERROR_IF(
        vertex_count_ == 0 ||
            vertex_count_ > static_cast<std::size_t>(
                                (std::numeric_limits<std::uint32_t>::max)()),
        "Vulkan triangle BLAS vertex_count must be in [1, "
        "UINT32_MAX].");
    TI_ERROR_IF(
        triangle_count_ == 0 ||
            triangle_count_ > static_cast<std::size_t>(
                                  (std::numeric_limits<std::uint32_t>::max)()),
        "Vulkan triangle BLAS triangle_count must be in [1, "
        "UINT32_MAX].");
    get_build_sizes_ = load_vulkan_device_function<
        PFN_vkGetAccelerationStructureBuildSizesKHR>(
        device_->vk_device(), "vkGetAccelerationStructureBuildSizesKHR");
    get_as_address_ = load_vulkan_device_function<
        PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        device_->vk_device(), "vkGetAccelerationStructureDeviceAddressKHR");
    cmd_build_ =
        load_vulkan_device_function<PFN_vkCmdBuildAccelerationStructuresKHR>(
            device_->vk_device(), "vkCmdBuildAccelerationStructuresKHR");

    VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties{};
    as_properties.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &as_properties;
    device_->vk_device().instance_functions().vkGetPhysicalDeviceProperties2(
        device_->vk_physical_device(), &properties);
    TI_ERROR_IF(triangle_count_ > as_properties.maxPrimitiveCount ||
                    as_properties.maxGeometryCount == 0,
                "Vulkan triangle BLAS exceeds the device's AS geometry or "
                "primitive capacity.");
    scratch_alignment_ =
        as_properties.minAccelerationStructureScratchOffsetAlignment;
    TI_ERROR_IF(scratch_alignment_ == 0,
                "Vulkan triangle BLAS reported zero scratch alignment.");

    vertex_bytes_ = checked_mul(vertex_count_, 3 * sizeof(float), "vertex");
    index_bytes_ =
        checked_mul(triangle_count_, 3 * sizeof(std::uint32_t), "index");
    try {
      vertex_buffer_ =
          allocate(vertex_bytes_, AllocUsage::AccelerationStructureBuildInput |
                                      AllocUsage::DeviceAddress);
      index_buffer_ =
          allocate(index_bytes_, AllocUsage::AccelerationStructureBuildInput |
                                     AllocUsage::DeviceAddress);
      create_acceleration_structure();
    } catch (...) {
      blas_.reset();
      release(scratch_);
      release(storage_);
      release(index_buffer_);
      release(vertex_buffer_);
      throw;
    }
  }

  ~State() {
    blas_.reset();
    release(scratch_);
    release(storage_);
    release(index_buffer_);
    release(vertex_buffer_);
  }

  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const {
    VulkanTriangleRaySceneMemoryStatistics result;
    result.geometry_input_requested_bytes = vertex_bytes_ + index_bytes_;
    result.acceleration_structure_requested_bytes = storage_bytes_;
    result.build_scratch_requested_bytes = scratch_bytes_;
    if (micromap_) {
      result.geometry_input_requested_bytes += micromap_->memory[1];
      result.acceleration_structure_requested_bytes += micromap_->memory[0];
    }
    result.known_requested_bytes =
        result.geometry_input_requested_bytes +
        result.acceleration_structure_requested_bytes +
        result.build_scratch_requested_bytes;
    result.known_allocation_count = 4;
    if (micromap_) {
      result.known_allocation_count +=
          (micromap_->memory[0] != 0) + (micromap_->memory[1] != 0);
    }
    return result;
  }

  vulkan::VulkanDevice *device() const {
    return device_;
  }

  std::size_t vertex_count() const {
    return vertex_count_;
  }

  std::size_t triangle_count() const {
    return triangle_count_;
  }

  VkDeviceAddress device_address() const {
    VkAccelerationStructureDeviceAddressInfoKHR info{};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    info.accelerationStructure = blas_->accel;
    const VkDeviceAddress address =
        get_as_address_(device_->vk_device(), &info);
    TI_ERROR_IF(address == 0,
                "Vulkan triangle BLAS returned a null device address.");
    return address;
  }

  void record_build(CommandList *command_list,
                    DevicePtr source_vertices,
                    DevicePtr source_indices,
                    bool update) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(command_list);
    auto command_buffer = vk_commands->vk_command_buffer();

    geometry_reuse_barrier(command_buffer);

    command_list->buffer_copy(vertex_buffer_.get_ptr(), source_vertices,
                              vertex_bytes_);
    if (!update) {
      command_list->buffer_copy(index_buffer_.get_ptr(), source_indices,
                                index_bytes_);
    }

    VkMemoryBarrier input_barrier{};
    input_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    input_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &input_barrier, 0, nullptr, 0, nullptr);

    const auto geometry = make_geometry();
    auto build = make_build_info(
        geometry, update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                         : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR);
    const VkAccelerationStructureBuildRangeInfoKHR range{
        static_cast<std::uint32_t>(triangle_count_), 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges[] = {&range};
    cmd_build_(command_buffer->buffer, 1, &build, ranges);

    VkMemoryBarrier completion_barrier{};
    completion_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    completion_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    completion_barrier.dstAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &completion_barrier, 0, nullptr, 0, nullptr);
    retain(command_buffer);
  }

  void retain(const vkapi::IVkCommandBuffer &command_buffer) const {
    const std::array<DeviceAllocation, 4> allocations{
        vertex_buffer_, index_buffer_, storage_, scratch_};
    for (const auto allocation : allocations) {
      command_buffer->refs.push_back(
          device_->get_vkbuffer(allocation.get_ptr()));
    }
    command_buffer->refs.push_back(blas_);
    if (micromap_) {
      micromap_->retain(command_buffer);
    }
  }

 private:
  DeviceAllocation allocate(std::size_t bytes, AllocUsage usage) {
    Device::AllocParams params;
    params.size = bytes;
    params.usage = usage;
    DeviceAllocation allocation{kDeviceNullAllocation};
    const auto result = device_->allocate_memory(params, &allocation);
    TI_ERROR_IF(result != RhiResult::success,
                "Failed to allocate Vulkan triangle BLAS buffer ({} bytes): "
                "RhiResult({}).",
                bytes, result);
    return allocation;
  }

  void release(DeviceAllocation &allocation) noexcept {
    if (allocation != kDeviceNullAllocation && device_ != nullptr) {
      device_->dealloc_memory(allocation);
      allocation = kDeviceNullAllocation;
    }
  }

  VkDeviceAddress scratch_address() const {
    return aligned_address(device_->get_buffer_device_address(scratch_),
                           scratch_alignment_);
  }

  std::size_t scratch_allocation_bytes(VkDeviceSize bytes) const {
    TI_ERROR_IF(
        bytes > (std::numeric_limits<std::size_t>::max)() - scratch_alignment_,
        "Vulkan triangle BLAS scratch size overflow.");
    return static_cast<std::size_t>(bytes + scratch_alignment_);
  }

  VkAccelerationStructureGeometryKHR make_geometry() const {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.pNext = micromap_ ? &micromap_->attachment : nullptr;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress =
        device_->get_buffer_device_address(vertex_buffer_);
    triangles.vertexStride = 3 * sizeof(float);
    triangles.maxVertex = static_cast<std::uint32_t>(vertex_count_ - 1);
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress =
        device_->get_buffer_device_address(index_buffer_);
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = opaque_ ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
    geometry.geometry.triangles = triangles;
    return geometry;
  }

  VkAccelerationStructureBuildGeometryInfoKHR make_build_info(
      const VkAccelerationStructureGeometryKHR &geometry,
      VkBuildAccelerationStructureModeKHR mode) const {
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                 VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    info.mode = mode;
    if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
      info.srcAccelerationStructure = blas_->accel;
    }
    info.dstAccelerationStructure = blas_->accel;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    info.scratchData.deviceAddress = scratch_address();
    return info;
  }

  void create_acceleration_structure() {
    const auto geometry = make_geometry();
    VkAccelerationStructureBuildGeometryInfoKHR build{};
    build.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                  VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geometry;
    const auto primitive_count = static_cast<std::uint32_t>(triangle_count_);
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    get_build_sizes_(device_->vk_device(),
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                     &primitive_count, &sizes);
    storage_bytes_ = static_cast<std::size_t>(sizes.accelerationStructureSize);
    storage_ =
        allocate(storage_bytes_, AllocUsage::AccelerationStructureStorage);
    blas_ = vkapi::create_acceleration_structure(
        0, device_->get_vkbuffer(storage_.get_ptr()), 0,
        sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
    TI_ERROR_IF(!blas_, "Failed to create independent Vulkan triangle BLAS.");
    scratch_bytes_ = scratch_allocation_bytes(
        std::max(sizes.buildScratchSize, sizes.updateScratchSize));
    scratch_ = allocate(scratch_bytes_,
                        AllocUsage::Storage | AllocUsage::DeviceAddress);
  }

  vulkan::VulkanDevice *device_{nullptr};
  std::size_t vertex_count_{0};
  std::size_t triangle_count_{0};
  std::size_t vertex_bytes_{0};
  std::size_t index_bytes_{0};
  std::size_t storage_bytes_{0};
  std::size_t scratch_bytes_{0};
  VkDeviceSize scratch_alignment_{1};
  DeviceAllocation vertex_buffer_{kDeviceNullAllocation};
  DeviceAllocation index_buffer_{kDeviceNullAllocation};
  DeviceAllocation storage_{kDeviceNullAllocation};
  DeviceAllocation scratch_{kDeviceNullAllocation};
  vkapi::IVkAccelerationStructureKHR blas_{nullptr};
  PFN_vkGetAccelerationStructureBuildSizesKHR get_build_sizes_{nullptr};
  PFN_vkGetAccelerationStructureDeviceAddressKHR get_as_address_{nullptr};
  PFN_vkCmdBuildAccelerationStructuresKHR cmd_build_{nullptr};
  std::mutex mutex_;
  std::shared_ptr<VulkanOpacityMicromap> micromap_;
  bool opaque_{true};
};

VulkanTriangleBlas::VulkanTriangleBlas(
    vulkan::VulkanDevice *device,
    std::size_t vertices,
    std::size_t triangles,
    std::shared_ptr<VulkanOpacityMicromap> micromap,
    bool opaque)
    : state_(std::make_unique<State>(device,
                                     vertices,
                                     triangles,
                                     std::move(micromap),
                                     opaque)) {
}
VulkanTriangleBlas::~VulkanTriangleBlas() = default;
vulkan::VulkanDevice *VulkanTriangleBlas::device() const {
  return state_->device();
}
VulkanTriangleRaySceneMemoryStatistics VulkanTriangleBlas::memory_statistics()
    const {
  return state_->memory_statistics();
}
std::size_t VulkanTriangleBlas::vertex_count() const {
  return state_->vertex_count();
}
std::size_t VulkanTriangleBlas::triangle_count() const {
  return state_->triangle_count();
}
VkDeviceAddress VulkanTriangleBlas::device_address() const {
  return state_->device_address();
}
void VulkanTriangleBlas::record_build(CommandList *commands,
                                      DevicePtr vertices,
                                      DevicePtr indices,
                                      bool update) {
  state_->record_build(commands, vertices, indices, update);
}
void VulkanTriangleBlas::retain(const vkapi::IVkCommandBuffer &commands) const {
  state_->retain(commands);
}
}  // namespace taichi::lang
#endif
