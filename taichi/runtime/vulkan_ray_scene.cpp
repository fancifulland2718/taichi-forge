#include "taichi/runtime/vulkan_ray_scene.h"

#if defined(TI_WITH_VULKAN)
#include <algorithm>
#include <cstring>
#include <mutex>
#include "taichi/runtime/vulkan_ray_build.h"
#include "taichi/runtime/vulkan_ray_query.h"

namespace taichi::lang {
using namespace vulkan_ray_detail;
struct VulkanTriangleRayScene::State {
  State(vulkan::VulkanDevice *device,
        std::size_t vertex_count,
        std::size_t triangle_count)
      : device_(device),
        vertex_count_(vertex_count),
        triangle_count_(triangle_count) {
  }

  // Initialization runs only after state_ owns this complete object. Allocation
  // or pipeline failures therefore release every successfully created resource.
  void initialize() {
    TI_ERROR_IF(device_ == nullptr ||
                    !device_->vk_caps().acceleration_structure ||
                    !device_->vk_caps().ray_query,
                "Vulkan triangle ray scene requires acceleration-structure "
                "and ray-query support.");
    TI_ERROR_IF(
        vertex_count_ == 0 ||
            vertex_count_ > static_cast<std::size_t>(
                                (std::numeric_limits<std::uint32_t>::max)()),
        "Vulkan triangle ray vertex_count must be in [1, UINT32_MAX].");
    TI_ERROR_IF(
        triangle_count_ == 0 ||
            triangle_count_ > static_cast<std::size_t>(
                                  (std::numeric_limits<std::uint32_t>::max)()),
        "Vulkan triangle ray triangle_count must be in [1, "
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
                    as_properties.maxGeometryCount == 0 ||
                    as_properties.maxInstanceCount == 0,
                "Vulkan triangle scene exceeds the device's AS geometry, "
                "primitive or instance capacity.");
    scratch_alignment_ =
        as_properties.minAccelerationStructureScratchOffsetAlignment;
    TI_ERROR_IF(scratch_alignment_ == 0,
                "Vulkan ray provider reported zero scratch alignment.");

    vertex_bytes_ = checked_mul(vertex_count_, 3 * sizeof(float), "vertex");
    index_bytes_ =
        checked_mul(triangle_count_, 3 * sizeof(std::uint32_t), "index");
    vertex_buffer_ =
        allocate(vertex_bytes_, AllocUsage::AccelerationStructureBuildInput |
                                    AllocUsage::DeviceAddress);
    index_buffer_ =
        allocate(index_bytes_, AllocUsage::AccelerationStructureBuildInput |
                                   AllocUsage::DeviceAddress);

    create_blas();
    create_tlas();
  }

  ~State() {
    query_.clear();
    tlas_.reset();
    blas_.reset();
    release(tlas_scratch_);
    release(blas_scratch_);
    release(tlas_storage_);
    release(blas_storage_);
    release(instance_buffer_);
    release(index_buffer_);
    release(vertex_buffer_);
  }

  State(const State &) = delete;
  State &operator=(const State &) = delete;

  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const {
    VulkanTriangleRaySceneMemoryStatistics result;
    result.geometry_input_requested_bytes =
        vertex_bytes_ + index_bytes_ + instance_buffer_bytes_;
    result.acceleration_structure_requested_bytes =
        blas_storage_bytes_ + tlas_storage_bytes_;
    result.build_scratch_requested_bytes =
        blas_scratch_bytes_ + tlas_scratch_bytes_;
    result.known_requested_bytes =
        result.geometry_input_requested_bytes +
        result.acceleration_structure_requested_bytes +
        result.build_scratch_requested_bytes;
    result.known_allocation_count = 7;
    return result;
  }

  void record_build(CommandList *command_list,
                    DevicePtr source_vertices,
                    DevicePtr source_indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(command_list);
    auto command_buffer = vk_commands->vk_command_buffer();

    geometry_reuse_barrier(command_buffer);
    command_list->buffer_copy(vertex_buffer_.get_ptr(), source_vertices,
                              vertex_bytes_);
    command_list->buffer_copy(index_buffer_.get_ptr(), source_indices,
                              index_bytes_);

    VkMemoryBarrier input_barrier{};
    input_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    // Vertex/index/instance inputs are shader reads in the AS build stage;
    // AS_READ instead describes AS storage and scratch, not geometry inputs.
    input_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &input_barrier, 0, nullptr, 0, nullptr);

    const auto blas_geometry = make_blas_geometry();
    auto blas_build = make_blas_build_info(blas_geometry);
    const VkAccelerationStructureBuildRangeInfoKHR blas_range{
        static_cast<std::uint32_t>(triangle_count_), 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *blas_ranges[] = {
        &blas_range};
    cmd_build_(command_buffer->buffer, 1, &blas_build, blas_ranges);

    VkMemoryBarrier blas_barrier{};
    blas_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    blas_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    blas_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &blas_barrier, 0, nullptr, 0, nullptr);

    const auto tlas_geometry = make_tlas_geometry();
    auto tlas_build = make_tlas_build_info(tlas_geometry);
    const VkAccelerationStructureBuildRangeInfoKHR tlas_range{1, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *tlas_ranges[] = {
        &tlas_range};
    cmd_build_(command_buffer->buffer, 1, &tlas_build, tlas_ranges);

    VkMemoryBarrier query_barrier{};
    query_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    query_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    query_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &query_barrier, 0, nullptr,
        0, nullptr);

    retain_build_resources(command_buffer);
  }

  void record_refit(CommandList *command_list, DevicePtr source_vertices) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(command_list);
    auto command_buffer = vk_commands->vk_command_buffer();

    geometry_reuse_barrier(command_buffer);
    command_list->buffer_copy(vertex_buffer_.get_ptr(), source_vertices,
                              vertex_bytes_);

    VkMemoryBarrier input_barrier{};
    input_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask =
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    input_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &input_barrier, 0, nullptr, 0, nullptr);

    const auto geometry = make_blas_geometry();
    auto build = make_blas_build_info(
        geometry, VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR);
    const VkAccelerationStructureBuildRangeInfoKHR range{
        static_cast<std::uint32_t>(triangle_count_), 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges[] = {&range};
    cmd_build_(command_buffer->buffer, 1, &build, ranges);

    // The BLAS may move outside the bounds encoded by the existing TLAS.
    // This owner contains both structures, so refit must refresh both. Reuse
    // its one-instance build storage/scratch; no host bounds readback is
    // needed.
    VkMemoryBarrier blas_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    blas_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    blas_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &blas_barrier, 0, nullptr, 0, nullptr);
    const auto tlas_geometry = make_tlas_geometry();
    auto tlas_build = make_tlas_build_info(tlas_geometry);
    const VkAccelerationStructureBuildRangeInfoKHR tlas_range{1, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *tlas_ranges[] = {
        &tlas_range};
    cmd_build_(command_buffer->buffer, 1, &tlas_build, tlas_ranges);

    VkMemoryBarrier query_barrier{};
    query_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    query_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    query_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &query_barrier, 0, nullptr,
        0, nullptr);

    retain_build_resources(command_buffer);
  }

  std::size_t vertex_count() const {
    return vertex_count_;
  }

  std::size_t triangle_count() const {
    return triangle_count_;
  }

  void prepare_query_variant(unsigned variant) {
    std::lock_guard<std::mutex> lock(mutex_);
    query_.prepare(device_, variant);
  }

  void record_query(CommandList *command_list,
                    const VulkanRayQueryPacket &packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    query_.record(command_list, tlas_, packet);
  }

 private:
  DeviceAllocation allocate(std::size_t bytes,
                            AllocUsage usage,
                            bool host_write = false) {
    Device::AllocParams params;
    params.size = bytes;
    params.host_write = host_write;
    params.usage = usage;
    DeviceAllocation allocation{kDeviceNullAllocation};
    const auto result = device_->allocate_memory(params, &allocation);
    TI_ERROR_IF(result != RhiResult::success,
                "Failed to allocate Vulkan ray buffer ({} bytes): "
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

  VkDeviceAddress scratch_address(DeviceAllocation allocation) const {
    return aligned_address(device_->get_buffer_device_address(allocation),
                           scratch_alignment_);
  }

  std::size_t scratch_allocation_bytes(VkDeviceSize bytes) const {
    TI_ERROR_IF(
        bytes > (std::numeric_limits<std::size_t>::max)() - scratch_alignment_,
        "Vulkan ray scratch size overflow.");
    return static_cast<std::size_t>(bytes + scratch_alignment_);
  }

  VkAccelerationStructureGeometryKHR make_blas_geometry() const {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
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
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles = triangles;
    return geometry;
  }

  VkAccelerationStructureBuildGeometryInfoKHR make_blas_build_info(
      const VkAccelerationStructureGeometryKHR &geometry,
      VkBuildAccelerationStructureModeKHR mode =
          VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR) const {
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                 VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    info.mode = mode;
    if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
      info.srcAccelerationStructure = blas_ ? blas_->accel : VK_NULL_HANDLE;
    }
    info.dstAccelerationStructure = blas_ ? blas_->accel : VK_NULL_HANDLE;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    if (blas_scratch_ != kDeviceNullAllocation) {
      info.scratchData.deviceAddress = scratch_address(blas_scratch_);
    }
    return info;
  }

  VkAccelerationStructureGeometryKHR make_tlas_geometry() const {
    VkAccelerationStructureGeometryInstancesDataKHR instances{};
    instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instances.arrayOfPointers = VK_FALSE;
    instances.data.deviceAddress =
        device_->get_buffer_device_address(instance_buffer_);

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instances;
    return geometry;
  }

  VkAccelerationStructureBuildGeometryInfoKHR make_tlas_build_info(
      const VkAccelerationStructureGeometryKHR &geometry) const {
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.dstAccelerationStructure = tlas_ ? tlas_->accel : VK_NULL_HANDLE;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    if (tlas_scratch_ != kDeviceNullAllocation) {
      info.scratchData.deviceAddress = scratch_address(tlas_scratch_);
    }
    return info;
  }

  void create_blas() {
    const auto geometry = make_blas_geometry();
    auto build_info = make_blas_build_info(geometry);
    const auto primitive_count = static_cast<std::uint32_t>(triangle_count_);
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    get_build_sizes_(device_->vk_device(),
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &build_info, &primitive_count, &sizes);
    blas_storage_bytes_ =
        static_cast<std::size_t>(sizes.accelerationStructureSize);
    blas_storage_ =
        allocate(blas_storage_bytes_, AllocUsage::AccelerationStructureStorage);
    blas_ = vkapi::create_acceleration_structure(
        0, device_->get_vkbuffer(blas_storage_.get_ptr()), 0,
        sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
    TI_ERROR_IF(!blas_, "Failed to create Vulkan triangle BLAS.");
    blas_scratch_bytes_ = scratch_allocation_bytes(
        std::max(sizes.buildScratchSize, sizes.updateScratchSize));
    blas_scratch_ = allocate(blas_scratch_bytes_,
                             AllocUsage::Storage | AllocUsage::DeviceAddress);
  }

  void create_tlas() {
    VkAccelerationStructureDeviceAddressInfoKHR address_info{};
    address_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address_info.accelerationStructure = blas_->accel;
    const VkDeviceAddress blas_address =
        get_as_address_(device_->vk_device(), &address_info);
    TI_ERROR_IF(blas_address == 0,
                "Vulkan triangle BLAS returned a null device address.");

    instance_buffer_bytes_ = sizeof(VkAccelerationStructureInstanceKHR);
    instance_buffer_ = allocate(
        instance_buffer_bytes_,
        AllocUsage::AccelerationStructureBuildInput | AllocUsage::DeviceAddress,
        true);
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform.matrix[0][0] = 1.0f;
    instance.transform.matrix[1][1] = 1.0f;
    instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xff;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = blas_address;
    void *mapped = nullptr;
    const auto map_result = device_->map(instance_buffer_, &mapped);
    TI_ERROR_IF(map_result != RhiResult::success || mapped == nullptr,
                "Failed to map Vulkan TLAS instance buffer: RhiResult({}).",
                map_result);
    std::memcpy(mapped, &instance, sizeof(instance));
    device_->unmap(instance_buffer_);

    const auto geometry = make_tlas_geometry();
    auto build_info = make_tlas_build_info(geometry);
    const std::uint32_t primitive_count = 1;
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    get_build_sizes_(device_->vk_device(),
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                     &build_info, &primitive_count, &sizes);
    tlas_storage_bytes_ =
        static_cast<std::size_t>(sizes.accelerationStructureSize);
    tlas_storage_ =
        allocate(tlas_storage_bytes_, AllocUsage::AccelerationStructureStorage);
    tlas_ = vkapi::create_acceleration_structure(
        0, device_->get_vkbuffer(tlas_storage_.get_ptr()), 0,
        sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
    TI_ERROR_IF(!tlas_, "Failed to create Vulkan triangle TLAS.");
    tlas_scratch_bytes_ = scratch_allocation_bytes(sizes.buildScratchSize);
    tlas_scratch_ = allocate(tlas_scratch_bytes_,
                             AllocUsage::Storage | AllocUsage::DeviceAddress);
  }

  void retain_build_resources(
      const vkapi::IVkCommandBuffer &command_buffer) const {
    const std::array<DeviceAllocation, 7> allocations{
        vertex_buffer_, index_buffer_, instance_buffer_, blas_storage_,
        tlas_storage_,  blas_scratch_, tlas_scratch_};
    for (const auto allocation : allocations) {
      command_buffer->refs.push_back(
          device_->get_vkbuffer(allocation.get_ptr()));
    }
    command_buffer->refs.push_back(blas_);
    command_buffer->refs.push_back(tlas_);
  }

  vulkan::VulkanDevice *device_{nullptr};
  std::size_t vertex_count_{0};
  std::size_t triangle_count_{0};
  std::size_t vertex_bytes_{0};
  std::size_t index_bytes_{0};
  std::size_t instance_buffer_bytes_{0};
  std::size_t blas_storage_bytes_{0};
  std::size_t tlas_storage_bytes_{0};
  std::size_t blas_scratch_bytes_{0};
  std::size_t tlas_scratch_bytes_{0};
  VkDeviceSize scratch_alignment_{1};
  DeviceAllocation vertex_buffer_{kDeviceNullAllocation};
  DeviceAllocation index_buffer_{kDeviceNullAllocation};
  DeviceAllocation instance_buffer_{kDeviceNullAllocation};
  DeviceAllocation blas_storage_{kDeviceNullAllocation};
  DeviceAllocation tlas_storage_{kDeviceNullAllocation};
  DeviceAllocation blas_scratch_{kDeviceNullAllocation};
  DeviceAllocation tlas_scratch_{kDeviceNullAllocation};
  vkapi::IVkAccelerationStructureKHR blas_{nullptr};
  vkapi::IVkAccelerationStructureKHR tlas_{nullptr};
  VulkanRayQueryPipelines query_;
  PFN_vkGetAccelerationStructureBuildSizesKHR get_build_sizes_{nullptr};
  PFN_vkGetAccelerationStructureDeviceAddressKHR get_as_address_{nullptr};
  PFN_vkCmdBuildAccelerationStructuresKHR cmd_build_{nullptr};
  std::mutex mutex_;
};

VulkanTriangleRayScene::VulkanTriangleRayScene(vulkan::VulkanDevice *device,
                                               std::size_t vertex_count,
                                               std::size_t triangle_count)
    : state_(std::make_unique<State>(device, vertex_count, triangle_count)) {
  state_->initialize();
}
VulkanTriangleRayScene::~VulkanTriangleRayScene() = default;
VulkanTriangleRaySceneMemoryStatistics
VulkanTriangleRayScene::memory_statistics() const {
  return state_->memory_statistics();
}
std::size_t VulkanTriangleRayScene::vertex_count() const {
  return state_->vertex_count();
}
std::size_t VulkanTriangleRayScene::triangle_count() const {
  return state_->triangle_count();
}
void VulkanTriangleRayScene::record_build(CommandList *commands,
                                          DevicePtr vertices,
                                          DevicePtr indices) {
  state_->record_build(commands, vertices, indices);
}
void VulkanTriangleRayScene::record_refit(CommandList *commands,
                                          DevicePtr vertices) {
  state_->record_refit(commands, vertices);
}
void VulkanTriangleRayScene::prepare_query_variant(unsigned variant) {
  state_->prepare_query_variant(variant);
}
void VulkanTriangleRayScene::record_query(CommandList *commands,
                                          const VulkanRayQueryPacket &packet) {
  state_->record_query(commands, packet);
}

}  // namespace taichi::lang
#endif
