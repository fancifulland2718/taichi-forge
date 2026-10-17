#include "taichi/runtime/vulkan_ray_tlas.h"
#if defined(TI_WITH_VULKAN)
#include "taichi/runtime/vulkan_ray_build.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>
namespace taichi::lang {
using namespace vulkan_ray_detail;
namespace {
constexpr std::uint32_t kRayTransformWorkgroupSize = 128;
static const std::uint32_t kRayInstanceTransformsSpv[] =
#include "taichi/program/vulkan_sort_shaders/ray_instance_transforms.comp.spv.h"
    ;
static_assert(sizeof(VkAccelerationStructureInstanceKHR) == 16 * sizeof(float));
static_assert(offsetof(VkAccelerationStructureInstanceKHR, transform) == 0);
}  // namespace
struct VulkanInstanceTlas::State {
 public:
  State(vulkan::VulkanDevice *device,
        std::vector<std::shared_ptr<VulkanTriangleBlas>> blases,
        std::vector<std::uint32_t> sbt_record_offsets)
      : device_(device),
        blases_(std::move(blases)),
        sbt_record_offsets_(std::move(sbt_record_offsets)) {
  }

  // State is already owned when allocations start, including failure cleanup.
  void initialize() {
    TI_ERROR_IF(blases_.empty(),
                "Vulkan instance TLAS requires at least one BLAS instance.");
    TI_ERROR_IF(sbt_record_offsets_.size() != blases_.size(),
                "Vulkan TLAS requires one frozen SBT offset per instance.");
    for (auto offset : sbt_record_offsets_) {
      TI_ERROR_IF(
          offset > 0xffffff,
          "Vulkan instance SBT record offset exceeds its 24-bit field.");
      max_sbt_record_offset_ = std::max(max_sbt_record_offset_, offset);
    }
    TI_ERROR_IF(device_ == nullptr ||
                    !device_->vk_caps().acceleration_structure ||
                    !device_->vk_caps().buffer_device_address,
                "Vulkan instance TLAS requires acceleration-structure and "
                "device-address support.");
    for (const auto &blas : blases_) {
      TI_ERROR_IF(!blas || blas->device() != device_,
                  "Vulkan TLAS requires live BLAS cores on the same device.");
    }
    get_build_sizes_ = load_vulkan_device_function<
        PFN_vkGetAccelerationStructureBuildSizesKHR>(
        device_->vk_device(), "vkGetAccelerationStructureBuildSizesKHR");
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
    scratch_alignment_ =
        as_properties.minAccelerationStructureScratchOffsetAlignment;
    TI_ERROR_IF(scratch_alignment_ == 0,
                "Vulkan instance TLAS reported zero scratch alignment.");
    TI_ERROR_IF(
        blases_.size() > as_properties.maxInstanceCount ||
            blases_.size() > static_cast<std::size_t>(
                                 (std::numeric_limits<std::uint32_t>::max)()),
        "Vulkan instance TLAS instance count exceeds the device "
        "limit.");

    instance_bytes_ =
        checked_mul(blases_.size(), sizeof(VkAccelerationStructureInstanceKHR),
                    "instance input");
    instance_buffer_ = allocate(
        instance_bytes_, AllocUsage::AccelerationStructureBuildInput |
                             AllocUsage::DeviceAddress | AllocUsage::Storage);
    create_acceleration_structure();
    // Instance topology remains ordered, but repeated instances of one BLAS
    // need only one set of lifetime references in each recorded command.
    std::unordered_set<VulkanTriangleBlas *> retained;
    for (const auto &blas : blases_) {
      if (retained.insert(blas.get()).second) {
        retained_blases_.push_back(blas);
      }
    }
  }

  ~State() {
    transform_bindings_.reset();
    transform_pipeline_.reset();
    query_.clear();
    tlas_.reset();
    release(scratch_);
    release(storage_);
    release(instance_buffer_);
  }

  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const {
    VulkanTriangleRaySceneMemoryStatistics result;
    result.geometry_input_requested_bytes = instance_bytes_;
    result.acceleration_structure_requested_bytes = storage_bytes_;
    result.build_scratch_requested_bytes = scratch_bytes_;
    result.known_requested_bytes =
        result.geometry_input_requested_bytes +
        result.acceleration_structure_requested_bytes +
        result.build_scratch_requested_bytes;
    result.known_allocation_count = 3;
    return result;
  }

  std::size_t instance_count() const {
    return blases_.size();
  }

  std::uint32_t max_sbt_record_offset() const {
    return max_sbt_record_offset_;
  }

  const std::vector<std::shared_ptr<VulkanTriangleBlas>> &blases() const {
    return blases_;
  }

  void record_build(CommandList *command_list,
                    const std::vector<VulkanRayInstanceInfo> &instances,
                    bool update) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(instances.size() != blases_.size(),
                "Vulkan instance TLAS build must preserve instance count {}.",
                blases_.size());
    std::vector<VkAccelerationStructureInstanceKHR> packed(instances.size());
    for (std::size_t index = 0; index < instances.size(); ++index) {
      const auto &source = instances[index];
      TI_ERROR_IF(source.mask > 0xff || source.custom_index > 0xffffff,
                  "Vulkan instance mask/custom index exceeds packed limits.");
      for (float value : source.transform) {
        TI_ERROR_IF(!std::isfinite(value),
                    "Vulkan instance transform must be finite.");
      }
      auto &destination = packed[index];
      std::memcpy(destination.transform.matrix, source.transform.data(),
                  sizeof(destination.transform.matrix));
      destination.instanceCustomIndex = source.custom_index;
      destination.mask = source.mask;
      destination.instanceShaderBindingTableRecordOffset =
          sbt_record_offsets_[index];
      destination.flags =
          VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      destination.accelerationStructureReference =
          blases_[index]->device_address();
    }

    auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(command_list);
    auto command_buffer = vk_commands->vk_command_buffer();
    geometry_reuse_barrier(command_buffer);

    const auto instance_vk_buffer =
        device_->get_vkbuffer(instance_buffer_.get_ptr());
    constexpr std::size_t kMaxUpdateBytes = 65536;
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(packed.data());
    for (std::size_t offset = 0; offset < instance_bytes_;) {
      const std::size_t chunk =
          std::min(kMaxUpdateBytes, instance_bytes_ - offset);
      command_buffer->device.functions().vkCmdUpdateBuffer(
          command_buffer->buffer, instance_vk_buffer->buffer, offset, chunk,
          bytes + offset);
      offset += chunk;
    }

    VkMemoryBarrier input_barrier{};
    input_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    input_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    input_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
        &input_barrier, 0, nullptr, 0, nullptr);

    record_acceleration_structure(command_buffer, update);
    initialized_ = true;
  }

  void prepare_transforms() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (transform_pipeline_) {
      return;
    }
    PipelineSourceDesc source{
        PipelineSourceType::spirv_binary, kRayInstanceTransformsSpv,
        sizeof(kRayInstanceTransformsSpv), PipelineStageType::compute};
    auto [pipeline, result] = device_->create_pipeline_unique(
        source, "vulkan_ray_instance_transforms");
    TI_ERROR_IF(result != RhiResult::success || !pipeline,
                "Failed to create Vulkan TLAS transform pipeline: {}.", result);
    std::unique_ptr<ShaderResourceSet> bindings(device_->create_resource_set());
    transform_pipeline_ = std::move(pipeline);
    transform_bindings_ = std::move(bindings);
  }

  void record_transforms(CommandList *commands,
                         const VulkanRayTransformPacket &packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(!initialized_,
                "Vulkan TLAS device transforms require an initial host build.");
    auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(commands);
    const auto command_buffer = vk_commands->vk_command_buffer();
    // Publish device producers and order reuse of the instance input, TLAS and
    // scratch after prior builds/queries, including repeated Graph executions.
    VkMemoryBarrier reuse{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    reuse.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                          VK_ACCESS_TRANSFER_WRITE_BIT |
                          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    reuse.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT |
                          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &reuse, 0, nullptr, 0, nullptr);

    transform_bindings_->rw_buffer(0, packet.shader_binding.pointer,
                                   packet.shader_binding.bytes);
    transform_bindings_->rw_buffer(1, instance_buffer_.get_ptr(),
                                   instance_bytes_);
    commands->bind_pipeline(transform_pipeline_.get());
    const auto bind_result =
        commands->bind_shader_resources(transform_bindings_.get(), 0);
    TI_ERROR_IF(bind_result != RhiResult::success,
                "Failed to bind Vulkan TLAS transform resources: {}.",
                bind_result);
    vk_commands->push_constants(packet.parameters.data(),
                                sizeof(packet.parameters));
    const auto dispatch_result = commands->dispatch(static_cast<std::uint32_t>(
        (blases_.size() + kRayTransformWorkgroupSize - 1) /
        kRayTransformWorkgroupSize));
    TI_ERROR_IF(dispatch_result != RhiResult::success,
                "Failed to dispatch Vulkan TLAS transform packing: {}.",
                dispatch_result);

    VkMemoryBarrier packed{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    packed.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    packed.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &packed,
        0, nullptr, 0, nullptr);
    record_acceleration_structure(command_buffer, true);
  }

  void prepare_query_variant(unsigned variant) {
    std::lock_guard<std::mutex> lock(mutex_);
    query_.prepare(device_, variant);
  }

  void record_query(CommandList *command_list,
                    const VulkanRayQueryPacket &packet) {
    std::lock_guard<std::mutex> lock(mutex_);
    query_.record(command_list, tlas_, packet);
    retain(static_cast<vulkan::VulkanCommandList *>(command_list)
               ->vk_command_buffer());
  }

  void bind_for_shader(ShaderResourceSet *bindings, int binding) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(!bindings || binding < 0,
                "Vulkan shader TLAS binding is invalid.");
    static_cast<vulkan::VulkanResourceSet *>(bindings)->acceleration_structure(
        binding, tlas_);
  }

  void record_shader_read(CommandList *commands,
                          VkPipelineStageFlags consumer_stages,
                          bool graphics_queue) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto command_buffer =
        static_cast<vulkan::VulkanCommandList *>(commands)->vk_command_buffer();
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    // Same-queue builds need an explicit shader dependency. With distinct
    // queues the existing runtime-ordered graphics semaphore supplies it;
    // an AS_BUILD stage is not legal on a graphics-only queue family.
    if (!graphics_queue ||
        device_->graphics_queue() == device_->compute_queue()) {
      command_buffer->device.functions().vkCmdPipelineBarrier(
          command_buffer->buffer,
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
          consumer_stages, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    retain(command_buffer);
  }

  void bind_for_kernel(ShaderResourceSet *bindings,
                       int binding,
                       CommandList *command_list) {
    std::lock_guard<std::mutex> lock(mutex_);
    TI_ERROR_IF(bindings == nullptr || command_list == nullptr || binding < 0,
                "Vulkan TLAS kernel binding is invalid.");
    auto *vulkan_bindings = static_cast<vulkan::VulkanResourceSet *>(bindings);
    auto *vulkan_commands =
        static_cast<vulkan::VulkanCommandList *>(command_list);
    vulkan_bindings->acceleration_structure(binding, tlas_);
    retain(vulkan_commands->vk_command_buffer());
  }

 private:
  void record_acceleration_structure(
      const vkapi::IVkCommandBuffer &command_buffer,
      bool update) {
    const auto geometry = make_geometry();
    auto build = make_build_info(
        geometry, update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                         : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR);
    const VkAccelerationStructureBuildRangeInfoKHR range{
        static_cast<std::uint32_t>(blases_.size()), 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *ranges[] = {&range};
    cmd_build_(command_buffer->buffer, 1, &build, ranges);

    VkMemoryBarrier query_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    query_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    query_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    command_buffer->device.functions().vkCmdPipelineBarrier(
        command_buffer->buffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &query_barrier, 0, nullptr, 0, nullptr);
    retain(command_buffer);
  }

  DeviceAllocation allocate(std::size_t bytes, AllocUsage usage) {
    Device::AllocParams params;
    params.size = bytes;
    params.usage = usage;
    DeviceAllocation allocation{kDeviceNullAllocation};
    const auto result = device_->allocate_memory(params, &allocation);
    TI_ERROR_IF(result != RhiResult::success,
                "Failed to allocate Vulkan instance TLAS buffer ({} bytes): "
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
        "Vulkan instance TLAS scratch size overflow.");
    return static_cast<std::size_t>(bytes + scratch_alignment_);
  }

  VkAccelerationStructureGeometryKHR make_geometry() const {
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

  VkAccelerationStructureBuildGeometryInfoKHR make_build_info(
      const VkAccelerationStructureGeometryKHR &geometry,
      VkBuildAccelerationStructureModeKHR mode) const {
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                 VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    info.mode = mode;
    if (mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) {
      info.srcAccelerationStructure = tlas_->accel;
    }
    info.dstAccelerationStructure = tlas_->accel;
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
    build.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                  VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geometry;
    const auto primitive_count = static_cast<std::uint32_t>(blases_.size());
    VkAccelerationStructureBuildSizesInfoKHR sizes{};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    get_build_sizes_(device_->vk_device(),
                     VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build,
                     &primitive_count, &sizes);
    storage_bytes_ = static_cast<std::size_t>(sizes.accelerationStructureSize);
    storage_ =
        allocate(storage_bytes_, AllocUsage::AccelerationStructureStorage);
    tlas_ = vkapi::create_acceleration_structure(
        0, device_->get_vkbuffer(storage_.get_ptr()), 0,
        sizes.accelerationStructureSize,
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
    TI_ERROR_IF(!tlas_, "Failed to create independent Vulkan instance TLAS.");
    scratch_bytes_ = scratch_allocation_bytes(
        std::max(sizes.buildScratchSize, sizes.updateScratchSize));
    scratch_ = allocate(scratch_bytes_,
                        AllocUsage::Storage | AllocUsage::DeviceAddress);
  }

  void retain(const vkapi::IVkCommandBuffer &command_buffer) const {
    const std::array<DeviceAllocation, 3> allocations{instance_buffer_,
                                                      storage_, scratch_};
    for (const auto allocation : allocations) {
      command_buffer->refs.push_back(
          device_->get_vkbuffer(allocation.get_ptr()));
    }
    command_buffer->refs.push_back(tlas_);
    for (const auto &blas : retained_blases_) {
      blas->retain(command_buffer);
    }
  }

  vulkan::VulkanDevice *device_{nullptr};
  std::vector<std::shared_ptr<VulkanTriangleBlas>> blases_;
  const std::vector<std::uint32_t> sbt_record_offsets_;
  std::uint32_t max_sbt_record_offset_{0};
  std::vector<std::shared_ptr<VulkanTriangleBlas>> retained_blases_;
  std::size_t instance_bytes_{0};
  std::size_t storage_bytes_{0};
  std::size_t scratch_bytes_{0};
  VkDeviceSize scratch_alignment_{1};
  DeviceAllocation instance_buffer_{kDeviceNullAllocation};
  DeviceAllocation storage_{kDeviceNullAllocation};
  DeviceAllocation scratch_{kDeviceNullAllocation};
  vkapi::IVkAccelerationStructureKHR tlas_{nullptr};
  VulkanRayQueryPipelines query_;
  std::unique_ptr<Pipeline> transform_pipeline_;
  std::unique_ptr<ShaderResourceSet> transform_bindings_;
  bool initialized_{false};
  PFN_vkGetAccelerationStructureBuildSizesKHR get_build_sizes_{nullptr};
  PFN_vkCmdBuildAccelerationStructuresKHR cmd_build_{nullptr};
  std::mutex mutex_;
};

VulkanRayTransformPacket prepare_vulkan_ray_transform_packet(
    vulkan::VulkanDevice *device,
    std::size_t instance_count,
    const VulkanRayQueryPacket::Binding &binding) {
  TI_ERROR_IF(!device || binding.pointer.device != device ||
                  instance_count == 0 ||
                  instance_count > (std::numeric_limits<std::uint32_t>::max)(),
              "Vulkan TLAS transforms require same-device storage and positive "
              "uint32 count.");
  TI_ERROR_IF(
      binding.bytes != checked_mul(instance_count, 12 * sizeof(float),
                                   "transform bytes") ||
          (binding.pointer.offset & 3u),
      "Vulkan TLAS transforms require compact four-byte-aligned storage.");
  TI_ERROR_IF(binding.pointer.offset >
                  (std::numeric_limits<std::uint64_t>::max)() - binding.bytes,
              "Vulkan TLAS transform range overflows.");
  const auto &limits = device->get_vk_physical_device_props().limits;
  TI_ERROR_IF((instance_count + kRayTransformWorkgroupSize - 1) /
                      kRayTransformWorkgroupSize >
                  limits.maxComputeWorkGroupCount[0],
              "Vulkan TLAS instance count exceeds the device dispatch limit.");
  const auto alignment =
      std::max<VkDeviceSize>(4, limits.minStorageBufferOffsetAlignment);
  const auto remainder = binding.pointer.offset % alignment;
  TI_ERROR_IF(
      remainder > limits.maxStorageBufferRange ||
          binding.bytes > limits.maxStorageBufferRange - remainder ||
          checked_mul(instance_count,
                      sizeof(VkAccelerationStructureInstanceKHR),
                      "packed instance bytes") > limits.maxStorageBufferRange,
      "Vulkan TLAS transforms exceed maxStorageBufferRange.");
  VulkanRayTransformPacket packet;
  packet.shader_binding = binding;
  packet.shader_binding.pointer.offset -= remainder;
  packet.shader_binding.bytes += remainder;
  packet.parameters = {static_cast<std::uint32_t>(instance_count),
                       static_cast<std::uint32_t>(remainder / sizeof(float))};
  return packet;
}
VulkanInstanceTlas::VulkanInstanceTlas(
    vulkan::VulkanDevice *device,
    std::vector<std::shared_ptr<VulkanTriangleBlas>> blases,
    std::vector<std::uint32_t> offsets)
    : state_(std::make_unique<State>(device,
                                     std::move(blases),
                                     std::move(offsets))) {
  state_->initialize();
}
VulkanInstanceTlas::~VulkanInstanceTlas() = default;
VulkanTriangleRaySceneMemoryStatistics VulkanInstanceTlas::memory_statistics()
    const {
  return state_->memory_statistics();
}
std::size_t VulkanInstanceTlas::instance_count() const {
  return state_->instance_count();
}
std::uint32_t VulkanInstanceTlas::max_sbt_record_offset() const {
  return state_->max_sbt_record_offset();
}
void VulkanInstanceTlas::record_build(
    CommandList *commands,
    const std::vector<VulkanRayInstanceInfo> &instances,
    bool update) {
  state_->record_build(commands, instances, update);
}
void VulkanInstanceTlas::prepare_transforms() {
  state_->prepare_transforms();
}
void VulkanInstanceTlas::record_transforms(
    CommandList *commands,
    const VulkanRayTransformPacket &packet) {
  state_->record_transforms(commands, packet);
}
void VulkanInstanceTlas::prepare_query_variant(unsigned variant) {
  state_->prepare_query_variant(variant);
}
void VulkanInstanceTlas::record_query(CommandList *commands,
                                      const VulkanRayQueryPacket &packet) {
  state_->record_query(commands, packet);
}
void VulkanInstanceTlas::bind_for_shader(ShaderResourceSet *bindings,
                                         int binding) {
  state_->bind_for_shader(bindings, binding);
}
void VulkanInstanceTlas::record_shader_read(CommandList *commands,
                                            VkPipelineStageFlags stages,
                                            bool graphics) {
  state_->record_shader_read(commands, stages, graphics);
}
void VulkanInstanceTlas::bind_for_kernel(ShaderResourceSet *bindings,
                                         int binding,
                                         CommandList *commands) {
  state_->bind_for_kernel(bindings, binding, commands);
}
}  // namespace taichi::lang
#endif
