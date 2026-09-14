#include "taichi/rhi/vulkan/vulkan_ray_pipeline.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace taichi::lang::vulkan {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > (std::numeric_limits<std::size_t>::max)() - a) {
    throw std::invalid_argument("Vulkan SBT byte size overflow");
  }
  return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > (std::numeric_limits<std::size_t>::max)() / a) {
    throw std::invalid_argument("Vulkan SBT byte size overflow");
  }
  return a * b;
}

std::size_t aligned_size(std::size_t size, std::size_t alignment) {
  if (!alignment || (alignment & (alignment - 1))) {
    throw std::invalid_argument("Invalid Vulkan SBT alignment");
  }
  return checked_add(size, alignment - 1) & ~(alignment - 1);
}

}  // namespace

VulkanPipeline::VulkanPipeline(const Params &params,
                               const VulkanRayTracingPipelineParams &ray_params)
    : ti_device_(*params.device),
      device_(params.device->vk_device()),
      name_(params.name),
      bind_point_(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR),
      cache_(params.cache) {
  const auto &caps = ti_device_.vk_caps();
  if (!caps.ray_tracing_pipeline) {
    throw std::invalid_argument("Vulkan ray-tracing pipeline is unavailable");
  }
  if (params.code.empty() || ray_params.groups.empty() ||
      params.code.size() > UINT32_MAX ||
      ray_params.groups.size() > UINT32_MAX ||
      !ray_params.max_recursion_depth ||
      ray_params.max_recursion_depth >
          caps.ray_tracing_properties.maxRayRecursionDepth) {
    throw std::invalid_argument(
        "Invalid Vulkan ray stages, groups or recursion depth");
  }
  if (ray_params.opacity_micromap && !caps.opacity_micromap) {
    throw std::invalid_argument("Vulkan opacity micromap is unavailable");
  }
  for (const auto &code : params.code) {
    if (code.stage != VK_SHADER_STAGE_RAYGEN_BIT_KHR &&
        code.stage != VK_SHADER_STAGE_MISS_BIT_KHR &&
        code.stage != VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR &&
        code.stage != VK_SHADER_STAGE_ANY_HIT_BIT_KHR) {
      throw std::invalid_argument("Unsupported Vulkan ray shader stage");
    }
  }
  const auto check_shader = [&](std::uint32_t index,
                                VkShaderStageFlagBits stage) {
    if (index >= params.code.size() || params.code[index].stage != stage) {
      throw std::invalid_argument(
          "Vulkan ray group shader index/stage mismatch");
    }
  };
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
  groups.reserve(ray_params.groups.size());
  bool has_raygen = false;
  for (const auto &input : ray_params.groups) {
    VkRayTracingShaderGroupCreateInfoKHR group{
        VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.generalShader = group.closestHitShader = group.anyHitShader =
        group.intersectionShader = VK_SHADER_UNUSED_KHR;
    VkShaderStageFlags group_stages = 0;
    if (input.kind == VulkanRayTracingGroup::Kind::triangles) {
      if (input.general != VK_SHADER_UNUSED_KHR ||
          (input.closest_hit == VK_SHADER_UNUSED_KHR &&
           input.any_hit == VK_SHADER_UNUSED_KHR)) {
        throw std::invalid_argument(
            "Triangle group requires closest-hit or any-hit, without a general "
            "shader");
      }
      group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
      if (input.closest_hit != VK_SHADER_UNUSED_KHR) {
        check_shader(input.closest_hit, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
        group.closestHitShader = input.closest_hit;
        group_stages |= VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
      }
      if (input.any_hit != VK_SHADER_UNUSED_KHR) {
        check_shader(input.any_hit, VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
        group.anyHitShader = input.any_hit;
        group_stages |= VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
      }
    } else {
      if (input.kind != VulkanRayTracingGroup::Kind::raygen &&
          input.kind != VulkanRayTracingGroup::Kind::miss) {
        throw std::invalid_argument("Unsupported Vulkan ray group kind");
      }
      if (input.closest_hit != VK_SHADER_UNUSED_KHR ||
          input.any_hit != VK_SHADER_UNUSED_KHR) {
        throw std::invalid_argument(
            "General ray group cannot contain hit shaders");
      }
      const auto stage = input.kind == VulkanRayTracingGroup::Kind::raygen
                             ? VK_SHADER_STAGE_RAYGEN_BIT_KHR
                             : VK_SHADER_STAGE_MISS_BIT_KHR;
      check_shader(input.general, stage);
      group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      group.generalShader = input.general;
      group_stages = stage;
      has_raygen |= stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    }
    groups.push_back(group);
    ray_group_stages_.push_back(
        static_cast<VkShaderStageFlagBits>(group_stages));
  }
  if (!has_raygen) {
    throw std::invalid_argument("Vulkan ray pipeline requires a raygen group");
  }

  // A failed constructor does not call ~VulkanPipeline. Release raw modules
  // here; the layout/cache/pipeline members already have shared RAII owners.
  const auto release_modules = [&] {
    for (auto module : shader_modules_) {
      vkDestroyShaderModule(device_, module, kNoVkAllocCallbacks);
    }
    shader_modules_.clear();
  };
  try {
    create_descriptor_set_layout(params);
    create_shader_stages(params);
    create_pipeline_layout();
    VkRayTracingPipelineCreateInfoKHR info{
        VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    info.stageCount = static_cast<std::uint32_t>(shader_stages_.size());
    info.pStages = shader_stages_.data();
    info.groupCount = static_cast<std::uint32_t>(groups.size());
    info.pGroups = groups.data();
    info.maxPipelineRayRecursionDepth = ray_params.max_recursion_depth;
    if (ray_params.opacity_micromap) {
      info.flags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
    }
    // No dynamic stack state: Vulkan computes the stack from these shaders.
    std::vector<vkapi::IVkPipeline> libraries;
    pipeline_ = vkapi::create_raytracing_pipeline(
        device_, &info, pipeline_layout_, libraries, VK_NULL_HANDLE, cache_);
    if (!pipeline_ || !pipeline_->pipeline) {
      throw std::runtime_error("Failed to create Vulkan ray-tracing pipeline");
    }
    ray_group_handles_.resize(checked_mul(
        groups.size(), caps.ray_tracing_properties.shaderGroupHandleSize));
    if (vkGetRayTracingShaderGroupHandlesKHR(
            device_, pipeline_->pipeline, 0, info.groupCount,
            ray_group_handles_.size(),
            ray_group_handles_.data()) != VK_SUCCESS) {
      throw std::runtime_error(
          "Failed to read Vulkan ray shader group handles");
    }
  } catch (...) {
    release_modules();
    throw;
  }
  release_modules();
}

VulkanShaderBindingTable::VulkanShaderBindingTable(
    VulkanDevice &device,
    const VulkanPipeline &pipeline,
    const VulkanShaderBindingRecord &raygen_record,
    const std::vector<VulkanShaderBindingRecord> &miss_records,
    const std::vector<VulkanShaderBindingRecord> &hit_records) {
  if (&pipeline.ti_device_ != &device ||
      pipeline.bind_point() != VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR) {
    throw std::invalid_argument(
        "SBT requires a ray pipeline on the same device");
  }
  const auto &props = device.vk_caps().ray_tracing_properties;
  const auto &stages = pipeline.ray_group_stages();
  const std::array<const VulkanShaderBindingRecord *, 3> records{
      &raygen_record, miss_records.data(), hit_records.data()};
  const std::array<std::size_t, 3> counts{1, miss_records.size(),
                                          hit_records.size()};
  const std::array<VkShaderStageFlags, 3> expected_stages{
      VK_SHADER_STAGE_RAYGEN_BIT_KHR, VK_SHADER_STAGE_MISS_BIT_KHR,
      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR};
  std::array<std::size_t, 3> offsets{};
  std::size_t bytes = 0;
  for (unsigned region = 0; region < 3; ++region) {
    if (!counts[region]) {
      continue;
    }
    std::size_t record_bytes = props.shaderGroupHandleSize;
    for (std::size_t i = 0; i < counts[region]; ++i) {
      const auto &record = records[region][i];
      if (record.group >= stages.size() ||
          !(stages[record.group] & expected_stages[region])) {
        throw std::invalid_argument(
            "SBT record references the wrong ray group kind");
      }
      record_bytes = std::max(
          record_bytes,
          checked_add(props.shaderGroupHandleSize, record.data.size()));
    }
    const auto stride =
        aligned_size(record_bytes, props.shaderGroupHandleAlignment);
    if (stride > props.maxShaderGroupStride) {
      throw std::invalid_argument("SBT record exceeds maxShaderGroupStride");
    }
    bytes = aligned_size(bytes, props.shaderGroupBaseAlignment);
    offsets[region] = bytes;
    regions_[region].stride = stride;
    regions_[region].size = checked_mul(stride, counts[region]);
    bytes = checked_add(bytes, regions_[region].size);
  }
  // Align the device address, not merely the allocation-relative offset.
  allocated_bytes_ = checked_add(bytes, props.shaderGroupBaseAlignment - 1);
  auto [allocation, allocation_result] = device.allocate_memory_unique(
      {allocated_bytes_, false, false, false, AllocUsage::ShaderBindingTable});
  if (allocation_result != RhiResult::success || !allocation) {
    throw std::runtime_error("Failed to allocate Vulkan SBT");
  }
  allocation_ = std::move(allocation);
  buffer_ = device.get_vkbuffer(*allocation_);
  const auto address = device.get_buffer_device_address(*allocation_);
  if (!address) {
    throw std::runtime_error("Vulkan SBT has no device address");
  }
  const auto base = aligned_size(address, props.shaderGroupBaseAlignment);
  const auto padding = base - address;
  std::vector<std::uint8_t> packed(allocated_bytes_, 0);
  for (unsigned region = 0; region < 3; ++region) {
    if (!counts[region]) {
      continue;
    }
    regions_[region].deviceAddress = checked_add(base, offsets[region]);
    for (std::size_t i = 0; i < counts[region]; ++i) {
      const auto &record = records[region][i];
      auto *destination = packed.data() + padding + offsets[region] +
                          i * regions_[region].stride;
      std::memcpy(destination,
                  pipeline.ray_group_handles().data() +
                      record.group * std::size_t(props.shaderGroupHandleSize),
                  props.shaderGroupHandleSize);
      if (!record.data.empty()) {
        std::memcpy(destination + props.shaderGroupHandleSize,
                    record.data.data(), record.data.size());
      }
    }
  }

  // Packing is host-only. Do not flush unrelated runtime work or introduce a
  // hidden submit/wait from construction or Graph binding preparation.
  auto [staging, staging_result] = device.allocate_memory_unique(
      {packed.size(), true, false, false, AllocUsage::Upload});
  if (staging_result != RhiResult::success || !staging) {
    throw std::runtime_error("Failed to allocate Vulkan SBT upload storage");
  }
  void *mapped = nullptr;
  if (device.map(*staging, &mapped) != RhiResult::success) {
    throw std::runtime_error("Failed to map Vulkan SBT upload storage");
  }
  std::memcpy(mapped, packed.data(), packed.size());
  device.unmap(*staging);
  staging_ = std::move(staging);
}

void VulkanShaderBindingTable::record_initialization(
    VulkanCommandList &commands) {
  if (!staging_) {
    throw std::logic_error("Vulkan SBT initialization was already recorded");
  }
  auto command_buffer = commands.vk_command_buffer();
  if (command_buffer->device != buffer_->device) {
    throw std::invalid_argument(
        "Vulkan SBT initialization requires the same device");
  }
  commands.buffer_copy(allocation_->get_ptr(0), staging_->get_ptr(0),
                       allocated_bytes_);
  VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex =
      VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = buffer_->buffer;
  barrier.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(command_buffer->buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0,
                       nullptr, 1, &barrier, 0, nullptr);
  // buffer_copy retains the staging Vk allocation until command retirement.
  // No persistent upload buffer and no completion wait are required here.
  staging_.reset();
}

void validate_ray_dispatch(const VulkanDevice &device,
                           std::uint32_t width,
                           std::uint32_t height,
                           std::uint32_t depth) {
  const auto &caps = device.vk_caps();
  const std::array<std::uint32_t, 3> dimensions{width, height, depth};
  std::uint64_t total = 1;
  for (unsigned i = 0; i < 3; ++i) {
    if (!caps.ray_tracing_pipeline || !dimensions[i] ||
        dimensions[i] > caps.max_ray_dispatch_dimensions[i] ||
        total > caps.ray_tracing_properties.maxRayDispatchInvocationCount /
                    dimensions[i]) {
      throw std::invalid_argument(
          "Ray dispatch dimensions exceed Vulkan device limits");
    }
    total *= dimensions[i];
  }
}

void record_ray_program_begin(VulkanCommandList &commands) {
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask =
      VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(commands.vk_command_buffer()->buffer,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1,
                       &barrier, 0, nullptr, 0, nullptr);
}

void record_ray_program_end(VulkanCommandList &commands) {
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
      VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
      VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  // The execution dependency also orders RT AS/geometry reads before a later
  // build/refit overwrites the same storage; no readback or host wait is
  // needed.
  vkCmdPipelineBarrier(
      commands.vk_command_buffer()->buffer,
      VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
      0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void VulkanCommandList::trace_rays(const VulkanShaderBindingTable &sbt,
                                   std::uint32_t width,
                                   std::uint32_t height,
                                   std::uint32_t depth) {
  const VkStridedDeviceAddressRegionKHR callable{};
  buffer_->refs.push_back(sbt.buffer());
  vkCmdTraceRaysKHR(buffer_->buffer, &sbt.raygen(), &sbt.miss(), &sbt.hit(),
                    &callable, width, height, depth);
}

}  // namespace taichi::lang::vulkan
