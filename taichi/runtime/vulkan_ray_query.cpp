#include "taichi/runtime/vulkan_ray_query.h"

#if defined(TI_WITH_VULKAN)
#include <algorithm>
#include <limits>

namespace taichi::lang {
namespace {
constexpr std::uint32_t kRayQueryWorkgroupSize = 128;
static const std::uint32_t kRayQueryTrianglesSpv[] =
#include "taichi/program/vulkan_sort_shaders/ray_query_triangles.comp.spv.h"
    ;

static const std::uint32_t kRayQueryTrianglesTypedSpv[] =
#include "taichi/program/vulkan_sort_shaders/ray_query_triangles_typed.comp.spv.h"
    ;

static const std::uint32_t kRayQueryTrianglesSubrangeSpv[] =
#include "taichi/program/vulkan_sort_shaders/ray_query_triangles_subrange.comp.spv.h"
    ;
static const std::uint32_t kRayQueryTrianglesTypedSubrangeSpv[] =
#include "taichi/program/vulkan_sort_shaders/ray_query_triangles_typed_subrange.comp.spv.h"
    ;

}  // namespace

VulkanRayQueryPacket prepare_vulkan_ray_query_packet(
    vulkan::VulkanDevice *device,
    std::size_t ray_count,
    bool typed,
    const std::array<VulkanRayQueryPacket::Binding, 3> &bindings) {
  TI_ERROR_IF(device == nullptr, "Vulkan ray query requires a device.");
  TI_ERROR_IF(
      ray_count == 0 || ray_count > (std::numeric_limits<std::uint32_t>::max)(),
      "Vulkan ray count must be in [1, UINT32_MAX].");
  const auto &limits = device->get_vk_physical_device_props().limits;
  TI_ERROR_IF(
      (ray_count + kRayQueryWorkgroupSize - 1) / kRayQueryWorkgroupSize >
          limits.maxComputeWorkGroupCount[0],
      "Vulkan ray count exceeds the device's one-dimensional dispatch limit.");
  VulkanRayQueryPacket packet;
  packet.ray_count = ray_count;
  packet.parameters[0] = static_cast<std::uint32_t>(ray_count);
  packet.variant = typed ? 1u : 0u;
  const auto alignment =
      std::max<VkDeviceSize>(16, limits.minStorageBufferOffsetAlignment);
  for (std::size_t i = 0; i < (typed ? 3u : 2u); ++i) {
    const auto &binding = bindings[i];
    TI_ERROR_IF(binding.pointer.device != device,
                "Vulkan ray storage belongs to a different device.");
    const std::size_t record_bytes = (i == 0 ? 8 : 4) * sizeof(float);
    TI_ERROR_IF(
        ray_count > (std::numeric_limits<std::size_t>::max)() / record_bytes,
        "Vulkan ray query bytes size overflow.");
    TI_ERROR_IF(
        binding.bytes != ray_count * record_bytes ||
            (binding.pointer.offset & 3u),
        "Vulkan ray storage must be a compact four-byte-aligned record range.");
    TI_ERROR_IF(binding.pointer.offset >
                    (std::numeric_limits<std::uint64_t>::max)() - binding.bytes,
                "Vulkan ray storage range overflow.");
    for (std::size_t j = 0; j < i; ++j) {
      const auto &other = bindings[j];
      TI_ERROR_IF(
          static_cast<const DeviceAllocation &>(binding.pointer) ==
                  static_cast<const DeviceAllocation &>(other.pointer) &&
              binding.pointer.offset < other.pointer.offset + other.bytes &&
              other.pointer.offset < binding.pointer.offset + binding.bytes,
          "Vulkan ray output ranges must not alias rays or each other.");
    }
    const auto remainder = binding.pointer.offset % alignment;
    auto shader_binding = binding;
    shader_binding.pointer.offset -= remainder;
    TI_ERROR_IF(remainder > limits.maxStorageBufferRange ||
                    binding.bytes > limits.maxStorageBufferRange - remainder,
                "Vulkan ray descriptor range exceeds maxStorageBufferRange.");
    shader_binding.bytes += remainder;
    packet.shader_bindings[i] = shader_binding;
    packet.parameters[i + 1] =
        static_cast<std::uint32_t>(remainder / sizeof(float));
    if (remainder)
      packet.variant |= 2u;
  }
  return packet;
}
void VulkanRayQueryPipelines::clear() {
  bindings_ = {};
  pipelines_ = {};
}

void VulkanRayQueryPipelines::prepare(vulkan::VulkanDevice *device,
                                      unsigned variant) {
  TI_ERROR_IF(!device->vk_caps().ray_query,
              "Vulkan inline/batch ray query is unavailable; AS construction "
              "alone does not enable it.");
  TI_ASSERT(variant < pipelines_.size());
  if (pipelines_[variant]) {
    return;
  }
  const std::array<std::pair<const std::uint32_t *, std::size_t>, 4> code{
      {{kRayQueryTrianglesSpv, sizeof(kRayQueryTrianglesSpv)},
       {kRayQueryTrianglesTypedSpv, sizeof(kRayQueryTrianglesTypedSpv)},
       {kRayQueryTrianglesSubrangeSpv, sizeof(kRayQueryTrianglesSubrangeSpv)},
       {kRayQueryTrianglesTypedSubrangeSpv,
        sizeof(kRayQueryTrianglesTypedSubrangeSpv)}}};
  PipelineSourceDesc source{PipelineSourceType::spirv_binary,
                            code[variant].first, code[variant].second,
                            PipelineStageType::compute};
  auto [pipeline, result] = device->create_pipeline_unique(
      source, "vulkan_ray_query_triangles_" + std::to_string(variant));
  TI_ERROR_IF(result != RhiResult::success || !pipeline,
              "Failed to create Vulkan ray query pipeline: {}.", result);
  std::unique_ptr<ShaderResourceSet> bindings(device->create_resource_set());
  pipelines_[variant] = std::move(pipeline);
  bindings_[variant] = std::move(bindings);
}

void VulkanRayQueryPipelines::record(
    CommandList *commands,
    const vkapi::IVkAccelerationStructureKHR &tlas,
    const VulkanRayQueryPacket &packet) {
  const bool typed = (packet.variant & 1u) != 0;
  auto *bindings =
      static_cast<vulkan::VulkanResourceSet *>(bindings_[packet.variant].get());
  commands->buffer_barrier(packet.shader_bindings[0].pointer);
  bindings->acceleration_structure(0, tlas);
  for (unsigned i = 0; i < (typed ? 3u : 2u); ++i) {
    const auto &binding = packet.shader_bindings[i];
    bindings->rw_buffer(i + 1, binding.pointer, binding.bytes);
  }
  commands->bind_pipeline(pipelines_[packet.variant].get());
  const auto bind_result = commands->bind_shader_resources(bindings, 0);
  TI_ERROR_IF(bind_result != RhiResult::success,
              "Failed to bind Vulkan ray query resources: {}.", bind_result);
  auto *vk_commands = static_cast<vulkan::VulkanCommandList *>(commands);
  vk_commands->push_constants(packet.parameters.data(),
                              (packet.variant & 2u) ? 16 : 4);
  const auto dispatch_result = commands->dispatch(static_cast<std::uint32_t>(
      (packet.ray_count + kRayQueryWorkgroupSize - 1) /
      kRayQueryWorkgroupSize));
  TI_ERROR_IF(dispatch_result != RhiResult::success,
              "Failed to dispatch Vulkan ray query: {}.", dispatch_result);
  commands->buffer_barrier(packet.shader_bindings[1].pointer);
  if (typed) {
    commands->buffer_barrier(packet.shader_bindings[2].pointer);
  }
}

}  // namespace taichi::lang
#endif
