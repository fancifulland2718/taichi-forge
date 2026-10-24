#pragma once

#include "taichi/runtime/vulkan_ray_query.h"
#include "taichi/runtime/vulkan_ray_blas.h"

#if defined(TI_WITH_VULKAN)
namespace taichi::lang {
// The consumer chooses the submission contract. A selected domain must never
// infer its build dependency from the runtime's unrelated default queues.
enum class VulkanRayShaderReadOrder {
  compute,
  runtime_graphics,
  execution_domain,
};

// Cold recording policy; queue capabilities come from the actual command pool.
bool vulkan_ray_shader_read_needs_barrier(
    VkQueueFlags queue_capabilities,
    VkPipelineStageFlags consumer_stages,
    VulkanRayShaderReadOrder order,
    bool runtime_queues_match);

struct VulkanRayTransformPacket {
  VulkanRayQueryPacket::Binding shader_binding{};
  std::array<std::uint32_t, 2> parameters{};
};

VulkanRayTransformPacket prepare_vulkan_ray_transform_packet(
    vulkan::VulkanDevice *device,
    std::size_t instance_count,
    const VulkanRayQueryPacket::Binding &binding);

// Ordered instance topology owns its BLAS cores; recorded commands retain the
// native buffers and AS objects. The enclosing owner retires work before close.
class VulkanInstanceTlas {
 public:
  VulkanInstanceTlas(vulkan::VulkanDevice *device,
                     std::vector<std::shared_ptr<VulkanTriangleBlas>> blases,
                     std::vector<std::uint32_t> sbt_record_offsets);
  ~VulkanInstanceTlas();
  VulkanInstanceTlas(const VulkanInstanceTlas &) = delete;
  VulkanInstanceTlas &operator=(const VulkanInstanceTlas &) = delete;

  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const;
  std::size_t instance_count() const;
  std::uint32_t max_sbt_record_offset() const;
  void record_build(CommandList *commands,
                    const std::vector<VulkanRayInstanceInfo> &instances,
                    bool update);
  // Seed immutable IDs/BLAS addresses, pack device matrices, then build once.
  // prepare_transforms() must have completed before recording this operation.
  void record_build_device(CommandList *commands,
                           const std::vector<VulkanRayInstanceInfo> &instances,
                           const VulkanRayTransformPacket &packet);
  void prepare_transforms();
  void record_transforms(CommandList *commands,
                         const VulkanRayTransformPacket &packet);
  void prepare_query_variant(unsigned variant);
  void record_query(CommandList *commands, const VulkanRayQueryPacket &packet);
  void bind_for_shader(ShaderResourceSet *bindings, int binding);
  void record_shader_read(CommandList *commands,
                          VkPipelineStageFlags consumer_stages,
                          VulkanRayShaderReadOrder order);
  void bind_for_kernel(ShaderResourceSet *bindings,
                       int binding,
                       CommandList *commands);

 private:
  struct State;
  std::unique_ptr<State> state_;
};
}  // namespace taichi::lang
#endif
