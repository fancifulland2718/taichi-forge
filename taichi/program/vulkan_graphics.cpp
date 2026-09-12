#include "taichi/program/program.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "taichi/program/ndarray.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/texture.h"

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
class VulkanGraphicsPipelineResource;

namespace {

constexpr std::size_t kMaximumShaderBytes = 16u * 1024u * 1024u;
constexpr std::size_t kMaximumVertexBindings = 16;
constexpr std::size_t kMaximumVertexAttributes = 32;
constexpr std::size_t kMaximumPassDraws = 1u << 20;

struct RecordedGraphicsShaderBuffer {
  std::uint32_t set_index{0};
  std::uint32_t binding{0};
  DeviceAllocation allocation{kDeviceNullAllocation};
  std::vector<DeviceAllocation> array_elements;
  bool storage{false};
};

struct RecordedGraphicsShaderImage {
  std::uint32_t set_index{0};
  std::uint32_t binding{0};
  DeviceAllocation allocation{kDeviceNullAllocation};
  ImageSamplerConfig sampler{};
};

struct RecordedGraphicsIndirect {
  DeviceAllocation command_buffer{kDeviceNullAllocation};
  DeviceAllocation count_buffer{kDeviceNullAllocation};
  std::uint32_t command_offset{0};
  std::uint32_t count_offset{0};
  std::uint32_t max_draw_count{1};
  std::uint32_t stride{0};
  std::uint32_t vertex_record_limit{0};
  std::uint32_t instance_record_limit{0};
  std::uint32_t index_element_limit{0};
  bool first_instance_may_be_nonzero{false};
};

struct RecordedGraphicsDraw {
  std::weak_ptr<VulkanGraphicsPipelineResource> pipeline;
  std::vector<std::pair<std::uint32_t, DeviceAllocation>> vertex_buffers;
  DeviceAllocation index_buffer{kDeviceNullAllocation};
  std::vector<RecordedGraphicsShaderBuffer> shader_buffers;
  std::vector<RecordedGraphicsShaderImage> shader_images;
  VulkanGraphicsDrawInfo draw;
  std::optional<RecordedGraphicsIndirect> indirect;
  std::optional<VulkanGraphicsMeshDrawInfo> mesh;
};

struct PreparedVulkanGraphicsDrawResources {
  std::vector<std::pair<std::uint32_t, std::unique_ptr<ShaderResourceSet>>>
      shader_resource_sets;
  std::unique_ptr<RasterResources> raster;
};

struct PreparedVulkanGraphicsResourcePayload {
  std::vector<PreparedVulkanGraphicsDrawResources> draws;
};

class PreparedVulkanGraphicsResourceLease;

struct PreparedVulkanGraphicsResourceStats {
  std::unordered_set<const PreparedVulkanGraphicsResourceLease *> counted;
  std::uint64_t leases{0};
  std::uint64_t draws{0};
  std::uint64_t descriptor_sets{0};
  std::uint64_t raster_resources{0};
};

class PreparedVulkanGraphicsResourceLease {
 public:
  explicit PreparedVulkanGraphicsResourceLease(
      std::shared_ptr<const PreparedVulkanGraphicsResourcePayload> payload)
      : payload_(std::move(payload)) {
  }

  std::shared_ptr<const PreparedVulkanGraphicsResourcePayload> acquire() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return payload_;
  }

  void clear() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    payload_.reset();
  }

  void append_debug_stats(
      PreparedVulkanGraphicsResourceStats &result) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!payload_ || !result.counted.insert(this).second) {
      return;
    }
    ++result.leases;
    result.draws += payload_->draws.size();
    for (const auto &draw : payload_->draws) {
      result.descriptor_sets += draw.shader_resource_sets.size();
      result.raster_resources += draw.raster ? 1 : 0;
    }
  }

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<const PreparedVulkanGraphicsResourcePayload> payload_;
};

class PreparedVulkanGraphicsResourceDomain {
 public:
  bool register_lease(
      const std::shared_ptr<PreparedVulkanGraphicsResourceLease> &lease) {
    TI_ERROR_IF(!lease,
                "Prepared Vulkan graphics resource lease must not be null.");
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return false;
    }
    leases_.erase(std::remove_if(leases_.begin(), leases_.end(),
                                 [](const auto &item) {
                                   return item.expired();
                                 }),
                  leases_.end());
    leases_.push_back(lease);
    return true;
  }

  void close() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    closed_ = true;
    for (const auto &item : leases_) {
      if (auto lease = item.lock()) {
        lease->clear();
      }
    }
    leases_.clear();
  }

  void append_debug_stats(
      PreparedVulkanGraphicsResourceStats &result) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &item : leases_) {
      if (auto lease = item.lock()) {
        lease->append_debug_stats(result);
      }
    }
  }

 private:
  mutable std::mutex mutex_;
  bool closed_{false};
  std::vector<std::weak_ptr<PreparedVulkanGraphicsResourceLease>> leases_;
};

std::size_t vertex_format_bytes(BufferFormat format) {
  switch (format) {
    case BufferFormat::r8:
    case BufferFormat::r8u:
    case BufferFormat::r8i:
      return 1;
    case BufferFormat::rg8:
    case BufferFormat::rg8u:
    case BufferFormat::rg8i:
    case BufferFormat::r16:
    case BufferFormat::r16u:
    case BufferFormat::r16i:
    case BufferFormat::r16f:
      return 2;
    case BufferFormat::rgba8:
    case BufferFormat::rgba8u:
    case BufferFormat::rgba8i:
    case BufferFormat::rg16:
    case BufferFormat::rg16u:
    case BufferFormat::rg16i:
    case BufferFormat::rg16f:
    case BufferFormat::r32u:
    case BufferFormat::r32i:
    case BufferFormat::r32f:
      return 4;
    case BufferFormat::rgb16:
    case BufferFormat::rgb16u:
    case BufferFormat::rgb16i:
    case BufferFormat::rgb16f:
      return 6;
    case BufferFormat::rgba16:
    case BufferFormat::rgba16u:
    case BufferFormat::rgba16i:
    case BufferFormat::rgba16f:
    case BufferFormat::rg32u:
    case BufferFormat::rg32i:
    case BufferFormat::rg32f:
      return 8;
    case BufferFormat::rgb32u:
    case BufferFormat::rgb32i:
    case BufferFormat::rgb32f:
      return 12;
    case BufferFormat::rgba32u:
    case BufferFormat::rgba32i:
    case BufferFormat::rgba32f:
      return 16;
    default:
      TI_ERROR("Unsupported Vulkan graphics vertex format {}.",
               static_cast<std::uint32_t>(format));
  }
}

TopologyType decode_topology(int value) {
  switch (value) {
    case 0:
      return TopologyType::Triangles;
    case 1:
      return TopologyType::Lines;
    case 2:
      return TopologyType::Points;
    default:
      TI_ERROR("Vulkan graphics topology must be 0, 1, or 2.");
  }
}

PolygonMode decode_polygon_mode(int value) {
  switch (value) {
    case 0:
      return PolygonMode::Fill;
    case 1:
      return PolygonMode::Line;
    case 2:
      return PolygonMode::Point;
    default:
      TI_ERROR("Vulkan graphics polygon mode must be 0, 1, or 2.");
  }
}

std::size_t ndarray_bytes(const Ndarray *array, const char *role) {
  TI_ERROR_IF(array == nullptr,
              "Vulkan graphics {} binding must not be null.", role);
  const std::size_t elements = array->get_nelement();
  const std::size_t element_bytes = array->get_element_size();
  TI_ERROR_IF(element_bytes != 0 &&
                  elements >
                      (std::numeric_limits<std::size_t>::max)() / element_bytes,
              "Vulkan graphics {} byte size overflows size_t.", role);
  return elements * element_bytes;
}

void append_graphics_allocation_key(
    std::vector<std::uint64_t> &key,
    const vulkan::VulkanDevice *device,
    DeviceAllocation allocation) {
  key.push_back(allocation.alloc_id);
  key.push_back(device->allocation_generation(allocation));
}

std::uint32_t graphics_float_bits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

}  // namespace

struct PreparedVulkanGraphicsPass::State {
  Program *owner{nullptr};
  std::uint64_t generation{0};
  std::vector<RuntimeResourceHandle> arrays;
  std::vector<RuntimeResourceHandle> textures;
  std::vector<std::uint64_t> pipelines;
  std::function<void(GraphicsDevice *, CommandList *)> record;
  std::vector<ComputeOpImageRef> images;
  std::vector<std::uint64_t> replay_key;
  std::shared_ptr<PreparedVulkanGraphicsResourceLease> resources;
};

class VulkanGraphicsPipelineResource {
 public:
  ~VulkanGraphicsPipelineResource() {
    close_prepared_resources();
  }

  VulkanGraphicsPipelineResource(
      Program *program,
      const std::vector<std::uint32_t> &vertex_spirv,
      const std::vector<std::uint32_t> &fragment_spirv,
      const std::vector<VulkanGraphicsVertexBinding> &vertex_bindings,
      const std::vector<VulkanGraphicsVertexAttribute> &vertex_attributes,
      int topology,
      int polygon_mode,
      bool front_face_cull,
      bool back_face_cull,
      bool depth_test,
      bool depth_write,
      bool blending,
      const std::string &name)
      : program_(program), bindings_(vertex_bindings) {
    TI_ERROR_IF(program_ == nullptr,
                "Vulkan graphics pipeline requires a live Program.");
    TI_ERROR_IF(vertex_spirv.empty() || fragment_spirv.empty(),
                "Vulkan graphics pipeline requires vertex and fragment "
                "SPIR-V.");
    TI_ERROR_IF(vertex_spirv.size() * sizeof(std::uint32_t) >
                        kMaximumShaderBytes ||
                    fragment_spirv.size() * sizeof(std::uint32_t) >
                        kMaximumShaderBytes,
                "Vulkan graphics shader exceeds the 16 MiB safety limit.");
    TI_ERROR_IF(vertex_spirv.front() != 0x07230203u ||
                    fragment_spirv.front() != 0x07230203u,
                "Vulkan graphics shaders must contain SPIR-V binary magic.");
    TI_ERROR_IF(bindings_.empty() ||
                    bindings_.size() > kMaximumVertexBindings,
                "Vulkan graphics pipeline requires 1 to {} vertex bindings.",
                kMaximumVertexBindings);
    TI_ERROR_IF(vertex_attributes.empty() ||
                    vertex_attributes.size() > kMaximumVertexAttributes,
                "Vulkan graphics pipeline requires 1 to {} vertex "
                "attributes.",
                kMaximumVertexAttributes);

    std::unordered_map<std::uint32_t, std::size_t> strides;
    std::unordered_set<std::uint32_t> locations;
    std::vector<VertexInputBinding> rhi_bindings;
    rhi_bindings.reserve(bindings_.size());
    for (const auto &binding : bindings_) {
      TI_ERROR_IF(binding.stride == 0 || binding.stride > (1u << 20),
                  "Vulkan graphics vertex stride must be in [1, 1 MiB].");
      TI_ERROR_IF(!strides.emplace(binding.binding, binding.stride).second,
                  "Vulkan graphics vertex binding {} is duplicated.",
                  binding.binding);
      rhi_bindings.push_back(
          {binding.binding, binding.stride, binding.instance});
    }

    std::vector<VertexInputAttribute> rhi_attributes;
    rhi_attributes.reserve(vertex_attributes.size());
    for (const auto &attribute : vertex_attributes) {
      const auto found = strides.find(attribute.binding);
      TI_ERROR_IF(found == strides.end(),
                  "Vulkan graphics attribute {} references undeclared "
                  "binding {}.",
                  attribute.location, attribute.binding);
      TI_ERROR_IF(!locations.insert(attribute.location).second,
                  "Vulkan graphics attribute location {} is duplicated.",
                  attribute.location);
      const std::size_t format_bytes = vertex_format_bytes(attribute.format);
      TI_ERROR_IF(attribute.offset > found->second ||
                      format_bytes > found->second - attribute.offset,
                  "Vulkan graphics attribute {} exceeds binding {} stride "
                  "{}.",
                  attribute.location, attribute.binding, found->second);
      rhi_attributes.push_back({attribute.location, attribute.binding,
                                attribute.format, attribute.offset});
    }

    auto *device = dynamic_cast<GraphicsDevice *>(program_->get_graphics_device());
    TI_ERROR_IF(device == nullptr,
                "Vulkan graphics pipeline has no graphics device.");

    std::vector<PipelineSourceDesc> sources(2);
    sources[0] = {PipelineSourceType::spirv_binary,
                  const_cast<std::uint32_t *>(fragment_spirv.data()),
                  fragment_spirv.size() * sizeof(std::uint32_t),
                  PipelineStageType::fragment};
    sources[1] = {PipelineSourceType::spirv_binary,
                  const_cast<std::uint32_t *>(vertex_spirv.data()),
                  vertex_spirv.size() * sizeof(std::uint32_t),
                  PipelineStageType::vertex};

    RasterParams params;
    params.prim_topology = decode_topology(topology);
    params.polygon_mode = decode_polygon_mode(polygon_mode);
    params.front_face_cull = front_face_cull;
    params.back_face_cull = back_face_cull;
    params.depth_test = depth_test;
    params.depth_write = depth_write;
    if (blending) {
      params.blending.emplace_back();
    }
    pipeline_ = device->create_raster_pipeline(
        sources, params, rhi_bindings, rhi_attributes,
        name.empty() ? "Forge VulkanGraphicsPipeline" : name);
    TI_ERROR_IF(!pipeline_, "Vulkan graphics pipeline creation failed.");
  }

  VulkanGraphicsPipelineResource(
      Program *program,
      const std::vector<std::uint32_t> &task_spirv,
      const std::vector<std::uint32_t> &mesh_spirv,
      const std::vector<std::uint32_t> &fragment_spirv,
      int topology,
      int polygon_mode,
      bool front_face_cull,
      bool back_face_cull,
      bool depth_test,
      bool depth_write,
      bool blending,
      const std::string &name)
      : program_(program), mesh_pipeline_(true), task_shader_(!task_spirv.empty()) {
    TI_ERROR_IF(program_ == nullptr,
                "Vulkan mesh pipeline requires a live Program.");
    const auto caps = program_->vulkan_mesh_shader_capabilities();
    TI_ERROR_IF(caps.at("mesh_shader") == 0 ||
                    (task_shader_ && caps.at("task_shader") == 0),
                "Vulkan mesh/task shader features are unavailable on the "
                "active device.");
    TI_ERROR_IF(mesh_spirv.empty() || fragment_spirv.empty(),
                "Vulkan mesh pipelines require mesh and fragment SPIR-V.");
    TI_ERROR_IF(mesh_spirv.size() * sizeof(std::uint32_t) >
                        kMaximumShaderBytes ||
                    fragment_spirv.size() * sizeof(std::uint32_t) >
                        kMaximumShaderBytes ||
                    task_spirv.size() * sizeof(std::uint32_t) >
                        kMaximumShaderBytes,
                "Vulkan mesh pipeline shader exceeds the 16 MiB safety "
                "limit.");
    TI_ERROR_IF(mesh_spirv.front() != 0x07230203u ||
                    fragment_spirv.front() != 0x07230203u ||
                    (task_shader_ && task_spirv.front() != 0x07230203u),
                "Vulkan mesh pipeline shaders must contain SPIR-V binary "
                "magic.");

    auto *device = dynamic_cast<GraphicsDevice *>(program_->get_graphics_device());
    TI_ERROR_IF(device == nullptr,
                "Vulkan mesh pipeline has no graphics device.");
    std::vector<PipelineSourceDesc> sources;
    sources.reserve(task_shader_ ? 3 : 2);
    sources.push_back({PipelineSourceType::spirv_binary,
                       const_cast<std::uint32_t *>(fragment_spirv.data()),
                       fragment_spirv.size() * sizeof(std::uint32_t),
                       PipelineStageType::fragment});
    sources.push_back({PipelineSourceType::spirv_binary,
                       const_cast<std::uint32_t *>(mesh_spirv.data()),
                       mesh_spirv.size() * sizeof(std::uint32_t),
                       PipelineStageType::mesh});
    if (task_shader_) {
      sources.push_back({PipelineSourceType::spirv_binary,
                         const_cast<std::uint32_t *>(task_spirv.data()),
                         task_spirv.size() * sizeof(std::uint32_t),
                         PipelineStageType::task});
    }

    RasterParams params;
    params.prim_topology = decode_topology(topology);
    params.polygon_mode = decode_polygon_mode(polygon_mode);
    params.front_face_cull = front_face_cull;
    params.back_face_cull = back_face_cull;
    params.depth_test = depth_test;
    params.depth_write = depth_write;
    if (blending) {
      params.blending.emplace_back();
    }
    pipeline_ = device->create_raster_pipeline(
        sources, params, {}, {},
        name.empty() ? "Forge VulkanMeshPipeline" : name);
    TI_ERROR_IF(!pipeline_, "Vulkan mesh pipeline creation failed.");
  }

  Pipeline *pipeline() const noexcept {
    return pipeline_.get();
  }

  const std::vector<VulkanGraphicsVertexBinding> &bindings() const noexcept {
    return bindings_;
  }

  bool mesh_pipeline() const noexcept {
    return mesh_pipeline_;
  }

  bool task_shader() const noexcept {
    return task_shader_;
  }

  bool register_prepared_resources(
      const std::shared_ptr<PreparedVulkanGraphicsResourceLease> &resources) {
    return prepared_resource_domain_.register_lease(resources);
  }

  void close_prepared_resources() noexcept {
    prepared_resource_domain_.close();
  }

  void append_prepared_resource_debug_stats(
      PreparedVulkanGraphicsResourceStats &result) const {
    prepared_resource_domain_.append_debug_stats(result);
  }

 private:
  Program *program_{nullptr};
  std::vector<VulkanGraphicsVertexBinding> bindings_;
  bool mesh_pipeline_{false};
  bool task_shader_{false};
  PreparedVulkanGraphicsResourceDomain prepared_resource_domain_;
  std::unique_ptr<Pipeline> pipeline_;
};

bool Program::vulkan_graphics_pipeline_available() const {
  return compile_config().arch == Arch::vulkan && program_impl_ &&
         const_cast<Program *>(this)->get_graphics_device() != nullptr;
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_graphics_indirect_capabilities() const {
  std::unordered_map<std::string, std::uint64_t> result{
      {"fixed_count", 0},
      {"multi_draw", 0},
      {"first_instance", 0},
      {"count_buffer", 0},
      {"max_draw_count", 0},
  };
  if (!vulkan_graphics_pipeline_available()) {
    return result;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_graphics_device());
  if (device == nullptr) {
    return result;
  }
  const auto &caps = device->vk_caps();
  result["fixed_count"] = 1;
  result["multi_draw"] = caps.multi_draw_indirect;
  result["first_instance"] = caps.draw_indirect_first_instance;
  result["count_buffer"] = caps.draw_indirect_count;
  result["max_draw_count"] = caps.max_draw_indirect_count;
  return result;
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_bindless_buffer_capabilities() const {
  std::unordered_map<std::string, std::uint64_t> result{
      {"descriptor_indexing", 0},
      {"storage_buffer_non_uniform_indexing", 0},
      {"fixed_count", 0},
      {"partially_bound", 0},
      {"update_after_bind", 0},
      {"variable_count", 0},
      {"runtime_array", 0},
      {"update_unused_while_pending", 0},
      {"max_fixed_count", 0},
      {"max_update_after_bind_descriptors_in_all_pools", 0},
      {"max_per_stage_update_after_bind_storage_buffers", 0},
      {"max_descriptor_set_update_after_bind_storage_buffers", 0},
  };
  if (!vulkan_graphics_pipeline_available()) {
    return result;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_graphics_device());
  if (device == nullptr) {
    return result;
  }
  const auto &caps = device->vk_caps();
  const auto max_fixed_count =
      std::min(caps.max_per_stage_descriptor_storage_buffers,
               caps.max_descriptor_set_storage_buffers);
  result["descriptor_indexing"] = caps.descriptor_indexing;
  result["storage_buffer_non_uniform_indexing"] =
      caps.descriptor_storage_buffer_array_non_uniform_indexing;
  result["fixed_count"] =
      caps.descriptor_storage_buffer_array_non_uniform_indexing &&
      max_fixed_count > 0;
  result["partially_bound"] = caps.descriptor_binding_partially_bound;
  result["update_after_bind"] =
      caps.descriptor_storage_buffer_update_after_bind;
  result["variable_count"] = caps.descriptor_binding_variable_count &&
                             caps.runtime_descriptor_array;
  result["runtime_array"] = caps.runtime_descriptor_array;
  result["update_unused_while_pending"] =
      caps.descriptor_update_unused_while_pending;
  result["max_fixed_count"] = max_fixed_count;
  result["max_update_after_bind_descriptors_in_all_pools"] =
      caps.max_update_after_bind_descriptors_in_all_pools;
  result["max_per_stage_update_after_bind_storage_buffers"] =
      caps.max_per_stage_descriptor_update_after_bind_storage_buffers;
  result["max_descriptor_set_update_after_bind_storage_buffers"] =
      caps.max_descriptor_set_update_after_bind_storage_buffers;
  return result;
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_mesh_shader_capabilities() const {
  std::unordered_map<std::string, std::uint64_t> result{
      {"mesh_shader", 0},
      {"task_shader", 0},
      {"max_task_group_count_x", 0},
      {"max_task_group_count_y", 0},
      {"max_task_group_count_z", 0},
      {"max_task_group_total_count", 0},
      {"max_task_group_invocations", 0},
      {"max_mesh_group_count_x", 0},
      {"max_mesh_group_count_y", 0},
      {"max_mesh_group_count_z", 0},
      {"max_mesh_group_total_count", 0},
      {"max_mesh_group_invocations", 0},
      {"max_mesh_output_vertices", 0},
      {"max_mesh_output_primitives", 0},
  };
  if (!vulkan_graphics_pipeline_available()) {
    return result;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_graphics_device());
  if (device == nullptr) {
    return result;
  }
  const auto &caps = device->vk_caps();
  result["mesh_shader"] = caps.mesh_shader;
  result["task_shader"] = caps.task_shader;
  result["max_task_group_count_x"] = caps.max_task_work_group_count[0];
  result["max_task_group_count_y"] = caps.max_task_work_group_count[1];
  result["max_task_group_count_z"] = caps.max_task_work_group_count[2];
  result["max_task_group_total_count"] =
      caps.max_task_work_group_total_count;
  result["max_task_group_invocations"] =
      caps.max_task_work_group_invocations;
  result["max_mesh_group_count_x"] = caps.max_mesh_work_group_count[0];
  result["max_mesh_group_count_y"] = caps.max_mesh_work_group_count[1];
  result["max_mesh_group_count_z"] = caps.max_mesh_work_group_count[2];
  result["max_mesh_group_total_count"] =
      caps.max_mesh_work_group_total_count;
  result["max_mesh_group_invocations"] =
      caps.max_mesh_work_group_invocations;
  result["max_mesh_output_vertices"] = caps.max_mesh_output_vertices;
  result["max_mesh_output_primitives"] = caps.max_mesh_output_primitives;
  return result;
}

std::size_t Program::debug_vulkan_graphics_pipeline_count() {
  std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
  return vulkan_graphics_pipelines_.size();
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_vulkan_graphics_resource_stats() {
  std::uint64_t live = 0;
  std::uint64_t queued_for_completion = 0;
  PreparedVulkanGraphicsResourceStats prepared;
  {
    std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
    live = vulkan_graphics_pipelines_.size();
    queued_for_completion = vulkan_graphics_pipeline_retirements_.size();
    for (const auto &item : vulkan_graphics_pipelines_) {
      item.second->append_prepared_resource_debug_stats(prepared);
    }
    for (const auto &resource : vulkan_graphics_pipeline_retirements_) {
      resource->append_prepared_resource_debug_stats(prepared);
    }
  }
  const auto completion_retained = runtime_completion_resource_count(
      kVulkanGraphicsPipelineResourceKind);
  std::unordered_map<std::string, std::uint64_t> result{
      {"live", live},
      {"retiring", queued_for_completion + completion_retained},
      {"queued_for_completion", queued_for_completion},
      {"completion_retained", completion_retained},
      {"prepared_resource_leases", prepared.leases},
      {"prepared_draw_resources", prepared.draws},
      {"prepared_descriptor_sets", prepared.descriptor_sets},
      {"prepared_raster_resources", prepared.raster_resources}};
  if (program_impl_) {
    const auto replay =
        program_impl_->debug_graphics_command_replay_stats();
    result.insert(replay.begin(), replay.end());
  }
  return result;
}

std::uint64_t Program::create_vulkan_graphics_pipeline(
    const std::vector<std::uint32_t> &vertex_spirv,
    const std::vector<std::uint32_t> &fragment_spirv,
    const std::vector<VulkanGraphicsVertexBinding> &vertex_bindings,
    const std::vector<VulkanGraphicsVertexAttribute> &vertex_attributes,
    int topology,
    int polygon_mode,
    bool front_face_cull,
    bool back_face_cull,
    bool depth_test,
    bool depth_write,
    bool blending,
    const std::string &name) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_graphics_pipeline_available(),
              "Vulkan graphics pipelines require the Vulkan backend.");
  auto resource = std::make_shared<VulkanGraphicsPipelineResource>(
      this, vertex_spirv, fragment_spirv, vertex_bindings, vertex_attributes,
      topology, polygon_mode, front_face_cull, back_face_cull, depth_test,
      depth_write, blending, name);
  std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
  TI_ERROR_IF(next_vulkan_graphics_pipeline_handle_ == 0,
              "Vulkan graphics pipeline handle space exhausted.");
  const std::uint64_t handle = next_vulkan_graphics_pipeline_handle_++;
  vulkan_graphics_pipelines_.emplace(handle, std::move(resource));
  return handle;
}

std::uint64_t Program::create_vulkan_mesh_pipeline(
    const std::vector<std::uint32_t> &task_spirv,
    const std::vector<std::uint32_t> &mesh_spirv,
    const std::vector<std::uint32_t> &fragment_spirv,
    int topology,
    int polygon_mode,
    bool front_face_cull,
    bool back_face_cull,
    bool depth_test,
    bool depth_write,
    bool blending,
    const std::string &name) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_graphics_pipeline_available(),
              "Vulkan mesh pipelines require the Vulkan backend.");
  auto resource = std::make_shared<VulkanGraphicsPipelineResource>(
      this, task_spirv, mesh_spirv, fragment_spirv, topology, polygon_mode,
      front_face_cull, back_face_cull, depth_test, depth_write, blending,
      name);
  std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
  TI_ERROR_IF(next_vulkan_graphics_pipeline_handle_ == 0,
              "Vulkan graphics pipeline handle space exhausted.");
  const std::uint64_t handle = next_vulkan_graphics_pipeline_handle_++;
  vulkan_graphics_pipelines_.emplace(handle, std::move(resource));
  return handle;
}

std::size_t Program::vulkan_graphics_draw(
    std::uint64_t handle,
    Texture *color,
    Texture *depth,
    const std::vector<std::pair<std::uint32_t, Ndarray *>> &vertex_buffers,
    Ndarray *index_buffer,
    const VulkanGraphicsDrawInfo &draw) {
  VulkanGraphicsDrawCommand command;
  command.pipeline_handle = handle;
  command.vertex_buffers = vertex_buffers;
  command.index_buffer = index_buffer;
  command.draw = draw;
  VulkanGraphicsPassInfo pass;
  pass.clear_color = draw.clear_color;
  pass.viewport = draw.viewport;
  return vulkan_graphics_pass(color, depth, {std::move(command)}, pass);
}

std::shared_ptr<PreparedVulkanGraphicsPass> Program::prepare_vulkan_graphics_pass(
    Texture *color,
    Texture *depth,
    const std::vector<VulkanGraphicsDrawCommand> &commands,
    const VulkanGraphicsPassInfo &pass) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  auto packet = std::make_shared<PreparedVulkanGraphicsPass>();
  packet->state = std::make_shared<PreparedVulkanGraphicsPass::State>();
  auto &prepared = *packet->state;
  prepared.owner = this;
  prepared.generation = runtime_program_generation();
  TI_ERROR_IF(!color,
              "Vulkan graphics pass requires a color attachment Texture.");
  TI_ERROR_IF(commands.empty() || commands.size() > kMaximumPassDraws,
              "Vulkan graphics pass requires 1 to {} draws.",
              kMaximumPassDraws);
  TI_ERROR_IF(color->owning_program() != this,
              "Vulkan graphics color attachment belongs to another Program.");
  const auto color_size = color->get_size();
  TI_ERROR_IF(color_size[0] <= 0 || color_size[1] <= 0 || color_size[2] != 1,
              "Vulkan graphics color attachment must be a nonempty 2D "
              "Texture.");
  TI_ERROR_IF(color->get_buffer_format() == BufferFormat::depth16 ||
                  color->get_buffer_format() ==
                      BufferFormat::depth24stencil8 ||
                  color->get_buffer_format() == BufferFormat::depth32f,
              "Vulkan graphics color attachment cannot use a depth format.");
  if (depth) {
    TI_ERROR_IF(depth->owning_program() != this,
                "Vulkan graphics depth attachment belongs to another "
                "Program.");
    const auto depth_size = depth->get_size();
    TI_ERROR_IF(depth_size != color_size,
                "Vulkan graphics color and depth attachments must have the "
                "same 2D shape.");
    TI_ERROR_IF(depth->get_buffer_format() != BufferFormat::depth32f,
                "Vulkan graphics P0 depth attachments require depth32f.");
  }
  std::array<std::uint32_t, 4> viewport = pass.viewport;
  if (viewport[2] == 0 && viewport[3] == 0) {
    viewport = {0, 0, static_cast<std::uint32_t>(color_size[0]),
                static_cast<std::uint32_t>(color_size[1])};
  }
  const std::uint64_t viewport_x_end =
      static_cast<std::uint64_t>(viewport[0]) + viewport[2];
  const std::uint64_t viewport_y_end =
      static_cast<std::uint64_t>(viewport[1]) + viewport[3];
  TI_ERROR_IF(viewport[2] == 0 || viewport[3] == 0 ||
                  viewport_x_end >
                      static_cast<std::uint32_t>(color_size[0]) ||
                  viewport_y_end >
                      static_cast<std::uint32_t>(color_size[1]),
              "Vulkan graphics viewport must be a nonempty rectangle inside "
              "the color attachment.");

  std::vector<std::shared_ptr<VulkanGraphicsPipelineResource>> pipelines;
  pipelines.reserve(commands.size());
  {
    std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
    for (const auto &command : commands) {
      const auto found =
          vulkan_graphics_pipelines_.find(command.pipeline_handle);
      TI_ERROR_IF(found == vulkan_graphics_pipelines_.end(),
                  "Vulkan graphics pipeline handle is stale or closed.");
      pipelines.push_back(found->second);
    }
  }

  std::vector<const Ndarray *> arrays;
  std::vector<const Texture *> textures{color};
  if (depth) {
    textures.push_back(depth);
  }
  std::vector<RecordedGraphicsDraw> recorded_draws;
  recorded_draws.reserve(commands.size());
  auto *device = static_cast<vulkan::VulkanDevice *>(get_graphics_device());
  for (std::size_t draw_index = 0; draw_index < commands.size(); ++draw_index) {
    const auto &command = commands[draw_index];
    const auto &draw = command.draw;
    const auto &resource = pipelines[draw_index];
    const bool mesh_draw = command.mesh.has_value();
    TI_ERROR_IF(mesh_draw != resource->mesh_pipeline(),
                "Vulkan graphics pipeline and draw command kinds must agree.");
    TI_ERROR_IF(mesh_draw &&
                    (command.indirect.has_value() || command.index_buffer ||
                     !command.vertex_buffers.empty()),
                "Vulkan mesh draws cannot bind vertex, index, or classic "
                "indirect draw inputs.");
    TI_ERROR_IF(!mesh_draw && !command.indirect.has_value() &&
                    (draw.element_count == 0 || draw.instance_count == 0),
                "Vulkan graphics direct draw counts must be positive.");
    TI_ERROR_IF(draw.indexed != (command.index_buffer != nullptr),
                "Vulkan graphics indexed draw and index-buffer binding must "
                "agree.");

    RecordedGraphicsDraw recorded;
    recorded.pipeline = resource;
    recorded.draw = draw;
    if (mesh_draw) {
      const auto &mesh = *command.mesh;
      const auto caps = vulkan_mesh_shader_capabilities();
      const std::string prefix =
          resource->task_shader() ? "max_task_" : "max_mesh_";
      const std::uint64_t total =
          static_cast<std::uint64_t>(mesh.group_count_x) *
          mesh.group_count_y * mesh.group_count_z;
      TI_ERROR_IF(mesh.group_count_x == 0 || mesh.group_count_y == 0 ||
                      mesh.group_count_z == 0 ||
                      mesh.group_count_x > caps.at(prefix + "group_count_x") ||
                      mesh.group_count_y > caps.at(prefix + "group_count_y") ||
                      mesh.group_count_z > caps.at(prefix + "group_count_z") ||
                      total > caps.at(prefix + "group_total_count"),
                  "Vulkan mesh draw group counts exceed the active device "
                  "limits.");
      recorded.mesh = mesh;
    }
    if (command.indirect.has_value()) {
      const auto &indirect = *command.indirect;
      const auto caps = vulkan_graphics_indirect_capabilities();
      TI_ERROR_IF(caps.at("fixed_count") == 0,
                  "Vulkan indirect graphics commands are unavailable.");
      TI_ERROR_IF(indirect.command_buffer == nullptr,
                  "Vulkan indirect graphics requires a command buffer.");
      TI_ERROR_IF(indirect.max_draw_count == 0 ||
                      indirect.max_draw_count > caps.at("max_draw_count"),
                  "Vulkan indirect graphics draw count exceeds the active "
                  "device limit.");
      TI_ERROR_IF(indirect.max_draw_count > 1 && caps.at("multi_draw") == 0,
                  "Vulkan multi-draw indirect is unavailable on the active "
                  "device.");
      TI_ERROR_IF(indirect.first_instance_may_be_nonzero &&
                      caps.at("first_instance") == 0,
                  "Vulkan indirect firstInstance is unavailable on the "
                  "active device.");
      TI_ERROR_IF(indirect.count_buffer != nullptr &&
                      caps.at("count_buffer") == 0,
                  "Vulkan indirect-count commands are unavailable on the "
                  "active device.");
      const std::size_t command_size =
          draw.indexed ? sizeof(VkDrawIndexedIndirectCommand)
                       : sizeof(VkDrawIndirectCommand);
      TI_ERROR_IF(indirect.stride < command_size ||
                      indirect.stride % sizeof(std::uint32_t) != 0 ||
                      indirect.command_offset % sizeof(std::uint32_t) != 0 ||
                      indirect.count_offset % sizeof(std::uint32_t) != 0,
                  "Vulkan indirect command offsets and stride violate the "
                  "four-byte command ABI.");
      TI_ERROR_IF(indirect.vertex_record_limit == 0 ||
                      indirect.instance_record_limit == 0 ||
                      (draw.indexed && indirect.index_element_limit == 0) ||
                      (!draw.indexed && indirect.index_element_limit != 0),
                  "Vulkan indirect graphics requires positive declared "
                  "vertex/instance limits and an indexed-only index limit.");

      auto validate_indirect_array = [&](Ndarray *array,
                                         const char *role,
                                         std::size_t required_bytes,
                                         std::size_t offset) {
        TI_ERROR_IF(array == nullptr || array->owning_program() != this ||
                        array->get_device_allocation().device != device,
                    "Vulkan graphics {} buffer belongs to another runtime "
                    "or device.",
                    role);
        TI_ERROR_IF(array->get_element_data_type() != PrimitiveType::u32,
                    "Vulkan graphics {} buffer must use u32.", role);
        TI_ERROR_IF(!int(device->allocation_usage(
                             array->get_device_allocation()) &
                         AllocUsage::Indirect),
                    "Vulkan graphics {} buffer lacks indirect usage.", role);
        const auto available = ndarray_bytes(array, role);
        TI_ERROR_IF(offset > available || required_bytes > available - offset,
                    "Vulkan graphics {} buffer is too small for the declared "
                    "command range.",
                    role);
      };
      const std::size_t command_count = indirect.max_draw_count;
      TI_ERROR_IF(
          command_count - 1 >
              ((std::numeric_limits<std::size_t>::max)() - command_size) /
                  indirect.stride,
          "Vulkan indirect command byte range overflows size_t.");
      const std::size_t required_command_bytes =
          (command_count - 1) * indirect.stride + command_size;
      validate_indirect_array(indirect.command_buffer, "indirect command",
                              required_command_bytes,
                              indirect.command_offset);
      if (indirect.count_buffer != nullptr) {
        validate_indirect_array(indirect.count_buffer, "indirect count",
                                sizeof(std::uint32_t),
                                indirect.count_offset);
      }
      arrays.push_back(indirect.command_buffer);
      if (indirect.count_buffer != nullptr) {
        arrays.push_back(indirect.count_buffer);
      }
      recorded.indirect = RecordedGraphicsIndirect{
          indirect.command_buffer->get_device_allocation(),
          indirect.count_buffer == nullptr
              ? kDeviceNullAllocation
              : indirect.count_buffer->get_device_allocation(),
          indirect.command_offset,
          indirect.count_offset,
          indirect.max_draw_count,
          indirect.stride,
          indirect.vertex_record_limit,
          indirect.instance_record_limit,
          indirect.index_element_limit,
          indirect.first_instance_may_be_nonzero};
    }
    std::unordered_map<std::uint32_t, Ndarray *> supplied;
    for (const auto &[binding, array] : command.vertex_buffers) {
      TI_ERROR_IF(!array,
                  "Vulkan graphics vertex binding {} is null.", binding);
      TI_ERROR_IF(!supplied.emplace(binding, array).second,
                  "Vulkan graphics vertex binding {} is duplicated.",
                  binding);
      TI_ERROR_IF(array->owning_program() != this ||
                      array->get_device_allocation().device != device,
                  "Vulkan graphics vertex binding {} belongs to another "
                  "runtime or device.",
                  binding);
      TI_ERROR_IF(!int(device->allocation_usage(
                           array->get_device_allocation()) &
                       AllocUsage::Vertex),
                  "Vulkan graphics vertex binding {} was not allocated for "
                  "vertex input.",
                  binding);
      arrays.push_back(array);
      recorded.vertex_buffers.emplace_back(binding,
                                           array->get_device_allocation());
    }

    TI_ERROR_IF(supplied.size() != resource->bindings().size(),
                "Vulkan graphics draw must bind every declared vertex buffer.");
    for (const auto &binding : resource->bindings()) {
      const auto found = supplied.find(binding.binding);
      TI_ERROR_IF(found == supplied.end(),
                  "Vulkan graphics draw is missing vertex binding {}.",
                  binding.binding);
      const std::size_t available = ndarray_bytes(found->second, "vertex");
      std::uint64_t records = 0;
      if (recorded.indirect.has_value()) {
        records = binding.instance
                      ? recorded.indirect->instance_record_limit
                      : recorded.indirect->vertex_record_limit;
      } else if (binding.instance) {
        records = static_cast<std::uint64_t>(draw.first_instance) +
                  draw.instance_count;
      } else if (draw.indexed) {
        const std::int64_t first_record =
            static_cast<std::int64_t>(draw.index_min) + draw.vertex_offset;
        const std::int64_t last_record =
            static_cast<std::int64_t>(draw.index_max) + draw.vertex_offset;
        TI_ERROR_IF(first_record < 0 || last_record < first_record,
                    "Vulkan graphics indexed vertex binding {} has an invalid "
                    "declared index range after applying vertex_offset.",
                    binding.binding);
        records = static_cast<std::uint64_t>(last_record) + 1;
      } else {
        records = static_cast<std::uint64_t>(draw.first_vertex) +
                  draw.element_count;
      }
      TI_ERROR_IF(
          records >
                  (std::numeric_limits<std::size_t>::max)() / binding.stride ||
              static_cast<std::size_t>(records) * binding.stride > available,
          "Vulkan graphics vertex binding {} is too small for the declared "
          "draw range.",
          binding.binding);
    }

    if (command.index_buffer) {
      Ndarray *index_buffer = command.index_buffer;
      TI_ERROR_IF(index_buffer->owning_program() != this ||
                      index_buffer->get_device_allocation().device != device,
                  "Vulkan graphics index buffer belongs to another runtime or "
                  "device.");
      TI_ERROR_IF(index_buffer->get_element_data_type() != PrimitiveType::u32,
                  "Vulkan graphics index buffer must use u32.");
      TI_ERROR_IF(!int(device->allocation_usage(
                           index_buffer->get_device_allocation()) &
                       AllocUsage::Index),
                  "Vulkan graphics index buffer was not allocated for index "
                  "input.");
      const std::uint64_t index_end = recorded.indirect.has_value()
                                          ? recorded.indirect->index_element_limit
                                          : static_cast<std::uint64_t>(
                                                draw.first_index) +
                                                draw.element_count;
      TI_ERROR_IF(
          index_end > (std::numeric_limits<std::size_t>::max)() /
                          sizeof(std::uint32_t) ||
              static_cast<std::size_t>(index_end) * sizeof(std::uint32_t) >
                  ndarray_bytes(index_buffer, "index"),
          "Vulkan graphics index buffer is too small for the declared draw "
          "range.");
      arrays.push_back(index_buffer);
      recorded.index_buffer = index_buffer->get_device_allocation();
    }

    std::unordered_set<std::uint64_t> shader_bindings;
    for (const auto &shader : command.shader_buffers) {
      const std::uint64_t key =
          (static_cast<std::uint64_t>(shader.set_index) << 32) |
          shader.binding;
      TI_ERROR_IF(!shader_bindings.insert(key).second,
                  "Vulkan graphics shader buffer set {} binding {} is "
                  "duplicated.",
                  shader.set_index, shader.binding);
      if (!shader.array_elements.empty()) {
        const auto bindless_caps = vulkan_bindless_buffer_capabilities();
        constexpr std::size_t kMaximumProviderBindlessBuffers = 64;
        const std::size_t limit = std::min<std::size_t>(
            bindless_caps.at("max_fixed_count"),
            kMaximumProviderBindlessBuffers);
        TI_ERROR_IF(bindless_caps.at("fixed_count") == 0 || !shader.storage,
                    "Vulkan bindless graphics requires fixed-count storage "
                    "buffer descriptor indexing.");
        TI_ERROR_IF(shader.array_elements.size() < 2 ||
                        shader.array_elements.size() > limit,
                    "Vulkan bindless graphics table size must be between 2 "
                    "and the provider/device limit {}.",
                    limit);
        RecordedGraphicsShaderBuffer recorded_shader;
        recorded_shader.set_index = shader.set_index;
        recorded_shader.binding = shader.binding;
        recorded_shader.storage = true;
        recorded_shader.array_elements.reserve(shader.array_elements.size());
        for (Ndarray *element : shader.array_elements) {
          TI_ERROR_IF(element == nullptr || element->owning_program() != this ||
                          element->get_device_allocation().device != device,
                      "Vulkan bindless graphics buffer set {} binding {} "
                      "contains an element from another runtime or device.",
                      shader.set_index, shader.binding);
          const DeviceAllocation allocation =
              element->get_device_allocation();
          TI_ERROR_IF(!int(device->allocation_usage(allocation) &
                           AllocUsage::Storage),
                      "Vulkan bindless graphics buffer set {} binding {} "
                      "contains an element without storage usage.",
                      shader.set_index, shader.binding);
          arrays.push_back(element);
          recorded_shader.array_elements.push_back(allocation);
        }
        recorded.shader_buffers.push_back(std::move(recorded_shader));
        continue;
      }
      TI_ERROR_IF(shader.array == nullptr,
                  "Vulkan graphics shader buffer set {} binding {} is null.",
                  shader.set_index, shader.binding);
      const DeviceAllocation allocation =
          shader.array->get_device_allocation();
      TI_ERROR_IF(shader.array->owning_program() != this ||
                      allocation.device != device,
                  "Vulkan graphics shader buffer set {} binding {} belongs "
                  "to another runtime or device.",
                  shader.set_index, shader.binding);
      const AllocUsage required_usage =
          shader.storage ? AllocUsage::Storage : AllocUsage::Uniform;
      TI_ERROR_IF(!int(device->allocation_usage(allocation) & required_usage),
                  "Vulkan graphics shader buffer set {} binding {} was not "
                  "allocated for {} input.",
                  shader.set_index, shader.binding,
                  shader.storage ? "storage" : "uniform");
      arrays.push_back(shader.array);
      recorded.shader_buffers.push_back(
          {shader.set_index, shader.binding, allocation, {}, shader.storage});
    }
    for (const auto &shader : command.shader_images) {
      const auto key = (static_cast<std::uint64_t>(shader.set_index) << 32) |
                       shader.binding;
      TI_ERROR_IF(!shader_bindings.insert(key).second,
                  "Vulkan graphics descriptor set {} binding {} is duplicated",
                  shader.set_index, shader.binding);
      auto *image = shader.texture;
      TI_ERROR_IF(image == nullptr || image->owning_program() != this,
                  "Vulkan graphics sampled image must belong to this Program");
      const auto allocation = image->get_device_allocation();
      TI_ERROR_IF(allocation.device != device ||
                      allocation == color->get_device_allocation() ||
                      (depth && allocation == depth->get_device_allocation()),
                  "Vulkan sampled images must be on this device and cannot "
                  "alias render-pass attachments");
      const auto format = image->get_buffer_format();
      TI_ERROR_IF(format == BufferFormat::depth16 ||
                      format == BufferFormat::depth24stencil8 ||
                      format == BufferFormat::depth32f ||
                      !is_float_sampled_texture_format(format),
                  "Graphics sampled images require floating-point or normalized color textures");
      textures.push_back(image);
      recorded.shader_images.push_back(
          {shader.set_index, shader.binding, allocation, image->sampler_config_});
    }
    recorded_draws.push_back(std::move(recorded));
  }

  auto ndarray_leases = acquire_ndarray_leases(arrays);
  auto texture_leases = acquire_texture_leases(textures);
  const DeviceAllocation color_allocation = color->get_device_allocation();
  const DeviceAllocation depth_allocation =
      depth ? depth->get_device_allocation() : kDeviceNullAllocation;
  const int width = color_size[0];
  const int height = color_size[1];

  std::vector<std::uint64_t> replay_key;
  const bool replay_eligible =
      pass.retained_replay && pass.color_clear &&
      (depth == nullptr || pass.depth_clear) && !compile_config().debug &&
      !compile_config().kernel_profiler;
  if (replay_eligible) {
    // Exact vector equality, including allocation generations, is the replay
    // contract. Hash-only identity would permit a stale descriptor or
    // attachment collision and is deliberately not used.
    replay_key.reserve(32 + commands.size() * 32);
    replay_key.push_back(0x4652475250415353ull);  // "FRGRPASS"
    replay_key.push_back(1);
    replay_key.push_back(reinterpret_cast<std::uintptr_t>(this));
    append_graphics_allocation_key(replay_key, device, color_allocation);
    replay_key.push_back(depth == nullptr ? 0 : 1);
    if (depth != nullptr) {
      append_graphics_allocation_key(replay_key, device, depth_allocation);
    }
    replay_key.push_back(static_cast<std::uint64_t>(width));
    replay_key.push_back(static_cast<std::uint64_t>(height));
    for (const auto component : pass.clear_color) {
      replay_key.push_back(graphics_float_bits(component));
    }
    for (const auto component : viewport) {
      replay_key.push_back(component);
    }
    replay_key.push_back(commands.size());
    for (std::size_t draw_index = 0; draw_index < commands.size();
         ++draw_index) {
      const auto &command = commands[draw_index];
      const auto &recorded = recorded_draws[draw_index];
      replay_key.push_back(command.pipeline_handle);
      replay_key.push_back(recorded.mesh.has_value() ? 1 : 0);
      if (recorded.mesh.has_value()) {
        replay_key.push_back(recorded.mesh->group_count_x);
        replay_key.push_back(recorded.mesh->group_count_y);
        replay_key.push_back(recorded.mesh->group_count_z);
      }
      replay_key.push_back(recorded.vertex_buffers.size());
      for (const auto &[binding, allocation] : recorded.vertex_buffers) {
        replay_key.push_back(binding);
        append_graphics_allocation_key(replay_key, device, allocation);
      }
      replay_key.push_back(recorded.draw.indexed ? 1 : 0);
      if (recorded.draw.indexed) {
        append_graphics_allocation_key(replay_key, device,
                                       recorded.index_buffer);
      }
      replay_key.push_back(recorded.shader_buffers.size());
      for (const auto &shader : recorded.shader_buffers) {
        replay_key.push_back(shader.set_index);
        replay_key.push_back(shader.binding);
        replay_key.push_back(shader.storage ? 1 : 0);
        replay_key.push_back(shader.array_elements.size());
        if (shader.array_elements.empty()) {
          append_graphics_allocation_key(replay_key, device,
                                         shader.allocation);
        } else {
          for (const auto allocation : shader.array_elements) {
            append_graphics_allocation_key(replay_key, device, allocation);
          }
        }
      }
      replay_key.push_back(recorded.shader_images.size());
      for (const auto &image : recorded.shader_images) {
        replay_key.push_back(image.set_index);
        replay_key.push_back(image.binding);
        append_graphics_allocation_key(replay_key, device, image.allocation);
      }
      replay_key.push_back(recorded.indirect.has_value() ? 1 : 0);
      if (recorded.indirect.has_value()) {
        const auto &indirect = *recorded.indirect;
        append_graphics_allocation_key(replay_key, device,
                                       indirect.command_buffer);
        replay_key.push_back(indirect.count_buffer == kDeviceNullAllocation
                                 ? 0
                                 : 1);
        if (indirect.count_buffer != kDeviceNullAllocation) {
          append_graphics_allocation_key(replay_key, device,
                                         indirect.count_buffer);
        }
        replay_key.push_back(indirect.command_offset);
        replay_key.push_back(indirect.count_offset);
        replay_key.push_back(indirect.max_draw_count);
        replay_key.push_back(indirect.stride);
        replay_key.push_back(indirect.vertex_record_limit);
        replay_key.push_back(indirect.instance_record_limit);
        replay_key.push_back(indirect.index_element_limit);
        replay_key.push_back(indirect.first_instance_may_be_nonzero ? 1 : 0);
      } else if (!recorded.mesh.has_value()) {
        const auto &draw = recorded.draw;
        replay_key.push_back(draw.element_count);
        replay_key.push_back(draw.instance_count);
        replay_key.push_back(draw.first_vertex);
        replay_key.push_back(draw.first_index);
        replay_key.push_back(draw.first_instance);
        replay_key.push_back(static_cast<std::uint32_t>(draw.vertex_offset));
        replay_key.push_back(draw.index_min);
        replay_key.push_back(draw.index_max);
      }
    }
  }

  auto resource_payload =
      std::make_shared<PreparedVulkanGraphicsResourcePayload>();
  resource_payload->draws.reserve(recorded_draws.size());
  for (const auto &recorded : recorded_draws) {
    PreparedVulkanGraphicsDrawResources draw_resources;
    std::unordered_map<std::uint32_t, std::unique_ptr<ShaderResourceSet>>
        resource_sets;
    for (const auto &shader : recorded.shader_buffers) {
      auto &resource_set = resource_sets[shader.set_index];
      if (!resource_set) {
        resource_set = device->create_resource_set_unique();
        TI_ERROR_IF(!resource_set,
                    "Vulkan graphics shader resource set creation failed.");
      }
      if (!shader.array_elements.empty()) {
        resource_set->rw_buffer_array(shader.binding, shader.array_elements);
      } else if (shader.storage) {
        resource_set->rw_buffer(shader.binding, shader.allocation);
      } else {
        resource_set->buffer(shader.binding, shader.allocation);
      }
    }
    for (const auto &image : recorded.shader_images) {
      auto &resource_set = resource_sets[image.set_index];
      if (!resource_set) {
        resource_set = device->create_resource_set_unique();
        TI_ERROR_IF(!resource_set,
                    "Vulkan graphics shader resource set creation failed.");
      }
      resource_set->image(image.binding, image.allocation, image.sampler);
    }
    draw_resources.shader_resource_sets.reserve(resource_sets.size());
    for (auto &[set_index, resource_set] : resource_sets) {
      const RhiResult prepare_result =
          resource_set->prepare_for_replay(/*patch_existing=*/false);
      TI_ERROR_IF(
          prepare_result != RhiResult::success,
          "Vulkan graphics shader resource set {} preparation failed: "
          "RhiResult({}).",
          set_index, prepare_result);
      draw_resources.shader_resource_sets.emplace_back(set_index,
                                                        std::move(resource_set));
    }
    std::sort(draw_resources.shader_resource_sets.begin(),
              draw_resources.shader_resource_sets.end(),
              [](const auto &lhs, const auto &rhs) {
                return lhs.first < rhs.first;
              });
    if (!recorded.mesh.has_value()) {
      draw_resources.raster = device->create_raster_resources_unique();
      TI_ERROR_IF(!draw_resources.raster,
                  "Vulkan graphics raster resource creation failed.");
      for (const auto &[binding, allocation] : recorded.vertex_buffers) {
        draw_resources.raster->vertex_buffer(allocation.get_ptr(), binding);
      }
      if (recorded.draw.indexed) {
        draw_resources.raster->index_buffer(recorded.index_buffer.get_ptr(),
                                            32);
      }
    }
    resource_payload->draws.push_back(std::move(draw_resources));
  }
  auto resource_lease =
      std::make_shared<PreparedVulkanGraphicsResourceLease>(resource_payload);
  std::unordered_set<const VulkanGraphicsPipelineResource *>
      registered_pipelines;
  for (const auto &pipeline : pipelines) {
    if (!registered_pipelines.insert(pipeline.get()).second) {
      continue;
    }
    TI_ERROR_IF(!pipeline->register_prepared_resources(resource_lease),
                "Vulkan graphics pipeline closed during pass preparation.");
  }
  prepared.resources = resource_lease;

  prepared.record =
      [recorded_draws = std::move(recorded_draws),
       resource_lease, pass, viewport,
       viewport_x_end, viewport_y_end, color_allocation, depth_allocation,
       width, height](GraphicsDevice *, CommandList *commands) {
        auto resource_payload = resource_lease->acquire();
        TI_ERROR_IF(!resource_payload,
                    "Prepared Vulkan graphics resources are stale or closed.");
        auto *vulkan_commands =
            static_cast<vulkan::VulkanCommandList *>(commands);
        vulkan_commands->set_next_renderpass_color_final_layout(
            ImageLayout::color_attachment);
        bool clear = pass.color_clear;
        std::vector<float> clear_color(pass.clear_color.begin(),
                                       pass.clear_color.end());
        DeviceAllocation color_target = color_allocation;
        DeviceAllocation depth_target = depth_allocation;
        DeviceAllocation *depth_target_ptr =
            depth_target == kDeviceNullAllocation ? nullptr : &depth_target;
        const BufferTransition indirect_transition{
            BufferBarrierStage::Transfer | BufferBarrierStage::Compute,
            BufferBarrierAccess::TransferWrite |
                BufferBarrierAccess::ShaderWrite,
            BufferBarrierStage::IndirectCommand,
            BufferBarrierAccess::IndirectCommandRead};
        for (const auto &recorded : recorded_draws) {
          if (!recorded.indirect.has_value()) {
            continue;
          }
          const auto &indirect = *recorded.indirect;
          const std::size_t command_size =
              recorded.draw.indexed ? sizeof(VkDrawIndexedIndirectCommand)
                                    : sizeof(VkDrawIndirectCommand);
          const std::size_t command_bytes =
              (static_cast<std::size_t>(indirect.max_draw_count) - 1) *
                  indirect.stride +
              command_size;
          commands->buffer_transition(
              indirect.command_buffer.get_ptr(indirect.command_offset),
              command_bytes, indirect_transition);
          if (indirect.count_buffer != kDeviceNullAllocation) {
            commands->buffer_transition(
                indirect.count_buffer.get_ptr(indirect.count_offset),
                sizeof(std::uint32_t), indirect_transition);
          }
        }
        commands->begin_renderpass(0, 0, width, height, 1, &color_target,
                                   &clear, &clear_color, depth_target_ptr,
                                   depth_target_ptr != nullptr &&
                                       pass.depth_clear);
        commands->set_raster_viewport_and_scissor(
            static_cast<int>(viewport[0]), static_cast<int>(viewport[1]),
            static_cast<int>(viewport_x_end),
            static_cast<int>(viewport_y_end));
        for (std::size_t draw_index = 0; draw_index < recorded_draws.size();
             ++draw_index) {
          const auto &recorded = recorded_draws[draw_index];
          const auto &draw_resources = resource_payload->draws[draw_index];
          auto pipeline = recorded.pipeline.lock();
          TI_ERROR_IF(!pipeline,
                      "Vulkan graphics pipeline closed or retired while recording");
          commands->bind_pipeline(pipeline->pipeline());
          if (!recorded.mesh.has_value()) {
            const RhiResult raster_result =
                commands->bind_raster_resources(draw_resources.raster.get());
            TI_ERROR_IF(
                raster_result != RhiResult::success,
                "Vulkan graphics raster resource binding failed: "
                "RhiResult({}).",
                raster_result);
          }
          for (const auto &[set_index, resource_set] :
               draw_resources.shader_resource_sets) {
            const RhiResult shader_result =
                commands->bind_shader_resources(resource_set.get(),
                                                 set_index);
            TI_ERROR_IF(
                shader_result != RhiResult::success,
                "Vulkan graphics shader resource set {} binding failed: "
                "RhiResult({}).",
                set_index, shader_result);
          }

          const auto &draw = recorded.draw;
          if (recorded.mesh.has_value()) {
            const auto &mesh = *recorded.mesh;
            const RhiResult mesh_result = commands->draw_mesh_tasks(
                mesh.group_count_x, mesh.group_count_y, mesh.group_count_z,
                pipeline->task_shader());
            TI_ERROR_IF(mesh_result != RhiResult::success,
                        "Vulkan mesh draw failed: RhiResult({}).",
                        mesh_result);
          } else if (recorded.indirect.has_value()) {
            const auto &indirect = *recorded.indirect;
            const auto command_ptr =
                indirect.command_buffer.get_ptr(indirect.command_offset);
            RhiResult indirect_result = RhiResult::not_supported;
            if (indirect.count_buffer != kDeviceNullAllocation) {
              const auto count_ptr =
                  indirect.count_buffer.get_ptr(indirect.count_offset);
              indirect_result = draw.indexed
                                    ? commands->draw_indexed_indirect_count(
                                          command_ptr, count_ptr,
                                          indirect.max_draw_count,
                                          indirect.stride)
                                    : commands->draw_indirect_count(
                                          command_ptr, count_ptr,
                                          indirect.max_draw_count,
                                          indirect.stride);
            } else {
              indirect_result = draw.indexed
                                    ? commands->draw_indexed_indirect(
                                          command_ptr,
                                          indirect.max_draw_count,
                                          indirect.stride)
                                    : commands->draw_indirect(
                                          command_ptr,
                                          indirect.max_draw_count,
                                          indirect.stride);
            }
            TI_ERROR_IF(
                indirect_result != RhiResult::success,
                "Vulkan graphics indirect draw failed: RhiResult({}).",
                indirect_result);
          } else if (draw.indexed &&
              (draw.instance_count > 1 || draw.first_instance != 0)) {
            commands->draw_indexed_instance(
                draw.element_count, draw.instance_count, draw.vertex_offset,
                draw.first_index, draw.first_instance);
          } else if (draw.indexed) {
            commands->draw_indexed(draw.element_count, draw.vertex_offset,
                                   draw.first_index);
          } else if (draw.instance_count > 1 || draw.first_instance != 0) {
            commands->draw_instance(draw.element_count, draw.instance_count,
                                    draw.first_vertex, draw.first_instance);
          } else {
            commands->draw(draw.element_count, draw.first_vertex);
          }
        }
        commands->end_renderpass();
      };
  prepared.images = depth ? std::vector<ComputeOpImageRef>{
                  {color_allocation, ImageLayout::color_attachment,
                   ImageLayout::shader_read},
                  {depth_allocation, ImageLayout::depth_attachment,
                   ImageLayout::shader_read}}
            : std::vector<ComputeOpImageRef>{
                  {color_allocation, ImageLayout::color_attachment,
                   ImageLayout::shader_read}};
  auto append_handle = [](auto &handles, const auto handle) {
    if (std::find(handles.begin(), handles.end(), handle) == handles.end()) {
      handles.push_back(handle);
    }
  };
  for (const auto *array : arrays) {
    append_handle(prepared.arrays, array->runtime_resource_handle());
  }
  for (const auto *texture : textures) {
    append_handle(prepared.textures, texture->runtime_resource_handle());
    const auto allocation = texture->get_device_allocation();
    if (allocation == color_allocation || allocation == depth_allocation) {
      continue;
    }
    if (std::none_of(prepared.images.begin(), prepared.images.end(),
                     [&](const auto &ref) { return ref.image == allocation; })) {
      prepared.images.push_back({allocation, ImageLayout::shader_read,
                                  ImageLayout::shader_read});
    }
  }
  for (const auto &command : commands) {
    append_handle(prepared.pipelines, command.pipeline_handle);
  }
  prepared.replay_key = std::move(replay_key);
  return packet;
}

std::size_t Program::execute_vulkan_graphics_pass(
    const std::shared_ptr<PreparedVulkanGraphicsPass> &packet) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!packet || !packet->state || packet->state->owner != this ||
                  packet->state->generation != runtime_program_generation(),
              "Prepared graphics pass belongs to another or retired runtime");
  const auto &prepared = *packet->state;
  {
    std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
    for (const auto handle : prepared.pipelines) {
      TI_ERROR_IF(vulkan_graphics_pipelines_.find(handle) ==
                      vulkan_graphics_pipelines_.end(),
                  "Prepared graphics pipeline is stale or closed");
    }
  }
  NdarrayLaunchLeases arrays;
  for (const auto handle : prepared.arrays) {
    TI_ERROR_IF(handle.index >= ndarray_view_slots_.size(),
                "Prepared graphics ndarray is retired");
    const auto &slot = ndarray_view_slots_[handle.index];
    TI_ERROR_IF(slot.handle != handle || !slot.view || !slot.resource,
                "Prepared graphics ndarray is retired");
    if (ndarray_inflight_leases_.find(ndarray_lease_key(handle)) ==
        ndarray_inflight_leases_.end()) {
      auto lease = slot.resource->lease.clone();
      TI_ERROR_IF(!lease, "Cannot retain prepared graphics ndarray");
      arrays.add(std::move(lease));
    }
  }
  TextureLaunchLeases textures;
  for (const auto handle : prepared.textures) {
    TI_ERROR_IF(handle.index >= texture_view_slots_.size(),
                "Prepared graphics texture is retired");
    const auto &slot = texture_view_slots_[handle.index];
    TI_ERROR_IF(slot.handle != handle || !slot.view,
                "Prepared graphics texture is retired");
    if (texture_inflight_leases_.find(texture_lease_key(handle)) ==
        texture_inflight_leases_.end()) {
      auto acquired = texture_resources_.acquire(handle);
      TI_ERROR_IF(acquired.first != TextureResourceRegistry::Result::kSuccess,
                  "Cannot retain prepared graphics texture");
      textures.add(std::move(acquired.second));
    }
  }
  // Lease acquisition is the existing lifetime boundary; layouts, ranges,
  // descriptors and replay identity are not reconstructed on bound execution.
  pin_ndarray_launch_leases(arrays);
  pin_texture_launch_leases(textures);
  // The graphics submission can succeed before its compute bridge fails.
  // Publish pending work before entering that partially committing operation.
  mark_runtime_submission_pending();
  enqueue_graphics_op_lambda(prepared.record, prepared.images,
                              prepared.replay_key);
  return 0;
}

std::size_t Program::vulkan_graphics_pass(
    Texture *color, Texture *depth,
    const std::vector<VulkanGraphicsDrawCommand> &commands,
    const VulkanGraphicsPassInfo &pass) {
  return execute_vulkan_graphics_pass(
      prepare_vulkan_graphics_pass(color, depth, commands, pass));
}

void Program::destroy_vulkan_graphics_pipeline(std::uint64_t handle) {
  std::shared_ptr<VulkanGraphicsPipelineResource> resource;
  bool record_retirement = false;
  if (runtime_has_fatal_fault()) {
    {
      std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
      const auto found = vulkan_graphics_pipelines_.find(handle);
      if (found == vulkan_graphics_pipelines_.end()) {
        return;
      }
      resource = std::move(found->second);
      vulkan_graphics_pipelines_.erase(found);
    }
    resource->close_prepared_resources();
    // A faulted backend rejects new completion submissions. Native command
    // buffers already carry their Vulkan object references through the stream,
    // so detach the host-side replay owner without attempting another wait or
    // retirement marker.
    if (program_impl_) {
      program_impl_->invalidate_graphics_command_replay();
    }
    return;
  }
  {
    auto submission_guard = acquire_runtime_resource_submission_guard();
    std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
    const auto found = vulkan_graphics_pipelines_.find(handle);
    if (found == vulkan_graphics_pipelines_.end()) {
      return;
    }
    resource = found->second;
    vulkan_graphics_pipelines_.erase(found);
    resource->close_prepared_resources();
    if (!runtime_has_fatal_fault() &&
        runtime_submission_pending_.load(std::memory_order_acquire)) {
      vulkan_graphics_pipeline_retirements_.push_back(std::move(resource));
      record_retirement = true;
    }
  }
  if (program_impl_) {
    program_impl_->invalidate_graphics_command_replay();
  }
  if (record_retirement) {
    record_runtime_completion();
  }
}

void Program::vulkan_clear_graphics_pipelines() {
  if (program_impl_) {
    program_impl_->invalidate_graphics_command_replay();
  }
  std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
  for (const auto &item : vulkan_graphics_pipelines_) {
    item.second->close_prepared_resources();
  }
  for (const auto &resource : vulkan_graphics_pipeline_retirements_) {
    resource->close_prepared_resources();
  }
  vulkan_graphics_pipelines_.clear();
  vulkan_graphics_pipeline_retirements_.clear();
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

bool Program::vulkan_graphics_pipeline_available() const {
  return false;
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_graphics_indirect_capabilities() const {
  return {{"fixed_count", 0},
          {"multi_draw", 0},
          {"first_instance", 0},
          {"count_buffer", 0},
          {"max_draw_count", 0}};
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_bindless_buffer_capabilities() const {
  return {{"descriptor_indexing", 0},
          {"storage_buffer_non_uniform_indexing", 0},
          {"fixed_count", 0},
          {"partially_bound", 0},
          {"update_after_bind", 0},
          {"variable_count", 0},
          {"runtime_array", 0},
          {"update_unused_while_pending", 0},
          {"max_fixed_count", 0},
          {"max_update_after_bind_descriptors_in_all_pools", 0},
          {"max_per_stage_update_after_bind_storage_buffers", 0},
          {"max_descriptor_set_update_after_bind_storage_buffers", 0}};
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_mesh_shader_capabilities() const {
  return {{"mesh_shader", 0},
          {"task_shader", 0},
          {"max_task_group_count_x", 0},
          {"max_task_group_count_y", 0},
          {"max_task_group_count_z", 0},
          {"max_task_group_total_count", 0},
          {"max_task_group_invocations", 0},
          {"max_mesh_group_count_x", 0},
          {"max_mesh_group_count_y", 0},
          {"max_mesh_group_count_z", 0},
          {"max_mesh_group_total_count", 0},
          {"max_mesh_group_invocations", 0},
          {"max_mesh_output_vertices", 0},
          {"max_mesh_output_primitives", 0}};
}

std::size_t Program::debug_vulkan_graphics_pipeline_count() {
  return 0;
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_vulkan_graphics_resource_stats() {
  return {{"live", 0},
          {"retiring", 0},
          {"queued_for_completion", 0},
          {"completion_retained", 0},
          {"prepared_resource_leases", 0},
          {"prepared_draw_resources", 0},
          {"prepared_descriptor_sets", 0},
          {"prepared_raster_resources", 0},
          {"retained_replay_attempts", 0},
          {"retained_replay_prewarms", 0},
          {"retained_replay_records", 0},
          {"retained_replay_replays", 0},
          {"retained_replay_fallbacks", 0},
          {"retained_replay_busy_fallbacks", 0},
          {"retained_replay_binding_misses", 0},
          {"retained_replay_layout_misses", 0},
          {"retained_replay_graphics_submissions", 0},
          {"retained_replay_bridge_submissions", 0},
          {"retained_replay_bridge_failures", 0},
          {"retained_replay_submit_failures", 0},
          {"retained_replay_invalidations", 0},
          {"retained_replay_slots", 0},
          {"retained_replay_slot_capacity", 0},
          {"retained_replay_peak_slots", 0},
          {"retained_replay_inflight_slots", 0},
          {"retained_replay_last_path", 0}};
}

std::uint64_t Program::create_vulkan_graphics_pipeline(
    const std::vector<std::uint32_t> &,
    const std::vector<std::uint32_t> &,
    const std::vector<VulkanGraphicsVertexBinding> &,
    const std::vector<VulkanGraphicsVertexAttribute> &,
    int,
    int,
    bool,
    bool,
    bool,
    bool,
    bool,
    const std::string &) {
  TI_ERROR("Vulkan graphics pipelines are unavailable in this build.");
}

std::uint64_t Program::create_vulkan_mesh_pipeline(
    const std::vector<std::uint32_t> &,
    const std::vector<std::uint32_t> &,
    const std::vector<std::uint32_t> &,
    int,
    int,
    bool,
    bool,
    bool,
    bool,
    bool,
    const std::string &) {
  TI_ERROR("Vulkan mesh pipelines are unavailable in this build.");
}

std::size_t Program::vulkan_graphics_draw(
    std::uint64_t,
    Texture *,
    Texture *,
    const std::vector<std::pair<std::uint32_t, Ndarray *>> &,
    Ndarray *,
    const VulkanGraphicsDrawInfo &) {
  TI_ERROR("Vulkan graphics draws are unavailable in this build.");
}

std::size_t Program::vulkan_graphics_pass(
    Texture *,
    Texture *,
    const std::vector<VulkanGraphicsDrawCommand> &,
    const VulkanGraphicsPassInfo &) {
  TI_ERROR("Vulkan graphics passes are unavailable in this build.");
}

void Program::destroy_vulkan_graphics_pipeline(std::uint64_t) {
}

std::shared_ptr<PreparedVulkanGraphicsPass> Program::prepare_vulkan_graphics_pass(
    Texture *, Texture *, const std::vector<VulkanGraphicsDrawCommand> &,
    const VulkanGraphicsPassInfo &) {
  TI_ERROR("Vulkan graphics passes are unavailable in this build.");
}

std::size_t Program::execute_vulkan_graphics_pass(
    const std::shared_ptr<PreparedVulkanGraphicsPass> &) {
  TI_ERROR("Vulkan graphics passes are unavailable in this build.");
}

void Program::vulkan_clear_graphics_pipelines() {
}

}  // namespace taichi::lang

#endif
