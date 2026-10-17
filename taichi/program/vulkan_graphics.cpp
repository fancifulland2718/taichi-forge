#include "taichi/program/program.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "taichi/program/ndarray.h"
#include "taichi/program/program_impl.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/gfx/graph_recording.h"
#include "taichi/runtime/vulkan_graphics_pipeline.h"
#include "taichi/runtime/vulkan_graphics_pass.h"

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/runtime/gfx/kernel_launcher.h"

namespace taichi::lang {
class VulkanGraphicsPipelineResource;

namespace {

using RecordedGraphicsShaderBuffer = VulkanGraphicsShaderBufferView;

using RecordedGraphicsShaderImage = VulkanGraphicsShaderImageView;

using RecordedGraphicsIndirect = VulkanGraphicsIndirectPacket;

struct RecordedGraphicsDraw {
  std::weak_ptr<VulkanGraphicsPipelineResource> pipeline;
  std::vector<std::pair<std::uint32_t, DeviceAllocation>> vertex_buffers;
  DeviceAllocation index_buffer{kDeviceNullAllocation};
  std::vector<RecordedGraphicsShaderBuffer> shader_buffers;
  std::vector<RecordedGraphicsShaderImage> shader_images;
  std::vector<VulkanGraphicsShaderAccelerationStructureBinding>
      shader_acceleration_structures;
  VulkanGraphicsDrawInfo draw;
  std::optional<RecordedGraphicsIndirect> indirect;
  std::optional<VulkanGraphicsMeshDrawInfo> mesh;
};

using PreparedVulkanGraphicsDrawResources = VulkanGraphicsDrawResources;

struct PreparedVulkanGraphicsResourcePayload {
  std::vector<PreparedVulkanGraphicsDrawResources> draws;
  std::vector<std::function<void(CommandList *)>> acceleration_structure_reads;
};

class PreparedVulkanGraphicsResourceLease;

struct PreparedVulkanGraphicsResourceStats {
  std::unordered_set<const PreparedVulkanGraphicsResourceLease *> counted;
  std::uint64_t leases{0};
  std::uint64_t draws{0};
  std::uint64_t descriptor_sets{0};
  std::uint64_t raster_resources{0};
};

class PreparedVulkanGraphicsResourceLease : public PreparedResourceLease {
 public:
  explicit PreparedVulkanGraphicsResourceLease(
      std::shared_ptr<const PreparedVulkanGraphicsResourcePayload> payload)
      : payload_(std::move(payload)) {
  }

  std::shared_ptr<const PreparedVulkanGraphicsResourcePayload> acquire() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return payload_;
  }

  void clear() noexcept override {
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
  std::vector<std::uint64_t> ray_resources;
  std::vector<DeviceAllocation> buffers;
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
      const std::string &name,
      const RasterDepthParams &depth_params,
      const std::vector<BlendingParams> &color_targets)
      : program_(program) {
    TI_ERROR_IF(program_ == nullptr,
                "Vulkan graphics pipeline requires a live Program.");
    for (const auto &binding : vertex_bindings)
      bindings_.push_back({binding.binding, binding.stride, binding.instance});
    std::vector<VertexInputAttribute> rhi_attributes;
    for (const auto &attribute : vertex_attributes)
      rhi_attributes.push_back({attribute.location, attribute.binding,
                                attribute.format, attribute.offset});
    auto *device = dynamic_cast<vulkan::VulkanDevice *>(program_->get_graphics_device());
    TI_ERROR_IF(!device, "Vulkan graphics pipeline has no Vulkan device.");

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
    params.depth = depth_params;
    params.blending = color_targets;
    if (blending && color_targets.empty()) {
      params.blending.emplace_back();
    }
    pipeline_ = prepare_vulkan_graphics_pipeline(
        *device, sources, params, bindings_, rhi_attributes,
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
      const std::string &name,
      const RasterDepthParams &depth_params,
      const std::vector<BlendingParams> &color_targets)
      : program_(program), mesh_pipeline_(true), task_shader_(!task_spirv.empty()) {
    TI_ERROR_IF(program_ == nullptr,
                "Vulkan mesh pipeline requires a live Program.");
    auto *device = dynamic_cast<vulkan::VulkanDevice *>(program_->get_graphics_device());
    TI_ERROR_IF(!device, "Vulkan mesh pipeline has no Vulkan device.");
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
    params.depth = depth_params;
    params.blending = color_targets;
    if (blending && color_targets.empty()) {
      params.blending.emplace_back();
    }
    pipeline_ = prepare_vulkan_graphics_pipeline(
        *device, sources, params, {}, {},
        name.empty() ? "Forge VulkanMeshPipeline" : name);
    TI_ERROR_IF(!pipeline_, "Vulkan mesh pipeline creation failed.");
  }

  Pipeline *pipeline() const noexcept {
    return pipeline_.get();
  }

  const std::vector<VertexInputBinding> &bindings() const noexcept {
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
  std::vector<VertexInputBinding> bindings_;
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
    const std::string &name,
    const RasterDepthParams &depth_params,
    const std::vector<BlendingParams> &color_targets) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_graphics_pipeline_available(),
              "Vulkan graphics pipelines require the Vulkan backend.");
  auto resource = std::make_shared<VulkanGraphicsPipelineResource>(
      this, vertex_spirv, fragment_spirv, vertex_bindings, vertex_attributes,
      topology, polygon_mode, front_face_cull, back_face_cull, depth_test,
      depth_write, blending, name, depth_params, color_targets);
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
    const std::string &name,
    const RasterDepthParams &depth_params,
    const std::vector<BlendingParams> &color_targets) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_graphics_pipeline_available(),
              "Vulkan mesh pipelines require the Vulkan backend.");
  auto resource = std::make_shared<VulkanGraphicsPipelineResource>(
      this, task_spirv, mesh_spirv, fragment_spirv, topology, polygon_mode,
      front_face_cull, back_face_cull, depth_test, depth_write, blending,
      name, depth_params, color_targets);
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
  pass.clear_depth = draw.clear_depth;
  pass.viewport = draw.viewport;
  return vulkan_graphics_pass(color, depth, {std::move(command)}, pass);
}

std::shared_ptr<PreparedVulkanGraphicsPass> Program::prepare_vulkan_graphics_pass(
    const std::vector<VulkanGraphicsColorAttachment> &colors,
    Texture *depth,
    const std::vector<VulkanGraphicsDrawCommand> &commands,
    const VulkanGraphicsPassInfo &pass) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  auto packet = std::make_shared<PreparedVulkanGraphicsPass>();
  packet->state = std::make_shared<PreparedVulkanGraphicsPass::State>();
  auto &prepared = *packet->state;
  prepared.owner = this;
  prepared.generation = runtime_program_generation();
  TI_ERROR_IF(!vulkan_graphics_pipeline_available(), "Graphics passes require Vulkan");
  auto *device = static_cast<vulkan::VulkanDevice *>(get_graphics_device());
  TI_ERROR_IF(commands.empty(), "Vulkan graphics pass requires at least one draw.");
  std::vector<VulkanGraphicsColorTarget> color_targets;
  std::vector<const Ndarray *> arrays;
  std::vector<const Texture *> textures;
  for (const auto &attachment : colors) {
    const auto *target = attachment.texture;
    TI_ERROR_IF(!target || target->owning_program() != this,
                "Vulkan graphics attachment belongs to another Program or is null.");
    color_targets.push_back({target->get_device_allocation(), attachment.clear,
                             attachment.clear_bits});
    textures.push_back(target);
  }
  TI_ERROR_IF(depth && depth->owning_program() != this,
              "Vulkan graphics depth attachment belongs to another Program.");
  const DeviceAllocation depth_allocation =
      depth ? depth->get_device_allocation() : kDeviceNullAllocation;
  if (depth)
    textures.push_back(depth);
  auto targets = std::make_shared<const VulkanGraphicsPassTargets>(
      prepare_vulkan_graphics_pass_targets(*device, color_targets,
                                           depth_allocation, pass.depth_clear,
                                           pass.clear_depth, pass.viewport));
  const auto &color_allocations = targets->colors;
  const auto &viewport = targets->viewport;
  const auto width = targets->width;
  const auto height = targets->height;

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

  std::vector<RecordedGraphicsDraw> recorded_draws;
  recorded_draws.reserve(commands.size());
  for (std::size_t draw_index = 0; draw_index < commands.size(); ++draw_index) {
    const auto &command = commands[draw_index];
    const auto &draw = command.draw;
    const auto &resource = pipelines[draw_index];
    auto *native_pipeline = static_cast<vulkan::VulkanPipeline *>(resource->pipeline());
    validate_vulkan_graphics_pipeline_targets(*native_pipeline, *targets);
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
      TI_ERROR_IF(!device->vk_caps().mesh_group_count_valid(
                      mesh.group_count_x, mesh.group_count_y, mesh.group_count_z,
                      resource->task_shader()),
                  "Vulkan mesh draw group counts exceed the active device "
                  "limits.");
      recorded.mesh = mesh;
    }
    if (command.indirect.has_value()) {
      const auto &indirect = *command.indirect;
      auto array_range = [&](Ndarray *array, const char *role) {
        TI_ERROR_IF(array == nullptr || array->owning_program() != this ||
                        array->get_device_allocation().device != device,
                    "Vulkan graphics {} buffer belongs to another runtime "
                    "or device.",
                    role);
        TI_ERROR_IF(array->get_element_data_type() != PrimitiveType::u32,
                    "Vulkan graphics {} buffer must use u32.", role);
        return DeviceBufferRange{array->get_device_allocation().get_ptr(),
                                 ndarray_bytes(array, role)};
      };
      const auto command_range =
          array_range(indirect.command_buffer, "indirect command");
      std::optional<DeviceBufferRange> count_range;
      if (indirect.count_buffer != nullptr) {
        count_range = array_range(indirect.count_buffer, "indirect count");
      }
      recorded.indirect = prepare_vulkan_graphics_indirect(
          *device, draw.indexed,
          {indirect.command_offset, indirect.count_offset,
           indirect.max_draw_count, indirect.stride,
           indirect.vertex_record_limit, indirect.instance_record_limit,
           indirect.index_element_limit,
           indirect.first_instance_may_be_nonzero},
          command_range, count_range);
      arrays.push_back(indirect.command_buffer);
      if (indirect.count_buffer != nullptr) {
        arrays.push_back(indirect.count_buffer);
      }
    }
    std::vector<VulkanGraphicsVertexRange> vertex_ranges;
    for (const auto &[binding, array] : command.vertex_buffers) {
      TI_ERROR_IF(!array || array->owning_program() != this,
                  "Vulkan graphics vertex binding {} belongs to another "
                  "Program or is null.",
                  binding);
      const auto allocation = array->get_device_allocation();
      vertex_ranges.push_back(
          {binding, {allocation.get_ptr(), ndarray_bytes(array, "vertex")}});
      arrays.push_back(array);
      recorded.vertex_buffers.emplace_back(binding, allocation);
    }
    std::optional<VulkanGraphicsBufferRange> index_range;
    if (command.index_buffer) {
      auto *index_buffer = command.index_buffer;
      TI_ERROR_IF(index_buffer->owning_program() != this,
                  "Vulkan graphics index buffer belongs to another Program.");
      TI_ERROR_IF(index_buffer->get_element_data_type() != PrimitiveType::u32,
                  "Vulkan graphics index buffer must use u32.");
      recorded.index_buffer = index_buffer->get_device_allocation();
      index_range = VulkanGraphicsBufferRange{
          recorded.index_buffer.get_ptr(), ndarray_bytes(index_buffer, "index")};
      arrays.push_back(index_buffer);
    }
    if (!mesh_draw)
      validate_vulkan_graphics_vertex_ranges(*device, resource->bindings(), vertex_ranges,
                                             index_range, draw, recorded.indirect);

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
        const std::size_t limit = bindless_caps.at("max_fixed_count");
        TI_ERROR_IF(bindless_caps.at("fixed_count") == 0 || !shader.storage,
                    "Vulkan bindless graphics requires fixed-count storage "
                    "buffer descriptor indexing.");
        TI_ERROR_IF(shader.array_elements.size() < 2 ||
                        shader.array_elements.size() > limit,
                    "Vulkan bindless graphics table size must be between 2 "
                    "and the device limit {}.",
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
          recorded_shader.array_elements.push_back(
              {allocation.get_ptr(), ndarray_bytes(element, "shader buffer array")});
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
                      std::find(color_allocations.begin(), color_allocations.end(), allocation) != color_allocations.end() ||
                      (depth && allocation == depth->get_device_allocation()),
                  "Vulkan sampled images must be on this device and cannot "
                  "alias render-pass attachments");
      const auto format = image->get_buffer_format();
      TI_ERROR_IF(!is_float_sampled_texture_format(format),
                  "Graphics sampled images require floating-point, normalized or depth textures");
      native_pipeline->validate_sampled_texture(shader.set_index, shader.binding,
                                                image->sampler_config_.compare_op >= 0);
      textures.push_back(image);
      recorded.shader_images.push_back(
          {shader.set_index, shader.binding, allocation, image->sampler_config_});
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> as_bindings;
    for (const auto &shader : command.shader_acceleration_structures) {
      const auto key =
          (static_cast<std::uint64_t>(shader.set_index) << 32) | shader.binding;
      TI_ERROR_IF(!shader_bindings.insert(key).second,
                  "Vulkan graphics descriptor set {} binding {} is duplicated",
                  shader.set_index, shader.binding);
      as_bindings.emplace_back(shader.set_index, shader.binding);
    }
    native_pipeline->validate_acceleration_structure_bindings(as_bindings);
    recorded.shader_acceleration_structures =
        command.shader_acceleration_structures;
    for (auto &as : recorded.shader_acceleration_structures) {
      as.consumer_stages = native_pipeline->acceleration_structure_stages(as.set_index, as.binding);
    }
    recorded_draws.push_back(std::move(recorded));
  }

  auto ndarray_leases = acquire_ndarray_leases(arrays);
  auto texture_leases = acquire_texture_leases(textures);

  std::vector<std::uint64_t> replay_key;
  const bool replay_eligible =
      pass.retained_replay && std::all_of(colors.begin(), colors.end(), [](const auto &x) { return x.clear; }) &&
      (depth == nullptr || pass.depth_clear) && !compile_config().debug &&
      !compile_config().kernel_profiler;
  if (replay_eligible) {
    // Exact vector equality, including allocation generations, is the replay
    // contract. Hash-only identity would permit a stale descriptor or
    // attachment collision and is deliberately not used.
    replay_key.reserve(32 + commands.size() * 32);
    replay_key.push_back(0x4652475250415353ull);  // "FRGRPASS"
    replay_key.push_back(3);
    replay_key.push_back(reinterpret_cast<std::uintptr_t>(this));
    replay_key.push_back(colors.size());
    for (std::size_t i = 0; i < colors.size(); ++i) {
      append_graphics_allocation_key(replay_key, device, color_allocations[i]);
      replay_key.push_back(colors[i].clear);
      for (const auto component : colors[i].clear_bits) {
        replay_key.push_back(component);
      }
    }
    replay_key.push_back(depth == nullptr ? 0 : 1);
    if (depth != nullptr) {
      append_graphics_allocation_key(replay_key, device, depth_allocation);
    }
    replay_key.push_back(static_cast<std::uint64_t>(width));
    replay_key.push_back(static_cast<std::uint64_t>(height));
    replay_key.push_back(graphics_float_bits(pass.clear_depth));
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
          for (const auto &range : shader.array_elements) {
            append_graphics_allocation_key(replay_key, device, range.pointer);
            replay_key.push_back(range.pointer.offset);
            replay_key.push_back(range.bytes);
          }
        }
      }
      replay_key.push_back(recorded.shader_images.size());
      replay_key.push_back(recorded.shader_acceleration_structures.size());
      for (const auto &as : recorded.shader_acceleration_structures) {
        replay_key.push_back(as.set_index);
        replay_key.push_back(as.binding);
        replay_key.push_back(as.handle);
      }
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
  auto resource_lease =
      std::make_shared<PreparedVulkanGraphicsResourceLease>(resource_payload);
  resource_payload->draws.reserve(recorded_draws.size());
  for (const auto &recorded : recorded_draws) {
    std::vector<std::pair<uint32_t, DevicePtr>> vertex_views;
    vertex_views.reserve(recorded.vertex_buffers.size());
    for (const auto &[binding, allocation] : recorded.vertex_buffers)
      vertex_views.emplace_back(binding, allocation.get_ptr());
    auto pipeline = recorded.pipeline.lock();
    TI_ERROR_IF(!pipeline, "Vulkan graphics pipeline closed during preparation.");
    auto draw_resources = prepare_vulkan_graphics_draw_resources(
        *device, *static_cast<vulkan::VulkanPipeline *>(pipeline->pipeline()), recorded.shader_buffers, recorded.shader_images, vertex_views,
        recorded.index_buffer.get_ptr(), recorded.draw.indexed,
        recorded.mesh.has_value(), [&](VulkanGraphicsResourceSets &resource_sets) {
          for (const auto &as : recorded.shader_acceleration_structures) {
            auto &resource_set = resource_sets[as.set_index];
            if (!resource_set) {
              resource_set = device->create_resource_set_unique();
              TI_ERROR_IF(!resource_set,
                          "Vulkan graphics AS resource set creation failed.");
            }
            resource_payload->acceleration_structure_reads.push_back(
                prepare_vulkan_ray_graphics_binding(
                    as.handle, resource_set.get(), as.binding, resource_lease,
                    as.consumer_stages));
          }
        });
    resource_payload->draws.push_back(std::move(draw_resources));
  }
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
       resource_lease, targets](GraphicsDevice *, CommandList *commands) {
        auto resource_payload = resource_lease->acquire();
        TI_ERROR_IF(!resource_payload,
                    "Prepared Vulkan graphics resources are stale or closed.");
        for (const auto &read :
             resource_payload->acceleration_structure_reads) {
          read(commands);
        }
        for (const auto &recorded : recorded_draws)
          if (recorded.indirect)
            record_vulkan_graphics_indirect_barriers(
                commands, recorded.draw.indexed, *recorded.indirect);
        begin_vulkan_graphics_pass(commands, *targets);
        for (std::size_t draw_index = 0; draw_index < recorded_draws.size();
             ++draw_index) {
          const auto &recorded = recorded_draws[draw_index];
          const auto &draw_resources = resource_payload->draws[draw_index];
          auto pipeline = recorded.pipeline.lock();
          TI_ERROR_IF(!pipeline,
                      "Vulkan graphics pipeline closed or retired while recording");
          record_vulkan_graphics_draw(
              commands, pipeline->pipeline(), pipeline->task_shader(),
              recorded.draw, recorded.indirect, recorded.mesh, draw_resources);
        }
        end_vulkan_graphics_pass(commands);
      };
  for (const auto allocation : color_allocations) {
    prepared.images.push_back({allocation, ImageLayout::color_attachment, ImageLayout::shader_read});
  }
  if (depth) {
    prepared.images.push_back({depth_allocation, ImageLayout::depth_attachment, ImageLayout::shader_read});
  }
  auto append_handle = [](auto &handles, const auto handle) {
    if (std::find(handles.begin(), handles.end(), handle) == handles.end()) {
      handles.push_back(handle);
    }
  };
  for (const auto *array : arrays) {
    append_handle(prepared.arrays, array->runtime_resource_handle());
    if (std::find(prepared.buffers.begin(), prepared.buffers.end(),
                  array->get_device_allocation()) == prepared.buffers.end()) {
      prepared.buffers.push_back(array->get_device_allocation());
    }
  }
  for (const auto *texture : textures) {
    append_handle(prepared.textures, texture->runtime_resource_handle());
    const auto allocation = texture->get_device_allocation();
    if (std::find(color_allocations.begin(), color_allocations.end(), allocation) != color_allocations.end()
        || allocation == depth_allocation) {
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
    for (const auto &as : command.shader_acceleration_structures) {
      append_handle(prepared.ray_resources, as.handle);
    }
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

namespace {
class PreparedGraphicsGraphCommand final : public gfx::ExternalGraphCommand {
 public:
  PreparedGraphicsGraphCommand(std::shared_ptr<PreparedVulkanGraphicsPass> packet,
                              std::shared_ptr<void> arrays,
                              std::shared_ptr<void> textures)
      : packet_(std::move(packet)),
        arrays_(std::move(arrays)),
        textures_(std::move(textures)) {
  }

  std::vector<aot::Arg> arguments() const override {
    return {};
  }
  bool requires_graphics_queue() const override {
    return true;
  }
  std::optional<std::vector<DeviceAllocation>> buffer_uses() const override {
    return packet_->state->buffers;
  }
  std::vector<std::uint64_t> graphics_pipeline_dependencies() const override {
    return packet_->state->pipelines;
  }
  std::vector<std::uint64_t> ray_resource_dependencies() const override {
    return packet_->state->ray_resources;
  }
  void validate(
      Program &program,
      const std::unordered_map<std::string, aot::IValue> &) const override {
    const auto &state = *packet_->state;
    TI_ERROR_IF(state.owner != &program ||
                    state.generation != program.runtime_program_generation() ||
                    !state.resources->acquire(),
                "Prepared graphics command is closed or belongs to another runtime");
  }
  std::vector<std::pair<DeviceAllocation, ImageLayout>> image_uses()
      const override {
    std::vector<std::pair<DeviceAllocation, ImageLayout>> result;
    for (const auto &image : packet_->state->images) {
      result.emplace_back(image.image, image.initial_layout);
    }
    return result;
  }
  void record(Device *device, CommandList *commands) const override {
    // These barriers replace cross-queue semaphore dependencies, not lifecycle
    // checks. Ordinary compute barriers do not cover vertex/index/uniform reads
    // or attachment accesses. Recording happens only at binding publication.
    auto buffer =
        static_cast<vulkan::VulkanCommandList *>(commands)->vk_command_buffer();
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    buffer->device.functions().vkCmdPipelineBarrier(
        buffer->buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &barrier, 0, nullptr, 0,
        nullptr);
    packet_->state->record(static_cast<GraphicsDevice *>(device), commands);
    // Also order read-before-write reuse of vertex/uniform buffers in a later
    // compute operation (including the next replay of this immutable frame).
    buffer->device.functions().vkCmdPipelineBarrier(
        buffer->buffer, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
  }

 private:
  std::shared_ptr<PreparedVulkanGraphicsPass> packet_;
  std::shared_ptr<void> arrays_;
  std::shared_ptr<void> textures_;
};
}  // namespace

std::shared_ptr<gfx::ExternalGraphCommand> Program::vulkan_graphics_graph_command(
    const std::shared_ptr<PreparedVulkanGraphicsPass> &packet) {
  auto guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!packet || !packet->state || packet->state->owner != this ||
                  packet->state->generation != runtime_program_generation(),
              "Prepared graphics pass belongs to another or retired runtime");
  const auto &prepared = *packet->state;
  TI_ERROR_IF(!prepared.resources->acquire(),
              "Prepared graphics pipeline is closed");
  NdarrayLaunchLeases arrays;
  for (const auto handle : prepared.arrays) {
    TI_ERROR_IF(handle.index >= ndarray_view_slots_.size(),
                "Prepared graphics ndarray is retired");
    const auto &slot = ndarray_view_slots_[handle.index];
    TI_ERROR_IF(slot.handle != handle || !slot.view || !slot.resource,
                "Prepared graphics ndarray is retired");
    auto lease = slot.resource->lease.clone();
    TI_ERROR_IF(!lease, "Cannot retain prepared graphics ndarray");
    arrays.add(std::move(lease));
  }
  TextureLaunchLeases textures;
  for (const auto handle : prepared.textures) {
    auto acquired = texture_resources_.acquire(handle);
    TI_ERROR_IF(acquired.first != TextureResourceRegistry::Result::kSuccess,
                "Cannot retain prepared graphics texture");
    textures.add(std::move(acquired.second));
  }
  return std::make_shared<PreparedGraphicsGraphCommand>(
      packet, std::make_shared<NdarrayLaunchLeases>(std::move(arrays)),
      std::make_shared<TextureLaunchLeases>(std::move(textures)));
}

std::size_t Program::vulkan_graphics_pass(
    Texture *color, Texture *depth,
    const std::vector<VulkanGraphicsDrawCommand> &commands,
    const VulkanGraphicsPassInfo &pass) {
  VulkanGraphicsColorAttachment attachment;
  attachment.texture = color;
  attachment.clear = pass.color_clear;
  std::memcpy(attachment.clear_bits.data(), pass.clear_color.data(), sizeof(attachment.clear_bits));
  TI_ERROR_IF(color && !is_float_sampled_texture_format(color->get_buffer_format()),
              "Integer graphics attachments require a typed record_pass ColorAttachment");
  return execute_vulkan_graphics_pass(
      prepare_vulkan_graphics_pass({attachment}, depth, commands, pass));
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
    // Cold owner invalidation, analogous to fixed dense root retirement. The
    // existing replay registration rejects later runs without scanning owners.
    if (auto *launcher =
            dynamic_cast<gfx::KernelLauncher *>(&get_kernel_launcher())) {
      launcher->runtime()->retire_graphics_pipeline_recordings(handle);
    }
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
    const std::string &,
    const RasterDepthParams &, const std::vector<BlendingParams> &) {
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
    const std::string &,
    const RasterDepthParams &, const std::vector<BlendingParams> &) {
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

std::shared_ptr<gfx::ExternalGraphCommand> Program::vulkan_graphics_graph_command(
    const std::shared_ptr<PreparedVulkanGraphicsPass> &) {
  TI_ERROR("Vulkan graphics recording is unavailable in this build");
}

void Program::destroy_vulkan_graphics_pipeline(std::uint64_t) {
}

std::shared_ptr<PreparedVulkanGraphicsPass> Program::prepare_vulkan_graphics_pass(
    const std::vector<VulkanGraphicsColorAttachment> &, Texture *, const std::vector<VulkanGraphicsDrawCommand> &,
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
