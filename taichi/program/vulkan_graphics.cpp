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
  bool storage{false};
};

struct RecordedGraphicsDraw {
  std::shared_ptr<VulkanGraphicsPipelineResource> pipeline;
  std::vector<std::pair<std::uint32_t, DeviceAllocation>> vertex_buffers;
  DeviceAllocation index_buffer{kDeviceNullAllocation};
  std::vector<RecordedGraphicsShaderBuffer> shader_buffers;
  VulkanGraphicsDrawInfo draw;
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

class VulkanGraphicsPipelineResource {
 public:
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

  Pipeline *pipeline() const noexcept {
    return pipeline_.get();
  }

  const std::vector<VulkanGraphicsVertexBinding> &bindings() const noexcept {
    return bindings_;
  }

 private:
  Program *program_{nullptr};
  std::vector<VulkanGraphicsVertexBinding> bindings_;
  std::unique_ptr<Pipeline> pipeline_;
};

bool Program::vulkan_graphics_pipeline_available() const {
  return compile_config().arch == Arch::vulkan && program_impl_ &&
         const_cast<Program *>(this)->get_graphics_device() != nullptr;
}

std::size_t Program::debug_vulkan_graphics_pipeline_count() {
  std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
  return vulkan_graphics_pipelines_.size();
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_vulkan_graphics_resource_stats() {
  std::uint64_t live = 0;
  std::uint64_t queued_for_completion = 0;
  {
    std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
    live = vulkan_graphics_pipelines_.size();
    queued_for_completion = vulkan_graphics_pipeline_retirements_.size();
  }
  const auto completion_retained = runtime_completion_resource_count(
      kVulkanGraphicsPipelineResourceKind);
  std::unordered_map<std::string, std::uint64_t> result{
      {"live", live},
      {"retiring", queued_for_completion + completion_retained},
      {"queued_for_completion", queued_for_completion},
      {"completion_retained", completion_retained}};
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

std::size_t Program::vulkan_graphics_pass(
    Texture *color,
    Texture *depth,
    const std::vector<VulkanGraphicsDrawCommand> &commands,
    const VulkanGraphicsPassInfo &pass) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
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
  std::vector<RecordedGraphicsDraw> recorded_draws;
  recorded_draws.reserve(commands.size());
  auto *device = static_cast<vulkan::VulkanDevice *>(get_graphics_device());
  for (std::size_t draw_index = 0; draw_index < commands.size(); ++draw_index) {
    const auto &command = commands[draw_index];
    const auto &draw = command.draw;
    const auto &resource = pipelines[draw_index];
    TI_ERROR_IF(draw.element_count == 0 || draw.instance_count == 0,
                "Vulkan graphics draw counts must be positive.");
    TI_ERROR_IF(draw.indexed != (command.index_buffer != nullptr),
                "Vulkan graphics indexed draw and index-buffer binding must "
                "agree.");

    RecordedGraphicsDraw recorded;
    recorded.pipeline = resource;
    recorded.draw = draw;
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
      if (binding.instance) {
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
      const std::uint64_t index_end =
          static_cast<std::uint64_t>(draw.first_index) + draw.element_count;
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
      TI_ERROR_IF(!shader.array,
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
          {shader.set_index, shader.binding, allocation, shader.storage});
    }
    recorded_draws.push_back(std::move(recorded));
  }

  auto ndarray_leases = acquire_ndarray_leases(arrays);
  std::vector<const Texture *> textures{color};
  if (depth) {
    textures.push_back(depth);
  }
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
        append_graphics_allocation_key(replay_key, device,
                                       shader.allocation);
      }
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

  enqueue_graphics_op_lambda(
      [recorded_draws = std::move(recorded_draws), pass, viewport,
       viewport_x_end, viewport_y_end, color_allocation, depth_allocation,
       width, height](GraphicsDevice *graphics, CommandList *commands) {
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
        commands->begin_renderpass(0, 0, width, height, 1, &color_target,
                                   &clear, &clear_color, depth_target_ptr,
                                   depth_target_ptr != nullptr &&
                                       pass.depth_clear);
        commands->set_raster_viewport_and_scissor(
            static_cast<int>(viewport[0]), static_cast<int>(viewport[1]),
            static_cast<int>(viewport_x_end),
            static_cast<int>(viewport_y_end));
        for (const auto &recorded : recorded_draws) {
          auto raster = graphics->create_raster_resources_unique();
          for (const auto &[binding, allocation] :
               recorded.vertex_buffers) {
            raster->vertex_buffer(allocation.get_ptr(), binding);
          }
          if (recorded.draw.indexed) {
            raster->index_buffer(recorded.index_buffer.get_ptr(), 32);
          }

          std::unordered_map<std::uint32_t,
                             std::unique_ptr<ShaderResourceSet>> resource_sets;
          for (const auto &shader : recorded.shader_buffers) {
            auto &resource_set = resource_sets[shader.set_index];
            if (!resource_set) {
              resource_set = graphics->create_resource_set_unique();
            }
            if (shader.storage) {
              resource_set->rw_buffer(shader.binding, shader.allocation);
            } else {
              resource_set->buffer(shader.binding, shader.allocation);
            }
          }

          commands->bind_pipeline(recorded.pipeline->pipeline());
          const RhiResult raster_result =
              commands->bind_raster_resources(raster.get());
          TI_ERROR_IF(
              raster_result != RhiResult::success,
              "Vulkan graphics raster resource binding failed: "
              "RhiResult({}).",
              raster_result);
          for (const auto &[set_index, resource_set] : resource_sets) {
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
          if (draw.indexed &&
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
      },
      depth ? std::vector<ComputeOpImageRef>{
                  {color_allocation, ImageLayout::color_attachment,
                   ImageLayout::shader_read},
                  {depth_allocation, ImageLayout::depth_attachment,
                   ImageLayout::shader_read}}
            : std::vector<ComputeOpImageRef>{
                  {color_allocation, ImageLayout::color_attachment,
                   ImageLayout::shader_read}},
      replay_key);
  mark_runtime_submission_pending();
  pin_ndarray_launch_leases(ndarray_leases);
  pin_texture_launch_leases(texture_leases);
  return 0;
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
  vulkan_graphics_pipelines_.clear();
  vulkan_graphics_pipeline_retirements_.clear();
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

bool Program::vulkan_graphics_pipeline_available() const {
  return false;
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

void Program::vulkan_clear_graphics_pipelines() {
}

}  // namespace taichi::lang

#endif
