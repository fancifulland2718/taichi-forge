#include "taichi/program/program.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"

namespace taichi::lang {
namespace {

constexpr std::size_t kMaximumShaderBytes = 16u * 1024u * 1024u;
constexpr std::size_t kMaximumVertexBindings = 16;
constexpr std::size_t kMaximumVertexAttributes = 32;

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
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!color,
              "Vulkan graphics draw requires a color attachment Texture.");
  TI_ERROR_IF(draw.element_count == 0 || draw.instance_count == 0,
              "Vulkan graphics draw counts must be positive.");
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
  TI_ERROR_IF(draw.indexed != (index_buffer != nullptr),
              "Vulkan graphics indexed draw and index-buffer binding must "
              "agree.");

  std::shared_ptr<VulkanGraphicsPipelineResource> resource;
  {
    std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
    const auto found = vulkan_graphics_pipelines_.find(handle);
    TI_ERROR_IF(found == vulkan_graphics_pipelines_.end(),
                "Vulkan graphics pipeline handle is stale or closed.");
    resource = found->second;
  }

  std::unordered_map<std::uint32_t, Ndarray *> supplied;
  std::vector<const Ndarray *> arrays;
  arrays.reserve(vertex_buffers.size() + (index_buffer ? 1 : 0));
  auto *device = static_cast<vulkan::VulkanDevice *>(get_graphics_device());
  for (const auto &[binding, array] : vertex_buffers) {
    TI_ERROR_IF(!array,
                "Vulkan graphics vertex binding {} is null.", binding);
    TI_ERROR_IF(!supplied.emplace(binding, array).second,
                "Vulkan graphics vertex binding {} is duplicated.", binding);
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
      records = static_cast<std::uint64_t>(draw.first_vertex) + 1;
    } else {
      records = static_cast<std::uint64_t>(draw.first_vertex) +
                draw.element_count;
    }
    TI_ERROR_IF(records >
                    (std::numeric_limits<std::size_t>::max)() / binding.stride ||
                    static_cast<std::size_t>(records) * binding.stride >
                        available,
                "Vulkan graphics vertex binding {} is too small for the "
                "declared draw range.",
                binding.binding);
  }

  if (index_buffer) {
    TI_ERROR_IF(index_buffer->owning_program() != this ||
                    index_buffer->get_device_allocation().device != device,
                "Vulkan graphics index buffer belongs to another runtime or "
                "device.");
    TI_ERROR_IF(index_buffer->get_element_data_type() != PrimitiveType::i32 &&
                    index_buffer->get_element_data_type() != PrimitiveType::u32,
                "Vulkan graphics index buffer must use i32 or u32.");
    TI_ERROR_IF(!int(device->allocation_usage(
                         index_buffer->get_device_allocation()) &
                     AllocUsage::Index),
                "Vulkan graphics index buffer was not allocated for index "
                "input.");
    const std::uint64_t index_end =
        static_cast<std::uint64_t>(draw.first_index) + draw.element_count;
    TI_ERROR_IF(index_end >
                    (std::numeric_limits<std::size_t>::max)() / sizeof(uint32_t) ||
                    static_cast<std::size_t>(index_end) * sizeof(uint32_t) >
                        ndarray_bytes(index_buffer, "index"),
                "Vulkan graphics index buffer is too small for the declared "
                "draw range.");
    arrays.push_back(index_buffer);
  }

  std::array<std::uint32_t, 4> viewport = draw.viewport;
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

  auto ndarray_leases = acquire_ndarray_leases(arrays);
  std::vector<const Texture *> textures{color};
  if (depth) {
    textures.push_back(depth);
  }
  auto texture_leases = acquire_texture_leases(textures);
  std::vector<std::pair<std::uint32_t, DeviceAllocation>> allocations;
  allocations.reserve(vertex_buffers.size());
  for (const auto &[binding, array] : vertex_buffers) {
    allocations.emplace_back(binding, array->get_device_allocation());
  }
  const DeviceAllocation index_allocation =
      index_buffer ? index_buffer->get_device_allocation()
                   : kDeviceNullAllocation;
  const DeviceAllocation color_allocation = color->get_device_allocation();
  const DeviceAllocation depth_allocation =
      depth ? depth->get_device_allocation() : kDeviceNullAllocation;
  const int width = color_size[0];
  const int height = color_size[1];

  enqueue_graphics_op_lambda(
      [resource, allocations = std::move(allocations), index_allocation, draw,
       viewport, color_allocation, depth_allocation, width,
       height](GraphicsDevice *graphics, CommandList *commands) {
        auto *vulkan_commands =
            static_cast<vulkan::VulkanCommandList *>(commands);
        vulkan_commands->set_next_renderpass_color_final_layout(
            ImageLayout::color_attachment);
        auto raster = graphics->create_raster_resources_unique();
        for (const auto &[binding, allocation] : allocations) {
          raster->vertex_buffer(allocation.get_ptr(), binding);
        }
        if (draw.indexed) {
          raster->index_buffer(index_allocation.get_ptr(), 32);
        }

        bool clear = true;
        std::vector<float> clear_color(draw.clear_color.begin(),
                                       draw.clear_color.end());
        DeviceAllocation color_target = color_allocation;
        DeviceAllocation depth_target = depth_allocation;
        DeviceAllocation *depth_target_ptr =
            depth_target == kDeviceNullAllocation ? nullptr : &depth_target;
        commands->begin_renderpass(0, 0, width, height, 1, &color_target,
                                   &clear, &clear_color, depth_target_ptr,
                                   depth_target_ptr != nullptr);
        commands->set_raster_viewport_and_scissor(
            static_cast<int>(viewport[0]), static_cast<int>(viewport[1]),
            static_cast<int>(viewport[2]), static_cast<int>(viewport[3]));
        commands->bind_pipeline(resource->pipeline());
        const RhiResult bind_result =
            commands->bind_raster_resources(raster.get());
        TI_ERROR_IF(bind_result != RhiResult::success,
                    "Vulkan graphics resource binding failed: RhiResult({}).",
                    bind_result);
        if (draw.indexed && draw.instance_count > 1) {
          commands->draw_indexed_instance(
              draw.element_count, draw.instance_count, draw.first_vertex,
              draw.first_index, draw.first_instance);
        } else if (draw.indexed) {
          commands->draw_indexed(draw.element_count, draw.first_vertex,
                                 draw.first_index);
        } else if (draw.instance_count > 1) {
          commands->draw_instance(draw.element_count, draw.instance_count,
                                  draw.first_vertex, draw.first_instance);
        } else {
          commands->draw(draw.element_count, draw.first_vertex);
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
                   ImageLayout::shader_read}});
  mark_runtime_submission_pending();
  pin_ndarray_launch_leases(ndarray_leases);
  pin_texture_launch_leases(texture_leases);
  return 0;
}

void Program::destroy_vulkan_graphics_pipeline(std::uint64_t handle) {
  std::shared_ptr<VulkanGraphicsPipelineResource> resource;
  {
    auto submission_guard = acquire_runtime_resource_submission_guard();
    std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
    const auto found = vulkan_graphics_pipelines_.find(handle);
    if (found == vulkan_graphics_pipelines_.end()) {
      return;
    }
    resource = found->second;
    vulkan_graphics_pipelines_.erase(found);
  }
  if (!runtime_has_fatal_fault()) {
    synchronize();
  }
}

void Program::vulkan_clear_graphics_pipelines() {
  std::lock_guard<std::mutex> lock(vulkan_graphics_pipeline_mutex_);
  vulkan_graphics_pipelines_.clear();
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

bool Program::vulkan_graphics_pipeline_available() const {
  return false;
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

void Program::destroy_vulkan_graphics_pipeline(std::uint64_t) {
}

void Program::vulkan_clear_graphics_pipelines() {
}

}  // namespace taichi::lang

#endif
