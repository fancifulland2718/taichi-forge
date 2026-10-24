#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_acceleration_structure.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/runtime/native_image.h"
#include "taichi/runtime/vulkan_graphics_pipeline.h"
#include "taichi/runtime/vulkan_graphics_pass.h"
#include "taichi/common/operation_json.h"
#include "taichi/program/texture.h"
#include "picosha2.h"

#include <cstring>
#include <limits>
#include <set>
#include <unordered_map>
#ifdef TI_WITH_VULKAN
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#endif

namespace taichi::lang {
namespace {
using Json = liong::json::JsonValue;
using namespace operation_json;

uint32_t u32(const Json &v) {
  auto n = integer(v);
  if (n > UINT32_MAX)
    throw std::invalid_argument("Graphics integer exceeds uint32 ABI");
  return uint32_t(n);
}
uint32_t opt_u32(const Json &v, const char *key, uint32_t fallback) {
  return v.obj.inner.count(key) ? u32(v[key]) : fallback;
}
int32_t opt_i32(const Json &v, const char *key) {
  if (!v.obj.inner.count(key))
    return 0;
  const auto &n = v[key];
  if (n.ty != liong::json::L_JSON_INT || n.num_int < INT32_MIN ||
      n.num_int > INT32_MAX)
    throw std::invalid_argument("Graphics vertex offset exceeds int32 ABI");
  return int32_t(n.num_int);
}
const std::vector<Json> &array(const Json &v, const char *key) {
  if (!v[key].is_arr())
    throw std::invalid_argument(std::string("Graphics requires array: ") + key);
  return v[key].arr.inner;
}
template <typename T>
const T &indexed(const std::vector<T> &values, uint32_t index) {
  if (index >= values.size())
    throw std::invalid_argument(
        "Graphics resource index is outside its binding group");
  return values[index];
}
BufferFormat format(const Json &v) {
  const auto label = name(v);
#define PER_BUFFER_FORMAT(value) \
  if (label == #value)           \
    return BufferFormat::value;
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_BUFFER_FORMAT
  throw std::invalid_argument("Unknown graphics format: " + label);
}
uint32_t enum_value(const Json &v, const std::vector<std::string> &labels) {
  const auto label = name(v);
  const auto found = std::find(labels.begin(), labels.end(), label);
  if (found == labels.end())
    throw std::invalid_argument("Unsupported graphics enum: " + label);
  return uint32_t(found - labels.begin());
}
std::string expected_contract(const Json &p) {
  if (!p.obj.inner.count("expected_pipeline"))
    return {};
  auto value = name(p["expected_pipeline"]);
  if (value.size() != 64 ||
      value.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument(
        "Expected graphics pipeline must be a SHA256 contract");
  return value;
}
void match_contract(const std::string &expected, const std::string &actual) {
  if (!expected.empty() && expected != actual)
    throw std::invalid_argument(
        "Graphics pipeline code or layout contract mismatch");
}
void same_device(Device &device,
                 const std::shared_ptr<NativeResource> &resource) {
  if (!resource || resource->arch() != Arch::vulkan ||
      &resource->device() != &device)
    throw std::invalid_argument(
        "Graphics resource requires the same Vulkan device");
}

struct CodeInput {
  PipelineStageType stage;
  std::shared_ptr<NativeHostBlobResource> code;
};
struct PipelineDescription {
  static constexpr const char *kind = "vulkan_graphics_pipeline";
  static constexpr uint32_t version = 1;
  std::vector<CodeInput> shaders;
  RasterParams raster;
  std::vector<VertexInputBinding> vertices;
  std::vector<VertexInputAttribute> attributes;
  std::string expected;
};
struct MeshPipelineDescription : PipelineDescription {
  static constexpr uint32_t version = 2;
};

// Typed, length-delimited contract, independent of JSON spelling/defaults.
std::string pipeline_contract(const PipelineDescription &d) {
  picosha2::hash256_one_by_one hash;
  auto number = [&](uint64_t value) {
    uint8_t bytes[8];
    for (size_t i = 0; i < 8; ++i)
      bytes[i] = uint8_t(value >> (8 * i));
    hash.process(bytes, bytes + 8);
  };
  auto text = [&](const std::string &value) {
    number(value.size());
    hash.process(value.begin(), value.end());
  };
  auto scalar = [&](float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    number(bits);
  };
  text("forge-vulkan-graphics-v1");
  number(d.shaders.size());
  for (const auto &shader : d.shaders) {
    number(uint32_t(shader.stage));
    const auto &bytes = shader.code->bytes();
    text(picosha2::hash256_hex_string(bytes.begin(), bytes.end()));
  }
  const auto &r = d.raster;
  for (uint32_t v : {uint32_t(r.prim_topology), uint32_t(r.polygon_mode),
                     uint32_t(r.front_face_cull), uint32_t(r.back_face_cull),
                     uint32_t(r.depth_test), uint32_t(r.depth_write),
                     uint32_t(r.depth.compare_op)})
    number(v);
  scalar(r.depth.bias_constant);
  scalar(r.depth.bias_slope);
  number(r.blending.size());
  for (const auto &b : r.blending) {
    number(b.enable);
    number(b.write_mask);
    for (const auto &f : {b.color, b.alpha}) {
      number(uint32_t(f.op));
      number(uint32_t(f.src_factor));
      number(uint32_t(f.dst_factor));
    }
  }
  number(d.vertices.size());
  for (const auto &v : d.vertices) {
    number(v.binding);
    number(v.stride);
    number(v.instance);
  }
  number(d.attributes.size());
  for (const auto &v : d.attributes) {
    number(v.location);
    number(v.binding);
    number(uint32_t(v.format));
    number(v.offset);
  }
  hash.finish();
  return picosha2::get_hash_hex_string(hash);
}
class PipelineAccess : public NativeResource {
 public:
  virtual const std::string &contract() const = 0;
  virtual const std::vector<VertexInputBinding> &vertices() const = 0;
  virtual bool mesh_pipeline() const = 0;
  virtual bool task_shader() const = 0;
#ifdef TI_WITH_VULKAN
  virtual vulkan::VulkanPipeline &pipeline() const = 0;
#endif
};
struct DrawDescription {
  uint32_t pipeline{0};
  VulkanGraphicsDrawInfo draw;
  std::optional<VulkanGraphicsMeshDrawInfo> mesh;
  struct Indirect {
    VulkanGraphicsIndirectParameters parameters;
    uint32_t commands{0};
    std::optional<uint32_t> count;
  };
  std::optional<Indirect> indirect;
  std::string expected;
  struct Vertex {
    uint32_t binding, array;
  };
  std::vector<Vertex> vertices;
  std::optional<uint32_t> indices;
  struct Buffer {
    uint32_t set, binding, array{0};
    bool storage{false};
    std::vector<uint32_t> arrays;
  };
  std::vector<Buffer> buffers;
  struct Image {
    uint32_t set, binding, image;
  };
  std::vector<Image> images;
  struct Scene {
    uint32_t set, binding, scene;
  };
  std::vector<Scene> scenes;
};
struct PassFeatures {
  bool scenes, arrays, mesh, indirect;
  bool attachment_contracts{false};
};
struct PassDescription {
  static constexpr const char *kind = "vulkan_graphics_pass_direct";
  static constexpr const char *operation = "pass_direct";
  static constexpr uint32_t version = 1;
  static constexpr PassFeatures features{false, false, false, false};
  struct Color {
    bool clear;
    std::array<uint32_t, 4> bits;
  };
  std::vector<Color> colors;
  bool depth_clear{true};
  float clear_depth{0};
  std::array<uint32_t, 4> viewport{};
  struct Attachments {
    uint32_t width, height;
    std::vector<BufferFormat> colors;
    std::optional<BufferFormat> depth;
  };
  std::optional<Attachments> attachments;
  std::vector<DrawDescription> draws;
  std::vector<std::shared_ptr<PipelineAccess>> pipelines;
  std::vector<NativeArrayBinding> arrays;
  std::vector<std::shared_ptr<NativeSampledImage>> targets, images;
  std::shared_ptr<NativeSampledImage> depth;
  std::vector<std::shared_ptr<NativeShaderAccelerationStructure>> scenes;
};
struct ShaderPassDescription : PassDescription {
  static constexpr uint32_t version = 2;
  static constexpr PassFeatures features{true, false, false, false};
};
struct ArrayPassDescription : PassDescription {
  static constexpr uint32_t version = 3;
  static constexpr PassFeatures features{true, true, false, false};
};
struct MeshPassDescription : PassDescription {
  static constexpr uint32_t version = 4;
  static constexpr PassFeatures features{true, true, true, false};
};
struct ContractPassDescription : PassDescription {
  static constexpr uint32_t version = 5;
  static constexpr PassFeatures features{true, true, true, false, true};
};
struct IndirectPassDescription : PassDescription {
  static constexpr const char *kind = "vulkan_graphics_pass_indirect";
  static constexpr const char *operation = "pass_indirect";
  static constexpr PassFeatures features{true, true, false, true};
};

#ifdef TI_WITH_VULKAN
vulkan::VulkanDevice &owner_device(NativePreparationContext &c) {
  auto *device = dynamic_cast<vulkan::VulkanDevice *>(&c.device);
  if (!device || !c.gfx_runtime || c.gfx_runtime->get_ti_device() != device ||
      !c.fault_domain)
    throw std::invalid_argument(
        "Graphics requires its owner's Vulkan GFX runtime");
  return *device;
}
class PreparedPipeline final : public PipelineAccess {
 public:
  PreparedPipeline(NativePreparationContext &c, const PipelineDescription &d)
      : device_(owner_device(c)),
        fault_(c.fault_domain),
        vertices_(d.vertices),
        contract_(pipeline_contract(d)) {
    match_contract(d.expected, contract_);
    std::vector<std::vector<uint32_t>> words;
    std::vector<PipelineSourceDesc> sources;
    words.reserve(d.shaders.size());
    for (const auto &shader : d.shaders) {
      mesh_ |= shader.stage == PipelineStageType::mesh;
      task_ |= shader.stage == PipelineStageType::task;
      const auto &bytes = shader.code->bytes();
      if (bytes.size() < 20 || bytes.size() % 4)
        throw std::invalid_argument(
            "Graphics shader blob requires complete SPIR-V words");
      words.emplace_back(bytes.size() / 4);
      std::memcpy(words.back().data(), bytes.data(), bytes.size());
      sources.push_back({PipelineSourceType::spirv_binary, words.back().data(),
                         bytes.size(), shader.stage});
    }
    pipeline_ = prepare_vulkan_graphics_pipeline(
        device_, sources, d.raster, d.vertices, d.attributes, "aot_graphics");
  }
  ~PreparedPipeline() override {
    if (!fault_->backend_calls_safe() ||
        fault_->state() == RuntimeLifecycleState::kFinalized)
      pipeline_.release();
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  bool retains_input(const std::string &) const override {
    return false;
  }
  const std::string &contract() const override {
    return contract_;
  }
  const std::vector<VertexInputBinding> &vertices() const override {
    return vertices_;
  }
  bool mesh_pipeline() const override {
    return mesh_;
  }
  bool task_shader() const override {
    return task_;
  }
  vulkan::VulkanPipeline &pipeline() const override {
    return *static_cast<vulkan::VulkanPipeline *>(pipeline_.get());
  }

 private:
  vulkan::VulkanDevice &device_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::vector<VertexInputBinding> vertices_;
  std::string contract_;
  bool mesh_{false}, task_{false};
  std::unique_ptr<Pipeline> pipeline_;
};

class PreparedPass final : public NativeOperation {
  struct Draw {
    std::shared_ptr<PipelineAccess> pipeline;
    VulkanGraphicsDrawInfo info;
    std::optional<VulkanGraphicsMeshDrawInfo> mesh;
    std::optional<VulkanGraphicsIndirectPacket> indirect;
    VulkanGraphicsDrawResources resources;
  };
  struct Payload {
    VulkanGraphicsPassTargets targets;
    std::vector<Draw> draws;
    std::vector<size_t> indirect_draws;
    std::vector<std::shared_ptr<NativeSampledImage>> reads, clears;
    std::vector<std::pair<std::shared_ptr<NativeShaderAccelerationStructure>,
                          NativeShaderStages>>
        scene_reads;
  };

 public:
  PreparedPass(NativePreparationContext &c, const PassDescription &d)
      : device_(owner_device(c)),
        fault_(c.fault_domain),
        payload_(std::make_unique<Payload>()) {
    std::unordered_set<DeviceAllocationId> seen_buffers;
    const auto array = [&](uint32_t index) -> const NativeArrayBinding & {
      const auto &binding = indexed(d.arrays, index);
      if (binding.pointer.device != &device_)
        throw std::invalid_argument(
            "Graphics buffer belongs to another runtime device");
      if (seen_buffers.insert(binding.pointer.alloc_id).second)
        buffers_.push_back(binding.pointer);
      return binding;
    };
    std::map<VkImage, ImageLayout> image_layouts;
    std::unordered_map<const NativeSampledImage *, ImageLayout>
        resource_layouts;
    std::unordered_map<const NativeShaderAccelerationStructure *, size_t>
        scene_reads;
    std::vector<VulkanGraphicsColorTarget> colors;
    for (size_t i = 0; i < d.colors.size(); ++i)
      colors.push_back({d.targets[i]->gfx_sampled_allocation(),
                        d.colors[i].clear, d.colors[i].bits});
    payload_->targets = prepare_vulkan_graphics_pass_targets(
        device_, colors,
        d.depth ? d.depth->gfx_sampled_allocation() : kDeviceNullAllocation,
        d.depth_clear, d.clear_depth, d.viewport);
    auto retain_image = [&](const std::shared_ptr<NativeSampledImage> &image,
                            ImageLayout layout, bool clear = false) {
      const auto [resource, first_use] =
          resource_layouts.emplace(image.get(), layout);
      if (!first_use) {
        if (resource->second != layout)
          throw std::invalid_argument(
              "Graphics pass cannot sample an attachment alias");
        return;
      }
      const auto allocation = image->gfx_sampled_allocation();
      if (allocation.device != &device_)
        throw std::invalid_argument(
            "Graphics image allocation belongs to another device");
      const auto handle = std::get<0>(device_.get_vk_image(allocation))->image;
      auto [found, inserted] = image_layouts.emplace(handle, layout);
      if (!inserted && found->second != layout)
        throw std::invalid_argument(
            "Graphics pass cannot sample an attachment alias");
      if (inserted)
        images_.emplace_back(allocation, layout);
      image_uses_.push_back({image, layout == ImageLayout::shader_read});
      (clear ? payload_->clears : payload_->reads).push_back(image);
    };
    for (size_t i = 0; i < d.targets.size(); ++i) {
      retain_image(d.targets[i], ImageLayout::color_attachment,
                   d.colors[i].clear);
    }
    if (d.depth) {
      retain_image(d.depth, ImageLayout::depth_attachment, d.depth_clear);
    }
    VulkanGraphicsIndirectAccesses accesses;
    const bool has_indirect =
        std::any_of(d.draws.begin(), d.draws.end(),
                    [](const auto &draw) { return draw.indirect.has_value(); });
    for (const auto &input : d.draws) {
      auto pipeline = indexed(d.pipelines, input.pipeline);
      match_contract(input.expected, pipeline->contract());
      if (input.mesh.has_value() != pipeline->mesh_pipeline())
        throw std::invalid_argument(
            "Graphics pipeline and draw command kinds must agree");
      if (input.mesh) {
        const auto &mesh = *input.mesh;
        if (!device_.vk_caps().mesh_group_count_valid(
                mesh.group_count_x, mesh.group_count_y, mesh.group_count_z,
                pipeline->task_shader()))
          throw std::invalid_argument(
              "Vulkan mesh draw group counts exceed the active device limits");
      }
      validate_vulkan_graphics_pipeline_targets(pipeline->pipeline(),
                                                payload_->targets);
      std::vector<VulkanGraphicsVertexRange> ranges;
      std::vector<std::pair<uint32_t, DevicePtr>> vertices;
      for (const auto &v : input.vertices) {
        const auto &a = array(v.array);
        ranges.push_back({v.binding, {a.pointer, size_t(a.bytes)}});
        vertices.emplace_back(v.binding, a.pointer);
      }
      std::optional<VulkanGraphicsBufferRange> indices;
      if (input.indices) {
        const auto &a = array(*input.indices);
        if (a.scalar_type != PrimitiveType::u32 &&
            a.scalar_type != PrimitiveType::i32)
          throw std::invalid_argument("Graphics index array must use i32 or u32");
        // Both source APIs use the same Vulkan uint32 index representation.
        // Signed storage still promises nonnegative values within index bounds.
        indices = VulkanGraphicsBufferRange{a.pointer, size_t(a.bytes)};
      }
      std::optional<VulkanGraphicsIndirectPacket> indirect;
      if (input.indirect) {
        const auto command_view = [&](uint32_t index) {
          const auto &a = array(index);
          if (a.scalar_type != PrimitiveType::u32)
            throw std::invalid_argument(
                "Graphics indirect command/count arrays must use u32");
          return DeviceBufferRange{a.pointer, size_t(a.bytes)};
        };
        const auto &spec = *input.indirect;
        std::optional<DeviceBufferRange> count;
        if (spec.count)
          count = command_view(*spec.count);
        indirect = prepare_vulkan_graphics_indirect(
            device_, input.draw.indexed, spec.parameters,
            command_view(spec.commands), count);
        accesses.add_indirect(input.draw.indexed, *indirect);
        payload_->indirect_draws.push_back(payload_->draws.size());
      }
      if (!input.mesh)
        validate_vulkan_graphics_vertex_ranges(device_, pipeline->vertices(),
                                               ranges, indices, input.draw,
                                               indirect);
      std::vector<VulkanGraphicsShaderBufferView> buffers;
      for (const auto &v : input.buffers) {
        VulkanGraphicsShaderBufferView buffer;
        buffer.set_index = v.set;
        buffer.binding = v.binding;
        buffer.storage = v.storage;
        if (v.arrays.empty()) {
          const auto &a = array(v.array);
          buffer.allocation = a.pointer;
          buffer.offset = size_t(a.pointer.offset);
          buffer.bytes = size_t(a.bytes);
        } else {
          const auto &caps = device_.vk_caps();
          if (!caps.descriptor_storage_buffer_array_non_uniform_indexing ||
              v.arrays.size() >
                  std::min(caps.max_per_stage_descriptor_storage_buffers,
                           caps.max_descriptor_set_storage_buffers))
            throw std::invalid_argument(
                "Graphics storage descriptor array exceeds the device "
                "capability");
          buffer.array_elements.reserve(v.arrays.size());
          for (const auto index : v.arrays) {
            const auto &a = array(index);
            buffer.array_elements.push_back({a.pointer, size_t(a.bytes)});
          }
        }
        buffers.push_back(std::move(buffer));
      }
      if (has_indirect)
        accesses.add_shader_buffers(pipeline->pipeline(), buffers);
      std::vector<VulkanGraphicsShaderImageView> sampled;
      for (const auto &v : input.images) {
        auto image = indexed(d.images, v.image);
        if (!is_float_sampled_texture_format(image->parameters().format))
          throw std::invalid_argument(
              "Graphics sampled images require a floating/normalized/depth "
              "format");
        retain_image(image, ImageLayout::shader_read);
        sampled.push_back({v.set, v.binding, image->gfx_sampled_allocation(),
                           image->parameters().sampler_config});
      }
      std::vector<std::pair<uint32_t, uint32_t>> scene_points;
      for (const auto &v : input.scenes)
        scene_points.emplace_back(v.set, v.binding);
      pipeline->pipeline().validate_acceleration_structure_bindings(
          scene_points);
      auto resources = prepare_vulkan_graphics_draw_resources(
          device_, pipeline->pipeline(), buffers, sampled, vertices,
          indices ? indices->pointer : kDeviceNullPtr, input.draw.indexed,
          input.mesh.has_value(), [&](VulkanGraphicsResourceSets &sets) {
            for (const auto &v : input.scenes) {
              auto scene = indexed(d.scenes, v.scene);
              auto &set = sets[v.set];
              if (!set)
                set = device_.create_resource_set_unique();
              if (!set)
                throw std::runtime_error(
                    "Graphics AS descriptor allocation failed");
              scene->bind_for_shader(set.get(), v.binding);
              const auto stages =
                  pipeline->pipeline().acceleration_structure_stages(v.set,
                                                                     v.binding);
              if (!stages ||
                  (stages & ~(VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)))
                throw std::invalid_argument(
                    "Graphics AS requires qualified vertex/fragment stages");
              NativeShaderStages qualified = 0;
              if (stages & VK_PIPELINE_STAGE_VERTEX_SHADER_BIT)
                qualified |= native_shader_stage(PipelineStageType::vertex);
              if (stages & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                qualified |= native_shader_stage(PipelineStageType::fragment);
              auto [it, inserted] = scene_reads.emplace(
                  scene.get(), payload_->scene_reads.size());
              if (inserted)
                payload_->scene_reads.emplace_back(scene, qualified);
              else
                payload_->scene_reads[it->second].second |= qualified;
            }
          });
      payload_->draws.push_back({std::move(pipeline), input.draw, input.mesh,
                                 indirect, std::move(resources)});
    }
    accesses.validate(device_);
  }
  ~PreparedPass() override {
    if (!fault_->backend_calls_safe() ||
        fault_->state() == RuntimeLifecycleState::kFinalized)
      payload_.release();
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void validate_recording(NativeRecordingState &state) const override {
    for (const auto &image : payload_->reads)
      image->validate_sampling(state);
    for (const auto &image : payload_->clears)
      state.initialized.insert(image.get());
  }
  NativeVulkanRecordingRequirements vulkan_recording_requirements()
      const override {
    return {images_, true, buffers_, image_uses_};
  }
  void validate_vulkan_execution_domain() const override {
    if (!payload_->scene_reads.empty())
      throw std::invalid_argument(
          "Independent Vulkan graphics has no acceleration-structure lease contract");
  }
  void record_vulkan(Device *, CommandList *commands) override {
    for (const auto &[scene, stages] : payload_->scene_reads)
      scene->record_shader_read(commands, stages, true);
    for (const auto index : payload_->indirect_draws) {
      const auto &draw = payload_->draws[index];
      record_vulkan_graphics_indirect_barriers(commands, draw.info.indexed,
                                               *draw.indirect);
    }
    begin_vulkan_graphics_pass(commands, payload_->targets);
    for (const auto &draw : payload_->draws)
      record_vulkan_graphics_draw(commands, &draw.pipeline->pipeline(),
                                  draw.pipeline->task_shader(), draw.info,
                                  draw.indirect, draw.mesh, draw.resources);
    end_vulkan_graphics_pass(commands);
  }

 private:
  vulkan::VulkanDevice &device_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::unique_ptr<Payload> payload_;
  std::vector<std::pair<DeviceAllocation, ImageLayout>> images_;
  std::vector<DeviceAllocation> buffers_;
  std::vector<NativeVulkanRecordingRequirements::ImageUse> image_uses_;
};
#endif

PipelineDescription pipeline_parameters(const Json &p, uint32_t version) {
  fields(p, {"shader_count", "shaders", "vertices", "attributes", "topology",
             "polygon_mode", "front_face_cull", "back_face_cull", "depth_test",
             "depth_write", "depth_compare", "depth_bias_constant",
             "depth_bias_slope", "color_targets", "expected_pipeline"});
  PipelineDescription d;
  d.expected = expected_contract(p);
  for (const auto &s : array(p, "shaders")) {
    fields(s, {"stage"});
    const auto stage = name(s["stage"]);
    if (version == 1 && stage != "vertex" && stage != "fragment")
      throw std::invalid_argument(
          "AOT graphics v1 requires vertex/fragment shaders");
    PipelineStageType type;
    if (stage == "vertex")
      type = PipelineStageType::vertex;
    else if (stage == "fragment")
      type = PipelineStageType::fragment;
    else if (stage == "mesh")
      type = PipelineStageType::mesh;
    else if (stage == "task")
      type = PipelineStageType::task;
    else
      throw std::invalid_argument("Unsupported AOT graphics shader stage");
    d.shaders.push_back({type, {}});
  }
  if (d.shaders.size() != u32(p["shader_count"]))
    throw std::invalid_argument("Graphics shader count differs from its array");
  for (const auto &v : array(p, "vertices")) {
    fields(v, {"binding", "stride", "instance"});
    d.vertices.push_back(
        {u32(v["binding"]), u32(v["stride"]), optional_bool(v, "instance")});
  }
  for (const auto &v : array(p, "attributes")) {
    fields(v, {"location", "binding", "format", "offset"});
    d.attributes.push_back({u32(v["location"]), u32(v["binding"]),
                            format(v["format"]), u32(v["offset"])});
  }
  auto &r = d.raster;
  if (p.obj.inner.count("topology"))
    r.prim_topology = TopologyType(
        enum_value(p["topology"], {"triangles", "lines", "points"}));
  if (p.obj.inner.count("polygon_mode"))
    r.polygon_mode =
        PolygonMode(enum_value(p["polygon_mode"], {"fill", "line", "point"}));
  r.front_face_cull = optional_bool(p, "front_face_cull");
  r.back_face_cull = optional_bool(p, "back_face_cull");
  r.depth_test = optional_bool(p, "depth_test");
  r.depth_write = optional_bool(p, "depth_write");
  if (p.obj.inner.count("depth_compare"))
    r.depth.compare_op = DepthCompareOp(enum_value(
        p["depth_compare"], {"never", "less", "equal", "less_equal", "greater",
                             "not_equal", "greater_equal", "always"}));
  r.depth.bias_constant = optional_f32(p, "depth_bias_constant", 0);
  r.depth.bias_slope = optional_f32(p, "depth_bias_slope", 0);
  if (p.obj.inner.count("color_targets")) {
    for (const auto &v : array(p, "color_targets")) {
      fields(v, {"blending", "write_mask", "color_op", "alpha_op",
                 "src_color_factor", "dst_color_factor", "src_alpha_factor",
                 "dst_alpha_factor"});
      BlendingParams b;
      b.enable = optional_bool(v, "blending");
      b.write_mask = opt_u32(v, "write_mask", 15);
      if (b.write_mask > 15)
        throw std::invalid_argument(
            "Graphics color write mask exceeds RGBA bits");
      for (const auto &item : {std::make_pair("color_op", &b.color.op),
                               std::make_pair("alpha_op", &b.alpha.op)})
        if (v.obj.inner.count(item.first))
          *item.second = BlendOp(enum_value(
              v[item.first],
              {"add", "subtract", "reverse_subtract", "min", "max"}));
      for (const auto &item :
           {std::make_pair("src_color_factor", &b.color.src_factor),
            std::make_pair("dst_color_factor", &b.color.dst_factor),
            std::make_pair("src_alpha_factor", &b.alpha.src_factor),
            std::make_pair("dst_alpha_factor", &b.alpha.dst_factor)})
        if (v.obj.inner.count(item.first))
          *item.second = BlendFactor(enum_value(
              v[item.first],
              {"zero", "one", "src_color", "one_minus_src_color", "dst_color",
               "one_minus_dst_color", "src_alpha", "one_minus_src_alpha",
               "dst_alpha", "one_minus_dst_alpha"}));
      r.blending.push_back(b);
    }
  }
  return d;
}

PassDescription pass_parameters(const Json &p, PassFeatures features) {
  fields(p, {"pipeline_count", "array_count", "image_count", "color_count",
             "colors", "draws", "viewport", "depth_clear", "clear_depth",
             "scene_count", "attachment_contract"});
  if (!features.scenes && p.obj.inner.count("scene_count"))
    throw std::invalid_argument(
        "Graphics AS bindings require pass_direct version 2");
  PassDescription d;
  if (p.obj.inner.count("attachment_contract")) {
    if (!features.attachment_contracts)
      throw std::invalid_argument(
          "Graphics attachment contracts require pass_direct version 5");
    const auto &value = p["attachment_contract"];
    fields(value, {"width", "height", "colors", "depth"});
    PassDescription::Attachments contract{u32(value["width"]),
                                          u32(value["height"]), {}, {}};
    if (!contract.width || !contract.height)
      throw std::invalid_argument("Graphics attachment extent must be positive");
    for (const auto &color : array(value, "colors"))
      contract.colors.push_back(format(color));
    if (value.obj.inner.count("depth"))
      contract.depth = format(value["depth"]);
    if (contract.colors.size() != u32(p["color_count"]))
      throw std::invalid_argument("Graphics attachment contract color count mismatch");
    d.attachments = std::move(contract);
  }
  d.depth_clear = optional_bool(p, "depth_clear", true);
  d.clear_depth = optional_f32(p, "clear_depth", 0);
  if (p.obj.inner.count("viewport")) {
    const auto &v = array(p, "viewport");
    if (v.size() != 4)
      throw std::invalid_argument(
          "Graphics viewport requires x/y/width/height");
    for (size_t i = 0; i < 4; ++i)
      d.viewport[i] = u32(v[i]);
  }
  for (const auto &v : array(p, "colors")) {
    fields(v, {"clear", "clear_bits"});
    PassDescription::Color color{optional_bool(v, "clear", true), {}};
    if (v.obj.inner.count("clear_bits")) {
      const auto &bits = array(v, "clear_bits");
      if (bits.size() != 4)
        throw std::invalid_argument(
            "Graphics clear_bits requires four uint32 values");
      for (size_t i = 0; i < 4; ++i)
        color.bits[i] = u32(bits[i]);
    }
    d.colors.push_back(color);
  }
  if (d.colors.size() != u32(p["color_count"]))
    throw std::invalid_argument("Graphics color count differs from its array");
  for (const auto &v : array(p, "draws")) {
    fields(v, {"pipeline", "vertices", "index", "buffers", "images",
               "element_count", "instance_count", "first_vertex", "first_index",
               "first_instance", "vertex_offset", "index_min", "index_max",
               "expected_pipeline", "scenes", "mesh", "indirect"});
    if (!features.scenes && v.obj.inner.count("scenes"))
      throw std::invalid_argument(
          "Graphics AS bindings require pass_direct version 2");
    DrawDescription draw;
    std::set<std::pair<uint32_t, uint32_t>> descriptors;
    auto claim = [&](uint32_t set, uint32_t binding) {
      if (set > INT32_MAX || binding > INT32_MAX ||
          !descriptors.emplace(set, binding).second)
        throw std::invalid_argument(
            "Graphics descriptor binding is duplicated or exceeds int32 ABI");
    };
    draw.pipeline = u32(v["pipeline"]);
    draw.expected = expected_contract(v);
    if (v.obj.inner.count("mesh")) {
      if (!features.mesh)
        throw std::invalid_argument(
            "Graphics mesh draws require pass_direct version 4");
      for (const auto *key : {"element_count", "instance_count", "first_vertex",
                              "first_index", "first_instance", "vertex_offset",
                              "index_min", "index_max", "index", "indirect"})
        if (v.obj.inner.count(key))
          throw std::invalid_argument(
              "Graphics mesh draws cannot declare vertex/index draw fields");
      const auto &groups = array(v, "mesh");
      if (groups.size() != 3)
        throw std::invalid_argument(
            "Graphics mesh requires three group counts");
      draw.mesh = VulkanGraphicsMeshDrawInfo{u32(groups[0]), u32(groups[1]),
                                             u32(groups[2])};
    } else if (v.obj.inner.count("indirect")) {
      if (!features.indirect)
        throw std::invalid_argument(
            "Graphics indirect draws require pass_indirect");
      for (const auto *key :
           {"element_count", "instance_count", "first_vertex", "first_index",
            "first_instance", "vertex_offset", "index_min", "index_max"})
        if (v.obj.inner.count(key))
          throw std::invalid_argument(
              "Graphics indirect draws cannot declare direct draw fields");
      const auto &spec = v["indirect"];
      fields(spec, {"commands", "count", "command_offset", "count_offset",
                    "max_draw_count", "stride", "vertex_record_limit",
                    "instance_record_limit", "index_element_limit",
                    "first_instance_may_be_nonzero"});
      DrawDescription::Indirect input;
      input.commands = u32(spec["commands"]);
      if (spec.obj.inner.count("count"))
        input.count = u32(spec["count"]);
      auto &parameters = input.parameters;
      parameters.command_offset =
          optional_integer(spec, "command_offset").value_or(0);
      parameters.count_offset =
          optional_integer(spec, "count_offset").value_or(0);
      parameters.max_draw_count = u32(spec["max_draw_count"]);
      parameters.stride = u32(spec["stride"]);
      parameters.vertex_record_limit = u32(spec["vertex_record_limit"]);
      parameters.instance_record_limit =
          opt_u32(spec, "instance_record_limit", 1);
      parameters.index_element_limit = opt_u32(spec, "index_element_limit", 0);
      parameters.first_instance_may_be_nonzero =
          optional_bool(spec, "first_instance_may_be_nonzero");
      draw.indirect = input;
    } else {
      draw.draw.element_count = u32(v["element_count"]);
    }
    draw.draw.instance_count = opt_u32(v, "instance_count", 1);
    draw.draw.first_vertex = opt_u32(v, "first_vertex", 0);
    draw.draw.first_index = opt_u32(v, "first_index", 0);
    draw.draw.first_instance = opt_u32(v, "first_instance", 0);
    draw.draw.vertex_offset = opt_i32(v, "vertex_offset");
    draw.draw.index_min = opt_u32(v, "index_min", 0);
    draw.draw.index_max = opt_u32(v, "index_max", 0);
    if (v.obj.inner.count("index")) {
      if (!draw.indirect &&
          (!v.obj.inner.count("index_min") || !v.obj.inner.count("index_max")))
        throw std::invalid_argument(
            "Indexed graphics requires explicit index_min and index_max "
            "bounds");
      draw.indices = u32(v["index"]);
      draw.draw.indexed = true;
    }
    for (const auto &vertex : array(v, "vertices")) {
      fields(vertex, {"binding", "array"});
      draw.vertices.push_back({u32(vertex["binding"]), u32(vertex["array"])});
    }
    if (draw.mesh && !draw.vertices.empty())
      throw std::invalid_argument(
          "Graphics mesh draws cannot bind vertex inputs");
    for (const auto &b : array(v, "buffers")) {
      fields(b, {"set", "binding", "array", "arrays", "storage"});
      claim(u32(b["set"]), u32(b["binding"]));
      const bool elements = b.obj.inner.count("arrays");
      if (elements && !features.arrays)
        throw std::invalid_argument(
            "Graphics descriptor arrays require pass_direct version 3");
      if (bool(b.obj.inner.count("array")) == elements)
        throw std::invalid_argument(
            "Graphics buffer requires exactly one of array or arrays");
      DrawDescription::Buffer buffer;
      buffer.set = u32(b["set"]);
      buffer.binding = u32(b["binding"]);
      buffer.storage = optional_bool(b, "storage");
      if (elements) {
        const auto &indices = array(b, "arrays");
        if (!buffer.storage || indices.size() < 2 ||
            indices.size() > (std::numeric_limits<uint32_t>::max)())
          throw std::invalid_argument(
              "Graphics descriptor arrays require at least two storage buffers "
              "and a uint32 count");
        for (const auto &index : indices)
          buffer.arrays.push_back(u32(index));
      } else {
        buffer.array = u32(b["array"]);
      }
      draw.buffers.push_back(std::move(buffer));
    }
    for (const auto &i : array(v, "images")) {
      fields(i, {"set", "binding", "image"});
      claim(u32(i["set"]), u32(i["binding"]));
      draw.images.push_back(
          {u32(i["set"]), u32(i["binding"]), u32(i["image"])});
    }
    if (v.obj.inner.count("scenes")) {
      for (const auto &s : array(v, "scenes")) {
        fields(s, {"set", "binding", "scene"});
        claim(u32(s["set"]), u32(s["binding"]));
        draw.scenes.push_back(
            {u32(s["set"]), u32(s["binding"]), u32(s["scene"])});
      }
    }
    d.draws.push_back(std::move(draw));
  }
  if (d.draws.empty() && !features.attachment_contracts)
    throw std::invalid_argument("Graphics pass requires at least one draw");
  return d;
}
}  // namespace

template <typename Descriptor>
static void register_graphics_pass(NativeOperationRegistry &registry) {
  std::vector<NativeOperationRegistry::BindingGroup> groups{
      {"pipeline_count", {"pipeline"}},
      {"array_count", {"array"}},
      {"image_count", {"image"}},
      {"color_count", {"color"}}};
  if constexpr (Descriptor::features.scenes)
    groups.push_back({"scene_count", {"scene"}});
  registry.define<Descriptor>([](Device &device, const Descriptor &d) {
    for (const auto &p : d.pipelines)
      same_device(device, p);
    for (const auto &i : d.targets)
      same_device(device, i);
    for (const auto &i : d.images)
      same_device(device, i);
    for (const auto &scene : d.scenes)
      same_device(device, scene);
    if (d.depth)
      same_device(device, d.depth);
    if (d.attachments) {
      const auto &contract = *d.attachments;
      const auto match = [&](const NativeSampledImage &image,
                             BufferFormat format) {
        const auto &p = image.parameters();
        if (p.dimension != ImageDimension::d2D || p.x != contract.width ||
            p.y != contract.height || p.z != 1 || p.mip_levels != 1 ||
            p.format != format)
          throw std::invalid_argument("Graphics attachment layout contract mismatch");
      };
      for (size_t i = 0; i < d.targets.size(); ++i)
        match(*d.targets[i], contract.colors[i]);
      if (bool(d.depth) != contract.depth.has_value())
        throw std::invalid_argument("Graphics attachment depth contract mismatch");
      if (d.depth)
        match(*d.depth, *contract.depth);
    }
    for (const auto &a : d.arrays) {
      if (a.pointer.device != &device ||
          a.bytes > (std::numeric_limits<size_t>::max)() ||
          a.pointer.offset > (std::numeric_limits<size_t>::max)())
        throw std::invalid_argument(
            "Graphics array requires a local size_t range");
      resolve_native_storage_range(Arch::vulkan, a.pointer, a.bytes);
    }
  });
  registry.register_schema<Descriptor>(
      "vulkan_graphics", Descriptor::operation, {},
      [](const Json &p) {
        Descriptor d;
        static_cast<PassDescription &>(d) =
            pass_parameters(p, Descriptor::features);
        return [d, p](const NativeOperationBindings &b) mutable {
          auto result = d;
          for (uint32_t i = 0; i < u32(p["pipeline_count"]); ++i)
            result.pipelines.push_back(b.at("pipeline_" + std::to_string(i))
                                           .resource<PipelineAccess>());
          for (uint32_t i = 0; i < u32(p["array_count"]); ++i)
            result.arrays.push_back(b.at("array_" + std::to_string(i)).array());
          for (uint32_t i = 0; i < u32(p["image_count"]); ++i)
            result.images.push_back(b.at("image_" + std::to_string(i))
                                        .resource<NativeSampledImage>());
          for (uint32_t i = 0; i < u32(p["color_count"]); ++i)
            result.targets.push_back(b.at("color_" + std::to_string(i))
                                         .resource<NativeSampledImage>());
          if constexpr (Descriptor::features.scenes) {
            for (uint32_t i = 0; i < u32(p["scene_count"]); ++i)
              result.scenes.push_back(
                  b.at("scene_" + std::to_string(i))
                      .resource<NativeShaderAccelerationStructure>());
          }
          if (b.count("depth"))
            result.depth = b.at("depth").resource<NativeSampledImage>();
          return result;
        };
      },
      {}, {"depth"}, groups);
#ifdef TI_WITH_VULKAN
  registry.register_context_factory<Descriptor>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &c, const Descriptor &d) {
        return std::make_shared<PreparedPass>(c, d);
      });
#endif
}

template <typename Descriptor>
static void register_graphics_pipeline(NativeOperationRegistry &registry) {
  registry.define<Descriptor>(
      [](Device &device, const Descriptor &d) {
        for (const auto &s : d.shaders)
          same_device(device, s.code);
      },
      NativeObjectKind::resource);
  registry.register_schema<Descriptor>(
      "vulkan_graphics", "pipeline", {},
      [](const Json &p) {
        Descriptor d;
        static_cast<PipelineDescription &>(d) =
            pipeline_parameters(p, Descriptor::version);
        return [d](const NativeOperationBindings &b) mutable {
          auto result = d;
          for (size_t i = 0; i < result.shaders.size(); ++i)
            result.shaders[i].code = b.at("shader_" + std::to_string(i))
                                         .resource<NativeHostBlobResource>();
          return result;
        };
      },
      {}, {}, {{"shader_count", {"shader"}}});
#ifdef TI_WITH_VULKAN
  registry.register_context_factory<Descriptor>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &c, const Descriptor &d) {
        return std::make_shared<PreparedPipeline>(c, d);
      });
#endif
}

void register_vulkan_graphics_operations(NativeOperationRegistry &registry) {
  register_graphics_pipeline<PipelineDescription>(registry);
  register_graphics_pipeline<MeshPipelineDescription>(registry);
  register_graphics_pass<PassDescription>(registry);
  register_graphics_pass<ShaderPassDescription>(registry);
  register_graphics_pass<ArrayPassDescription>(registry);
  register_graphics_pass<MeshPassDescription>(registry);
  register_graphics_pass<ContractPassDescription>(registry);
  register_graphics_pass<IndirectPassDescription>(registry);
}
}  // namespace taichi::lang
