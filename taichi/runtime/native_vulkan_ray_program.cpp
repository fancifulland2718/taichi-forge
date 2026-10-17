#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_acceleration_structure.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/runtime/native_image.h"
#include "taichi/runtime/vulkan_ray_program.h"
#include "taichi/common/operation_json.h"

#include <cstring>
#include <algorithm>
#include <limits>
#include <set>
#include "picosha2.h"
#ifdef TI_WITH_VULKAN
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/gfx/runtime.h"
#endif

namespace taichi::lang {
namespace {
using Json = liong::json::JsonValue;
using operation_json::fields;
using operation_json::integer;
using operation_json::name;
using operation_json::optional_bool;

uint32_t u32(const Json &value) {
  const auto n = integer(value);
  if (n > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Ray program value exceeds uint32 ABI");
  return uint32_t(n);
}
uint32_t optional_u32(const Json &p, const char *key, uint32_t fallback) {
  const auto found = p.obj.inner.find(key);
  return found == p.obj.inner.end() ? fallback : u32(found->second);
}
const std::vector<Json> &array(const Json &p, const char *key) {
  const auto &v = p[key];
  if (!v.is_arr())
    throw std::invalid_argument(std::string("Ray program requires array: ") +
                                key);
  return v.arr.inner;
}
const std::vector<Json> &group_array(const Json &p,
                                     const char *key,
                                     const char *count) {
  const auto &values = array(p, key);
  if (values.size() != u32(p[count]))
    throw std::invalid_argument(
        "Ray descriptor count does not match its array");
  return values;
}
void same_device(Device &device,
                 const std::shared_ptr<NativeResource> &resource) {
  if (!resource || resource->arch() != Arch::vulkan ||
      &resource->device() != &device)
    throw std::invalid_argument(
        "Ray program resource requires the same Vulkan device");
}
struct CodeInput {
  VulkanRayProgramShader::Stage stage;
  std::string entry;
  std::shared_ptr<NativeHostBlobResource> code;
};
struct PipelineDescription {
  static constexpr const char *kind = "vulkan_ray_program_pipeline";
  static constexpr uint32_t version = 1;
  std::vector<CodeInput> shaders;
  std::vector<VulkanRayProgramGroup> groups;
  uint32_t depth{1};
  bool micromap{false};
  std::string expected_pipeline;
};

// Stable, length-delimited little-endian contract, independent of JSON
// spelling, driver handles and build paths. Hash compiled code only at cold
// creation.
std::string pipeline_contract(const PipelineDescription &d) {
  picosha2::hash256_one_by_one hash;
  auto integer = [&](uint64_t value, size_t width) {
    uint8_t bytes[8];
    for (size_t i = 0; i < width; ++i)
      bytes[i] = uint8_t(value >> (8 * i));
    hash.process(bytes, bytes + width);
  };
  auto text = [&](const std::string &value) {
    integer(value.size(), 8);
    hash.process(value.begin(), value.end());
  };
  text("forge-vulkan-ray-program-v1");
  integer(d.shaders.size(), 8);
  for (const auto &shader : d.shaders) {
    integer(uint32_t(shader.stage), 4);
    text(shader.entry);
    const auto &bytes = shader.code->bytes();
    text(picosha2::hash256_hex_string(bytes.begin(), bytes.end()));
  }
  integer(d.groups.size(), 8);
  for (const auto &group : d.groups) {
    integer(uint32_t(group.kind), 4);
    integer(group.general, 4);
    integer(group.closest_hit, 4);
    integer(group.any_hit, 4);
  }
  integer(d.depth, 4);
  integer(d.micromap, 4);
  hash.finish();
  return picosha2::get_hash_hex_string(hash);
}

std::string expected_pipeline(const Json &p) {
  if (!p.obj.inner.count("expected_pipeline"))
    return {};
  const auto value = name(p["expected_pipeline"]);
  if (value.size() != 64 ||
      value.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::invalid_argument(
        "Expected ray pipeline must be a SHA256 contract");
  return value;
}
void validate_pipeline(const std::string &expected, const std::string &actual) {
  if (!expected.empty() && expected != actual)
    throw std::invalid_argument(
        "Ray pipeline code or layout contract mismatch");
}
class ProgramAccess : public NativeResource {
 public:
  virtual const std::string &contract() const = 0;
#ifdef TI_WITH_VULKAN
  virtual vulkan::VulkanPipeline &pipeline() const = 0;
#endif
};
struct Point {
  uint32_t set, binding;
};
Point point(const Json &p) {
  Point result{u32(p["set"]), u32(p["binding"])};
  if (result.binding > std::numeric_limits<int>::max())
    throw std::invalid_argument(
        "Shader binding exceeds the signed RHI binding ABI");
  return result;
}
struct Access {
  bool read, write;
};
Access access(const Json &p) {
  const auto a = name(p["access"]);
  if (a == "read")
    return {true, false};
  if (a == "write")
    return {false, true};
  if (a == "read_write")
    return {true, true};
  throw std::invalid_argument(
      "Ray descriptor access must be read, write or read_write");
}
struct BufferInput {
  Point point;
  bool uniform;
  NativeArrayBinding array;
};
struct ImageInput {
  Point point;
  bool storage, read;
  uint32_t mip;
  std::shared_ptr<NativeSampledImage> image;
};
struct SceneInput {
  Point point;
  std::shared_ptr<NativeShaderAccelerationStructure> scene;
};
struct Section {
  uint64_t offset{0}, size{0};
  explicit Section(const Json &p) {
    offset = operation_json::optional_integer(p, "offset").value_or(0);
    size = operation_json::optional_integer(p, "size").value_or(0);
  }
  std::vector<uint8_t> read(const NativeHostBlobResource &blob) const {
    const auto &bytes = blob.bytes();
    if (offset > bytes.size() || size > bytes.size() - offset)
      throw std::invalid_argument(
          "Ray scalar payload section exceeds its host blob");
    return {bytes.begin() + size_t(offset),
            bytes.begin() + size_t(offset + size)};
  }
};
VulkanRayProgramRecord record(const Json &p,
                              const NativeHostBlobResource &blob) {
  fields(p, {"group", "offset", "size"});
  return {u32(p["group"]), Section(p).read(blob)};
}
struct LaunchDescription {
  static constexpr const char *kind = "vulkan_ray_program_launch";
  static constexpr uint32_t version = 1;
  std::shared_ptr<ProgramAccess> program;
  std::shared_ptr<NativeHostBlobResource> payload;
  std::array<uint32_t, 3> dimensions;
  VulkanRayProgramRecord raygen;
  std::vector<VulkanRayProgramRecord> miss, hit;
  std::vector<uint8_t> push;
  std::vector<BufferInput> buffers;
  std::vector<ImageInput> images;
  std::vector<SceneInput> scenes;
  std::string expected_pipeline;
};

#ifdef TI_WITH_VULKAN
vulkan::VulkanDevice &owner_device(NativePreparationContext &c) {
  auto *device = dynamic_cast<vulkan::VulkanDevice *>(&c.device);
  if (!device || !c.gfx_runtime || c.gfx_runtime->get_ti_device() != device ||
      !c.fault_domain)
    throw std::invalid_argument(
        "Ray program requires its owner's Vulkan GFX runtime");
  if (!device->vk_caps().ray_tracing_pipeline)
    throw std::invalid_argument("Vulkan RT pipelines are unavailable");
  return *device;
}
class PreparedPipeline final : public ProgramAccess {
 public:
  PreparedPipeline(NativePreparationContext &c, const PipelineDescription &d)
      : device_(owner_device(c)),
        fault_(c.fault_domain),
        contract_(pipeline_contract(d)) {
    validate_pipeline(d.expected_pipeline, contract_);
    std::vector<VulkanRayProgramShader> shaders;
    shaders.reserve(d.shaders.size());
    for (const auto &input : d.shaders) {
      const auto &bytes = input.code->bytes();
      if (bytes.size() < 20 || bytes.size() % 4)
        throw std::invalid_argument(
            "Ray shader blob must contain complete SPIR-V words");
      VulkanRayProgramShader shader;
      shader.stage = input.stage;
      shader.entry = input.entry;
      shader.words.resize(bytes.size() / 4);
      std::memcpy(shader.words.data(), bytes.data(), bytes.size());
      shaders.push_back(std::move(shader));
    }
    pipeline_ = prepare_vulkan_ray_program_pipeline(device_, shaders, d.groups,
                                                    d.depth, d.micromap);
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
  vulkan::VulkanPipeline &pipeline() const override {
    return *pipeline_;
  }

 private:
  vulkan::VulkanDevice &device_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  const std::string contract_;
  std::unique_ptr<vulkan::VulkanPipeline> pipeline_;
};
class PreparedLaunch final : public NativeOperation {
  struct Payload {
    std::shared_ptr<ProgramAccess> program;
    VulkanRayProgramResourceSets sets;
    std::unique_ptr<vulkan::VulkanShaderBindingTable> sbt;
    std::vector<std::function<void(CommandList *)>> reads;
    std::vector<std::shared_ptr<NativeSampledImage>> read_images;
    std::vector<std::shared_ptr<NativeSampledImage>> image_owners;
  };

 public:
  PreparedLaunch(NativePreparationContext &c, const LaunchDescription &d)
      : device_(owner_device(c)),
        runtime_(*c.gfx_runtime),
        fault_(c.fault_domain),
        dimensions_(d.dimensions),
        push_(d.push),
        payload_(std::make_unique<Payload>()) {
    validate_pipeline(d.expected_pipeline, d.program->contract());
    payload_->program = d.program;
    auto &pipeline = d.program->pipeline();
    validate_vulkan_ray_program_launch(device_, pipeline, dimensions_, push_);
    std::set<std::pair<uint32_t, uint32_t>> claimed;
    auto get_set = [&](const Point &point) -> vulkan::VulkanResourceSet & {
      if (!claimed.emplace(point.set, point.binding).second)
        throw std::invalid_argument("Duplicate ray descriptor binding");
      auto &set = payload_->sets[point.set];
      if (!set)
        set = std::make_unique<vulkan::VulkanResourceSet>(&device_);
      return *set;
    };
    const auto &limits = device_.get_vk_physical_device_props().limits;
    for (const auto &input : d.buffers) {
      const auto &a = input.array;
      resolve_native_storage_range(Arch::vulkan, a.pointer, a.bytes);
      const auto alignment = input.uniform
                                 ? limits.minUniformBufferOffsetAlignment
                                 : limits.minStorageBufferOffsetAlignment;
      const auto maximum = input.uniform ? limits.maxUniformBufferRange
                                         : limits.maxStorageBufferRange;
      const auto usage =
          input.uniform ? AllocUsage::Uniform : AllocUsage::Storage;
      if (!a.bytes || a.bytes > maximum || a.pointer.offset % alignment ||
          !int(device_.allocation_usage({&device_, a.pointer.alloc_id}) &
               usage))
        throw std::invalid_argument(
            "Ray buffer range, alignment or usage is incompatible");
      auto &set = get_set(input.point);
      if (input.uniform)
        set.buffer(input.point.binding, a.pointer, a.bytes);
      else
        set.rw_buffer(input.point.binding, a.pointer, a.bytes);
    }
    for (const auto &input : d.images) {
      const auto &p = input.image->parameters();
      const auto image = input.image->gfx_sampled_allocation();
      if (image.device != &device_ || input.mip >= p.mip_levels ||
          (!input.storage && input.mip))
        throw std::invalid_argument("Ray image device or mip is incompatible");
      const auto layout = input.storage ? ImageLayout::shader_read_write
                                        : ImageLayout::shader_read;
      auto &set = get_set(input.point);
      if (input.storage) {
        if (!(p.usage & ImageAllocUsage::Storage))
          throw std::invalid_argument(
              "Ray storage image requires explicit storage usage");
        VkFormatProperties properties{};
        device_.vk_device()
            .instance_functions()
            .vkGetPhysicalDeviceFormatProperties(
                device_.vk_physical_device(),
                std::get<2>(device_.get_vk_image(image)), &properties);
        if (!(properties.optimalTilingFeatures &
              VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
          throw std::invalid_argument(
              "Ray storage image format lacks storage support");
        set.rw_image(input.point.binding, image, input.mip);
      } else {
        pipeline.validate_sampled_texture(input.point.set, input.point.binding,
                                          p.sampler_config.compare_op >= 0);
        set.image(input.point.binding, image, p.sampler_config);
      }
      auto prior =
          std::find_if(images_.begin(), images_.end(),
                       [&](const auto &v) { return v.first == image; });
      if (prior != images_.end() && prior->second != layout)
        throw std::invalid_argument(
            "One ray launch cannot sample and store the same image");
      if (prior == images_.end())
        images_.emplace_back(image, layout);
      payload_->image_owners.push_back(input.image);
      if (input.read)
        payload_->read_images.push_back(input.image);
    }
    for (const auto &input : d.scenes) {
      if (!d.hit.empty() &&
          input.scene->max_sbt_record_offset() >= d.hit.size())
        throw std::invalid_argument(
            "Ray hit records do not cover bound instance SBT offsets");
      input.scene->bind_for_shader(&get_set(input.point), input.point.binding);
      payload_->reads.push_back([scene = input.scene](CommandList *commands) {
        scene->record_shader_read(commands, PipelineStageType::raytracing,
                                  false);
      });
    }
    prepare_vulkan_ray_program_bindings(pipeline, payload_->sets);
    payload_->sbt = prepare_vulkan_ray_program_sbt(device_, pipeline, d.raygen,
                                                   d.miss, d.hit);
  }
  void initialize() {
    initialization_pending_ = true;
    runtime_.enqueue_compute_op_lambda(
        [sbt = payload_->sbt.get()](Device *, CommandList *commands) {
          sbt->record_initialization(
              *static_cast<vulkan::VulkanCommandList *>(commands));
        },
        {});
    runtime_.synchronize();
    initialization_pending_ = false;
  }
  ~PreparedLaunch() override {
    if (!fault_->backend_calls_safe() ||
        fault_->state() == RuntimeLifecycleState::kFinalized) {
      payload_.release();
    } else if (initialization_pending_) {
      try {
        runtime_.synchronize();
      } catch (...) {
        payload_.release();
      }
    }
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return payload_->sbt->allocated_bytes();
  }
  bool retains_input(const std::string &port) const override {
    return port != "payload";
  }
  void validate_recording(NativeRecordingState &state) const override {
    for (const auto &image : payload_->read_images)
      image->validate_sampling(state);
  }
  NativeVulkanRecordingRequirements vulkan_recording_requirements()
      const override {
    return {images_, false};
  }
  void record_vulkan(Device *, CommandList *commands) override {
    record_vulkan_ray_program(commands, payload_->program->pipeline(),
                              *payload_->sbt, payload_->sets, payload_->reads,
                              dimensions_, push_);
  }

 private:
  vulkan::VulkanDevice &device_;
  gfx::GfxRuntime &runtime_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::array<uint32_t, 3> dimensions_;
  std::vector<uint8_t> push_;
  std::vector<std::pair<DeviceAllocation, ImageLayout>> images_;
  std::unique_ptr<Payload> payload_;
  bool initialization_pending_{false};
};
#endif
}  // namespace

void register_vulkan_ray_program_operations(NativeOperationRegistry &registry) {
  registry.define<PipelineDescription>(
      [](Device &device, const PipelineDescription &d) {
        for (const auto &shader : d.shaders)
          same_device(device, shader.code);
      },
      NativeObjectKind::resource);
  registry.register_schema<PipelineDescription>(
      "vulkan_ray_program", "pipeline", {},
      [](const Json &p) {
        fields(p, {"shader_count", "shaders", "groups", "max_trace_depth",
                   "allow_opacity_micromaps", "expected_pipeline"});
        PipelineDescription d;
        d.expected_pipeline = expected_pipeline(p);
        for (const auto &shader : group_array(p, "shaders", "shader_count")) {
          fields(shader, {"stage", "entry"});
          const auto stage = name(shader["stage"]);
          VulkanRayProgramShader::Stage native;
          if (stage == "raygen")
            native = VulkanRayProgramShader::Stage::raygen;
          else if (stage == "miss")
            native = VulkanRayProgramShader::Stage::miss;
          else if (stage == "closest_hit")
            native = VulkanRayProgramShader::Stage::closest_hit;
          else if (stage == "any_hit")
            native = VulkanRayProgramShader::Stage::any_hit;
          else
            throw std::invalid_argument("Invalid ray shader stage");
          d.shaders.push_back(
              {native,
               shader.obj.inner.count("entry") ? name(shader["entry"]) : "main",
               {}});
        }
        for (const auto &group : array(p, "groups")) {
          fields(group, {"kind", "general", "closest_hit", "any_hit"});
          const auto kind = name(group["kind"]);
          VulkanRayProgramGroup g;
          if (kind == "raygen")
            g.kind = VulkanRayProgramGroup::Kind::raygen;
          else if (kind == "miss")
            g.kind = VulkanRayProgramGroup::Kind::miss;
          else if (kind == "triangles")
            g.kind = VulkanRayProgramGroup::Kind::triangles;
          else
            throw std::invalid_argument("Invalid ray group kind");
          g.general = optional_u32(group, "general", UINT32_MAX);
          g.closest_hit = optional_u32(group, "closest_hit", UINT32_MAX);
          g.any_hit = optional_u32(group, "any_hit", UINT32_MAX);
          d.groups.push_back(g);
        }
        d.depth = optional_u32(p, "max_trace_depth", 1);
        d.micromap = optional_bool(p, "allow_opacity_micromaps");
        return [d](const NativeOperationBindings &b) mutable {
          auto result = d;
          for (size_t i = 0; i < result.shaders.size(); ++i)
            result.shaders[i].code = b.at("shader_" + std::to_string(i))
                                         .resource<NativeHostBlobResource>();
          return result;
        };
      },
      {}, {}, {{"shader_count", {"shader"}}});
  registry.define<LaunchDescription>(
      [](Device &device, const LaunchDescription &d) {
        same_device(device, d.program);
        same_device(device, d.payload);
        for (const auto &v : d.images)
          same_device(device, v.image);
        for (const auto &v : d.scenes)
          same_device(device, v.scene);
        for (const auto &v : d.buffers)
          if (v.array.pointer.device != &device)
            throw std::invalid_argument("Ray buffer belongs to another device");
      });
  registry.register_schema<LaunchDescription>(
      "vulkan_ray_program", "launch", {"pipeline", "payload"},
      [](const Json &p) {
        fields(p, {"dimensions", "raygen", "miss", "hit", "push_constants",
                   "buffer_count", "buffers", "image_count", "images",
                   "scene_count", "scenes", "expected_pipeline"});
        group_array(p, "buffers", "buffer_count");
        group_array(p, "images", "image_count");
        group_array(p, "scenes", "scene_count");
        return [p](const NativeOperationBindings &b) {
          const auto &buffers = array(p, "buffers");
          const auto &images = array(p, "images");
          const auto &scenes = array(p, "scenes");
          LaunchDescription d;
          d.expected_pipeline = expected_pipeline(p);
          d.program = b.at("pipeline").resource<ProgramAccess>();
          d.payload = b.at("payload").resource<NativeHostBlobResource>();
          const auto &dims = array(p, "dimensions");
          if (dims.size() != 3)
            throw std::invalid_argument(
                "Ray dimensions require three uint32 values");
          for (size_t i = 0; i < 3; ++i)
            d.dimensions[i] = u32(dims[i]);
          d.raygen = record(p["raygen"], *d.payload);
          for (const auto &v : array(p, "miss"))
            d.miss.push_back(record(v, *d.payload));
          for (const auto &v : array(p, "hit"))
            d.hit.push_back(record(v, *d.payload));
          if (p.obj.inner.count("push_constants")) {
            fields(p["push_constants"], {"offset", "size"});
            d.push = Section(p["push_constants"]).read(*d.payload);
          }
          for (size_t i = 0; i < buffers.size(); ++i) {
            const auto &v = buffers[i];
            fields(v, {"set", "binding", "kind", "access"});
            const auto kind = name(v["kind"]);
            const auto a = access(v);
            if (kind != "storage_buffer" && kind != "uniform_buffer")
              throw std::invalid_argument("Invalid ray buffer kind");
            if (kind == "uniform_buffer" && a.write)
              throw std::invalid_argument("Uniform ray buffers are read-only");
            d.buffers.push_back({point(v), kind == "uniform_buffer",
                                 b.at("buffer_" + std::to_string(i)).array()});
          }
          for (size_t i = 0; i < images.size(); ++i) {
            const auto &v = images[i];
            fields(v, {"set", "binding", "kind", "access", "mip_level"});
            const auto kind = name(v["kind"]);
            const auto a = access(v);
            if (kind != "sampled_image" && kind != "storage_image")
              throw std::invalid_argument("Invalid ray image kind");
            if (kind == "sampled_image" && a.write)
              throw std::invalid_argument("Sampled ray images are read-only");
            d.images.push_back({point(v), kind == "storage_image", a.read,
                                optional_u32(v, "mip_level", 0),
                                b.at("image_" + std::to_string(i))
                                    .resource<NativeSampledImage>()});
          }
          for (size_t i = 0; i < scenes.size(); ++i) {
            fields(scenes[i], {"set", "binding"});
            d.scenes.push_back(
                {point(scenes[i]),
                 b.at("scene_" + std::to_string(i))
                     .resource<NativeShaderAccelerationStructure>()});
          }
          return d;
        };
      },
      {}, {},
      {{"buffer_count", {"buffer"}},
       {"image_count", {"image"}},
       {"scene_count", {"scene"}}});
#ifdef TI_WITH_VULKAN
  registry.register_context_factory<PipelineDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &c, const PipelineDescription &d) {
        return std::make_shared<PreparedPipeline>(c, d);
      });
  registry.register_context_factory<LaunchDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &c, const LaunchDescription &d) {
        auto result = std::make_shared<PreparedLaunch>(c, d);
        result->initialize();
        return result;
      });
#endif
}
}  // namespace taichi::lang
