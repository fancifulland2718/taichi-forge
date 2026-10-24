#include "taichi/runtime/native_operation_registry.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_acceleration_structure.h"
#include "taichi/runtime/native_vulkan_ray_access.h"
#include "taichi/runtime/native_host_blob.h"
#include "taichi/runtime/vulkan_ray_types.h"
#ifdef TI_WITH_VULKAN
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/vulkan_ray_query.h"
#include "taichi/runtime/vulkan_ray_scene.h"
#include "taichi/runtime/vulkan_ray_tlas.h"
#include "taichi/runtime/vulkan_micromap.h"
#endif

namespace taichi::lang {
#ifdef TI_WITH_VULKAN
void native_vulkan_ray::record_shader_read(
    VulkanInstanceTlas &tlas, CommandList *commands,
    NativeShaderStages shader_stages, bool graphics_queue, bool independent) {
  VkPipelineStageFlags stages = 0;
  for (const auto &[source, target] :
       {std::make_pair(PipelineStageType::compute,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
        std::make_pair(PipelineStageType::vertex,
                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT),
        std::make_pair(PipelineStageType::fragment,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT),
        std::make_pair(PipelineStageType::raytracing,
                       VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR)}) {
    const auto bit = native_shader_stage(source);
    if (shader_stages & bit) {
      stages |= target;
      shader_stages &= ~bit;
    }
  }
  if (!stages || shader_stages)
    throw std::invalid_argument("Unqualified acceleration structure shader stages");
  tlas.record_shader_read(
      commands, stages,
      independent ? VulkanRayShaderReadOrder::execution_domain
                  : graphics_queue ? VulkanRayShaderReadOrder::runtime_graphics
                                   : VulkanRayShaderReadOrder::compute);
}
#endif
namespace {
using namespace native_vulkan_ray;

struct SceneDescription {
  static constexpr const char *kind = "vulkan_ray_triangle_scene";
  static constexpr uint32_t version = 1;
  uint32_t vertex_count, triangle_count;
  NativeArrayBinding vertices, indices;
};
class MicromapAccess : public NativeResource {
 public:
  virtual size_t triangle_count() const = 0;
#ifdef TI_WITH_VULKAN
  virtual VulkanOpacityMicromap &micromap_core() = 0;
#endif
};
struct MicromapDescription {
  static constexpr const char *kind = "vulkan_ray_opacity_micromap";
  static constexpr uint32_t version = 1;
  uint32_t triangle_count;
  std::shared_ptr<NativeHostBlobResource> data, descriptors, indices;
};
struct BlasDescription : SceneDescription {
  static constexpr const char *kind = "vulkan_ray_triangle_blas";
  bool opaque{true};
  std::shared_ptr<MicromapAccess> micromap;
};
struct TlasDescription {
  static constexpr const char *kind = "vulkan_ray_instance_tlas";
  static constexpr uint32_t version = 1;
  std::vector<std::shared_ptr<BlasAccess>> blases;
  std::vector<VulkanRayInstanceInfo> instances;
  std::vector<uint32_t> sbt_record_offsets;
  NativeArrayBinding transforms;
};
struct QueryDescription {
  static constexpr const char *kind = "vulkan_ray_query";
  static constexpr uint32_t version = 1;
  std::shared_ptr<QueryAccess> scene;
  uint32_t ray_count;
  NativeArrayBinding rays, hits;
  std::optional<NativeArrayBinding> hit_indices;
};
struct RefitDescription {
  static constexpr const char *kind = "vulkan_ray_refit";
  static constexpr uint32_t version = 1;
  std::shared_ptr<GeometryAccess> scene;
  NativeArrayBinding vertices;
};
struct InstanceRefitDescription {
  static constexpr const char *kind = "vulkan_ray_instance_refit";
  static constexpr uint32_t version = 1;
  std::shared_ptr<InstanceAccess> scene;
  NativeArrayBinding transforms;
};
struct BlasBuildDescription {
  static constexpr const char *kind = "vulkan_ray_blas_build";
  static constexpr uint32_t version = 1;
  std::shared_ptr<BlasAccess> blas;
  NativeArrayBinding vertices, indices;
};
struct InstanceUpdateDescription {
  static constexpr const char *kind = "vulkan_ray_instance_update";
  static constexpr uint32_t version = 1;
  std::shared_ptr<InstanceAccess> scene;
  std::vector<VulkanRayInstanceInfo> instances;
  bool update{true};
};

uint32_t count(const liong::json::JsonValue &value) {
  const auto n = operation_json::integer(value);
  if (!n || n > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Vulkan ray count must fit positive uint32");
  return uint32_t(n);
}

std::vector<uint32_t> instance_values(const liong::json::JsonValue &p,
                                      const char *name,
                                      uint32_t size,
                                      uint32_t maximum,
                                      uint32_t fallback) {
  const auto found = p.obj.inner.find(name);
  if (found == p.obj.inner.end())
    return std::vector<uint32_t>(size, fallback);
  if (!found->second.is_arr() || found->second.arr.inner.size() != size)
    throw std::invalid_argument(
        "Vulkan instance metadata must match instance_count");
  std::vector<uint32_t> result;
  result.reserve(size);
  for (const auto &value : found->second.arr.inner) {
    const auto n = operation_json::integer(value);
    if (n > maximum)
      throw std::invalid_argument(
          "Vulkan instance metadata exceeds its packed bit width");
    result.push_back(uint32_t(n));
  }
  return result;
}

std::vector<VulkanRayInstanceInfo> instance_records(
    const liong::json::JsonValue &p) {
  const auto &transforms = p["transforms"];
  if (!transforms.is_arr() || transforms.arr.inner.empty() ||
      transforms.arr.inner.size() > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument(
        "Vulkan captured transforms require a nonempty uint32-sized array");
  const auto size = uint32_t(transforms.arr.inner.size());
  const auto ids = instance_values(p, "custom_indices", size, 0xffffffu, 0);
  const auto masks = instance_values(p, "visibility_masks", size, 0xffu, 0xffu);
  std::vector<VulkanRayInstanceInfo> instances(size);
  for (uint32_t i = 0; i < size; ++i) {
    const auto &row = transforms.arr.inner[i];
    if (!row.is_arr() || row.arr.inner.size() != 12)
      throw std::invalid_argument(
          "Vulkan captured transforms require 12 scalars per instance");
    for (size_t j = 0; j < 12; ++j) {
      const auto &value = row.arr.inner[j];
      if (value.ty != liong::json::L_JSON_FLOAT &&
          value.ty != liong::json::L_JSON_INT)
        throw std::invalid_argument(
            "Vulkan captured transform scalars must be finite f32 numbers");
      const double number = value.ty == liong::json::L_JSON_FLOAT
                                ? value.num_float
                                : double(value.num_int);
      if (!std::isfinite(number) ||
          std::abs(number) > (std::numeric_limits<float>::max)())
        throw std::invalid_argument(
            "Vulkan captured transform scalars must be finite f32 numbers");
      instances[i].transform[j] = float(number);
    }
    instances[i].custom_index = ids[i];
    instances[i].mask = masks[i];
  }
  return instances;
}

size_t array_bytes(Device &device,
                   const NativeArrayBinding &array,
                   uint32_t count,
                   uint32_t width,
                   const char *dtype) {
  if (array.pointer.device != &device)
    throw std::invalid_argument("Vulkan ray array belongs to another device");
  NativeArrayLayoutRequirement required{{count}, {}, 4, 4, true, dtype};
  if (array.element_shape.empty()) {
    required.minimum_shape.push_back(width);
  } else {
    required.element_shape = {width};
    required.record_bytes *= width;
  }
  validate_native_array_layout(array, required);
  if (count > std::numeric_limits<size_t>::max() / (width * 4))
    throw std::overflow_error("Vulkan ray array byte size overflows");
  const auto bytes = size_t(count) * width * 4;
  if (array.bytes < bytes)
    throw std::invalid_argument("Vulkan ray array range is too small");
  return bytes;
}

size_t transform_bytes(Device &device,
                       const NativeArrayBinding &array,
                       size_t count) {
  if (array.pointer.device != &device)
    throw std::invalid_argument("Vulkan transforms belong to another device");
  NativeArrayLayoutRequirement required{{count}, {}, 4, 4, true, "f32"};
  if (array.element_shape.empty()) {
    if (array.shape.size() == 2)
      required.minimum_shape.push_back(12);
    else
      required.minimum_shape.insert(required.minimum_shape.end(), {3, 4});
  } else {
    required.element_shape = {3, 4};
    required.record_bytes = 48;
  }
  validate_native_array_layout(array, required);
  if (count > std::numeric_limits<size_t>::max() / 48)
    throw std::overflow_error("Vulkan transform byte size overflows");
  const auto bytes = count * 48;
  if (array.bytes < bytes)
    throw std::invalid_argument("Vulkan transform range is too small");
  return bytes;
}

template <typename Access>
void validate_scene(Device &device, const std::shared_ptr<Access> &scene) {
  if (!scene || &scene->device() != &device)
    throw std::invalid_argument("Vulkan ray scene belongs to another device");
}

#ifdef TI_WITH_VULKAN
vulkan::VulkanDevice &ray_device(NativePreparationContext &context) {
  if (!context.gfx_runtime || !context.fault_domain)
    throw std::invalid_argument("Vulkan ray resource requires its GFX owner");
  return dynamic_cast<vulkan::VulkanDevice &>(context.device);
}

// All cores have the same ordered initialization and fault-safe retirement.
// Native registry leases retain public input resources through dependent work.
template <typename Core, typename Owner = NativeResource>
class OwnedRayResource : public Owner, public virtual NativeRayResourceAccess {
 public:
  OwnedRayResource(NativePreparationContext &context,
                   std::unique_ptr<Core> core)
      : device_(ray_device(context)),
        runtime_(*context.gfx_runtime),
        fault_(context.fault_domain),
        registration_(context.defer_vulkan_initialization
                          ? nullptr : runtime_.register_ray_resource(fault_)),
        core_(std::move(core)),
        deferred_(bool(context.defer_vulkan_initialization)) {
  }
  template <typename Record>
  void initialize(Record record) {
    initialization_pending_ = true;
    runtime_.enqueue_compute_op_lambda(
        [record = std::move(record)](Device *, CommandList *commands) {
          record(commands);
        },
        {});
    runtime_.synchronize();
    initialization_pending_ = false;
  }
  ~OwnedRayResource() override {
    // Registry retirement has already retired recorded uses. Failed cold
    // construction can still have initialization in flight on the same queue.
    if (!fault_->backend_calls_safe() ||
        fault_->state() == RuntimeLifecycleState::kFinalized) {
      core_.release();
      return;
    }
    if (initialization_pending_) {
      try {
        runtime_.synchronize();
      } catch (...) {
        core_.release();
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
    return core_->memory_statistics().known_requested_bytes;
  }
  void require_initialized(const NativeRecordingState &state) const override {
    if (deferred_ && !state.initialized.count(this))
      throw std::invalid_argument(
          "Vulkan ray resource requires its recorded initialization phase");
  }
  const std::vector<std::shared_ptr<NativeResource>> &dependencies() const override {
    return dependencies_;
  }
  std::shared_ptr<NativeRayResourceRegistration> ray_registration() const override {
    return registration_;
  }

 protected:
  Device &device_;
  gfx::GfxRuntime &runtime_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::shared_ptr<NativeRayResourceRegistration> registration_;
  std::unique_ptr<Core> core_;
  bool initialization_pending_{false};
  const bool deferred_;
  std::vector<std::shared_ptr<NativeResource>> dependencies_;
};

class PreparedMicromap final
    : public OwnedRayResource<VulkanOpacityMicromap, MicromapAccess> {
 public:
  PreparedMicromap(NativePreparationContext &context,
                   const MicromapDescription &d)
      : OwnedRayResource(
            context,
            std::make_unique<VulkanOpacityMicromap>(&ray_device(context))),
        triangle_count_(d.triangle_count) {
  }
  void initialize(const MicromapDescription &d) {
    // Drain the owner's existing commands before the core's cold synchronized
    // import. Never synchronize recursively inside an enqueue callback.
    runtime_.synchronize();
    initialization_pending_ = true;
    auto view = [](const std::shared_ptr<NativeHostBlobResource> &blob) {
      if (!blob || blob->bytes().empty())
        return std::string_view{};
      const auto &bytes = blob->bytes();
      return std::string_view(reinterpret_cast<const char *>(bytes.data()),
                              bytes.size());
    };
    core_->build(view(d.data), view(d.descriptors), view(d.indices),
                 bool(d.indices), d.triangle_count);
    initialization_pending_ = false;
  }
  size_t triangle_count() const override {
    return triangle_count_;
  }
  bool retains_input(const std::string &) const override {
    return false;
  }
  VulkanOpacityMicromap &micromap_core() override {
    return *core_;
  }

 private:
  size_t triangle_count_;
};

class PreparedScene final : public OwnedRayResource<VulkanTriangleRayScene>,
                            public QueryAccess,
                            public GeometryAccess {
 public:
  PreparedScene(NativePreparationContext &context, const SceneDescription &d)
      : OwnedRayResource(
            context,
            std::make_unique<VulkanTriangleRayScene>(&ray_device(context),
                                                     d.vertex_count,
                                                     d.triangle_count)) {
  }
  std::function<void(CommandList *)> initialization_commands(const SceneDescription &d) {
    return [scene = core_.get(), vertices = d.vertices.pointer,
            indices = d.indices.pointer](CommandList *commands) {
      scene->record_build(commands, vertices, indices);
    };
  }
  Device &device() const override {
    return device_;
  }
  size_t vertex_count() const override {
    return core_->vertex_count();
  }
  bool retains_input(const std::string &port) const override {
    return deferred_ || (port != "vertices" && port != "indices");
  }
  void prepare_query_variant(unsigned variant) override {
    core_->prepare_query_variant(variant);
  }
  void record_query(CommandList *commands,
                    const VulkanRayQueryPacket &p) override {
    core_->record_query(commands, p);
  }
  void record_refit(CommandList *commands, DevicePtr vertices) override {
    core_->record_refit(commands, vertices);
  }
};

class PreparedBlas final : public OwnedRayResource<VulkanTriangleBlas>,
                           public BlasAccess {
 public:
  PreparedBlas(NativePreparationContext &context, const BlasDescription &d)
      : OwnedRayResource(
            context,
            std::make_unique<VulkanTriangleBlas>(
                &ray_device(context),
                d.vertex_count,
                d.triangle_count,
                d.micromap ? std::shared_ptr<VulkanOpacityMicromap>(
                                 d.micromap,
                                 &d.micromap->micromap_core())
                           : nullptr,
                d.opaque)),
        micromap_bytes_(d.micromap ? d.micromap->workspace_bytes() : 0) {
    if (d.micromap)
      dependencies_.push_back(d.micromap);
  }
  std::function<void(CommandList *)> initialization_commands(const BlasDescription &d) {
    return [blas = core_.get(), vertices = d.vertices.pointer,
            indices = d.indices.pointer](CommandList *commands) {
      blas->record_build(commands, vertices, indices, false);
    };
  }
  Device &device() const override {
    return device_;
  }
  size_t vertex_count() const override {
    return core_->vertex_count();
  }
  size_t triangle_count() const override {
    return core_->triangle_count();
  }
  size_t workspace_bytes() const override {
    // The dependency owner accounts for shared OMM storage exactly once.
    return OwnedRayResource::workspace_bytes() - micromap_bytes_;
  }
  bool retains_input(const std::string &port) const override {
    return deferred_ || (port != "vertices" && port != "indices");
  }
  VulkanTriangleBlas &blas_core() override {
    return *core_;
  }
  void record_build(CommandList *commands,
                    DevicePtr vertices,
                    DevicePtr indices) override {
    core_->record_build(commands, vertices, indices, false);
  }
  void record_refit(CommandList *commands, DevicePtr vertices) override {
    core_->record_build(commands, vertices, {}, true);
  }

 private:
  size_t micromap_bytes_;
};

std::unique_ptr<VulkanInstanceTlas> make_tlas(NativePreparationContext &context,
                                              const TlasDescription &d) {
  auto &device = ray_device(context);
  std::vector<std::shared_ptr<VulkanTriangleBlas>> blases;
  for (const auto &blas : d.blases)
    // Retain the original owner, including its fault-domain retirement policy.
    blases.emplace_back(blas, &blas->blas_core());
  return std::make_unique<VulkanInstanceTlas>(
      &device, std::move(blases), d.sbt_record_offsets);
}

VulkanRayTransformPacket transform_packet(Device &device,
                                          const NativeArrayBinding &array,
                                          size_t count) {
  const auto bytes = transform_bytes(device, array, count);
  resolve_native_storage_range(Arch::vulkan, array.pointer, bytes);
  return prepare_vulkan_ray_transform_packet(
      &dynamic_cast<vulkan::VulkanDevice &>(device), count,
      {array.pointer, bytes});
}

class PreparedTlas final
    : public OwnedRayResource<VulkanInstanceTlas,
                              NativeShaderAccelerationStructure>,
      public QueryAccess,
      public InstanceAccess {
 public:
  PreparedTlas(NativePreparationContext &context, const TlasDescription &d)
      : OwnedRayResource(context, make_tlas(context, d)) {
    for (const auto &blas : d.blases)
      dependencies_.push_back(std::dynamic_pointer_cast<NativeResource>(blas));
  }
  std::function<void(CommandList *)> initialization_commands(
      const TlasDescription &d, const VulkanRayTransformPacket &p) {
    core_->prepare_transforms();
    return [tlas = core_.get(), instances = d.instances, p](CommandList *commands) {
      tlas->record_build_device(commands, instances, p);
    };
  }
  Device &device() const override {
    return device_;
  }
  size_t instance_count() const override {
    return core_->instance_count();
  }
  const std::vector<std::shared_ptr<NativeResource>> &instance_blases()
      const override {
    return dependencies_;
  }
  bool retains_input(const std::string &port) const override {
    return deferred_ || port != "transforms";
  }
  void prepare_query_variant(unsigned variant) override {
    core_->prepare_query_variant(variant);
  }
  void record_query(CommandList *commands,
                    const VulkanRayQueryPacket &p) override {
    core_->record_query(commands, p);
  }
  void prepare_transforms() override {
    core_->prepare_transforms();
  }
  void bind_for_kernel(ShaderResourceSet *bindings,
                       int binding,
                       CommandList *commands) override {
    core_->bind_for_kernel(bindings, binding, commands);
  }
  void bind_for_shader(ShaderResourceSet *bindings, int binding) override {
    core_->bind_for_shader(bindings, binding);
  }
  void record_shader_read(CommandList *commands,
                          NativeShaderStages shader_stages,
                          bool graphics_queue,
                          bool independent) override {
    native_vulkan_ray::record_shader_read(*core_, commands, shader_stages,
                                          graphics_queue, independent);
  }
  std::uint32_t max_sbt_record_offset() const override {
    return core_->max_sbt_record_offset();
  }
  void record_transforms(CommandList *commands,
                         const VulkanRayTransformPacket &p) override {
    core_->record_transforms(commands, p);
  }
  void record_metadata(CommandList *commands,
                       const std::vector<VulkanRayInstanceInfo> &instances,
                       bool update) override {
    core_->record_build(commands, instances, update);
  }
};

// The same typed owner closure validates ordinary recording initialization and
// independent-domain ownership. Traversal is cold; commands retain these owners
// directly and never consult the registry during submission.
class RayOperation : public NativeOperation {
 public:
  explicit RayOperation(std::shared_ptr<NativeResource> resource,
                        bool writes_root = false, bool initializing = false)
      : resource_(std::move(resource)),
        initializing_(initializing),
        resources_(collect_native_ray_uses(resource_, writes_root)) {
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return resource_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void validate_recording(NativeRecordingState &state) const override {
    for (const auto &use : resources_)
      if (!initializing_ || use.resource != resource_)
        dynamic_cast<const NativeRayResourceAccess &>(*use.resource).require_initialized(state);
    if (initializing_)
      state.initialized.insert(resource_.get());
  }
  NativeVulkanRecordingRequirements vulkan_recording_requirements() const override {
    NativeVulkanRecordingRequirements result;
    result.buffers = buffers_;
    result.image_uses.emplace();
    result.stateful_resources = resources_;
    return result;
  }

 protected:
  std::vector<DeviceAllocation> buffers_;

 private:
  std::shared_ptr<NativeResource> resource_;
  bool initializing_;
  std::vector<NativeResourceUse> resources_;
};

class RayInitialization final : public RayOperation {
 public:
  RayInitialization(std::shared_ptr<NativeResource> owner,
                    std::function<void(CommandList *)> record,
                    std::vector<DeviceAllocation> buffers)
      : RayOperation(std::move(owner), true, true), record_(std::move(record)) {
    buffers_ = std::move(buffers);
  }
  void record_vulkan(Device *, CommandList *commands) override {
    record_(commands);
  }

 private:
  std::function<void(CommandList *)> record_;
};

template <typename Resource>
void initialize_ray_resource(NativePreparationContext &context,
                             const std::shared_ptr<Resource> &resource,
                             std::function<void(CommandList *)> record,
                             std::vector<DeviceAllocation> buffers) {
  if (context.defer_vulkan_initialization)
    context.defer_vulkan_initialization(std::make_shared<RayInitialization>(
        resource, std::move(record), std::move(buffers)));
  else {
    auto access = context.gfx_runtime->ordinary_ray_scope();
    context.gfx_runtime->require_ray_uses(collect_native_ray_uses(resource, true));
    resource->initialize(std::move(record));
  }
}

class PreparedQuery final : public RayOperation {
 public:
  explicit PreparedQuery(const QueryDescription &d)
      : RayOperation(std::dynamic_pointer_cast<NativeResource>(d.scene)),
        scene_(d.scene) {
    auto &device = dynamic_cast<vulkan::VulkanDevice &>(scene_->device());
    std::array<VulkanRayQueryPacket::Binding, 3> bindings{};
    const auto ray_bytes = array_bytes(device, d.rays, d.ray_count, 8, "f32");
    const auto hit_bytes = array_bytes(device, d.hits, d.ray_count, 4, "f32");
    bindings[0] = {d.rays.pointer, ray_bytes};
    bindings[1] = {d.hits.pointer, hit_bytes};
    buffers_ = {d.rays.pointer, d.hits.pointer};
    std::vector<NativeStorageRange> reads{
        resolve_native_storage_range(Arch::vulkan, d.rays.pointer, ray_bytes)};
    std::vector<NativeStorageWrite> writes{
        {resolve_native_storage_range(Arch::vulkan, d.hits.pointer, hit_bytes),
         std::nullopt}};
    if (d.hit_indices) {
      const auto &indices = *d.hit_indices;
      const auto bytes = array_bytes(
          device, indices, d.ray_count, 4,
          indices.scalar_type == PrimitiveType::u32 ? "u32" : "i32");
      bindings[2] = {indices.pointer, bytes};
      buffers_.push_back(indices.pointer);
      writes.push_back(
          {resolve_native_storage_range(Arch::vulkan, indices.pointer, bytes),
           std::nullopt});
    }
    // Imported buffers can alias even when their DeviceAllocation IDs differ.
    validate_native_disjoint_writes(reads, writes);
    packet_ = prepare_vulkan_ray_query_packet(&device, d.ray_count,
                                              bool(d.hit_indices), bindings);
    scene_->prepare_query_variant(packet_.variant);
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return scene_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void record_vulkan(Device *, CommandList *commands) override {
    scene_->record_query(commands, packet_);
  }

 private:
  std::shared_ptr<QueryAccess> scene_;
  VulkanRayQueryPacket packet_;
};

class PreparedRefit final : public RayOperation {
 public:
  explicit PreparedRefit(const RefitDescription &d)
      : RayOperation(std::dynamic_pointer_cast<NativeResource>(d.scene), true),
        scene_(d.scene), vertices_(d.vertices.pointer) {
    buffers_ = {vertices_};
    const auto bytes =
        array_bytes(device(), d.vertices, scene_->vertex_count(), 3, "f32");
    resolve_native_storage_range(Arch::vulkan, vertices_, bytes);
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return scene_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void record_vulkan(Device *, CommandList *commands) override {
    scene_->record_refit(commands, vertices_);
  }

 private:
  std::shared_ptr<GeometryAccess> scene_;
  DevicePtr vertices_;
};

class PreparedInstanceRefit final : public RayOperation {
 public:
  explicit PreparedInstanceRefit(const InstanceRefitDescription &d)
      : RayOperation(std::dynamic_pointer_cast<NativeResource>(d.scene), true),
        scene_(d.scene),
        packet_(transform_packet(device(),
                                 d.transforms,
                                 scene_->instance_count())) {
    buffers_ = {d.transforms.pointer};
    scene_->prepare_transforms();
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return scene_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void record_vulkan(Device *, CommandList *commands) override {
    scene_->record_transforms(commands, packet_);
  }

 private:
  std::shared_ptr<InstanceAccess> scene_;
  VulkanRayTransformPacket packet_;
};

class PreparedBlasBuild final : public RayOperation {
 public:
  explicit PreparedBlasBuild(const BlasBuildDescription &d)
      : RayOperation(std::dynamic_pointer_cast<NativeResource>(d.blas), true),
        blas_(d.blas),
        vertices_(d.vertices.pointer),
        indices_(d.indices.pointer) {
    buffers_ = {vertices_, indices_};
    resolve_native_storage_range(
        Arch::vulkan, vertices_,
        array_bytes(device(), d.vertices, blas_->vertex_count(), 3, "f32"));
    resolve_native_storage_range(
        Arch::vulkan, indices_,
        array_bytes(device(), d.indices, blas_->triangle_count(), 3, "i32"));
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return blas_->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void record_vulkan(Device *, CommandList *commands) override {
    blas_->record_build(commands, vertices_, indices_);
  }

 private:
  std::shared_ptr<BlasAccess> blas_;
  DevicePtr vertices_, indices_;
};

class PreparedInstanceUpdate final : public RayOperation {
 public:
  explicit PreparedInstanceUpdate(const InstanceUpdateDescription &d)
      : RayOperation(std::dynamic_pointer_cast<NativeResource>(d.scene), true),
        description_(d) {
  }
  Arch arch() const override {
    return Arch::vulkan;
  }
  Device &device() const override {
    return description_.scene->device();
  }
  size_t workspace_bytes() const override {
    return 0;
  }
  void record_vulkan(Device *, CommandList *commands) override {
    description_.scene->record_metadata(commands, description_.instances,
                                        description_.update);
  }

 private:
  InstanceUpdateDescription description_;
};
#endif
}  // namespace

void register_vulkan_ray_operations(NativeOperationRegistry &registry) {
  registry.define<MicromapDescription>(
      [](Device &device, const MicromapDescription &d) {
        for (const auto &blob : {d.data, d.descriptors, d.indices}) {
          if (blob &&
              (&blob->device() != &device || blob->arch() != Arch::vulkan))
            throw std::invalid_argument(
                "Vulkan micromap blob belongs to another device");
        }
        if (!d.data || !d.descriptors)
          throw std::invalid_argument(
              "Vulkan micromap requires baked data and descriptors");
      },
      NativeObjectKind::resource);
  registry.register_schema<MicromapDescription>(
      "vulkan_ray", "opacity_micromap", {"data", "descriptors"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<MicromapDescription> {
        operation_json::fields(p, {"triangle_count"});
        const auto triangles = count(p["triangle_count"]);
        return [triangles](const NativeOperationBindings &b) {
          return MicromapDescription{
              triangles, b.at("data").resource<NativeHostBlobResource>(),
              b.at("descriptors").resource<NativeHostBlobResource>(),
              b.count("indices")
                  ? b.at("indices").resource<NativeHostBlobResource>()
                  : nullptr};
        };
      },
      {}, {"indices"});
  registry.define<BlasBuildDescription>(
      [](Device &device, const BlasBuildDescription &d) {
        validate_scene(device, d.blas);
        array_bytes(device, d.vertices, d.blas->vertex_count(), 3, "f32");
        array_bytes(device, d.indices, d.blas->triangle_count(), 3, "i32");
      });
  registry.register_schema<BlasBuildDescription>(
      "vulkan_ray", "blas_build", {"scene", "vertices", "indices"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<BlasBuildDescription> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          return BlasBuildDescription{b.at("scene").resource<BlasAccess>(),
                                      b.at("vertices").array(),
                                      b.at("indices").array()};
        };
      });
  registry.define<InstanceUpdateDescription>([](Device &device,
                                                const InstanceUpdateDescription
                                                    &d) {
    validate_scene(device, d.scene);
    if (d.instances.size() != d.scene->instance_count())
      throw std::invalid_argument(
          "Vulkan captured instances must preserve the TLAS instance count");
  });
  registry.register_schema<InstanceUpdateDescription>(
      "vulkan_ray", "instance_update", {"scene"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<InstanceUpdateDescription> {
        operation_json::fields(
            p, {"transforms", "custom_indices", "visibility_masks", "update"});
        const auto instances = instance_records(p);
        const auto update = operation_json::optional_bool(p, "update", true);
        return [instances, update](const NativeOperationBindings &b) {
          return InstanceUpdateDescription{
              b.at("scene").resource<InstanceAccess>(), instances, update};
        };
      });
  registry.define<BlasDescription>(
      [](Device &device, const BlasDescription &d) {
        array_bytes(device, d.vertices, d.vertex_count, 3, "f32");
        array_bytes(device, d.indices, d.triangle_count, 3, "i32");
        if (d.micromap) {
          validate_scene(device, d.micromap);
          if (d.micromap->triangle_count() != d.triangle_count)
            throw std::invalid_argument(
                "Vulkan micromap triangle count must match its BLAS");
          if (d.opaque)
            throw std::invalid_argument(
                "Vulkan micromap BLAS cannot force opaque geometry");
        }
      },
      NativeObjectKind::resource);
  registry.register_schema<BlasDescription>(
      "vulkan_ray", "triangle_blas", {"vertices", "indices"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<BlasDescription> {
        operation_json::fields(p, {"vertex_count", "triangle_count", "opaque"});
        const auto vertices = count(p["vertex_count"]);
        const auto triangles = count(p["triangle_count"]);
        const auto opaque =
            p.obj.inner.count("opaque")
                ? std::optional<bool>(
                      operation_json::optional_bool(p, "opaque"))
                : std::nullopt;
        return [vertices, triangles, opaque](const NativeOperationBindings &b) {
          BlasDescription d;
          d.vertex_count = vertices;
          d.triangle_count = triangles;
          d.vertices = b.at("vertices").array();
          d.indices = b.at("indices").array();
          if (b.count("opacity_micromap"))
            d.micromap = b.at("opacity_micromap").resource<MicromapAccess>();
          d.opaque = opaque.value_or(!d.micromap);
          return d;
        };
      },
      {}, {"opacity_micromap"});
  registry.define<TlasDescription>(
      [](Device &device, const TlasDescription &d) {
        for (const auto &blas : d.blases)
          validate_scene(device, blas);
        transform_bytes(device, d.transforms, d.blases.size());
      },
      NativeObjectKind::resource);
  registry.register_schema<TlasDescription>(
      "vulkan_ray", "instance_tlas", {"transforms"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<TlasDescription> {
        operation_json::fields(p, {"instance_count", "custom_indices",
                                   "visibility_masks", "sbt_record_offsets"});
        const auto size = count(p["instance_count"]);
        const auto ids =
            instance_values(p, "custom_indices", size, 0xffffffu, 0);
        const auto masks =
            instance_values(p, "visibility_masks", size, 0xffu, 0xffu);
        const auto sbt =
            instance_values(p, "sbt_record_offsets", size, 0xffffffu, 0);
        return [size, ids, masks, sbt](const NativeOperationBindings &b) {
          TlasDescription d;
          d.transforms = b.at("transforms").array();
          d.sbt_record_offsets = sbt;
          d.instances.resize(size);
          d.blases.reserve(size);
          for (uint32_t i = 0; i < size; ++i) {
            d.blases.push_back(
                b.at("blas_" + std::to_string(i)).resource<BlasAccess>());
            d.instances[i].custom_index = ids[i];
            d.instances[i].mask = masks[i];
          }
          return d;
        };
      },
      {}, {}, {{"instance_count", {"blas"}}});
  registry.define<SceneDescription>(
      [](Device &device, const SceneDescription &d) {
        array_bytes(device, d.vertices, d.vertex_count, 3, "f32");
        array_bytes(device, d.indices, d.triangle_count, 3, "i32");
      },
      NativeObjectKind::resource);
  registry.register_schema<SceneDescription>(
      "vulkan_ray", "triangle_scene", {"vertices", "indices"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<SceneDescription> {
        operation_json::fields(p, {"vertex_count", "triangle_count"});
        const auto vertices = count(p["vertex_count"]);
        const auto triangles = count(p["triangle_count"]);
        return [vertices, triangles](const NativeOperationBindings &b) {
          return SceneDescription{vertices, triangles, b.at("vertices").array(),
                                  b.at("indices").array()};
        };
      });
  registry.define<QueryDescription>(
      [](Device &device, const QueryDescription &d) {
        validate_scene(device, d.scene);
        array_bytes(device, d.rays, d.ray_count, 8, "f32");
        array_bytes(device, d.hits, d.ray_count, 4, "f32");
        if (d.hit_indices)
          array_bytes(
              device, *d.hit_indices, d.ray_count, 4,
              d.hit_indices->scalar_type == PrimitiveType::u32 ? "u32" : "i32");
      });
  registry.register_schema<QueryDescription>(
      "vulkan_ray", "query", {"scene", "rays", "hits"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<QueryDescription> {
        operation_json::fields(p, {"ray_count"});
        const auto rays = count(p["ray_count"]);
        return [rays](const NativeOperationBindings &b) {
          QueryDescription d{b.at("scene").resource<QueryAccess>(), rays,
                             b.at("rays").array(), b.at("hits").array(),
                             std::nullopt};
          if (b.count("hit_indices"))
            d.hit_indices = b.at("hit_indices").array();
          return d;
        };
      },
      {}, {"hit_indices"});
  registry.define<RefitDescription>(
      [](Device &device, const RefitDescription &d) {
        validate_scene(device, d.scene);
        array_bytes(device, d.vertices, d.scene->vertex_count(), 3, "f32");
      });
  registry.register_schema<RefitDescription>(
      "vulkan_ray", "refit", {"scene", "vertices"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<RefitDescription> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          return RefitDescription{b.at("scene").resource<GeometryAccess>(),
                                  b.at("vertices").array()};
        };
      });
  registry.define<InstanceRefitDescription>(
      [](Device &device, const InstanceRefitDescription &d) {
        validate_scene(device, d.scene);
        transform_bytes(device, d.transforms, d.scene->instance_count());
      });
  registry.register_schema<InstanceRefitDescription>(
      "vulkan_ray", "instance_refit", {"scene", "transforms"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<InstanceRefitDescription> {
        operation_json::fields(p, {});
        return [](const NativeOperationBindings &b) {
          return InstanceRefitDescription{
              b.at("scene").resource<InstanceAccess>(),
              b.at("transforms").array()};
        };
      });
#ifdef TI_WITH_VULKAN
  registry.register_context_factory<MicromapDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &context, const MicromapDescription &d) {
        if (context.defer_vulkan_initialization)
          throw std::invalid_argument(
              "Independent Vulkan micromap initialization is not qualified");
        auto result = std::make_shared<PreparedMicromap>(context, d);
        result->initialize(d);
        return result;
      });
  registry.register_context_factory<BlasBuildDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &, const BlasBuildDescription &d) {
        return std::make_shared<PreparedBlasBuild>(d);
      });
  registry.register_context_factory<InstanceUpdateDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &, const InstanceUpdateDescription &d) {
        return std::make_shared<PreparedInstanceUpdate>(d);
      });
  registry.register_context_factory<BlasDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &context, const BlasDescription &d) {
        for (const auto &entry : {std::make_pair(&d.vertices, d.vertex_count),
                                  std::make_pair(&d.indices, d.triangle_count)})
          resolve_native_storage_range(Arch::vulkan, entry.first->pointer,
                                       size_t(entry.second) * 12);
        auto result = std::make_shared<PreparedBlas>(context, d);
        initialize_ray_resource(context, result, result->initialization_commands(d),
                                 {d.vertices.pointer, d.indices.pointer});
        return result;
      });
  registry.register_context_factory<TlasDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &context, const TlasDescription &d) {
        const auto packet =
            transform_packet(context.device, d.transforms, d.blases.size());
        auto result = std::make_shared<PreparedTlas>(context, d);
        initialize_ray_resource(context, result, result->initialization_commands(d, packet),
                                 {d.transforms.pointer});
        return result;
      });
  registry.register_context_factory<InstanceRefitDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &, const InstanceRefitDescription &d) {
        return std::make_shared<PreparedInstanceRefit>(d);
      });
  registry.register_context_factory<SceneDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &context, const SceneDescription &d) {
        if (!context.gfx_runtime || !context.fault_domain)
          throw std::invalid_argument(
              "Vulkan ray scene requires its GFX owner");
        for (const auto &entry : {std::make_pair(&d.vertices, d.vertex_count),
                                  std::make_pair(&d.indices, d.triangle_count)})
          resolve_native_storage_range(Arch::vulkan, entry.first->pointer,
                                       size_t(entry.second) * 12);
        auto result = std::make_shared<PreparedScene>(context, d);
        initialize_ray_resource(context, result, result->initialization_commands(d),
                                 {d.vertices.pointer, d.indices.pointer});
        return result;
      });
  registry.register_context_factory<QueryDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &, const QueryDescription &d) {
        return std::make_shared<PreparedQuery>(d);
      });
  registry.register_context_factory<RefitDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &, const RefitDescription &d) {
        return std::make_shared<PreparedRefit>(d);
      });
#endif
}
}  // namespace taichi::lang
