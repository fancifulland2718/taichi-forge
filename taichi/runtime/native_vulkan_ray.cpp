#include "taichi/runtime/native_operation_registry.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/vulkan_ray_types.h"
#ifdef TI_WITH_VULKAN
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/vulkan_ray_query.h"
#include "taichi/runtime/vulkan_ray_scene.h"
#include "taichi/runtime/vulkan_ray_tlas.h"
#endif

namespace taichi::lang {
namespace {
// Typed views share the NativeResource owner's control block. A TLAS is
// queryable but is not geometry; a BLAS is geometry but cannot be queried
// alone.
class QueryAccess {
 public:
  virtual ~QueryAccess() = default;
  virtual Device &device() const = 0;
#ifdef TI_WITH_VULKAN
  virtual void prepare_query_variant(unsigned variant) = 0;
  virtual void record_query(CommandList *, const VulkanRayQueryPacket &) = 0;
#endif
};
class GeometryAccess {
 public:
  virtual ~GeometryAccess() = default;
  virtual Device &device() const = 0;
  virtual size_t vertex_count() const = 0;
#ifdef TI_WITH_VULKAN
  virtual void record_refit(CommandList *, DevicePtr vertices) = 0;
#endif
};
class BlasAccess : public GeometryAccess {
 public:
  virtual size_t triangle_count() const = 0;
#ifdef TI_WITH_VULKAN
  virtual VulkanTriangleBlas &blas_core() = 0;
  virtual void record_build(CommandList *,
                            DevicePtr vertices,
                            DevicePtr indices) = 0;
#endif
};
class InstanceAccess {
 public:
  virtual ~InstanceAccess() = default;
  virtual Device &device() const = 0;
  virtual size_t instance_count() const = 0;
#ifdef TI_WITH_VULKAN
  virtual void prepare_transforms() = 0;
  virtual void record_transforms(CommandList *,
                                 const VulkanRayTransformPacket &) = 0;
  virtual void record_metadata(CommandList *,
                               const std::vector<VulkanRayInstanceInfo> &,
                               bool update) = 0;
#endif
};

struct SceneDescription {
  static constexpr const char *kind = "vulkan_ray_triangle_scene";
  static constexpr uint32_t version = 1;
  uint32_t vertex_count, triangle_count;
  NativeArrayBinding vertices, indices;
};
struct BlasDescription : SceneDescription {
  static constexpr const char *kind = "vulkan_ray_triangle_blas";
  bool opaque{true};
};
struct TlasDescription {
  static constexpr const char *kind = "vulkan_ray_instance_tlas";
  static constexpr uint32_t version = 1;
  std::vector<std::shared_ptr<BlasAccess>> blases;
  std::vector<VulkanRayInstanceInfo> instances;
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
template <typename Core>
class OwnedRayResource : public NativeResource {
 public:
  OwnedRayResource(NativePreparationContext &context,
                   std::unique_ptr<Core> core)
      : device_(ray_device(context)),
        runtime_(*context.gfx_runtime),
        fault_(context.fault_domain),
        core_(std::move(core)) {
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

 protected:
  Device &device_;
  gfx::GfxRuntime &runtime_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::unique_ptr<Core> core_;
  bool initialization_pending_{false};
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
  void initialize(const SceneDescription &d) {
    OwnedRayResource::initialize(
        [scene = core_.get(), vertices = d.vertices.pointer,
         indices = d.indices.pointer](CommandList *commands) {
          scene->record_build(commands, vertices, indices);
        });
  }
  Device &device() const override {
    return device_;
  }
  size_t vertex_count() const override {
    return core_->vertex_count();
  }
  bool retains_input(const std::string &port) const override {
    return port != "vertices" && port != "indices";
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
            std::make_unique<VulkanTriangleBlas>(&ray_device(context),
                                                 d.vertex_count,
                                                 d.triangle_count,
                                                 nullptr,
                                                 d.opaque)) {
  }
  void initialize(const BlasDescription &d) {
    OwnedRayResource::initialize(
        [blas = core_.get(), vertices = d.vertices.pointer,
         indices = d.indices.pointer](CommandList *commands) {
          blas->record_build(commands, vertices, indices, false);
        });
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
  bool retains_input(const std::string &port) const override {
    return port != "vertices" && port != "indices";
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
};

std::unique_ptr<VulkanInstanceTlas> make_tlas(NativePreparationContext &context,
                                              const TlasDescription &d) {
  auto &device = ray_device(context);
  std::vector<std::shared_ptr<VulkanTriangleBlas>> blases;
  for (const auto &blas : d.blases)
    // Retain the original owner, including its fault-domain retirement policy.
    blases.emplace_back(blas, &blas->blas_core());
  return std::make_unique<VulkanInstanceTlas>(
      &device, std::move(blases), std::vector<uint32_t>(d.blases.size(), 0));
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

class PreparedTlas final : public OwnedRayResource<VulkanInstanceTlas>,
                           public QueryAccess,
                           public InstanceAccess {
 public:
  PreparedTlas(NativePreparationContext &context, const TlasDescription &d)
      : OwnedRayResource(context, make_tlas(context, d)) {
  }
  void initialize(const TlasDescription &d, const VulkanRayTransformPacket &p) {
    core_->prepare_transforms();
    OwnedRayResource::initialize([tlas = core_.get(), instances = d.instances,
                                  p](CommandList *commands) {
      tlas->record_build_device(commands, instances, p);
    });
  }
  Device &device() const override {
    return device_;
  }
  size_t instance_count() const override {
    return core_->instance_count();
  }
  bool retains_input(const std::string &port) const override {
    return port != "transforms";
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

class PreparedQuery final : public NativeOperation {
 public:
  explicit PreparedQuery(const QueryDescription &d) : scene_(d.scene) {
    auto &device = dynamic_cast<vulkan::VulkanDevice &>(scene_->device());
    std::array<VulkanRayQueryPacket::Binding, 3> bindings{};
    const auto ray_bytes = array_bytes(device, d.rays, d.ray_count, 8, "f32");
    const auto hit_bytes = array_bytes(device, d.hits, d.ray_count, 4, "f32");
    bindings[0] = {d.rays.pointer, ray_bytes};
    bindings[1] = {d.hits.pointer, hit_bytes};
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

class PreparedRefit final : public NativeOperation {
 public:
  explicit PreparedRefit(const RefitDescription &d)
      : scene_(d.scene), vertices_(d.vertices.pointer) {
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

class PreparedInstanceRefit final : public NativeOperation {
 public:
  explicit PreparedInstanceRefit(const InstanceRefitDescription &d)
      : scene_(d.scene),
        packet_(transform_packet(device(),
                                 d.transforms,
                                 scene_->instance_count())) {
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

class PreparedBlasBuild final : public NativeOperation {
 public:
  explicit PreparedBlasBuild(const BlasBuildDescription &d)
      : blas_(d.blas),
        vertices_(d.vertices.pointer),
        indices_(d.indices.pointer) {
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

class PreparedInstanceUpdate final : public NativeOperation {
 public:
  explicit PreparedInstanceUpdate(const InstanceUpdateDescription &d)
      : description_(d) {
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
      },
      NativeObjectKind::resource);
  registry.register_schema<BlasDescription>(
      "vulkan_ray", "triangle_blas", {"vertices", "indices"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<BlasDescription> {
        operation_json::fields(p, {"vertex_count", "triangle_count", "opaque"});
        const auto vertices = count(p["vertex_count"]);
        const auto triangles = count(p["triangle_count"]);
        const auto opaque = operation_json::optional_bool(p, "opaque", true);
        return [vertices, triangles, opaque](const NativeOperationBindings &b) {
          BlasDescription d;
          d.vertex_count = vertices;
          d.triangle_count = triangles;
          d.vertices = b.at("vertices").array();
          d.indices = b.at("indices").array();
          d.opaque = opaque;
          return d;
        };
      });
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
        operation_json::fields(
            p, {"instance_count", "custom_indices", "visibility_masks"});
        const auto size = count(p["instance_count"]);
        const auto ids =
            instance_values(p, "custom_indices", size, 0xffffffu, 0);
        const auto masks =
            instance_values(p, "visibility_masks", size, 0xffu, 0xffu);
        return [size, ids, masks](const NativeOperationBindings &b) {
          TlasDescription d;
          d.transforms = b.at("transforms").array();
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
        result->initialize(d);
        return result;
      });
  registry.register_context_factory<TlasDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](NativePreparationContext &context, const TlasDescription &d) {
        const auto packet =
            transform_packet(context.device, d.transforms, d.blases.size());
        auto result = std::make_shared<PreparedTlas>(context, d);
        result->initialize(d, packet);
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
        result->initialize(d);
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
