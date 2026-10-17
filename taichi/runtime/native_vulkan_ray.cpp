#include "taichi/runtime/native_operation_registry.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#ifdef TI_WITH_VULKAN
#include "taichi/program/runtime_fault.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/runtime/vulkan_ray_query.h"
#include "taichi/runtime/vulkan_ray_scene.h"
#endif

namespace taichi::lang {
namespace {
class VulkanRaySceneResource : public NativeResource {
 public:
  explicit VulkanRaySceneResource(uint32_t vertices) : vertices_(vertices) {
  }
  uint32_t vertex_count() const {
    return vertices_;
  }
#ifdef TI_WITH_VULKAN
  virtual VulkanTriangleRayScene &core() = 0;
#endif
 private:
  uint32_t vertices_;
};

struct SceneDescription {
  static constexpr const char *kind = "vulkan_ray_triangle_scene";
  static constexpr uint32_t version = 1;
  uint32_t vertex_count, triangle_count;
  NativeArrayBinding vertices, indices;
};
struct QueryDescription {
  static constexpr const char *kind = "vulkan_ray_query";
  static constexpr uint32_t version = 1;
  std::shared_ptr<VulkanRaySceneResource> scene;
  uint32_t ray_count;
  NativeArrayBinding rays, hits;
  std::optional<NativeArrayBinding> hit_indices;
};
struct RefitDescription {
  static constexpr const char *kind = "vulkan_ray_refit";
  static constexpr uint32_t version = 1;
  std::shared_ptr<VulkanRaySceneResource> scene;
  NativeArrayBinding vertices;
};

uint32_t count(const liong::json::JsonValue &value) {
  const auto n = operation_json::integer(value);
  if (!n || n > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Vulkan ray count must fit positive uint32");
  return uint32_t(n);
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

void validate_scene(Device &device,
                    const std::shared_ptr<VulkanRaySceneResource> &scene) {
  if (!scene || scene->arch() != Arch::vulkan || &scene->device() != &device)
    throw std::invalid_argument("Vulkan ray scene belongs to another device");
}

#ifdef TI_WITH_VULKAN
class PreparedScene final : public VulkanRaySceneResource {
 public:
  PreparedScene(NativePreparationContext &context, const SceneDescription &d)
      : VulkanRaySceneResource(d.vertex_count),
        device_(context.device),
        runtime_(*context.gfx_runtime),
        fault_(context.fault_domain),
        core_(std::make_unique<VulkanTriangleRayScene>(
            &dynamic_cast<vulkan::VulkanDevice &>(device_),
            d.vertex_count,
            d.triangle_count)) {
  }
  void initialize(const SceneDescription &d) {
    auto *scene = core_.get();
    const auto vertices = d.vertices.pointer, indices = d.indices.pointer;
    initialization_pending_ = true;
    runtime_.enqueue_compute_op_lambda(
        [scene, vertices, indices](Device *, CommandList *commands) {
          scene->record_build(commands, vertices, indices);
        },
        {});
    runtime_.synchronize();
    initialization_pending_ = false;
  }
  ~PreparedScene() override {
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
  bool retains_input(const std::string &port) const override {
    return port != "vertices" && port != "indices";
  }
  VulkanTriangleRayScene &core() override {
    return *core_;
  }

 private:
  Device &device_;
  gfx::GfxRuntime &runtime_;
  std::shared_ptr<RuntimeFaultDomain> fault_;
  std::unique_ptr<VulkanTriangleRayScene> core_;
  bool initialization_pending_{false};
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
    scene_->core().prepare_query_variant(packet_.variant);
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
    scene_->core().record_query(commands, packet_);
  }

 private:
  std::shared_ptr<VulkanRaySceneResource> scene_;
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
    scene_->core().record_refit(commands, vertices_);
  }

 private:
  std::shared_ptr<VulkanRaySceneResource> scene_;
  DevicePtr vertices_;
};
#endif
}  // namespace

void register_vulkan_ray_operations(NativeOperationRegistry &registry) {
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
          QueryDescription d{b.at("scene").resource<VulkanRaySceneResource>(),
                             rays, b.at("rays").array(), b.at("hits").array(),
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
          return RefitDescription{
              b.at("scene").resource<VulkanRaySceneResource>(),
              b.at("vertices").array()};
        };
      });
#ifdef TI_WITH_VULKAN
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
