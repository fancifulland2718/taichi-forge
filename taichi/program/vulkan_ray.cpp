#include "taichi/program/program.h"
#include "taichi/program/storage_view.h"
#include "taichi/program/prepared_vulkan_ray_access.h"
#include "taichi/runtime/vulkan_ray_query.h"
#include "taichi/runtime/vulkan_ray_build.h"
#include "taichi/runtime/vulkan_ray_scene.h"
#include "taichi/runtime/vulkan_ray_blas.h"
#include "taichi/runtime/vulkan_ray_tlas.h"
#include "taichi/runtime/native_vulkan_ray_access.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_set>
#include <utility>

#if defined(TI_WITH_VULKAN)
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/runtime/vulkan_micromap.h"
#include "taichi/runtime/gfx/kernel_launcher.h"

namespace taichi::lang {
namespace {
using namespace vulkan_ray_detail;


void check_ray_storage(const storage::DenseStorageDescriptor &value,
                       std::size_t count,
                       unsigned width,
                       DataType dtype,
                       const char *name) {
  TI_ERROR_IF(count == 0 || count > (std::numeric_limits<std::uint32_t>::max)(),
              "Vulkan ray {} count must be in [1, UINT32_MAX].", name);
  const auto shape = value.index_shape();
  const auto element = value.element_shape();
  const bool scalar =
      element.empty() && shape == std::vector<std::int64_t>{
                                      static_cast<std::int64_t>(count), width};
  const bool vector =
      element == std::vector<std::int64_t>{width} &&
      shape == std::vector<std::int64_t>{static_cast<std::int64_t>(count)};
  TI_ERROR_IF(value.scalar_type() != dtype || (!scalar && !vector),
              "Vulkan ray {} requires dtype {} and {}-wide records.", name,
              dtype->to_string(), width);
}

}  // namespace

enum class VulkanRayResourceKind {
  kTriangleScene,
  kTriangleBlas,
  kInstanceTlas,
};

class VulkanRayResource {
 public:
  virtual ~VulkanRayResource() = default;
  virtual VulkanRayResourceKind kind() const = 0;
  virtual VulkanTriangleRaySceneMemoryStatistics memory_statistics() const = 0;

  void register_prepared_lease(
      const std::shared_ptr<PreparedResourceLease> &lease) {
    // The Program submission guard serializes preparation against close/reset.
    prepared_leases_.erase(
        std::remove_if(prepared_leases_.begin(), prepared_leases_.end(),
                       [](const auto &item) { return item.expired(); }),
        prepared_leases_.end());
    if (std::none_of(prepared_leases_.begin(), prepared_leases_.end(),
                     [&](const auto &item) { return item.lock() == lease; })) {
      prepared_leases_.push_back(lease);
    }
  }

  void invalidate_prepared_resources() noexcept {
    for (const auto &item : prepared_leases_) {
      if (auto lease = item.lock()) {
        lease->clear();
      }
    }
    prepared_leases_.clear();
  }

 private:
  std::vector<std::weak_ptr<PreparedResourceLease>> prepared_leases_;
};

// The Program retains the original physical cores and its existing prepared
// leases. A native view shares that owner, registration and dependency closure;
// it neither recreates storage nor changes Program retirement ordering.
template <class Base = NativeResource>
class VulkanRayNativeOwner : public VulkanRayResource,
                             public Base,
                             public virtual NativeRayResourceAccess {
 public:
  explicit VulkanRayNativeOwner(Program *program)
      : device_(*program->get_compute_device()),
        registration_(program->get_gfx_runtime_for_borrowing()
                          ->register_ray_resource(program->runtime_fault_domain())) {
  }
  Arch arch() const override { return Arch::vulkan; }
  Device &device() const override { return device_; }
  std::size_t workspace_bytes() const override {
    // This is borrowed Program storage; the Program still accounts for it.
    return 0;
  }
  void require_initialized(const NativeRecordingState &) const override {
    // Existing Program build ordering is captured by the exact producer tail.
  }
  const std::vector<std::shared_ptr<NativeResource>> &dependencies() const override {
    return dependencies_;
  }
  std::shared_ptr<NativeRayResourceRegistration> ray_registration() const override {
    return registration_;
  }

 protected:
  Device &device_;
  std::shared_ptr<NativeRayResourceRegistration> registration_;
  std::vector<std::shared_ptr<NativeResource>> dependencies_;
};

class VulkanTriangleSceneResource final : public VulkanRayNativeOwner<>,
                                         public native_vulkan_ray::QueryAccess,
                                         public native_vulkan_ray::GeometryAccess {
 public:
  VulkanTriangleSceneResource(Program *program, std::size_t vertices,
                               std::size_t triangles)
      : VulkanRayNativeOwner(program),
        core_(std::make_shared<VulkanTriangleRayScene>(
            static_cast<vulkan::VulkanDevice *>(&device_), vertices, triangles)) {
  }
  VulkanRayResourceKind kind() const override { return VulkanRayResourceKind::kTriangleScene; }
  Device &device() const override { return device_; }
  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const override {
    return core_->memory_statistics();
  }
  std::size_t vertex_count() const override { return core_->vertex_count(); }
  std::size_t triangle_count() const { return core_->triangle_count(); }
  void record_build(CommandList *commands, DevicePtr vertices, DevicePtr indices) {
    core_->record_build(commands, vertices, indices);
  }
  void record_refit(CommandList *commands, DevicePtr vertices) override {
    core_->record_refit(commands, vertices);
  }
  void prepare_query_variant(unsigned variant) override {
    core_->prepare_query_variant(variant);
  }
  void record_query(CommandList *commands, const VulkanRayQueryPacket &packet) override {
    core_->record_query(commands, packet);
  }

 private:
  std::shared_ptr<VulkanTriangleRayScene> core_;
};

// Program handles retain prepared leases; the AS core only owns device state.
class VulkanTriangleBlasResource final : public VulkanRayNativeOwner<>,
                                        public native_vulkan_ray::BlasAccess {
 public:
  VulkanTriangleBlasResource(
      Program *program,
      std::size_t vertices,
      std::size_t triangles,
      std::shared_ptr<VulkanOpacityMicromap> micromap = {},
      bool opaque = true) : VulkanRayNativeOwner(program) {
    TI_ERROR_IF(program == nullptr,
                "Vulkan triangle BLAS requires a live Program.");
    core_ = std::make_shared<VulkanTriangleBlas>(
        static_cast<vulkan::VulkanDevice *>(program->get_compute_device()),
        vertices, triangles, std::move(micromap), opaque);
  }
  VulkanRayResourceKind kind() const override {
    return VulkanRayResourceKind::kTriangleBlas;
  }
  Device &device() const override { return device_; }
  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const override {
    return core_->memory_statistics();
  }
  std::size_t vertex_count() const {
    return core_->vertex_count();
  }
  std::size_t triangle_count() const {
    return core_->triangle_count();
  }
  VkDeviceAddress device_address() const {
    return core_->device_address();
  }
  void record_build(CommandList *commands,
                    DevicePtr vertices,
                    DevicePtr indices,
                    bool update) {
    core_->record_build(commands, vertices, indices, update);
  }
  void record_build(CommandList *commands, DevicePtr vertices,
                    DevicePtr indices) override {
    record_build(commands, vertices, indices, false);
  }
  void record_refit(CommandList *commands, DevicePtr vertices) override {
    record_build(commands, vertices, {}, true);
  }
  VulkanTriangleBlas &blas_core() override { return *core_; }
  void retain(const vkapi::IVkCommandBuffer &commands) const {
    core_->retain(commands);
  }

  const std::shared_ptr<VulkanTriangleBlas> &core() const {
    return core_;
  }

 private:
  std::shared_ptr<VulkanTriangleBlas> core_;
};

class VulkanInstanceTlasResource final
    : public VulkanRayNativeOwner<NativeShaderAccelerationStructure>,
      public native_vulkan_ray::QueryAccess,
      public native_vulkan_ray::InstanceAccess {
 public:
  VulkanInstanceTlasResource(
      Program *program,
      std::vector<std::shared_ptr<VulkanTriangleBlasResource>> blases,
      std::vector<std::uint32_t> offsets) : VulkanRayNativeOwner(program) {
    TI_ERROR_IF(!program, "Vulkan instance TLAS requires a live Program.");
    std::vector<std::shared_ptr<VulkanTriangleBlas>> cores;
    std::unordered_set<VulkanTriangleBlasResource *> seen;
    for (const auto &blas : blases) {
      cores.push_back(blas->core());
      if (seen.insert(blas.get()).second) {
        dependencies_.push_back(blas);
      }
    }
    core_ = std::make_shared<VulkanInstanceTlas>(
        static_cast<vulkan::VulkanDevice *>(program->get_compute_device()),
        std::move(cores), std::move(offsets));
  }
  VulkanRayResourceKind kind() const override {
    return VulkanRayResourceKind::kInstanceTlas;
  }
  Device &device() const override { return device_; }
  VulkanTriangleRaySceneMemoryStatistics memory_statistics() const override {
    return core_->memory_statistics();
  }
  std::size_t instance_count() const {
    return core_->instance_count();
  }
  std::uint32_t max_sbt_record_offset() const {
    return core_->max_sbt_record_offset();
  }
  void record_build(CommandList *commands,
                    const std::vector<VulkanRayInstanceInfo> &instances,
                    bool update) {
    core_->record_build(commands, instances, update);
  }
  void record_metadata(CommandList *commands,
                        const std::vector<VulkanRayInstanceInfo> &instances,
                        bool update) override {
    core_->record_build(commands, instances, update);
  }
  void prepare_transforms() {
    core_->prepare_transforms();
  }
  void record_transforms(CommandList *commands,
                         const VulkanTLASTransformCommand &packet) {
    core_->record_transforms(
        commands, {{packet.shader_binding.pointer, packet.shader_binding.bytes},
                   packet.parameters});
  }
  void record_transforms(CommandList *commands,
                         const VulkanRayTransformPacket &packet) override {
    core_->record_transforms(commands, packet);
  }
  void prepare_query_variant(unsigned variant) {
    core_->prepare_query_variant(variant);
  }
  void record_query(CommandList *commands, const VulkanRayQueryPacket &packet) {
    core_->record_query(commands, packet);
  }
  void bind_for_shader(ShaderResourceSet *bindings, int binding) {
    core_->bind_for_shader(bindings, binding);
  }
  void record_shader_read(CommandList *commands,
                          VkPipelineStageFlags stages,
                          bool graphics) {
    core_->record_shader_read(commands, stages,
                             graphics ? VulkanRayShaderReadOrder::runtime_graphics
                                      : VulkanRayShaderReadOrder::compute);
  }
  void record_shader_read(CommandList *commands, NativeShaderStages stages,
                          bool graphics, bool independent) override {
    native_vulkan_ray::record_shader_read(*core_, commands, stages, graphics, independent);
  }
  void bind_for_kernel(ShaderResourceSet *bindings,
                       int binding,
                       CommandList *commands) {
    core_->bind_for_kernel(bindings, binding, commands);
  }

 private:
  // dependencies_ retains each original Program BLAS owner exactly once.
  std::shared_ptr<VulkanInstanceTlas> core_;
};

bool Program::vulkan_acceleration_structure_available() const {
  if (compile_config().arch != Arch::vulkan || !program_impl_) {
    return false;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_compute_device());
  return device && device->vk_caps().buffer_device_address &&
         device->vk_caps().acceleration_structure;
}

bool Program::vulkan_ray_query_available() const {
  if (compile_config().arch != Arch::vulkan || !program_impl_) {
    return false;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_compute_device());
  return device && device->vk_caps().buffer_device_address &&
         device->vk_caps().acceleration_structure &&
         device->vk_caps().ray_query;
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_ray_query_properties() const {
  std::unordered_map<std::string, std::uint64_t> result{
      {"available", 0},
      {"buffer_device_address", 0},
      {"acceleration_structure", 0},
      {"ray_query", 0},
      {"opacity_micromap", 0},
      {"max_geometry_count", 0},
      {"max_instance_count", 0},
      {"max_primitive_count", 0},
      {"max_per_stage_descriptor_acceleration_structures", 0},
      {"max_per_stage_descriptor_update_after_bind_acceleration_structures", 0},
      {"max_descriptor_set_acceleration_structures", 0},
      {"max_descriptor_set_update_after_bind_acceleration_structures", 0},
      {"min_acceleration_structure_scratch_offset_alignment", 0},
  };
  if (compile_config().arch != Arch::vulkan || !program_impl_) {
    return result;
  }
  auto *device = static_cast<vulkan::VulkanDevice *>(
      const_cast<Program *>(this)->get_compute_device());
  if (device == nullptr) {
    return result;
  }
  const auto &caps = device->vk_caps();
  result["buffer_device_address"] = caps.buffer_device_address;
  result["acceleration_structure"] = caps.acceleration_structure;
  result["ray_query"] = caps.ray_query;
  result["opacity_micromap"] = caps.opacity_micromap;
  result["available"] = vulkan_ray_query_available();
  if (!caps.acceleration_structure) {
    return result;
  }

  VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{};
  properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &properties;
  device->vk_device().instance_functions().vkGetPhysicalDeviceProperties2(
      device->vk_physical_device(), &properties2);
  result["max_geometry_count"] = properties.maxGeometryCount;
  result["max_instance_count"] = properties.maxInstanceCount;
  result["max_primitive_count"] = properties.maxPrimitiveCount;
  result["max_per_stage_descriptor_acceleration_structures"] =
      properties.maxPerStageDescriptorAccelerationStructures;
  result["max_per_stage_descriptor_update_after_bind_acceleration_structures"] =
      properties.maxPerStageDescriptorUpdateAfterBindAccelerationStructures;
  result["max_descriptor_set_acceleration_structures"] =
      properties.maxDescriptorSetAccelerationStructures;
  result["max_descriptor_set_update_after_bind_acceleration_structures"] =
      properties.maxDescriptorSetUpdateAfterBindAccelerationStructures;
  result["min_acceleration_structure_scratch_offset_alignment"] =
      properties.minAccelerationStructureScratchOffsetAlignment;
  return result;
}

std::uint64_t Program::create_vulkan_triangle_ray_scene(
    const storage::DenseStorageDescriptor &vertices,
    const storage::DenseStorageDescriptor &indices,
    std::size_t vertex_count,
    std::size_t triangle_count) {
  TI_ERROR_IF(!vulkan_ray_query_available(),
              "Vulkan triangle ray scenes require "
              "VK_KHR_acceleration_structure and VK_KHR_ray_query.");
  check_ray_storage(vertices, vertex_count, 3, PrimitiveType::f32, "vertices");
  check_ray_storage(indices, triangle_count, 3, PrimitiveType::i32, "indices");
  auto storage = prepare_native_storage({&vertices, &indices}, {false, false});
  std::uint64_t handle = 0;
  with_prepared_native_storage(*storage, [&] {
    auto scene = std::make_shared<VulkanTriangleSceneResource>(
        this, vertex_count, triangle_count);
    const auto vertex_pointer = storage->binding(0).pointer;
    const auto index_pointer = storage->binding(1).pointer;
    enqueue_compute_op_lambda(
        [scene, vertex_pointer, index_pointer](Device *,
                                               CommandList *commands) {
          scene->record_build(commands, vertex_pointer, index_pointer);
        },
        {});
    mark_runtime_submission_pending();
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    TI_ERROR_IF(next_vulkan_ray_scene_handle_ == 0,
                "Vulkan ray scene handle space exhausted.");
    handle = next_vulkan_ray_scene_handle_++;
    vulkan_ray_scenes_.emplace(handle, std::move(scene));
  });
  return handle;
}

VulkanRayQueryCommand Program::prepare_vulkan_ray_query(std::uint64_t handle,
                                                        bool instance_tlas,
                                                        Ndarray *rays,
                                                        Ndarray *hits,
                                                        std::size_t ray_count,
                                                        Ndarray *hit_indices) {
  TI_ERROR_IF(!rays || !hits, "Vulkan ray query requires rays and hits.");
  const auto ray_storage = storage::describe_ndarray_storage(*rays);
  const auto hit_storage = storage::describe_ndarray_storage(*hits);
  const auto index_storage =
      hit_indices ? storage::describe_ndarray_storage(*hit_indices)
                  : storage::DenseStorageBuildResult{};
  TI_ERROR_IF(!ray_storage || !hit_storage || (hit_indices && !index_storage),
              "Vulkan ray query requires describable dense storage.");
  return prepare_vulkan_ray_query_storage(
      handle, instance_tlas, *ray_storage.descriptor, *hit_storage.descriptor,
      ray_count, hit_indices ? &*index_storage.descriptor : nullptr);
}

VulkanRayQueryCommand Program::prepare_vulkan_ray_query_storage(
    std::uint64_t handle,
    bool instance_tlas,
    const storage::DenseStorageDescriptor &rays,
    const storage::DenseStorageDescriptor &hits,
    std::size_t ray_count,
    const storage::DenseStorageDescriptor *hit_indices) {
  TI_ERROR_IF(
      compile_config().arch != Arch::vulkan || ray_count == 0 ||
          ray_count > (std::numeric_limits<std::uint32_t>::max)(),
      "Vulkan ray query requires Vulkan storage and a positive uint32 count.");
  check_ray_storage(rays, ray_count, 8, PrimitiveType::f32, "rays");
  check_ray_storage(hits, ray_count, 4, PrimitiveType::f32, "hits");

  std::vector<const storage::DenseStorageDescriptor *> descriptors{&rays,
                                                                   &hits};
  std::vector<bool> writable{false, true};
  if (hit_indices) {
    const auto dtype = hit_indices->scalar_type();
    TI_ERROR_IF(dtype != PrimitiveType::i32 && dtype != PrimitiveType::u32,
                "Vulkan ray hit_indices requires dtype i32 or u32.");
    check_ray_storage(*hit_indices, ray_count, 4, dtype, "hit_indices");
    descriptors.push_back(hit_indices);
    writable.push_back(true);
  }
  VulkanRayQueryCommand packet;
  packet.storage = prepare_native_storage(descriptors, writable);
  packet.scene_handle = handle;
  packet.instance_tlas = instance_tlas;
  std::array<VulkanRayQueryPacket::Binding, 3> bindings{};
  for (std::size_t i = 0; i < descriptors.size(); ++i) {
    const auto &binding = packet.storage->binding(i);
    bindings[i] = {binding.pointer, binding.bytes};
  }
  packet.query = prepare_vulkan_ray_query_packet(
      static_cast<vulkan::VulkanDevice *>(get_compute_device()), ray_count,
      hit_indices != nullptr, bindings);
  // Build only the selected shader variant at the cold boundary.
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<NativeResource> root;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    if (instance_tlas) {
      const auto found = vulkan_ray_resources_.find(handle);
      TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                  "Vulkan instance TLAS handle is stale or closed.");
      auto resource =
          std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
      TI_ERROR_IF(!resource, "Vulkan ray resource is not an instance TLAS.");
      resource->prepare_query_variant(packet.query.variant);
      root = resource;
    } else {
      const auto found = vulkan_ray_scenes_.find(handle);
      TI_ERROR_IF(found == vulkan_ray_scenes_.end(),
                  "Vulkan triangle ray scene handle is stale or closed.");
      found->second->prepare_query_variant(packet.query.variant);
      root = found->second;
    }
  }
  packet.access = prepare_vulkan_ray_access({root});
  return packet;
}

std::size_t Program::execute_vulkan_ray_query(
    const VulkanRayQueryCommand &packet) {
  with_prepared_native_storage(*packet.storage, [&] {
    TI_ERROR_IF(!packet.access, "Prepared Vulkan AS access is missing");
    packet.access->submit([&] {
      if (packet.instance_tlas) {
        std::shared_ptr<VulkanInstanceTlasResource> resource;
        {
          std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
          const auto found = vulkan_ray_resources_.find(packet.scene_handle);
          TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                      "Vulkan instance TLAS handle is stale or closed.");
          resource = std::dynamic_pointer_cast<VulkanInstanceTlasResource>(
              found->second);
        }
        TI_ERROR_IF(!resource, "Vulkan ray resource is not an instance TLAS.");
        enqueue_compute_op_lambda(
            [resource, packet](Device *, CommandList *commands) {
              resource->record_query(commands, packet.query);
            },
            {});
      } else {
        std::shared_ptr<VulkanTriangleSceneResource> scene;
        {
          std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
          const auto found = vulkan_ray_scenes_.find(packet.scene_handle);
          TI_ERROR_IF(found == vulkan_ray_scenes_.end(),
                      "Vulkan triangle ray scene handle is stale or closed.");
          scene = found->second;
        }
        enqueue_compute_op_lambda(
            [scene, packet](Device *, CommandList *commands) {
              scene->record_query(commands, packet.query);
            },
            {});
      }
      mark_runtime_submission_pending();
    });
  });
  return 0;
}

std::size_t Program::vulkan_triangle_ray_query(std::uint64_t handle,
                                               Ndarray *rays,
                                               Ndarray *hits,
                                               std::size_t ray_count,
                                               Ndarray *hit_indices) {
  return execute_vulkan_ray_query(prepare_vulkan_ray_query(
      handle, false, rays, hits, ray_count, hit_indices));
}

VulkanRayGeometryCommand Program::prepare_vulkan_ray_geometry(
    std::uint64_t handle,
    bool independent_blas,
    const storage::DenseStorageDescriptor &vertices,
    const storage::DenseStorageDescriptor *indices,
    std::size_t vertex_count,
    std::size_t triangle_count) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan,
              "Vulkan ray geometry requires Vulkan storage.");
  TI_ERROR_IF(!independent_blas && indices,
              "TriangleScene refit cannot replace topology.");
  check_ray_storage(vertices, vertex_count, 3, PrimitiveType::f32, "vertices");
  std::vector<const storage::DenseStorageDescriptor *> descriptors{&vertices};
  if (indices) {
    check_ray_storage(*indices, triangle_count, 3, PrimitiveType::i32,
                      "indices");
    descriptors.push_back(indices);
  }
  VulkanRayGeometryCommand command;
  command.storage = prepare_native_storage(
      descriptors, std::vector<bool>(descriptors.size(), false));
  command.handle = handle;
  command.independent_blas = independent_blas;
  command.update = indices == nullptr;
  // Handle/count validation and the access closure are cold.
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<NativeResource> root;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    if (independent_blas) {
      const auto found = vulkan_ray_resources_.find(handle);
      TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                  "Vulkan triangle BLAS handle is stale or closed.");
      const auto blas =
          std::dynamic_pointer_cast<VulkanTriangleBlasResource>(found->second);
      TI_ERROR_IF(!blas, "Vulkan ray resource is not a triangle BLAS.");
      TI_ERROR_IF(vertex_count != blas->vertex_count() ||
                      triangle_count != blas->triangle_count(),
                  "Vulkan triangle BLAS build must preserve geometry counts.");
      root = blas;
    } else {
      const auto found = vulkan_ray_scenes_.find(handle);
      TI_ERROR_IF(found == vulkan_ray_scenes_.end(),
                  "Vulkan triangle ray scene handle is stale or closed.");
      TI_ERROR_IF(vertex_count != found->second->vertex_count() ||
                      triangle_count != found->second->triangle_count(),
                  "Vulkan triangle ray refit must preserve geometry counts.");
      root = found->second;
    }
  }
  command.access = prepare_vulkan_ray_access({root}, true);
  return command;
}

std::size_t Program::execute_vulkan_ray_geometry(
    const VulkanRayGeometryCommand &command) {
  with_prepared_native_storage(*command.storage, [&] {
    TI_ERROR_IF(!command.access, "Prepared Vulkan AS access is missing");
    command.access->submit([&] {
      const auto vertices = command.storage->binding(0).pointer;
      if (command.independent_blas) {
        std::shared_ptr<VulkanTriangleBlasResource> resource;
        {
          std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
          const auto found = vulkan_ray_resources_.find(command.handle);
          TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                      "Vulkan triangle BLAS handle is stale or closed.");
          resource =
              std::static_pointer_cast<VulkanTriangleBlasResource>(found->second);
        }
        const auto indices =
            command.update ? DevicePtr{} : command.storage->binding(1).pointer;
        enqueue_compute_op_lambda(
            [resource, vertices, indices, update = command.update](
                Device *, CommandList *commands) {
              resource->record_build(commands, vertices, indices, update);
            },
            {});
      } else {
        std::shared_ptr<VulkanTriangleSceneResource> scene;
        {
          std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
          const auto found = vulkan_ray_scenes_.find(command.handle);
          TI_ERROR_IF(found == vulkan_ray_scenes_.end(),
                      "Vulkan triangle ray scene handle is stale or closed.");
          scene = found->second;
        }
        enqueue_compute_op_lambda(
            [scene, vertices](Device *, CommandList *commands) {
              scene->record_refit(commands, vertices);
            },
            {});
      }
      mark_runtime_submission_pending();
    });
  });
  return 0;
}

std::uint64_t Program::create_vulkan_triangle_blas_resource(
    std::size_t vertex_count,
    std::size_t triangle_count) {
  return create_vulkan_triangle_blas_resource_with_opacity(
      vertex_count, triangle_count, true);
}

std::uint64_t Program::create_vulkan_triangle_blas_resource_with_opacity(
    std::size_t vertex_count,
    std::size_t triangle_count,
    bool opaque) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_acceleration_structure_available(),
              "Vulkan triangle BLAS resources require "
              "VK_KHR_acceleration_structure and device addresses.");
  auto resource = std::make_shared<VulkanTriangleBlasResource>(
      this, vertex_count, triangle_count, nullptr, opaque);
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  TI_ERROR_IF(next_vulkan_ray_resource_handle_ == 0,
              "Vulkan ray resource handle space exhausted.");
  const std::uint64_t handle = next_vulkan_ray_resource_handle_++;
  vulkan_ray_resources_.emplace(handle, std::move(resource));
  return handle;
}

std::pair<std::uint64_t, std::array<std::size_t, 3>>
Program::create_vulkan_triangle_blas_micromap_resource(
    std::size_t vertex_count,
    std::size_t triangle_count,
    const std::string &data,
    const std::string &descriptors,
    const std::string &indices,
    bool indexed) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_acceleration_structure_available(),
              "Vulkan acceleration structures are unavailable.");
  auto *device = static_cast<vulkan::VulkanDevice *>(get_compute_device());
  auto micromap = std::make_shared<VulkanOpacityMicromap>(device);
  micromap->build(data, descriptors, indices, indexed, triangle_count);
  auto resource = std::make_shared<VulkanTriangleBlasResource>(
      this, vertex_count, triangle_count, micromap);
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  TI_ERROR_IF(next_vulkan_ray_resource_handle_ == 0,
              "Vulkan ray resource handle space exhausted.");
  const auto handle = next_vulkan_ray_resource_handle_++;
  vulkan_ray_resources_.emplace(handle, std::move(resource));
  return {handle, micromap->memory};
}

std::uint64_t Program::create_vulkan_instance_tlas_resource(
    const std::vector<std::uint64_t> &blas_handles) {
  return create_vulkan_instance_tlas_resource_with_sbt(
      blas_handles, std::vector<std::uint32_t>(blas_handles.size(), 0));
}

std::uint64_t Program::create_vulkan_instance_tlas_resource_with_sbt(
    const std::vector<std::uint64_t> &blas_handles,
    const std::vector<std::uint32_t> &sbt_record_offsets) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(!vulkan_acceleration_structure_available(),
              "Vulkan instance TLAS resources require "
              "VK_KHR_acceleration_structure and device addresses.");
  TI_ERROR_IF(blas_handles.empty(),
              "Vulkan instance TLAS requires at least one BLAS handle.");
  std::vector<std::shared_ptr<VulkanTriangleBlasResource>> blases;
  blases.reserve(blas_handles.size());
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    for (const auto handle : blas_handles) {
      const auto found = vulkan_ray_resources_.find(handle);
      TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                  "Vulkan instance TLAS received a stale or closed BLAS "
                  "handle.");
      auto blas =
          std::dynamic_pointer_cast<VulkanTriangleBlasResource>(found->second);
      TI_ERROR_IF(!blas,
                  "Vulkan instance TLAS dependencies must be triangle BLAS "
                  "resources.");
      blases.push_back(std::move(blas));
    }
  }
  auto resource =
      std::make_shared<VulkanInstanceTlasResource>(this, std::move(blases), sbt_record_offsets);
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  TI_ERROR_IF(next_vulkan_ray_resource_handle_ == 0,
              "Vulkan ray resource handle space exhausted.");
  const std::uint64_t handle = next_vulkan_ray_resource_handle_++;
  vulkan_ray_resources_.emplace(handle, std::move(resource));
  return handle;
}

std::size_t Program::vulkan_instance_tlas_build(
    std::uint64_t handle,
    const std::vector<VulkanRayInstanceInfo> &instances,
    bool update) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<VulkanInstanceTlasResource> resource;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_resources_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                "Vulkan instance TLAS handle is stale or closed.");
    resource =
        std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
  }
  TI_ERROR_IF(!resource, "Vulkan ray resource is not an instance TLAS.");
  TI_ERROR_IF(instances.size() != resource->instance_count(),
              "Vulkan instance TLAS build must preserve instance count {}.",
              resource->instance_count());
  auto *runtime = get_gfx_runtime_for_borrowing();
  auto access = runtime->ordinary_ray_scope();
  runtime->require_ray_uses(collect_native_ray_uses(resource, true));
  enqueue_compute_op_lambda(
      [resource, instances, update](Device *, CommandList *commands) {
        resource->record_build(commands, instances, update);
      },
      {});
  mark_runtime_submission_pending();
  return 0;
}

VulkanTLASTransformCommand Program::prepare_vulkan_tlas_transforms(
    std::uint64_t handle,
    const storage::DenseStorageDescriptor &transforms,
    std::size_t instance_count) {
  TI_ERROR_IF(compile_config().arch != Arch::vulkan || instance_count == 0 ||
                  instance_count > (std::numeric_limits<std::uint32_t>::max)(),
              "Vulkan TLAS transforms require Vulkan storage and a positive "
              "uint32 instance count.");
  const auto shape = transforms.index_shape();
  const auto element = transforms.element_shape();
  const auto count = static_cast<std::int64_t>(instance_count);
  const bool scalar = element.empty() &&
                      (shape == std::vector<std::int64_t>{count, 3, 4} ||
                       shape == std::vector<std::int64_t>{count, 12});
  const bool matrix = element == std::vector<std::int64_t>{3, 4} &&
                      shape == std::vector<std::int64_t>{count};
  TI_ERROR_IF(transforms.scalar_type() != PrimitiveType::f32 ||
                  (!scalar && !matrix),
              "Vulkan TLAS transforms require f32 scalar (N, 3, 4)/(N, 12) "
              "or AOS matrix-3x4 (N,) storage with the fixed instance count.");
  VulkanTLASTransformCommand packet;
  packet.storage = prepare_native_storage({&transforms}, {false});
  packet.handle = handle;
  const auto &binding = packet.storage->binding(0);
  const auto qualified = prepare_vulkan_ray_transform_packet(
      static_cast<vulkan::VulkanDevice *>(get_compute_device()), instance_count,
      {binding.pointer, binding.bytes});
  packet.shader_binding = binding;
  packet.shader_binding.pointer = qualified.shader_binding.pointer;
  packet.shader_binding.bytes = qualified.shader_binding.bytes;
  packet.parameters = qualified.parameters;

  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::shared_ptr<VulkanInstanceTlasResource> resource;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_resources_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                "Vulkan instance TLAS handle is stale or closed.");
    resource =
        std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
  }
  TI_ERROR_IF(!resource, "Vulkan ray resource is not an instance TLAS.");
  TI_ERROR_IF(instance_count != resource->instance_count(),
              "Vulkan TLAS transforms must preserve the fixed instance count.");
  resource->prepare_transforms();
  packet.access = prepare_vulkan_ray_access({resource}, true);
  return packet;
}

std::size_t Program::execute_vulkan_tlas_transforms(
    const VulkanTLASTransformCommand &packet) {
  with_prepared_native_storage(*packet.storage, [&] {
    TI_ERROR_IF(!packet.access, "Prepared Vulkan AS access is missing");
    packet.access->submit([&] {
      std::shared_ptr<VulkanInstanceTlasResource> resource;
      {
        std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
        const auto found = vulkan_ray_resources_.find(packet.handle);
        TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                    "Vulkan instance TLAS handle is stale or closed.");
        resource =
            std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
      }
      TI_ERROR_IF(!resource, "Vulkan ray resource is not an instance TLAS.");
      enqueue_compute_op_lambda(
          [resource, packet](Device *, CommandList *commands) {
            resource->record_transforms(commands, packet);
          },
          {});
      mark_runtime_submission_pending();
    });
  });
  return 0;
}

std::size_t Program::vulkan_instance_tlas_query(std::uint64_t handle,
                                                Ndarray *rays,
                                                Ndarray *hits,
                                                std::size_t ray_count,
                                                Ndarray *hit_indices) {
  return execute_vulkan_ray_query(prepare_vulkan_ray_query(
      handle, true, rays, hits, ray_count, hit_indices));
}

VulkanTriangleRaySceneMemoryStatistics
Program::vulkan_ray_resource_memory_statistics(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  const auto found = vulkan_ray_resources_.find(handle);
  TI_ERROR_IF(found == vulkan_ray_resources_.end(),
              "Vulkan ray resource handle is stale or closed.");
  return found->second->memory_statistics();
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_ray_kernel_resource_properties(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  const auto found = vulkan_ray_resources_.find(handle);
  TI_ERROR_IF(found == vulkan_ray_resources_.end(),
              "Vulkan ray kernel resource handle is stale or closed.");
  auto resource =
      std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
  TI_ERROR_IF(!resource,
              "Vulkan ray kernel resources must be top-level instance AS.");
  return {{"handle", handle},
          {"top_level", 1},
          {"read_only", 1},
          {"exact_generation", 1},
          {"instance_count", resource->instance_count()},
          {"max_sbt_record_offset", resource->max_sbt_record_offset()}};
}

void Program::vulkan_bind_ray_kernel_resource(
    std::uint64_t handle,
    ShaderResourceSet *bindings,
    int binding,
    CommandList *command_list) {
  std::shared_ptr<VulkanInstanceTlasResource> resource;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_resources_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_resources_.end(),
                "Vulkan ray kernel resource handle is stale or closed.");
    resource =
        std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
  }
  TI_ERROR_IF(!resource,
              "Vulkan ray kernel resources must be top-level instance AS.");
  resource->bind_for_kernel(bindings, binding, command_list);
}

std::function<void(CommandList *)> Program::prepare_vulkan_ray_graphics_binding(
    std::uint64_t handle,
    ShaderResourceSet *bindings,
    int binding,
    const std::shared_ptr<PreparedResourceLease> &lease,
    std::uint32_t consumer_stages) {
  return prepare_vulkan_ray_shader_binding(handle, bindings, binding, lease,
                                            consumer_stages, true);
}

std::function<void(CommandList *)> Program::prepare_vulkan_ray_shader_binding(
    std::uint64_t handle,
    ShaderResourceSet *bindings,
    int binding,
    const std::shared_ptr<PreparedResourceLease> &lease,
    std::uint32_t consumer_stages,
    bool graphics_queue) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  const auto found = vulkan_ray_resources_.find(handle);
  TI_ERROR_IF(found == vulkan_ray_resources_.end(),
              "Vulkan shader acceleration structure is stale or closed.");
  auto resource =
      std::dynamic_pointer_cast<VulkanInstanceTlasResource>(found->second);
  TI_ERROR_IF(!resource || !lease,
              "Vulkan shader binding requires a top-level AS and a prepared lease.");
  resource->bind_for_shader(bindings, binding);
  resource->register_prepared_lease(lease);
  return [weak = std::weak_ptr<VulkanInstanceTlasResource>(resource),
          consumer_stages, graphics_queue](
             CommandList *commands) {
    auto owner = weak.lock();
    TI_ERROR_IF(!owner, "Prepared Vulkan shader AS is stale or closed.");
    owner->record_shader_read(commands, consumer_stages, graphics_queue);
  };
}

VulkanTriangleRaySceneMemoryStatistics
Program::vulkan_triangle_ray_scene_memory_statistics(
    std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  const auto found = vulkan_ray_scenes_.find(handle);
  TI_ERROR_IF(found == vulkan_ray_scenes_.end(),
              "Vulkan triangle ray scene handle is stale or closed.");
  return found->second->memory_statistics();
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_vulkan_ray_resource_stats() {
  std::uint64_t live = 0;
  std::uint64_t queued_for_completion = 0;
  std::uint64_t independent_live = 0;
  std::uint64_t blas_live = 0;
  std::uint64_t tlas_live = 0;
  std::uint64_t independent_queued_for_completion = 0;
  {
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    live = vulkan_ray_scenes_.size();
    queued_for_completion = vulkan_ray_scene_retirements_.size();
    independent_live = vulkan_ray_resources_.size();
    independent_queued_for_completion = vulkan_ray_resource_retirements_.size();
    for (const auto &[handle, resource] : vulkan_ray_resources_) {
      if (resource->kind() == VulkanRayResourceKind::kTriangleBlas) {
        ++blas_live;
      } else {
        ++tlas_live;
      }
    }
  }
  const auto completion_retained =
      runtime_completion_resource_count(kVulkanRaySceneResourceKind);
  const auto independent_completion_retained =
      runtime_completion_resource_count(kVulkanRayResourceKind);
  return {{"live", live},
          {"retiring", queued_for_completion + completion_retained},
          {"queued_for_completion", queued_for_completion},
          {"completion_retained", completion_retained},
          {"independent_live", independent_live},
          {"blas_live", blas_live},
          {"tlas_live", tlas_live},
          {"independent_retiring", independent_queued_for_completion +
                                       independent_completion_retained},
          {"independent_queued_for_completion",
           independent_queued_for_completion},
          {"independent_completion_retained",
           independent_completion_retained}};
}

void Program::destroy_vulkan_triangle_ray_scene(std::uint64_t handle) {
  std::shared_ptr<VulkanTriangleSceneResource> scene;
  bool record_retirement = false;
  {
    auto submission_guard = acquire_runtime_resource_submission_guard();
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_scenes_.find(handle);
    if (found == vulkan_ray_scenes_.end()) {
      return;
    }
    scene = found->second;
    scene->invalidate_prepared_resources();
    vulkan_ray_scenes_.erase(found);
    if (!runtime_has_fatal_fault() &&
        runtime_submission_pending_.load(std::memory_order_acquire)) {
      vulkan_ray_scene_retirements_.push_back(std::move(scene));
      record_retirement = true;
    }
  }
  if (record_retirement) {
    record_runtime_completion();
  }
}

void Program::destroy_vulkan_ray_resource(std::uint64_t handle) {
  std::shared_ptr<VulkanRayResource> resource;
  bool record_retirement = false;
  {
    auto submission_guard = acquire_runtime_resource_submission_guard();
    std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
    const auto found = vulkan_ray_resources_.find(handle);
    if (found == vulkan_ray_resources_.end()) {
      return;
    }
    resource = found->second;
    resource->invalidate_prepared_resources();
    vulkan_ray_resources_.erase(found);
    if (auto *launcher =
            dynamic_cast<gfx::KernelLauncher *>(&get_kernel_launcher())) {
      launcher->runtime()->retire_ray_resource_recordings(handle);
    }
    program_impl_->invalidate_graphics_command_replay();
    if (!runtime_has_fatal_fault() &&
        runtime_submission_pending_.load(std::memory_order_acquire)) {
      vulkan_ray_resource_retirements_.push_back(std::move(resource));
      record_retirement = true;
    }
  }
  if (record_retirement) {
    record_runtime_completion();
  }
}

std::shared_ptr<NativeResource> Program::acquire_vulkan_ray_native_resource(
    std::uint64_t handle, bool triangle_scene) {
  auto guard = acquire_runtime_resource_submission_guard();
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  if (triangle_scene) {
    const auto found = vulkan_ray_scenes_.find(handle);
    TI_ERROR_IF(found == vulkan_ray_scenes_.end(),
                "Vulkan triangle ray scene handle is stale or closed.");
    return found->second;
  }
  const auto found = vulkan_ray_resources_.find(handle);
  TI_ERROR_IF(found == vulkan_ray_resources_.end(),
              "Vulkan ray resource handle is stale or closed.");
  auto native = std::dynamic_pointer_cast<NativeResource>(found->second);
  TI_ASSERT(native);
  return native;
}

std::shared_ptr<PreparedVulkanRayAccess> Program::prepare_vulkan_ray_access(
    const std::vector<std::shared_ptr<NativeResource>> &roots, bool writes) {
  auto guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(roots.empty(), "Prepared Vulkan AS access requires an owner");
  std::vector<NativeResourceUse> uses;
  std::vector<std::shared_ptr<VulkanRayResource>> owners;
  for (const auto &root : roots) {
    auto owner = std::dynamic_pointer_cast<VulkanRayResource>(root);
    TI_ERROR_IF(!owner, "Prepared Vulkan AS access requires original Program owners");
    owners.push_back(std::move(owner));
    auto closure = collect_native_ray_uses(root, writes);
    uses.insert(uses.end(), closure.begin(), closure.end());
  }
  auto access = std::make_shared<PreparedVulkanRayAccess>(
      get_gfx_runtime_for_borrowing()->prepare_ray_uses(uses));
  // Dependencies participate in admission, but closing a public BLAS handle
  // does not invalidate a TLAS that still owns that BLAS. Only the packet's
  // original public roots invalidate its permit at close/reset.
  for (const auto &owner : owners)
    owner->register_prepared_lease(access);
  return access;
}

void Program::vulkan_clear_ray_scenes() {
  vulkan_clear_ray_programs();
  std::lock_guard<std::mutex> lock(vulkan_ray_scene_mutex_);
  for (const auto &[handle, scene] : vulkan_ray_scenes_)
    scene->invalidate_prepared_resources();
  vulkan_ray_scenes_.clear();
  vulkan_ray_scene_retirements_.clear();
  for (const auto &[handle, resource] : vulkan_ray_resources_) {
    resource->invalidate_prepared_resources();
  }
  vulkan_ray_resources_.clear();
  vulkan_ray_resource_retirements_.clear();
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

std::shared_ptr<PreparedVulkanRayAccess> Program::prepare_vulkan_ray_access(
    const std::vector<std::shared_ptr<NativeResource>> &, bool) {
  TI_ERROR("Vulkan AS access is unavailable in this build");
}

std::shared_ptr<NativeResource> Program::acquire_vulkan_ray_native_resource(
    std::uint64_t, bool) {
  TI_ERROR("Vulkan AS views require TI_WITH_VULKAN=ON.");
}

std::function<void(CommandList *)> Program::prepare_vulkan_ray_shader_binding(
    std::uint64_t,
    ShaderResourceSet *,
    int,
    const std::shared_ptr<PreparedResourceLease> &,
    std::uint32_t,
    bool) {
  TI_ERROR("Vulkan shader AS binding requires TI_WITH_VULKAN=ON.");
}

std::function<void(CommandList *)> Program::prepare_vulkan_ray_graphics_binding(
    std::uint64_t,
    ShaderResourceSet *,
    int,
    const std::shared_ptr<PreparedResourceLease> &,
    std::uint32_t) {
  TI_ERROR("Vulkan graphics AS binding requires TI_WITH_VULKAN=ON.");
}

bool Program::vulkan_ray_query_available() const {
  return false;
}

bool Program::vulkan_acceleration_structure_available() const {
  return false;
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_ray_query_properties() const {
  return {{"available", 0},
          {"buffer_device_address", 0},
          {"acceleration_structure", 0},
          {"ray_query", 0},
          {"opacity_micromap", 0},
          {"max_geometry_count", 0},
          {"max_instance_count", 0},
          {"max_primitive_count", 0},
          {"max_per_stage_descriptor_acceleration_structures", 0},
          {"max_per_stage_descriptor_update_after_bind_acceleration_structures", 0},
          {"max_descriptor_set_acceleration_structures", 0},
          {"max_descriptor_set_update_after_bind_acceleration_structures", 0},
          {"min_acceleration_structure_scratch_offset_alignment", 0}};
}

std::uint64_t Program::create_vulkan_triangle_ray_scene(
    const storage::DenseStorageDescriptor &,
    const storage::DenseStorageDescriptor &,
    std::size_t,
    std::size_t) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

VulkanRayQueryCommand Program::prepare_vulkan_ray_query(std::uint64_t,
                                                        bool,
                                                        Ndarray *,
                                                        Ndarray *,
                                                        std::size_t,
                                                        Ndarray *) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

VulkanRayQueryCommand Program::prepare_vulkan_ray_query_storage(
    std::uint64_t,
    bool,
    const storage::DenseStorageDescriptor &,
    const storage::DenseStorageDescriptor &,
    std::size_t,
    const storage::DenseStorageDescriptor *) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::size_t Program::execute_vulkan_ray_query(const VulkanRayQueryCommand &) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::size_t Program::vulkan_triangle_ray_query(std::uint64_t,
                                               Ndarray *,
                                               Ndarray *,
                                               std::size_t,
                                               Ndarray *) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

VulkanRayGeometryCommand Program::prepare_vulkan_ray_geometry(
    std::uint64_t,
    bool,
    const storage::DenseStorageDescriptor &,
    const storage::DenseStorageDescriptor *,
    std::size_t,
    std::size_t) {
  TI_ERROR("Vulkan ray geometry requires TI_WITH_VULKAN=ON.");
}

std::size_t Program::execute_vulkan_ray_geometry(
    const VulkanRayGeometryCommand &) {
  TI_ERROR("Vulkan ray geometry requires TI_WITH_VULKAN=ON.");
}

VulkanTriangleRaySceneMemoryStatistics
Program::vulkan_triangle_ray_scene_memory_statistics(std::uint64_t) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::uint64_t Program::create_vulkan_triangle_blas_resource(std::size_t,
                                                            std::size_t) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::uint64_t Program::create_vulkan_triangle_blas_resource_with_opacity(
    std::size_t, std::size_t, bool) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::pair<std::uint64_t, std::array<std::size_t, 3>>
Program::create_vulkan_triangle_blas_micromap_resource(
    std::size_t, std::size_t, const std::string &, const std::string &,
    const std::string &, bool) {
  TI_ERROR("Vulkan opacity micromap requires TI_WITH_VULKAN=ON.");
}

std::uint64_t Program::create_vulkan_instance_tlas_resource(
    const std::vector<std::uint64_t> &) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::uint64_t Program::create_vulkan_instance_tlas_resource_with_sbt(
    const std::vector<std::uint64_t> &,
    const std::vector<std::uint32_t> &) {
  TI_ERROR("Vulkan acceleration structures require TI_WITH_VULKAN=ON.");
}

std::size_t Program::vulkan_instance_tlas_build(
    std::uint64_t,
    const std::vector<VulkanRayInstanceInfo> &,
    bool) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

VulkanTLASTransformCommand Program::prepare_vulkan_tlas_transforms(
    std::uint64_t,
    const storage::DenseStorageDescriptor &,
    std::size_t) {
  TI_ERROR("Vulkan TLAS transforms require TI_WITH_VULKAN=ON.");
}

std::size_t Program::execute_vulkan_tlas_transforms(
    const VulkanTLASTransformCommand &) {
  TI_ERROR("Vulkan TLAS transforms require TI_WITH_VULKAN=ON.");
}

std::size_t Program::vulkan_instance_tlas_query(std::uint64_t,
                                                Ndarray *,
                                                Ndarray *,
                                                std::size_t,
                                                Ndarray *) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

VulkanTriangleRaySceneMemoryStatistics
Program::vulkan_ray_resource_memory_statistics(std::uint64_t) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::unordered_map<std::string, std::uint64_t>
Program::vulkan_ray_kernel_resource_properties(std::uint64_t) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

void Program::vulkan_bind_ray_kernel_resource(std::uint64_t,
                                              ShaderResourceSet *,
                                              int,
                                              CommandList *) {
  TI_ERROR("Vulkan ray query requires TI_WITH_VULKAN=ON.");
}

std::unordered_map<std::string, std::uint64_t>
Program::debug_vulkan_ray_resource_stats() {
  return {{"live", 0},
          {"retiring", 0},
          {"queued_for_completion", 0},
          {"completion_retained", 0},
          {"independent_live", 0},
          {"blas_live", 0},
          {"tlas_live", 0},
          {"independent_retiring", 0},
          {"independent_queued_for_completion", 0},
          {"independent_completion_retained", 0}};
}

void Program::destroy_vulkan_triangle_ray_scene(std::uint64_t) {
}

void Program::destroy_vulkan_ray_resource(std::uint64_t) {
}

void Program::vulkan_clear_ray_scenes() {
}

}  // namespace taichi::lang

#endif
