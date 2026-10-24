#pragma once

#include "taichi/runtime/native_acceleration_structure.h"
#include "taichi/runtime/vulkan_ray_types.h"

namespace taichi::lang {
class VulkanTriangleBlas;
class VulkanInstanceTlas;
struct VulkanRayQueryPacket;
struct VulkanRayTransformPacket;

namespace native_vulkan_ray {

#ifdef TI_WITH_VULKAN
void record_shader_read(VulkanInstanceTlas &tlas, CommandList *commands,
                        NativeShaderStages stages, bool graphics_queue,
                        bool independent);
#endif

// Typed views of the original resource owner. Both reconstructed resources and
// live Program resources implement these contracts; operations never rebuild
// an AS or infer its capabilities from a serialized kind or device address.
class QueryAccess : public virtual NativeRayResourceAccess {
 public:
  virtual ~QueryAccess() = default;
  virtual Device &device() const = 0;
#ifdef TI_WITH_VULKAN
  virtual void prepare_query_variant(unsigned variant) = 0;
  virtual void record_query(CommandList *, const VulkanRayQueryPacket &) = 0;
#endif
};

class GeometryAccess : public virtual NativeRayResourceAccess {
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
  virtual void record_build(CommandList *, DevicePtr vertices,
                            DevicePtr indices) = 0;
#endif
};

class InstanceAccess : public virtual NativeRayResourceAccess {
 public:
  virtual ~InstanceAccess() = default;
  virtual Device &device() const = 0;
  virtual size_t instance_count() const = 0;
  // Cold symbolic input projection. Preserve instance ordinals, including
  // repeated BLAS owners; dependencies() is an access closure and may dedup.
  virtual const std::vector<std::shared_ptr<NativeResource>> &
  instance_blases() const = 0;
#ifdef TI_WITH_VULKAN
  virtual void prepare_transforms() = 0;
  virtual void record_transforms(CommandList *,
                                 const VulkanRayTransformPacket &) = 0;
  virtual void record_metadata(CommandList *,
                               const std::vector<VulkanRayInstanceInfo> &,
                               bool update) = 0;
#endif
};

}  // namespace native_vulkan_ray
}  // namespace taichi::lang
