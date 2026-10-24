#pragma once

#include "taichi/runtime/native_resource.h"
#include "taichi/runtime/native_kernel_resource.h"
#include "taichi/runtime/native_ray_admission.h"

namespace taichi::lang {

// Cold recording facts shared by native traversal and kernel/shader bindings.
// The closure owns every mutable dependency; retaining it does not grant access
// from another execution instance. Submission never traverses this interface.
class NativeRayResourceAccess {
 public:
  virtual ~NativeRayResourceAccess() = default;
  virtual void require_initialized(const NativeRecordingState &) const = 0;
  virtual const std::vector<std::shared_ptr<NativeResource>> &
  dependencies() const = 0;
  // Private factory resources need no live admission identity. Public owners
  // and imported views must share the original registration, never a raw AS
  // address or a separately reconstructed scene.
  virtual std::shared_ptr<NativeRayResourceRegistration> ray_registration() const {
    return {};
  }
};

std::vector<std::shared_ptr<NativeResource>> collect_native_ray_resources(
    const std::shared_ptr<NativeResource> &root);
std::vector<NativeResourceUse> collect_native_ray_uses(
    const std::shared_ptr<NativeResource> &root, bool writes_root = false);

// Owner-local AS service. Native handles stay behind this interface; callers
// retain the resource through submitted work and never serialize a Program.
class NativeAccelerationStructure : public NativeResource,
                                    public virtual NativeRayResourceAccess {
 public:
  virtual void bind_for_kernel(ShaderResourceSet *bindings,
                               int binding,
                               CommandList *commands) = 0;
};

// Qualified shader reads are prepared once, then recorded on the owner's
// queue. Kernel-only resources do not implicitly acquire this capability.
using NativeShaderStages = std::uint32_t;
constexpr NativeShaderStages native_shader_stage(PipelineStageType stage) {
  return NativeShaderStages(1) << static_cast<unsigned>(stage);
}
class NativeShaderAccelerationStructure : public NativeAccelerationStructure {
 public:
  virtual void bind_for_shader(ShaderResourceSet *bindings, int binding) = 0;
  virtual void record_shader_read(CommandList *commands,
                                  NativeShaderStages stages,
                                  bool graphics_queue,
                                  bool independent = false) = 0;
  virtual std::uint32_t max_sbt_record_offset() const = 0;
};

class NativeAccelerationStructureKernelBinding final
    : public NativeKernelResourceBinding {
 public:
  NativeAccelerationStructureKernelBinding(
      std::shared_ptr<NativeAccelerationStructure> resource,
      std::vector<int> argument,
      Arch arch,
      Device &device,
      const CallableBase &kernel,
      LaunchContextBuilder &context);
  const std::vector<int> &argument() const override {
    return argument_;
  }
  void validate(Arch arch,
                Device &device,
                const CallableBase &kernel,
                const LaunchContextBuilder &context,
                const NativeRecordingState &state) const override;
  const std::vector<NativeResourceUse> &
  vulkan_stateful_resources() const override {
    return resources_;
  }

 private:
  void validate_parameter(Arch arch,
                          Device &device,
                          const CallableBase &kernel) const;
  const std::shared_ptr<NativeAccelerationStructure> resource_;
  const std::vector<NativeResourceUse> resources_;
  const std::vector<int> argument_;
  const CallableBase *kernel_;
  const LaunchContextBuilder *context_;
};
}  // namespace taichi::lang
