#pragma once

#include "taichi/runtime/native_resource.h"
#include "taichi/runtime/native_kernel_resource.h"

namespace taichi::lang {

// Owner-local AS service. Native handles stay behind this interface; callers
// retain the resource through submitted work and never serialize a Program.
class NativeAccelerationStructure : public NativeResource {
 public:
  virtual void bind_for_kernel(ShaderResourceSet *bindings,
                               int binding,
                               CommandList *commands) = 0;
};

// Qualified shader reads are prepared once, then recorded on the owner's
// queue. Kernel-only resources do not implicitly acquire this capability.
class NativeShaderAccelerationStructure : public NativeAccelerationStructure {
 public:
  virtual void bind_for_shader(ShaderResourceSet *bindings, int binding) = 0;
  virtual void record_shader_read(CommandList *commands,
                                  PipelineStageType stage,
                                  bool graphics_queue) = 0;
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

 private:
  void validate_parameter(Arch arch,
                          Device &device,
                          const CallableBase &kernel) const;
  const std::shared_ptr<NativeAccelerationStructure> resource_;
  const std::vector<int> argument_;
  const CallableBase *kernel_;
  const LaunchContextBuilder *context_;
};
}  // namespace taichi::lang
