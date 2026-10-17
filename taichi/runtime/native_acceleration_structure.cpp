#include "taichi/runtime/native_acceleration_structure.h"

#include <algorithm>
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang {
void NativeAccelerationStructureKernelBinding::validate_parameter(
    Arch arch,
    Device &device,
    const CallableBase &kernel) const {
  if (!resource_ || arch != Arch::vulkan || resource_->arch() != arch ||
      &resource_->device() != &device)
    throw std::invalid_argument(
        "Kernel acceleration structure requires a same-device Vulkan owner");
  const auto found = kernel.nested_parameters.find(argument_);
  if (found == kernel.nested_parameters.end() ||
      found->second.ptype != ParameterType::kAccelerationStructure ||
      found->second.needs_grad)
    throw std::invalid_argument(
        "Managed acceleration structure requires an acceleration_structure "
        "kernel parameter");
}

NativeAccelerationStructureKernelBinding::
    NativeAccelerationStructureKernelBinding(
        std::shared_ptr<NativeAccelerationStructure> resource,
        std::vector<int> argument,
        Arch arch,
        Device &device,
        const CallableBase &kernel,
        LaunchContextBuilder &context)
    : resource_(std::move(resource)),
      argument_(std::move(argument)),
      kernel_(&kernel),
      context_(&context) {
  validate_parameter(arch, device, kernel);
  context.set_arg_native_acceleration_structure(argument_, resource_.get());
}

void NativeAccelerationStructureKernelBinding::validate(
    Arch arch,
    Device &device,
    const CallableBase &kernel,
    const LaunchContextBuilder &context,
    const NativeRecordingState &) const {
  validate_parameter(arch, device, kernel);
  const auto kind = context.device_allocation_type.find(argument_);
  const auto offset = context.args_type->get_element_offset(argument_);
  const auto found = std::find_if(
      context.acceleration_structure_ptrs.begin(),
      context.acceleration_structure_ptrs.end(),
      [offset](const auto &ref) { return ref.arg_offset == offset; });
  if (kernel_ != &kernel || context_ != &context ||
      kind == context.device_allocation_type.end() ||
      kind->second !=
          LaunchContextBuilder::DevAllocType::kAccelerationStructure ||
      found == context.acceleration_structure_ptrs.end() || found->owner ||
      found->handle || found->native_resource != resource_.get())
    throw std::invalid_argument(
        "Kernel acceleration-structure binding changed after qualification");
}
}  // namespace taichi::lang
