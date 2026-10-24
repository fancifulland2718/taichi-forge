#include "taichi/runtime/native_acceleration_structure.h"

#include <algorithm>
#include <unordered_set>
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang {
std::vector<std::shared_ptr<NativeResource>> collect_native_ray_resources(
    const std::shared_ptr<NativeResource> &root) {
  std::vector<std::shared_ptr<NativeResource>> resources;
  std::vector<std::shared_ptr<NativeResource>> pending{root};
  std::unordered_set<const NativeResource *> seen;
  while (!pending.empty()) {
    auto current = std::move(pending.back());
    pending.pop_back();
    if (!current)
      throw std::invalid_argument("Vulkan ray dependency has no owner");
    if (!seen.insert(current.get()).second)
      continue;
    const auto *ray = dynamic_cast<const NativeRayResourceAccess *>(current.get());
    if (!ray)
      throw std::invalid_argument("Vulkan ray dependency has no recording contract");
    pending.insert(pending.end(), ray->dependencies().begin(),
                   ray->dependencies().end());
    resources.push_back(std::move(current));
  }
  return resources;
}

std::vector<NativeResourceUse> collect_native_ray_uses(
    const std::shared_ptr<NativeResource> &root, bool writes_root) {
  std::vector<NativeResourceUse> uses;
  for (auto &resource : collect_native_ray_resources(root)) {
    const bool writes = writes_root && resource == root;
    uses.push_back({std::move(resource), writes});
  }
  return uses;
}

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
      resources_(collect_native_ray_uses(resource_)),
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
    const NativeRecordingState &state) const {
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
  for (const auto &use : resources_)
    dynamic_cast<const NativeRayResourceAccess &>(*use.resource).require_initialized(state);
}
}  // namespace taichi::lang
