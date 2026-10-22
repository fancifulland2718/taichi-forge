#include "taichi/runtime/native_argpack_binding.h"
#include "taichi/program/launch_context_builder.h"

namespace taichi::lang {
NativeArgPackKernelBinding::NativeArgPackKernelBinding(
    std::shared_ptr<NativeArgPackStorage> storage,
    std::vector<int> argument,
    Arch arch,
    Device &device,
    const CallableBase &kernel,
    LaunchContextBuilder &context)
    : storage_(std::move(storage)),
      argument_(std::move(argument)),
      kernel_(&kernel),
      context_(&context) {
  if (!storage_ || storage_->arch() != arch || &storage_->device() != &device ||
      !storage_->layout().matches(kernel, argument_))
    throw std::invalid_argument("ArgPack storage does not match the kernel and device");
  const auto &packs = storage_->layout().storages();
  allocations_.reserve(packs.size());
  // Qualify the entire subtree before writing any launch packet state.
  for (std::size_t i = 0; i < packs.size(); ++i) {
    auto physical = packs[i].origin.physical;
    physical.front() = argument_.front();
    const auto parameter = kernel.nested_parameters.find(physical);
    if (parameter == kernel.nested_parameters.end() ||
        !parameter->second.is_argpack || context.argpack_ptrs.count(physical) ||
        context.device_allocation_type.count(physical))
      throw std::invalid_argument("ArgPack subtree is already bound or invalid");
    allocations_.push_back(storage_->allocation(i));
    paths_.push_back(std::move(physical));
  }
  for (std::size_t i = 0; i < packs.size(); ++i)
    context.set_arg_argpack_impl(paths_[i], &allocations_[i],
                                 storage_->address(i));
}

void NativeArgPackKernelBinding::validate(
    Arch arch,
    Device &device,
    const CallableBase &kernel,
    const LaunchContextBuilder &context,
    const NativeRecordingState &) const {
  if (kernel_ != &kernel || context_ != &context || storage_->arch() != arch ||
      &storage_->device() != &device)
    throw std::invalid_argument("ArgPack binding owner changed after qualification");
  const auto &packs = storage_->layout().storages();
  for (std::size_t i = 0; i < packs.size(); ++i) {
    const auto &physical = paths_[i];
    const auto pointer = context.array_ptrs.find(physical);
    const auto kind = context.device_allocation_type.find(physical);
    if (pointer == context.array_ptrs.end() || pointer->second != &allocations_[i] ||
        kind == context.device_allocation_type.end() ||
        kind->second != LaunchContextBuilder::DevAllocType::kResolvedArgPack ||
        context.argpack_ptrs.count(physical))
      throw std::invalid_argument("ArgPack binding changed after qualification");
  }
}
}  // namespace taichi::lang
