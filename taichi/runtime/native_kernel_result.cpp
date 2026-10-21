#include "taichi/runtime/native_kernel_result.h"

#include <stdexcept>
#include <cstring>
#include <vector>
#include "taichi/aot/value_layout.h"
#include "taichi/program/callable.h"
#include "taichi/runtime/native_array_contract.h"

namespace taichi::lang {
NativeKernelResultStorage::NativeKernelResultStorage(
    Arch arch, Device &device, const CallableBase &kernel)
    : NativeKernelResultStorage(arch, device, kernel, Use::host_result) {
}

NativeKernelResultStorage::NativeKernelResultStorage(
    Arch arch, Device &device, const CallableBase &kernel, Use use)
    : arch_(arch), device_(device), type_(kernel.ret_type), bytes_(kernel.ret_size),
      use_(use) {
  const bool host = arch_is_cpu(arch) || arch == Arch::cuda;
  if (!host && arch != Arch::vulkan)
    throw std::invalid_argument("Kernel results require CPU, CUDA or Vulkan");
  if (type_) {
    // Validate the actual compiled offsets and all pure-value leaves before
    // allocating. Return pointers/resources are not a serializable value ABI.
    const aot::ValueLayout layout(type_, bytes_, host ? aot::ValueStorage::Host
                                                    : aot::ValueStorage::Gfx);
  } else if (bytes_) {
    throw std::invalid_argument("Kernel result is missing its compiled type");
  }
  if (!bytes_)
    return;
  auto [allocation, status] = device.allocate_memory_unique(
      {bytes_, false, arch == Arch::vulkan && observable(), false, AllocUsage::Storage});
  if (status != RhiResult::success)
    throw std::runtime_error("Unable to allocate kernel result storage");
  allocation_ = std::move(allocation);
  const auto range = resolve_native_storage_range(arch_, allocation_->get_ptr(), bytes_);
  if (host)
    address_ = range.memory_offset;
  // Discarded results are write-only scratch. No observer can read their
  // padding, so they need neither cold initialization nor an upload/wait.
  if (!observable())
    return;
  // Kernels write leaves, not padding. Initialize only on cold construction so
  // raw C result copies cannot reveal earlier allocation contents.
  const std::vector<std::uint8_t> zero(bytes_, 0);
  auto pointer = allocation_->get_ptr();
  const void *source = zero.data();
  auto size = bytes_;
  if (device_.upload_data(&pointer, &source, &size, 1) != RhiResult::success)
    throw std::runtime_error("Unable to initialize kernel result storage");
}

bool NativeKernelResultStorage::matches(Arch arch, Device &device,
                                       const CallableBase &kernel) const {
  return arch_ == arch && &device_ == &device && type_ == kernel.ret_type &&
         bytes_ == kernel.ret_size;
}

void NativeKernelResultStorage::read_completed(void *destination,
                                               std::size_t bytes) const {
  if (!observable())
    throw std::invalid_argument("Discarded kernel results have no host consumer");
  if (bytes != bytes_ || (bytes && !destination))
    throw std::invalid_argument("Kernel result destination must match compiled size");
  if (!bytes)
    return;
  if (arch_ == Arch::vulkan) {
    // The prepared recording supplies ShaderWrite -> HostRead visibility and
    // the caller has waited. Reuse ordinary GFX return-buffer mapping instead
    // of allocating staging and submitting a second download for every read.
    void *source = nullptr;
    if (device_.map(*allocation_, &source) != RhiResult::success)
      throw std::runtime_error("Unable to map completed kernel result storage");
    std::memcpy(destination, source, bytes);
    device_.unmap(*allocation_);
    return;
  }
  auto source = allocation_->get_ptr();
  if (device_.readback_data(&source, &destination, &bytes, 1) != RhiResult::success)
    throw std::runtime_error("Unable to read completed kernel result storage");
}

std::shared_ptr<NativeKernelResultStorage> NativeKernelResultPool::acquire(
    const CallableBase &kernel) {
  if (!kernel.ret_size)
    return {};
  auto &layouts = frames_[kernel.ret_type];
  const auto found = layouts.find(kernel.ret_size);
  if (found != layouts.end())
    return found->second;
  auto result = std::make_shared<NativeKernelResultStorage>(
      arch_, device_, kernel, NativeKernelResultStorage::Use::discarded);
  resources_.push_back(result);
  try {
    layouts.emplace(kernel.ret_size, result);
  } catch (...) {
    resources_.pop_back();
    throw;
  }
  return result;
}
}  // namespace taichi::lang
