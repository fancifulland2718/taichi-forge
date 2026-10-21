#include "taichi/runtime/native_kernel_result.h"

#include <stdexcept>
#include <vector>
#include "taichi/aot/value_layout.h"
#include "taichi/program/callable.h"
#include "taichi/runtime/native_array_contract.h"

namespace taichi::lang {
NativeKernelResultStorage::NativeKernelResultStorage(
    Arch arch, Device &device, const CallableBase &kernel)
    : arch_(arch), device_(device), type_(kernel.ret_type), bytes_(kernel.ret_size) {
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
      {bytes_, false, false, false, AllocUsage::Storage});
  if (status != RhiResult::success)
    throw std::runtime_error("Unable to allocate kernel result storage");
  allocation_ = std::move(allocation);
  const auto range = resolve_native_storage_range(arch_, allocation_->get_ptr(), bytes_);
  if (host)
    address_ = range.memory_offset;
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
  if (bytes != bytes_ || (bytes && !destination))
    throw std::invalid_argument("Kernel result destination must match compiled size");
  if (!bytes)
    return;
  auto source = allocation_->get_ptr();
  if (device_.readback_data(&source, &destination, &bytes, 1) != RhiResult::success)
    throw std::runtime_error("Unable to read completed kernel result storage");
}
}  // namespace taichi::lang
