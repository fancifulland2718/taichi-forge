#include "taichi/runtime/native_operation.h"

#include <array>
#include <cstring>
#include <limits>
#include "taichi/rhi/arch.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

namespace taichi::lang {
namespace {
class CompactBase : public NativeOperation {
 public:
  CompactBase(Arch arch, Device &device, CompactOperation operation)
      : arch_(arch), device_(device), operation_(operation) {
  }
  Arch arch() const override {
    return arch_;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return bytes_;
  }

  std::array<void *, 4> pointers{};

 protected:
  Arch arch_;
  Device &device_;
  CompactOperation operation_;
  std::size_t bytes_{0};
};

void validate_pointers(const std::array<void *, 4> &pointers,
                       const std::array<std::size_t, 4> &sizes) {
  for (std::size_t i = 0; i < pointers.size(); ++i) {
    const auto a = reinterpret_cast<std::uintptr_t>(pointers[i]);
    if (!a || a % 4 ||
        sizes[i] > std::numeric_limits<std::uintptr_t>::max() - a)
      throw std::invalid_argument(
          "Compact requires live 4-byte-aligned storage");
    for (std::size_t j = 0; j < i; ++j) {
      const auto b = reinterpret_cast<std::uintptr_t>(pointers[j]);
      if ((i >= 2 || j >= 2) && sizes[i] && sizes[j] &&
          (a <= b ? b - a < sizes[i] : a - b < sizes[j]))
        throw std::invalid_argument("Compact writable ranges must not overlap");
    }
  }
}

#ifdef TI_WITH_LLVM
class CpuCompact final : public CompactBase {
 public:
  using CompactBase::CompactBase;
  void run_cpu() override {
    const auto *flags = static_cast<const std::int32_t *>(pointers[1]);
    const auto *source = static_cast<const char *>(pointers[0]);
    auto *destination = static_cast<char *>(pointers[2]);
    std::int32_t selected = 0;
    for (std::size_t i = 0; i < operation_.size; ++i) {
      if (flags[i]) {
        std::memcpy(destination + std::size_t(selected) * operation_.item_bytes,
                    source + i * operation_.item_bytes, operation_.item_bytes);
        ++selected;
      }
    }
    *static_cast<std::int32_t *>(pointers[3]) = selected;
  }
};
#endif
#ifdef TI_WITH_CUDA
class CudaCompact final : public CompactBase {
 public:
  using CompactBase::CompactBase;
  void prepare() {
    if (!operation_.size)
      return;
    auto &context = CUDAContext::get_instance();
    auto guard = context.get_guard();
    cuda::driver_prepare_compact();
    bytes_ =
        cuda::driver_compact_workspace_bytes(static_cast<int>(operation_.size));
    Device::AllocParams params{};
    params.size = bytes_;
    params.usage = AllocUsage::Storage;
    auto [allocation, result] = device_.allocate_memory_unique(params);
    if (result != RhiResult::success)
      throw std::runtime_error(
          "Unable to allocate fixed CUDA compact workspace");
    workspace_ = std::move(allocation);
    workspace_pointer_ =
        static_cast<cuda::CudaDevice &>(device_).get_memory_addr(*workspace_);
  }
  void record_cuda(void *stream) override {
    if (!operation_.size) {
      CUDADriver::get_instance().memsetd32_async(pointers[3], 0, 1, stream);
      return;
    }
    cuda::driver_compact_with_workspace(
        pointers[0], pointers[1], pointers[2], pointers[3],
        static_cast<int>(operation_.size),
        static_cast<int>(operation_.item_bytes / 4), workspace_pointer_,
        stream);
  }

 private:
  DeviceAllocationUnique workspace_;
  void *workspace_pointer_{nullptr};
};
#endif
}  // namespace

std::shared_ptr<NativeOperation> prepare_compact_operation(
    Arch arch,
    Device &device,
    const CompactOperation &operation) {
  if (operation.size > std::numeric_limits<std::int32_t>::max() ||
      !operation.item_bytes || operation.item_bytes % 4 ||
      operation.item_bytes / 4 > std::numeric_limits<int>::max() ||
      operation.size >
          std::numeric_limits<std::size_t>::max() / operation.item_bytes)
    throw std::invalid_argument(
        "Compact exceeds its i32 count or raw-word record ABI");
  const std::array<DevicePtr, 4> bindings{operation.values, operation.flags,
                                          operation.output, operation.count};
  const std::array<std::size_t, 4> sizes{operation.size * operation.item_bytes,
                                         operation.size * sizeof(std::int32_t),
                                         operation.size * operation.item_bytes,
                                         sizeof(std::int32_t)};
  for (const auto &binding : bindings) {
    if (binding.device != &device || binding.offset % 4)
      throw std::invalid_argument(
          "Compact requires aligned storage on one device");
  }
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch)) {
    auto *cpu_device = dynamic_cast<cpu::CpuDevice *>(&device);
    if (!cpu_device)
      throw std::invalid_argument("Compact requires a CPU device");
    auto result = std::make_shared<CpuCompact>(arch, device, operation);
    for (std::size_t i = 0; i < bindings.size(); ++i) {
      if (cpu_device->map_range_for_cpu_native(bindings[i], sizes[i],
                                               &result->pointers[i]) !=
          RhiResult::success)
        throw std::invalid_argument("Compact CPU storage range is invalid");
    }
    validate_pointers(result->pointers, sizes);
    return result;
  }
#endif
#ifdef TI_WITH_CUDA
  if (arch == Arch::cuda) {
    auto *cuda_device = dynamic_cast<cuda::CudaDevice *>(&device);
    if (!cuda_device)
      throw std::invalid_argument("Compact requires a CUDA device");
    auto result = std::make_shared<CudaCompact>(arch, device, operation);
    for (std::size_t i = 0; i < bindings.size(); ++i)
      result->pointers[i] =
          static_cast<char *>(cuda_device->get_memory_addr(bindings[i])) +
          bindings[i].offset;
    validate_pointers(result->pointers, sizes);
    result->prepare();
    return result;
  }
#endif
#ifdef TI_WITH_VULKAN
  if (arch == Arch::vulkan)
    return gfx::prepare_vulkan_compact_operation(device, operation);
#endif
  throw std::invalid_argument("Compact requires CPU, CUDA or Vulkan");
}
}  // namespace taichi::lang
