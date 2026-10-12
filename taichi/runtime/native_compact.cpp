#include "taichi/runtime/native_operation.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <tuple>
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
      : arch_(arch), device_(device), operation_(std::move(operation)) {
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

  // flags, count, then (input, output) per column.
  std::vector<void *> pointers;

 protected:
  Arch arch_;
  Device &device_;
  CompactOperation operation_;
  std::size_t bytes_{0};
};

struct AccessRange {
  std::uint64_t storage, begin, end;
  bool write;
};

void validate_ranges(std::vector<AccessRange> ranges) {
  std::sort(ranges.begin(), ranges.end(), [](const auto &a, const auto &b) {
    return std::tie(a.storage, a.begin) < std::tie(b.storage, b.begin);
  });
  std::uint64_t storage = 0, any_end = 0, write_end = 0;
  for (const auto &range : ranges) {
    if (range.begin == range.end)
      continue;
    if (range.storage != storage) {
      storage = range.storage;
      any_end = write_end = 0;
    }
    if (range.begin < (range.write ? any_end : write_end))
      throw std::invalid_argument("Compact writable ranges must not overlap");
    any_end = std::max(any_end, range.end);
    if (range.write)
      write_end = std::max(write_end, range.end);
  }
}

void validate_pointers(const std::vector<void *> &pointers,
                       const std::vector<std::size_t> &sizes) {
  std::vector<AccessRange> ranges;
  for (std::size_t i = 0; i < pointers.size(); ++i) {
    const auto a = reinterpret_cast<std::uintptr_t>(pointers[i]);
    if (!a || a % 4 ||
        sizes[i] > std::numeric_limits<std::uintptr_t>::max() - a)
      throw std::invalid_argument(
          "Compact requires live 4-byte-aligned storage");
    ranges.push_back({0, a, a + sizes[i], bool(i % 2)});
  }
  validate_ranges(std::move(ranges));
}

#ifdef TI_WITH_LLVM
class CpuCompact final : public CompactBase {
 public:
  using CompactBase::CompactBase;
  void prepare() {
    if (operation_.columns.size() > 1) {
      selected_.resize(operation_.size);
      bytes_ = selected_.size() * sizeof(std::uint32_t);
    }
  }
  template <std::size_t Width>
  void gather(void *output,
              const void *input,
              std::size_t count,
              std::size_t width) {
    const auto bytes = Width ? Width : width;
    auto *destination = static_cast<char *>(output);
    const auto *source = static_cast<const char *>(input);
    for (std::size_t i = 0; i < count; ++i)
      std::memcpy(destination + i * bytes, source + selected_[i] * bytes,
                  bytes);
  }
  void run_cpu() override {
    const auto *flags = static_cast<const std::int32_t *>(pointers[0]);
    std::int32_t selected = 0;
    if (operation_.columns.size() == 1) {
      const auto bytes = operation_.columns[0].item_bytes;
      const auto *source = static_cast<const char *>(pointers[2]);
      auto *destination = static_cast<char *>(pointers[3]);
      for (std::size_t i = 0; i < operation_.size; ++i) {
        if (flags[i]) {
          std::memcpy(destination + std::size_t(selected) * bytes,
                      source + i * bytes, bytes);
          ++selected;
        }
      }
    } else {
      for (std::size_t i = 0; i < operation_.size; ++i)
        if (flags[i])
          selected_[selected++] = static_cast<std::uint32_t>(i);
      for (std::size_t c = 0; c < operation_.columns.size(); ++c) {
        const auto bytes = operation_.columns[c].item_bytes;
        auto *output = pointers[3 + c * 2];
        auto *input = pointers[2 + c * 2];
        if (bytes == 4)
          gather<4>(output, input, selected, bytes);
        else if (bytes == 8)
          gather<8>(output, input, selected, bytes);
        else
          gather<0>(output, input, selected, bytes);
      }
    }
    *static_cast<std::int32_t *>(pointers[1]) = selected;
  }

 private:
  std::vector<std::uint32_t> selected_;
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
      CUDADriver::get_instance().memsetd32_async(pointers[1], 0, 1, stream);
      return;
    }
    for (std::size_t c = 0; c < operation_.columns.size(); ++c)
      cuda::driver_compact_with_workspace(
          pointers[2 + c * 2], pointers[0], pointers[3 + c * 2], pointers[1],
          static_cast<int>(operation_.size),
          static_cast<int>(operation_.columns[c].item_bytes / 4),
          workspace_pointer_, stream, c != 0);
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
      operation.columns.empty())
    throw std::invalid_argument("Compact requires columns and an i32 count");
  std::vector<DevicePtr> bindings{operation.flags, operation.count};
  std::vector<std::size_t> sizes{operation.size * sizeof(std::int32_t),
                                 sizeof(std::int32_t)};
  for (const auto &column : operation.columns) {
    if (!column.item_bytes || column.item_bytes % 4 ||
        column.item_bytes / 4 > std::numeric_limits<int>::max() ||
        operation.size >
            std::numeric_limits<std::size_t>::max() / column.item_bytes)
      throw std::invalid_argument("Compact exceeds its raw-word record ABI");
    bindings.insert(bindings.end(), {column.values, column.output});
    sizes.insert(sizes.end(), 2, operation.size * column.item_bytes);
  }
  std::vector<AccessRange> ranges;
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const auto &binding = bindings[i];
    if (binding.device != &device || binding.offset % 4 ||
        sizes[i] > std::numeric_limits<std::uint64_t>::max() - binding.offset)
      throw std::invalid_argument(
          "Compact requires aligned storage on one device");
    ranges.push_back({binding.alloc_id, binding.offset,
                      binding.offset + sizes[i], bool(i % 2)});
  }
  validate_ranges(std::move(ranges));
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch)) {
    auto *cpu_device = dynamic_cast<cpu::CpuDevice *>(&device);
    if (!cpu_device)
      throw std::invalid_argument("Compact requires a CPU device");
    auto result = std::make_shared<CpuCompact>(arch, device, operation);
    result->pointers.resize(bindings.size());
    for (std::size_t i = 0; i < bindings.size(); ++i) {
      if (cpu_device->map_range_for_cpu_native(bindings[i], sizes[i],
                                               &result->pointers[i]) !=
          RhiResult::success)
        throw std::invalid_argument("Compact CPU storage range is invalid");
    }
    validate_pointers(result->pointers, sizes);
    result->prepare();
    return result;
  }
#endif
#ifdef TI_WITH_CUDA
  if (arch == Arch::cuda) {
    auto *cuda_device = dynamic_cast<cuda::CudaDevice *>(&device);
    if (!cuda_device)
      throw std::invalid_argument("Compact requires a CUDA device");
    auto result = std::make_shared<CudaCompact>(arch, device, operation);
    result->pointers.resize(bindings.size());
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
