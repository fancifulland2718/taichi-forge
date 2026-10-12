#include "taichi/runtime/native_operation_registry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>
#include "taichi/rhi/arch.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

namespace taichi::lang {
namespace {
class SortBase : public NativeOperation {
 public:
  SortBase(Arch arch, Device &device, SortOperation operation)
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
  std::array<void *, 2> pointers{};

 protected:
  Arch arch_;
  Device &device_;
  SortOperation operation_;
  std::size_t bytes_{0};
};

#ifdef TI_WITH_LLVM
class CpuSort final : public SortBase {
 public:
  using SortBase::SortBase;
  void prepare(std::size_t key_bytes) {
    key_bytes_ = key_bytes;
    if (operation_.size <= 1)
      return;
    order_.resize(operation_.size);
    keys_.resize(operation_.size * key_bytes);
    values_.resize(operation_.size * operation_.value_bytes);
    bytes_ =
        order_.size() * sizeof(std::uint32_t) + keys_.size() + values_.size();
  }
  template <typename T>
  void sort_indices() {
    const auto *keys = static_cast<const T *>(pointers[0]);
    std::sort(
        order_.begin(), order_.end(), [&](std::uint32_t a, std::uint32_t b) {
          if constexpr (std::is_floating_point_v<T>) {
            const bool a_nan = std::isnan(keys[a]), b_nan = std::isnan(keys[b]);
            if (a_nan != b_nan)
              return !a_nan;
            if (a_nan)
              return a < b;
          }
          return keys[a] < keys[b] || (!(keys[b] < keys[a]) && a < b);
        });
  }
  void run_cpu() override {
    if (operation_.size <= 1)
      return;
    std::iota(order_.begin(), order_.end(), 0u);
    switch (operation_.key_type) {
      case NativeSortKeyType::u32:
        sort_indices<std::uint32_t>();
        break;
      case NativeSortKeyType::i32:
        sort_indices<std::int32_t>();
        break;
      case NativeSortKeyType::f32:
        sort_indices<float>();
        break;
      case NativeSortKeyType::u64:
        sort_indices<std::uint64_t>();
        break;
      case NativeSortKeyType::i64:
        sort_indices<std::int64_t>();
        break;
      case NativeSortKeyType::f64:
        sort_indices<double>();
        break;
    }
    for (std::size_t i = 0; i < operation_.size; ++i) {
      std::memcpy(keys_.data() + i * key_bytes_,
                  static_cast<char *>(pointers[0]) + order_[i] * key_bytes_,
                  key_bytes_);
      if (operation_.value_bytes)
        std::memcpy(values_.data() + i * operation_.value_bytes,
                    static_cast<char *>(pointers[1]) +
                        order_[i] * operation_.value_bytes,
                    operation_.value_bytes);
    }
    std::memcpy(pointers[0], keys_.data(), keys_.size());
    if (!values_.empty())
      std::memcpy(pointers[1], values_.data(), values_.size());
  }

 private:
  std::size_t key_bytes_{0};
  std::vector<std::uint32_t> order_;
  std::vector<char> keys_, values_;
};
#endif

#ifdef TI_WITH_CUDA
class CudaSort final : public SortBase {
 public:
  using SortBase::SortBase;
  void prepare() {
    if (operation_.size <= 1)
      return;
    auto guard = CUDAContext::get_instance().get_guard();
    cuda::driver_prepare_sort();
    bytes_ = cuda::driver_stable_sort_workspace_bytes(
        static_cast<int>(operation_.size),
        static_cast<cuda::CudaDriverSortKeyType>(operation_.key_type),
        static_cast<int>(operation_.value_bytes / 4));
    Device::AllocParams params{};
    params.size = bytes_;
    params.usage = AllocUsage::Storage;
    auto [allocation, status] = device_.allocate_memory_unique(params);
    if (status != RhiResult::success)
      throw std::runtime_error("Unable to allocate fixed CUDA sort workspace");
    workspace_ = std::move(allocation);
    workspace_pointer_ =
        static_cast<cuda::CudaDevice &>(device_).get_memory_addr(*workspace_);
  }
  void record_cuda(void *stream) override {
    if (operation_.size > 1)
      cuda::driver_stable_sort_with_workspace(
          pointers[0], pointers[1], static_cast<int>(operation_.size),
          static_cast<cuda::CudaDriverSortKeyType>(operation_.key_type),
          static_cast<int>(operation_.value_bytes / 4), workspace_pointer_,
          stream);
  }

 private:
  DeviceAllocationUnique workspace_;
  void *workspace_pointer_{nullptr};
};
#endif
void validate_sort(Device &device, const SortOperation &operation) {
  const auto type = static_cast<int>(operation.key_type);
  const std::size_t key_bytes = type >= 3 ? 8 : 4;
  if (type < 0 || type > 5 ||
      operation.size > std::numeric_limits<int>::max() ||
      operation.keys.device != &device || operation.keys.offset % key_bytes ||
      (operation.values.device &&
       (operation.values.device != &device || operation.values.offset % 4 ||
        !operation.value_bytes || operation.value_bytes % 4 ||
        !operation.value_scalar_bytes ||
        operation.value_bytes % operation.value_scalar_bytes)) ||
      (!operation.values.device && operation.value_bytes) ||
      operation.value_bytes / 4 > std::numeric_limits<int>::max() ||
      operation.size >
          std::numeric_limits<std::size_t>::max() /
              (key_bytes + operation.value_bytes + sizeof(std::uint32_t)))
    throw std::invalid_argument(
        "Sort exceeds its key, record or i32 index ABI");
}

template <typename Map>
void map_sort(SortBase &result, const SortOperation &operation, Map map) {
  const std::size_t key_bytes =
      static_cast<int>(operation.key_type) >= 3 ? 8 : 4;
  const std::array<DevicePtr, 2> bindings{operation.keys, operation.values};
  const std::array<std::size_t, 2> sizes{
      operation.size * key_bytes, operation.size * operation.value_bytes};
  for (std::size_t i = 0; i < bindings.size(); ++i)
    if (bindings[i].device)
      result.pointers[i] = map(bindings[i], sizes[i]);
  const auto a = reinterpret_cast<std::uintptr_t>(result.pointers[0]);
  const auto b = reinterpret_cast<std::uintptr_t>(result.pointers[1]);
  if (!a || a % key_bytes || (operation.values.device && (!b || b % 4)) ||
      sizes[0] > std::numeric_limits<std::uintptr_t>::max() - a ||
      sizes[1] > std::numeric_limits<std::uintptr_t>::max() - b ||
      (sizes[0] && sizes[1] && (a <= b ? b - a < sizes[0] : a - b < sizes[1])))
    throw std::invalid_argument(
        "Sort key and payload ranges must be aligned and nonoverlapping");
}

#ifdef TI_WITH_LLVM
std::shared_ptr<NativeOperation>
prepare_cpu_sort(Arch arch, Device &device, const SortOperation &operation) {
  auto *cpu_device = dynamic_cast<cpu::CpuDevice *>(&device);
  if (!cpu_device)
    throw std::invalid_argument("Sort requires a CPU device");
  auto result = std::make_shared<CpuSort>(arch, device, operation);
  map_sort(*result, operation, [&](DevicePtr ptr, std::size_t bytes) {
    void *mapped = nullptr;
    if (cpu_device->map_range_for_cpu_native(ptr, bytes, &mapped) !=
        RhiResult::success)
      throw std::invalid_argument("Sort CPU storage range is invalid");
    return mapped;
  });
  result->prepare(static_cast<int>(operation.key_type) >= 3 ? 8 : 4);
  return result;
}
#endif
#ifdef TI_WITH_CUDA
std::shared_ptr<NativeOperation>
prepare_cuda_sort(Arch arch, Device &device, const SortOperation &operation) {
  auto *cuda_device = dynamic_cast<cuda::CudaDevice *>(&device);
  if (!cuda_device)
    throw std::invalid_argument("Sort requires a CUDA device");
  auto result = std::make_shared<CudaSort>(arch, device, operation);
  map_sort(*result, operation, [&](DevicePtr ptr, std::size_t) {
    return static_cast<char *>(cuda_device->get_memory_addr(ptr)) + ptr.offset;
  });
  result->prepare();
  return result;
}
#endif
}  // namespace

void register_sort_operation(NativeOperationRegistry &registry) {
  registry.define<SortOperation>(validate_sort);
#ifdef TI_WITH_LLVM
  registry.register_factory<SortOperation>(
      host_arch(), NativeOperationMode::cpu, prepare_cpu_sort);
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<SortOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_cuda_sort);
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<SortOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const SortOperation &operation) {
        return gfx::prepare_vulkan_sort_operation(device, operation);
      });
#endif
}

std::shared_ptr<NativeOperation> prepare_sort_operation(
    Arch arch,
    Device &device,
    const SortOperation &operation) {
  return native_operation_registry().prepare(arch, device, operation);
}
}  // namespace taichi::lang
