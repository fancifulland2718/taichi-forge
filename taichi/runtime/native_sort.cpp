#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_sort.h"

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
#include "taichi/rhi/cpu/stable_sort.h"
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
      : arch_(arch),
        device_(device),
        operation_(operation),
        workspace_(operation.workspace
                       ? operation.workspace
                       : prepare_sort_workspace(
                             arch,
                             device,
                             SortWorkspaceDescription::from(operation))) {
    if (workspace_->arch() != arch || &workspace_->device() != &device ||
        !workspace_->matches(operation))
      throw std::invalid_argument(
          "Sort workspace does not match its operation");
  }
  Arch arch() const override {
    return arch_;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return operation_.workspace ? 0 : workspace_->workspace_bytes();
  }
  std::array<void *, 2> pointers{};

 protected:
  Arch arch_;
  Device &device_;
  SortOperation operation_;
  std::shared_ptr<NativeSortWorkspace> workspace_;
  template <typename T>
  T *typed_workspace() {
    auto *result = dynamic_cast<T *>(workspace_.get());
    if (!result)
      throw std::invalid_argument("Sort workspace backend/type mismatch");
    return result;
  }
};

#ifdef TI_WITH_LLVM
template <typename KeyT, typename ValueT>
class CpuTypedSortWorkspace final : public NativeSortWorkspace {
 public:
  CpuTypedSortWorkspace(Arch arch,
                        Device &device,
                        const SortWorkspaceDescription &description)
      : NativeSortWorkspace(arch, device, description) {
    storage.prepare(description.size, description.value_bytes != 0);
  }
  std::size_t workspace_bytes() const override {
    return storage.bytes();
  }
  cpu::StableSortWorkspace<KeyT, ValueT> storage;
};

class CpuRawSortWorkspace final : public NativeSortWorkspace {
 public:
  CpuRawSortWorkspace(Arch arch,
                      Device &device,
                      const SortWorkspaceDescription &description)
      : NativeSortWorkspace(arch, device, description) {
    if (description.size <= 1)
      return;
    order.resize(description.size);
    keys.resize(description.size *
                (static_cast<int>(description.key_type) >= 3 ? 8 : 4));
    values.resize(description.size * description.value_bytes);
  }
  std::size_t workspace_bytes() const override {
    return order.size() * sizeof(std::uint32_t) + keys.size() + values.size();
  }
  std::vector<std::uint32_t> order;
  std::vector<char> keys, values;
};

template <typename KeyT, typename ValueT>
class CpuTypedSort final : public SortBase {
 public:
  CpuTypedSort(Arch arch, Device &device, const SortOperation &operation)
      : SortBase(arch, device, operation),
        typed_(typed_workspace<CpuTypedSortWorkspace<KeyT, ValueT>>()) {
  }
  void run_cpu() override {
    cpu::cpu_stable_sort_impl(static_cast<KeyT *>(pointers[0]),
                              static_cast<ValueT *>(pointers[1]),
                              operation_.size, false, 0, &typed_->storage);
  }

 private:
  CpuTypedSortWorkspace<KeyT, ValueT> *typed_;
};

class CpuSort final : public SortBase {
 public:
  CpuSort(Arch arch, Device &device, const SortOperation &operation)
      : SortBase(arch, device, operation),
        typed_(typed_workspace<CpuRawSortWorkspace>()),
        key_bytes_(static_cast<int>(operation.key_type) >= 3 ? 8 : 4) {
  }
  template <typename T>
  void sort_indices() {
    auto &order_ = typed_->order;
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
    auto &order_ = typed_->order;
    auto &keys_ = typed_->keys;
    auto &values_ = typed_->values;
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
  CpuRawSortWorkspace *typed_;
  std::size_t key_bytes_;
};
#endif

#ifdef TI_WITH_CUDA
class CudaSortWorkspace final : public NativeSortWorkspace {
 public:
  CudaSortWorkspace(Arch arch,
                    Device &device,
                    const SortWorkspaceDescription &description)
      : NativeSortWorkspace(arch, device, description) {
    if (description.size <= 1)
      return;
    auto guard = CUDAContext::get_instance().get_guard();
    cuda::driver_prepare_sort();
    bytes_ = cuda::driver_stable_sort_workspace_bytes(
        static_cast<int>(description.size),
        static_cast<cuda::CudaDriverSortKeyType>(description.key_type),
        static_cast<int>(description.value_bytes / 4));
    Device::AllocParams params{};
    params.size = bytes_;
    params.usage = AllocUsage::Storage;
    auto [allocation, status] = device_.allocate_memory_unique(params);
    if (status != RhiResult::success)
      throw std::runtime_error("Unable to allocate fixed CUDA sort workspace");
    storage_ = std::move(allocation);
    pointer =
        static_cast<cuda::CudaDevice &>(device_).get_memory_addr(*storage_);
  }
  std::size_t workspace_bytes() const override {
    return bytes_;
  }
  void *pointer{nullptr};

 private:
  std::size_t bytes_{0};
  DeviceAllocationUnique storage_;
};

class CudaSort final : public SortBase {
 public:
  CudaSort(Arch arch, Device &device, const SortOperation &operation)
      : SortBase(arch, device, operation),
        typed_(typed_workspace<CudaSortWorkspace>()) {
  }
  void record_cuda(void *stream) override {
    if (operation_.size > 1)
      cuda::driver_stable_sort_with_workspace(
          pointers[0], pointers[1], static_cast<int>(operation_.size),
          static_cast<cuda::CudaDriverSortKeyType>(operation_.key_type),
          static_cast<int>(operation_.value_bytes / 4), typed_->pointer,
          stream);
  }

 private:
  CudaSortWorkspace *typed_;
};
#endif
void validate_sort_workspace(Device &, const SortWorkspaceDescription &d) {
  const auto type = static_cast<int>(d.key_type);
  const std::size_t key_bytes = type >= 3 ? 8 : 4;
  if (type < 0 || type > 5 || d.size > std::numeric_limits<int>::max() ||
      d.value_bytes % 4 ||
      (d.value_bytes &&
       (!d.value_scalar_bytes || d.value_bytes % d.value_scalar_bytes)) ||
      (!d.value_bytes && d.value_scalar_bytes) ||
      d.value_bytes / 4 > std::numeric_limits<int>::max() ||
      d.size > std::numeric_limits<std::size_t>::max() /
                   (key_bytes + d.value_bytes + sizeof(std::uint32_t)))
    throw std::invalid_argument(
        "Sort workspace exceeds its record or i32 index ABI");
}

void validate_sort(Device &device, const SortOperation &operation) {
  validate_sort_workspace(device, SortWorkspaceDescription::from(operation));
  if (operation.workspace && (&operation.workspace->device() != &device ||
                              !operation.workspace->matches(operation)))
    throw std::invalid_argument("Sort workspace does not match its operation");
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
template <typename KeyT>
std::shared_ptr<NativeResource> cpu_workspace_for_key(
    Arch arch,
    Device &device,
    const SortWorkspaceDescription &d) {
  if (!d.value_bytes || (d.value_bytes == 4 && d.value_scalar_bytes == 4))
    return std::make_shared<CpuTypedSortWorkspace<KeyT, std::uint32_t>>(
        arch, device, d);
  if (d.value_bytes == 8 && d.value_scalar_bytes == 8)
    return std::make_shared<CpuTypedSortWorkspace<KeyT, std::uint64_t>>(
        arch, device, d);
  return std::make_shared<CpuRawSortWorkspace>(arch, device, d);
}
std::shared_ptr<NativeResource> prepare_cpu_sort_workspace(
    Arch arch,
    Device &device,
    const SortWorkspaceDescription &d) {
  if (!dynamic_cast<cpu::CpuDevice *>(&device))
    throw std::invalid_argument("Sort workspace requires a CPU device");
  switch (d.key_type) {
    case NativeSortKeyType::u32:
      return cpu_workspace_for_key<std::uint32_t>(arch, device, d);
    case NativeSortKeyType::i32:
      return cpu_workspace_for_key<std::int32_t>(arch, device, d);
    case NativeSortKeyType::f32:
      return cpu_workspace_for_key<float>(arch, device, d);
    case NativeSortKeyType::u64:
      return cpu_workspace_for_key<std::uint64_t>(arch, device, d);
    case NativeSortKeyType::i64:
      return cpu_workspace_for_key<std::int64_t>(arch, device, d);
    case NativeSortKeyType::f64:
      return cpu_workspace_for_key<double>(arch, device, d);
  }
  throw std::invalid_argument("Unsupported CPU sort workspace key type");
}

template <typename KeyT>
std::shared_ptr<SortBase> make_cpu_sort(Arch arch,
                                        Device &device,
                                        const SortOperation &operation) {
  if (!operation.value_bytes ||
      (operation.value_bytes == 4 && operation.value_scalar_bytes == 4))
    return std::make_shared<CpuTypedSort<KeyT, std::uint32_t>>(arch, device,
                                                               operation);
  if (operation.value_bytes == 8 && operation.value_scalar_bytes == 8)
    return std::make_shared<CpuTypedSort<KeyT, std::uint64_t>>(arch, device,
                                                               operation);
  return std::make_shared<CpuSort>(arch, device, operation);
}

std::shared_ptr<NativeOperation>
prepare_cpu_sort(Arch arch, Device &device, const SortOperation &operation) {
  auto *cpu_device = dynamic_cast<cpu::CpuDevice *>(&device);
  if (!cpu_device)
    throw std::invalid_argument("Sort requires a CPU device");
  std::shared_ptr<SortBase> result;
  switch (operation.key_type) {
    case NativeSortKeyType::u32:
      result = make_cpu_sort<std::uint32_t>(arch, device, operation);
      break;
    case NativeSortKeyType::i32:
      result = make_cpu_sort<std::int32_t>(arch, device, operation);
      break;
    case NativeSortKeyType::f32:
      result = make_cpu_sort<float>(arch, device, operation);
      break;
    case NativeSortKeyType::u64:
      result = make_cpu_sort<std::uint64_t>(arch, device, operation);
      break;
    case NativeSortKeyType::i64:
      result = make_cpu_sort<std::int64_t>(arch, device, operation);
      break;
    case NativeSortKeyType::f64:
      result = make_cpu_sort<double>(arch, device, operation);
      break;
  }
  map_sort(*result, operation, [&](DevicePtr ptr, std::size_t bytes) {
    void *mapped = nullptr;
    if (cpu_device->map_range_for_cpu_native(ptr, bytes, &mapped) !=
        RhiResult::success)
      throw std::invalid_argument("Sort CPU storage range is invalid");
    return mapped;
  });
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
  return result;
}
#endif
}  // namespace

void register_sort_operation(NativeOperationRegistry &registry) {
  registry.define<SortWorkspaceDescription>(validate_sort_workspace,
                                            NativeObjectKind::resource);
  registry.define<SortOperation>(validate_sort);
#ifdef TI_WITH_LLVM
  registry.register_factory<SortWorkspaceDescription>(
      host_arch(), NativeOperationMode::cpu, prepare_cpu_sort_workspace);
  registry.register_factory<SortOperation>(
      host_arch(), NativeOperationMode::cpu, prepare_cpu_sort);
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<SortWorkspaceDescription>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch arch, Device &device, const SortWorkspaceDescription &d) {
        if (!dynamic_cast<cuda::CudaDevice *>(&device))
          throw std::invalid_argument("Sort workspace requires a CUDA device");
        return std::make_shared<CudaSortWorkspace>(arch, device, d);
      });
  registry.register_factory<SortOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_cuda_sort);
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<SortWorkspaceDescription>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const SortWorkspaceDescription &d) {
        return gfx::prepare_vulkan_sort_workspace(device, d);
      });
  registry.register_factory<SortOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const SortOperation &operation) {
        return gfx::prepare_vulkan_sort_operation(device, operation);
      });
#endif
}

std::shared_ptr<NativeSortWorkspace> prepare_sort_workspace(
    Arch arch,
    Device &device,
    const SortWorkspaceDescription &description) {
  return std::dynamic_pointer_cast<NativeSortWorkspace>(
      native_operation_registry().prepare_resource(arch, device, description));
}

std::shared_ptr<NativeOperation> prepare_sort_operation(
    Arch arch,
    Device &device,
    const SortOperation &operation) {
  return native_operation_registry().prepare(arch, device, operation);
}
}  // namespace taichi::lang
