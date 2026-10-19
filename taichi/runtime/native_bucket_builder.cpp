#include "taichi/runtime/native_bucket_builder.h"

#include <array>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cpu/bucket_builder.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

namespace taichi::lang {
namespace {
const std::array<DataType, 6> &payload_types() {
  static const std::array<DataType, 6> types{
      PrimitiveType::i32, PrimitiveType::f32, PrimitiveType::u32,
      PrimitiveType::u64, PrimitiveType::i64, PrimitiveType::f64};
  return types;
}

size_t payload_bytes(const BucketBuilderOperation &op) {
  if (op.value_type < 0 || op.value_type >= payload_types().size())
    throw std::invalid_argument("Bucket builder requires a numeric payload");
  size_t bytes = data_type_size(payload_types()[op.value_type]);
  for (const auto extent : op.element_shape) {
    if (!extent || extent > SIZE_MAX / bytes)
      throw std::invalid_argument("Bucket payload exceeds address space");
    bytes *= extent;
  }
  return bytes;
}

void validate(Device &device, const BucketBuilderOperation &op) {
  const auto item_bytes = payload_bytes(op);
  if (op.size > INT32_MAX || !op.num_bins ||
      op.num_bins >= SIZE_MAX / sizeof(int32_t) || op.cpu_max_threads < 0)
    throw std::invalid_argument(
        "Bucket builder requires addressable bins and i32 item counts");
  for (const auto *a : {&op.keys, &op.values, &op.offsets, &op.output}) {
    const bool payload = a == &op.values || a == &op.output;
    const auto dtype =
        payload ? payload_types()[op.value_type] : DataType(PrimitiveType::i32);
    const uint64_t width = payload ? item_bytes : 4;
    const auto count = a == &op.offsets ? op.num_bins + 1 : op.size;
    if (a->pointer.device != &device)
      throw std::invalid_argument("Bucket builder requires same-device arrays");
    validate_native_array_layout(
        *a, {{count},
             payload ? op.element_shape : std::vector<uint64_t>{},
             width,
             uint64_t(data_type_size(dtype)),
             false,
             dtype->to_string()});
    if (a->shape[0] > SIZE_MAX / width || a->bytes != a->shape[0] * width)
      throw std::invalid_argument(
          "Bucket builder requires packed payload storage");
  }
}

void validate_storage(Arch arch, const BucketBuilderOperation &op) {
  for (const auto *a : {&op.keys, &op.values, &op.offsets, &op.output}) {
    validate_native_shader_storage(arch, a->pointer);
    resolve_native_storage_range(arch, a->pointer, a->bytes);
  }
  const auto width = payload_bytes(op);
  std::vector<NativeStorageRange> reads;
  std::vector<NativeStorageWrite> writes{
      {resolve_native_storage_range(arch, op.offsets.pointer,
                                    (op.num_bins + 1) * 4),
       std::nullopt}};
  if (op.size) {
    reads = {
        resolve_native_storage_range(arch, op.keys.pointer, op.size * 4),
        resolve_native_storage_range(arch, op.values.pointer, op.size * width)};
    writes.push_back(
        {resolve_native_storage_range(arch, op.output.pointer, op.size * width),
         std::nullopt});
  }
  validate_native_disjoint_writes(reads, writes);
}

class BucketBase : public NativeOperation {
 public:
  BucketBase(Arch arch, Device &device, const BucketBuilderOperation &op)
      : arch_(arch),
        device_(device),
        size_(op.size),
        bins_(op.num_bins),
        item_bytes_(payload_bytes(op)) {
    validate_storage(arch, op);
  }
  Arch arch() const override {
    return arch_;
  }
  Device &device() const override {
    return device_;
  }
  size_t workspace_bytes() const override {
    return bytes_;
  }

 protected:
  Arch arch_;
  Device &device_;
  size_t size_, bins_, item_bytes_, bytes_{0};
  void *keys_{nullptr}, *values_{nullptr}, *offsets_{nullptr},
      *output_{nullptr};
};

#ifdef TI_WITH_LLVM
template <typename T, bool raw = false>
class CpuBucket final : public BucketBase {
 public:
  CpuBucket(NativePreparationContext &context, const BucketBuilderOperation &op)
      : BucketBase(context.arch, context.device, op),
        pool_(context.cpu_thread_pool),
        partitions_(cpu::bucket_builder_partitions(
            size_,
            bins_,
            op.cpu_max_threads ? op.cpu_max_threads : context.cpu_max_threads,
            raw ? 65536 : 32768)) {
    auto *cpu = dynamic_cast<cpu::CpuDevice *>(&device_);
    if (!cpu || (partitions_ > 1 && !pool_))
      throw std::invalid_argument("Bucket CPU device or scheduler is invalid");
    const auto map = [&](const NativeArrayBinding &array, void **address) {
      if (cpu->map_range_for_cpu_native(array.pointer, array.bytes, address) !=
          RhiResult::success)
        throw std::invalid_argument("Bucket CPU storage is invalid");
    };
    map(op.offsets, &offsets_);
    if (size_) {
      map(op.keys, &keys_);
      map(op.values, &values_);
      map(op.output, &output_);
      bytes_ = cpu::bucket_builder_workspace_bytes(bins_, partitions_);
      scratch_.resize(bytes_ / sizeof(int32_t));
    }
  }
  void run_cpu() override {
    std::fill(scratch_.begin(), scratch_.end(), int32_t{});
    if constexpr (raw)
      cpu::bucket_builder_bytes(
          static_cast<const int32_t *>(keys_),
          static_cast<const uint8_t *>(values_),
          static_cast<int32_t *>(offsets_), static_cast<uint8_t *>(output_),
          size_, bins_, item_bytes_, partitions_, scratch_.data(), pool_);
    else
      cpu::bucket_builder(
          static_cast<const int32_t *>(keys_), static_cast<const T *>(values_),
          static_cast<int32_t *>(offsets_), static_cast<T *>(output_), size_,
          bins_, partitions_, scratch_.data(), pool_);
  }

 private:
  ThreadPool *pool_;
  int partitions_;
  std::vector<int32_t> scratch_;
};

std::shared_ptr<NativeResource> prepare_cpu(NativePreparationContext &context,
                                            const BucketBuilderOperation &op) {
  if (payload_bytes(op) != data_type_size(payload_types()[op.value_type]))
    return std::make_shared<CpuBucket<uint8_t, true>>(context, op);
  switch (op.value_type) {
    case 0:
      return std::make_shared<CpuBucket<int32_t>>(context, op);
    case 1:
      return std::make_shared<CpuBucket<float>>(context, op);
    case 2:
      return std::make_shared<CpuBucket<uint32_t>>(context, op);
    case 3:
      return std::make_shared<CpuBucket<uint64_t>>(context, op);
    case 4:
      return std::make_shared<CpuBucket<int64_t>>(context, op);
    case 5:
      return std::make_shared<CpuBucket<double>>(context, op);
    default:
      throw std::invalid_argument("Unsupported bucket payload type");
  }
}
#endif

#ifdef TI_WITH_CUDA
class CudaBucket final : public BucketBase {
 public:
  CudaBucket(Device &device, const BucketBuilderOperation &op)
      : BucketBase(Arch::cuda, device, op) {
    if (bins_ > INT_MAX || item_bytes_ / 4 > INT_MAX)
      throw std::invalid_argument(
          "Bucket builder exceeds the CUDA int parameter ABI");
    auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device_);
    if (!cuda)
      throw std::invalid_argument("Bucket builder requires a CUDA device");
    const auto address = [&](const NativeArrayBinding &array) {
      return static_cast<char *>(cuda->get_memory_addr(array.pointer)) +
             array.pointer.offset;
    };
    offsets_ = address(op.offsets);
    if (size_) {
      keys_ = address(op.keys);
      values_ = address(op.values);
      output_ = address(op.output);
    }
    bytes_ = cuda::driver_bucket_builder_workspace_bytes(size_, bins_);
    if (bytes_) {
      auto allocation = device_.allocate_memory_unique({bytes_});
      if (allocation.second != RhiResult::success)
        throw std::runtime_error(
            "Failed to allocate fixed CUDA bucket scratch");
      workspace_ = std::move(allocation.first);
      scratch_ = cuda->get_memory_addr(*workspace_);
    }
    cuda::driver_prepare_bucket_builder();
  }
  void record_cuda(void *stream) override {
    cuda::driver_bucket_builder_with_workspace(
        keys_, values_, offsets_, output_, size_, bins_, item_bytes_ / 4,
        scratch_, stream);
  }

 private:
  DeviceAllocationUnique workspace_;
  void *scratch_{nullptr};
};
#endif
}  // namespace

void register_bucket_builder_operation(NativeOperationRegistry &registry) {
  registry.define<BucketBuilderOperation>(validate);
  registry.register_schema<BucketBuilderOperation>(
      "builtin", "bucket_builder", {"keys", "values", "offsets", "output"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<BucketBuilderOperation> {
        operation_json::fields(p, {"size", "num_bins", "dtype", "element_shape",
                                   "cpu_max_threads", "vulkan_policy"});
        BucketBuilderOperation op;
        const auto n = operation_json::integer(p["size"]);
        const auto bins = operation_json::integer(p["num_bins"]);
        if (n > INT32_MAX || !bins || bins >= SIZE_MAX / sizeof(int32_t))
          throw std::invalid_argument(
              "Bucket builder requires addressable bins and i32 item counts");
        op.size = n;
        op.num_bins = bins;
        const auto dtype = operation_json::name(p["dtype"]);
        op.value_type = -1;
        for (int i = 0; i < payload_types().size(); ++i)
          if (payload_types()[i]->to_string() == dtype)
            op.value_type = i;
        if (!p["element_shape"].is_arr())
          throw std::invalid_argument(
              "Bucket payload requires an element shape");
        for (const auto &axis : p["element_shape"].arr.inner)
          op.element_shape.push_back(operation_json::integer(axis));
        payload_bytes(op);
        if (const auto threads =
                operation_json::optional_integer(p, "cpu_max_threads")) {
          if (!*threads || *threads > INT_MAX)
            throw std::invalid_argument(
                "Bucket CPU thread count must fit the positive int ABI");
          op.cpu_max_threads = *threads;
        }
        if (const auto it = p.obj.inner.find("vulkan_policy");
            it != p.obj.inner.end()) {
          const auto &cfg = it->second;
          operation_json::fields(
              cfg, {"private_enabled", "private_min_items", "private_max_bins",
                    "private_max_bytes"});
          auto &policy = op.vulkan_policy;
          policy.private_enabled = operation_json::optional_bool(
              cfg, "private_enabled", policy.private_enabled);
          policy.private_min_items =
              operation_json::optional_integer(cfg, "private_min_items")
                  .value_or(policy.private_min_items);
          policy.private_max_bins =
              operation_json::optional_integer(cfg, "private_max_bins")
                  .value_or(policy.private_max_bins);
          policy.private_max_bytes =
              operation_json::optional_integer(cfg, "private_max_bytes")
                  .value_or(policy.private_max_bytes);
        }
        return [op](const NativeOperationBindings &bindings) {
          auto bound = op;
          bound.keys = bindings.at("keys").array();
          bound.values = bindings.at("values").array();
          bound.offsets = bindings.at("offsets").array();
          bound.output = bindings.at("output").array();
          return bound;
        };
      });
#ifdef TI_WITH_LLVM
  registry.register_context_factory<BucketBuilderOperation>(
      host_arch(), NativeOperationMode::cpu, prepare_cpu);
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<BucketBuilderOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const BucketBuilderOperation &op) {
        auto &context = CUDAContext::get_instance();
        auto lock = context.get_submission_lock_guard();
        auto guard = context.get_guard();
        return std::make_shared<CudaBucket>(device, op);
      },
      [] {
        return NativeOperationAvailability{
            cuda::driver_hierarchical_available(), 0};
      });
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<BucketBuilderOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const BucketBuilderOperation &op) {
        validate_storage(Arch::vulkan, op);
        const auto bytes = payload_bytes(op);
        return gfx::prepare_vulkan_bucket_builder_operation(
            device, {op.keys.pointer, 4}, {op.values.pointer, bytes},
            {op.offsets.pointer, 4}, {op.output.pointer, bytes}, op.size,
            op.num_bins, bytes, op.value_type, op.vulkan_policy);
      });
#endif
}
}  // namespace taichi::lang
