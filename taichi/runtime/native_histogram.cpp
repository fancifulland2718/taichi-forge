#include "taichi/runtime/native_histogram.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cpu/histogram.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

namespace taichi::lang {
namespace {
void validate(Device &device, const HistogramOperation &op) {
  if (!op.num_bins || (op.value_type != 0 && op.value_type != 2) ||
      (op.bin_type != 0 && op.bin_type != 4) || op.cpu_max_threads < 0)
    throw std::invalid_argument("Histogram requires integer IDs and counters");
  for (const auto *a : {&op.values, &op.bins}) {
    const bool output = a == &op.bins;
    const DataType type =
        output ? (op.bin_type == 4 ? PrimitiveType::i64 : PrimitiveType::i32)
               : (op.value_type == 2 ? PrimitiveType::u32 : PrimitiveType::i32);
    const uint64_t width = data_type_size(type);
    if (a->pointer.device != &device)
      throw std::invalid_argument("Histogram requires same-device arrays");
    validate_native_array_layout(*a, {{output ? op.num_bins : op.size},
                                      {},
                                      width,
                                      width,
                                      false,
                                      type->to_string()});
    if (a->shape[0] > std::numeric_limits<size_t>::max() / width ||
        a->bytes != a->shape[0] * width)
      throw std::invalid_argument(
          "Histogram requires exact packed scalar storage");
  }
}

class HistogramBase : public NativeOperation {
 public:
  HistogramBase(Arch arch, Device &device, const HistogramOperation &op)
      : arch_(arch), device_(device), size_(op.size), count_(op.num_bins) {
    validate_histogram_storage(arch, op);
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
  size_t size_, count_, bytes_{0};
  void *values_{nullptr}, *bins_{nullptr};
};

#ifdef TI_WITH_LLVM
template <typename Value, typename Counter>
class CpuHistogram final : public HistogramBase {
 public:
  CpuHistogram(NativePreparationContext &c, const HistogramOperation &op)
      : HistogramBase(c.arch, c.device, op),
        pool_(c.cpu_thread_pool),
        partitions_(cpu::histogram_partitions(
            size_,
            count_,
            op.cpu_max_threads ? op.cpu_max_threads : c.cpu_max_threads)) {
    auto *cpu = dynamic_cast<cpu::CpuDevice *>(&device_);
    if (!cpu ||
        cpu->map_range_for_cpu_native(op.values.pointer, op.values.bytes,
                                      &values_) != RhiResult::success ||
        cpu->map_range_for_cpu_native(op.bins.pointer, op.bins.bytes, &bins_) !=
            RhiResult::success ||
        (partitions_ > 1 && !pool_))
      throw std::invalid_argument(
          "Histogram CPU storage or scheduler is invalid");
    bytes_ = cpu::aggregation_workspace_bytes<Counter>(count_, partitions_);
    partial_.resize(bytes_ / sizeof(Counter));
  }
  void run_cpu() override {
    std::fill(partial_.begin(), partial_.end(), Counter{});
    cpu::histogram(static_cast<const Value *>(values_),
                   static_cast<Counter *>(bins_), size_, count_, partitions_,
                   partial_.data(), pool_);
  }

 private:
  ThreadPool *pool_;
  int partitions_;
  std::vector<Counter> partial_;
};

std::shared_ptr<NativeResource> prepare_cpu(NativePreparationContext &c,
                                            const HistogramOperation &op) {
  if (op.value_type == 2) {
    if (op.bin_type == 4)
      return std::make_shared<CpuHistogram<uint32_t, int64_t>>(c, op);
    return std::make_shared<CpuHistogram<uint32_t, int32_t>>(c, op);
  }
  if (op.bin_type == 4)
    return std::make_shared<CpuHistogram<int32_t, int64_t>>(c, op);
  return std::make_shared<CpuHistogram<int32_t, int32_t>>(c, op);
}
#endif

#ifdef TI_WITH_CUDA
class CudaHistogram final : public HistogramBase {
 public:
  CudaHistogram(Device &device, const HistogramOperation &op)
      : HistogramBase(Arch::cuda, device, op),
        value_type_(static_cast<cuda::CudaTransformValueType>(op.value_type)),
        bin_type_(static_cast<cuda::CudaTransformValueType>(op.bin_type)) {
    if (size_ > UINT32_MAX || count_ > UINT32_MAX)
      throw std::invalid_argument(
          "Histogram exceeds its CUDA uint32 count ABI");
    auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device_);
    if (!cuda)
      throw std::invalid_argument("Histogram requires a CUDA device");
    values_ = static_cast<char *>(cuda->get_memory_addr(op.values.pointer)) +
              op.values.pointer.offset;
    bins_ = static_cast<char *>(cuda->get_memory_addr(op.bins.pointer)) +
            op.bins.pointer.offset;
    cuda::driver_prepare_histogram(value_type_, bin_type_);
  }
  void record_cuda(void *stream) override {
    cuda::driver_histogram_on_stream(values_, bins_, size_, count_, value_type_,
                                     bin_type_, stream);
  }

 private:
  cuda::CudaTransformValueType value_type_, bin_type_;
};
#endif
}  // namespace

void validate_histogram_storage(Arch arch, const HistogramOperation &op) {
  for (const auto *a : {&op.values, &op.bins}) {
    validate_native_shader_storage(arch, a->pointer);
    resolve_native_storage_range(arch, a->pointer, a->bytes);
  }
  validate_native_read_write_disjoint(
      {resolve_native_storage_range(arch, op.values.pointer, op.size * 4)},
      {resolve_native_storage_range(arch, op.bins.pointer,
                                    op.num_bins * (op.bin_type == 4 ? 8 : 4))});
}

void register_histogram_operation(NativeOperationRegistry &registry) {
  registry.define<HistogramOperation>(validate);
  registry.register_schema<HistogramOperation>(
      "builtin", "histogram", {"values", "bins"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<HistogramOperation> {
        operation_json::fields(
            p, {"size", "num_bins", "value_dtype", "counter_dtype",
                "cpu_max_threads", "vulkan_policy"});
        const auto n = operation_json::integer(p["size"]);
        const auto bins = operation_json::integer(p["num_bins"]);
        if (!bins || n > std::numeric_limits<size_t>::max() ||
            bins > std::numeric_limits<size_t>::max())
          throw std::invalid_argument(
              "Histogram extents must be addressable with nonempty bins");
        const auto value = operation_json::name(p["value_dtype"]);
        const auto counter = operation_json::name(p["counter_dtype"]);
        if ((value != "i32" && value != "u32") ||
            (counter != "i32" && counter != "i64"))
          throw std::invalid_argument("Unsupported histogram ID/counter dtype");
        const auto threads =
            operation_json::optional_integer(p, "cpu_max_threads");
        if (threads && (!*threads || *threads > INT_MAX))
          throw std::invalid_argument(
              "Histogram CPU configuration must fit its positive int ABI");
        VulkanHistogramPolicy policy;
        if (const auto it = p.obj.inner.find("vulkan_policy");
            it != p.obj.inner.end()) {
          const auto &cfg = it->second;
          operation_json::fields(cfg, {"private_min_n", "private_max_bins",
                                       "single_shared_max_n"});
          policy.private_min_n =
              operation_json::optional_integer(cfg, "private_min_n")
                  .value_or(policy.private_min_n);
          policy.private_max_bins =
              operation_json::optional_integer(cfg, "private_max_bins")
                  .value_or(policy.private_max_bins);
          policy.single_shared_max_n =
              operation_json::optional_integer(cfg, "single_shared_max_n")
                  .value_or(policy.single_shared_max_n);
        }
        return [n, bins, value, counter, threads,
                policy](const NativeOperationBindings &bindings) {
          return HistogramOperation{size_t(n),
                                    size_t(bins),
                                    value == "u32" ? 2 : 0,
                                    counter == "i64" ? 4 : 0,
                                    static_cast<int>(threads.value_or(0)),
                                    policy,
                                    bindings.at("values").array(),
                                    bindings.at("bins").array()};
        };
      });
#ifdef TI_WITH_LLVM
  registry.register_context_factory<HistogramOperation>(
      host_arch(), NativeOperationMode::cpu, prepare_cpu);
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<HistogramOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const HistogramOperation &op) {
        auto &context = CUDAContext::get_instance();
        auto lock = context.get_submission_lock_guard();
        auto guard = context.get_guard();
        return std::make_shared<CudaHistogram>(device, op);
      },
      [] {
        return NativeOperationAvailability{
            cuda::driver_hierarchical_available(), 0};
      });
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<HistogramOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const HistogramOperation &op) {
        return gfx::prepare_vulkan_histogram_operation(device, op);
      });
#endif
}
}  // namespace taichi::lang
