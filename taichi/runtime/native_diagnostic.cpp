#include "taichi/runtime/native_diagnostic.h"

#include <array>
#include <limits>
#include <type_traits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cpu/diagnostic_reduce.h"
#include "taichi/rhi/cpu/reduce.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/linear_ptx.h"
#endif
#ifdef TI_WITH_VULKAN
#include "taichi/runtime/vulkan_diagnostic.h"
#endif

namespace taichi::lang {
namespace {
template <typename Op>
constexpr bool is_metric = std::is_same_v<Op, MetricReduceOperation>;

const std::array<DataType, 6> &diagnostic_types() {
  static const std::array<DataType, 6> types{
      PrimitiveType::i32, PrimitiveType::f32, PrimitiveType::u32,
      PrimitiveType::u64, PrimitiveType::i64, PrimitiveType::f64};
  return types;
}

template <typename Op>
std::vector<const NativeArrayBinding *> diagnostic_inputs(const Op &op) {
  std::vector<const NativeArrayBinding *> result{&op.values};
  if constexpr (is_metric<Op>) {
    if (op.other)
      result.push_back(&*op.other);
  }
  return result;
}

template <typename Op>
void validate_diagnostic(Device &device, const Op &op) {
  const auto &types = diagnostic_types();
  if (op.value_type < 0 || op.value_type >= types.size() || op.op < 0 ||
      op.op > (is_metric<Op> ? 1 : 5) || op.cpu_max_threads < 0)
    throw std::invalid_argument(
        "Diagnostic type, operation or CPU policy is invalid");
  if constexpr (is_metric<Op>) {
    if ((op.value_type != 1 && op.value_type != 5) ||
        bool(op.other) != (op.op == 1))
      throw std::invalid_argument(
          "Metric requires floating input and rhs only for delta");
  } else if (op.op == 5 && (op.value_type == 1 || op.value_type == 5)) {
    throw std::invalid_argument("Bounds checks require integer inputs");
  }
  auto arrays = diagnostic_inputs(op);
  arrays.push_back(&op.output);
  for (const auto *array : arrays) {
    const bool output = array == &op.output;
    const auto type = output && !is_metric<Op> ? DataType(PrimitiveType::i32)
                                               : types[op.value_type];
    const uint64_t width = data_type_size(type);
    const bool scalar = output && array->shape.empty();
    if (array->pointer.device != &device)
      throw std::invalid_argument("Diagnostic requires same-device storage");
    validate_native_array_layout(
        *array, {scalar ? std::vector<uint64_t>{}
                        : std::vector<uint64_t>{output ? 1 : op.size},
                 {},
                 width,
                 width,
                 false,
                 type->to_string()});
    const auto count = scalar ? 1 : array->shape[0];
    if (count > std::numeric_limits<size_t>::max() / width ||
        array->bytes != count * width)
      throw std::invalid_argument(
          "Diagnostic requires exact packed scalar storage");
  }
}

template <typename Op>
void validate_diagnostic_storage(Arch arch, const Op &op) {
  auto inputs = diagnostic_inputs(op);
  auto arrays = inputs;
  arrays.push_back(&op.output);
  for (const auto *array : arrays) {
    validate_native_shader_storage(arch, array->pointer);
    resolve_native_storage_range(arch, array->pointer, array->bytes);
  }
  const auto width = op.value_type < 3 ? 4 : 8;
  std::vector<NativeStorageRange> reads;
  for (const auto *array : inputs)
    reads.push_back(
        resolve_native_storage_range(arch, array->pointer, op.size * width));
  validate_native_read_write_disjoint(
      reads, {resolve_native_storage_range(arch, op.output.pointer,
                                           is_metric<Op> ? width : 4)});
}

template <typename Op>
class DiagnosticBase : public NativeOperation {
 public:
  DiagnosticBase(Arch arch, Device &device, const Op &op)
      : arch_(arch),
        device_(device),
        size_(op.size),
        type_(op.value_type),
        op_(op.op) {
    validate_diagnostic_storage(arch, op);
    if constexpr (!is_metric<Op>) {
      lower_ = op.lower;
      upper_ = op.upper;
    }
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
  size_t size_, bytes_{0};
  int type_, op_;
  int32_t lower_{0}, upper_{0};
  void *values_{nullptr}, *other_{nullptr}, *output_{nullptr};
};

#ifdef TI_WITH_LLVM
template <typename Op, typename T>
class CpuDiagnostic final : public DiagnosticBase<Op> {
  using Base = DiagnosticBase<Op>;
  using Partial = std::conditional_t<is_metric<Op>, T, uint64_t>;

 public:
  CpuDiagnostic(NativePreparationContext &context, const Op &op)
      : Base(context.arch, context.device, op),
        pool_(context.cpu_thread_pool),
        partitions_(cpu::reduce_partitions(op.size,
                                           op.cpu_max_threads
                                               ? op.cpu_max_threads
                                               : context.cpu_max_threads)) {
    auto *device = dynamic_cast<cpu::CpuDevice *>(&context.device);
    const auto map = [&](const NativeArrayBinding &array, void **pointer) {
      if (!device ||
          device->map_range_for_cpu_native(array.pointer, array.bytes,
                                           pointer) != RhiResult::success)
        throw std::invalid_argument("Diagnostic CPU storage is invalid");
    };
    map(op.values, &this->values_);
    map(op.output, &this->output_);
    if constexpr (is_metric<Op>) {
      if (op.other)
        map(*op.other, &this->other_);
    }
    if (partitions_ > 1) {
      if (!pool_)
        throw std::invalid_argument(
            "Diagnostic requires the owner's CPU scheduler");
      partial_.resize(partitions_);
    }
    this->bytes_ = partial_.size() * sizeof(Partial);
  }
  void run_cpu() override {
    const cpu::DiagnosticPackedRead<T> values{
        static_cast<const T *>(this->values_)};
    if constexpr (is_metric<Op>) {
      const cpu::DiagnosticPackedRead<T> other{
          static_cast<const T *>(this->other_)};
      *static_cast<T *>(this->output_) =
          cpu::metric_reduce(values, other, this->size_, this->op_, partitions_,
                             partial_.data(), pool_);
    } else {
      *static_cast<int32_t *>(this->output_) = cpu::encode_check_count_i32(
          cpu::check_count(values, this->size_, this->op_, this->lower_,
                           this->upper_, partitions_, partial_.data(), pool_));
    }
  }

 private:
  ThreadPool *pool_;
  int partitions_;
  std::vector<Partial> partial_;
};

template <typename Op>
std::shared_ptr<NativeResource> prepare_cpu_diagnostic(
    NativePreparationContext &c,
    const Op &op) {
  if (op.value_type == 1)
    return std::make_shared<CpuDiagnostic<Op, float>>(c, op);
  if (op.value_type == 5)
    return std::make_shared<CpuDiagnostic<Op, double>>(c, op);
  if constexpr (!is_metric<Op>) {
    switch (op.value_type) {
      case 0:
        return std::make_shared<CpuDiagnostic<Op, int32_t>>(c, op);
      case 2:
        return std::make_shared<CpuDiagnostic<Op, uint32_t>>(c, op);
      case 3:
        return std::make_shared<CpuDiagnostic<Op, uint64_t>>(c, op);
      case 4:
        return std::make_shared<CpuDiagnostic<Op, int64_t>>(c, op);
    }
  }
  throw std::invalid_argument("Unsupported CPU diagnostic dtype");
}
#endif

#ifdef TI_WITH_CUDA
template <typename Op>
class CudaDiagnostic final : public DiagnosticBase<Op> {
 public:
  CudaDiagnostic(Device &device, const Op &op)
      : DiagnosticBase<Op>(Arch::cuda, device, op) {
    if (op.size > UINT32_MAX)
      throw std::invalid_argument(
          "CUDA diagnostic exceeds its uint32 count ABI");
    auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device);
    if (!cuda)
      throw std::invalid_argument("Diagnostic requires a CUDA device");
    const auto address = [&](const NativeArrayBinding &array) {
      return static_cast<char *>(cuda->get_memory_addr(array.pointer)) +
             array.pointer.offset;
    };
    this->values_ = address(op.values);
    this->output_ = address(op.output);
    if constexpr (is_metric<Op>) {
      if (op.other)
        this->other_ = address(*op.other);
      cuda::driver_prepare_metric_reduce(
          static_cast<cuda::CudaTransformValueType>(op.value_type),
          static_cast<cuda::CudaMetricOp>(op.op));
    } else {
      cuda::driver_prepare_check_count(
          static_cast<cuda::CudaTransformValueType>(op.value_type),
          static_cast<cuda::CudaCheckOp>(op.op));
    }
  }
  void record_cuda(void *stream) override {
    const auto type = static_cast<cuda::CudaTransformValueType>(this->type_);
    const size_t width = this->type_ < 3 ? 4 : 8;
    if constexpr (is_metric<Op>) {
      cuda::driver_metric_reduce_on_stream(
          this->values_, this->other_, this->output_, this->size_, type, 0,
          width, 0, width, static_cast<cuda::CudaMetricOp>(this->op_), stream);
    } else {
      cuda::driver_check_count_on_stream(
          this->values_, this->output_, this->size_, type, 0, width,
          static_cast<cuda::CudaCheckOp>(this->op_), this->lower_, this->upper_,
          stream);
    }
  }
};
#endif

int diagnostic_type(const liong::json::JsonValue &value, bool metric) {
  const auto name = operation_json::name(value);
  const auto &types = diagnostic_types();
  for (int i = 0; i < types.size(); ++i)
    if (types[i]->to_string() == name && (!metric || i == 1 || i == 5))
      return i;
  throw std::invalid_argument("Unsupported diagnostic dtype");
}

int32_t diagnostic_bound(const liong::json::JsonValue &p, const char *key) {
  const auto it = p.obj.inner.find(key);
  if (it == p.obj.inner.end())
    return 0;
  const auto &value = it->second;
  if (value.ty != liong::json::L_JSON_INT || value.num_int < INT32_MIN ||
      value.num_int > INT32_MAX)
    throw std::invalid_argument(
        "Diagnostic bounds must fit the signed i32 ABI");
  return static_cast<int32_t>(value.num_int);
}

template <typename Op>
void register_diagnostic(NativeOperationRegistry &registry) {
  registry.define<Op>(validate_diagnostic<Op>);
  registry.register_schema<Op>(
      "builtin", Op::kind, {"values", "output"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<Op> {
        if constexpr (is_metric<Op>)
          operation_json::fields(p, {"size", "dtype", "op", "cpu_max_threads",
                                     "vulkan_items_per_group"});
        else
          operation_json::fields(
              p, {"size", "dtype", "op", "lower", "upper", "cpu_max_threads",
                  "vulkan_items_per_group"});
        Op op;
        const auto n = operation_json::integer(p["size"]);
        if (n > std::numeric_limits<size_t>::max())
          throw std::invalid_argument("Diagnostic count must be addressable");
        op.size = static_cast<size_t>(n);
        op.value_type = diagnostic_type(p["dtype"], is_metric<Op>);
        const auto name = operation_json::name(p["op"]);
        if constexpr (is_metric<Op>) {
          op.op = name == "max_abs" ? 0 : name == "max_abs_delta" ? 1 : -1;
        } else {
          const std::array<const char *, 6> names{
              "nonzero", "zero", "nan", "inf", "not_finite", "index_oob"};
          op.op = -1;
          for (int i = 0; i < names.size(); ++i)
            if (name == names[i])
              op.op = i;
          op.lower = diagnostic_bound(p, "lower");
          op.upper = diagnostic_bound(p, "upper");
          if (op.op == 5 && (op.value_type == 1 || op.value_type == 5))
            throw std::invalid_argument("Bounds checks require integer input");
        }
        if (op.op < 0)
          throw std::invalid_argument("Unknown diagnostic operation");
        const auto threads =
            operation_json::optional_integer(p, "cpu_max_threads");
        if (threads && (!*threads || *threads > INT_MAX))
          throw std::invalid_argument(
              "Diagnostic CPU policy must fit its positive int ABI");
        op.cpu_max_threads = static_cast<int>(threads.value_or(0));
        op.vulkan_items_per_group =
            operation_json::optional_integer(p, "vulkan_items_per_group")
                .value_or(16384);
        return [op](const NativeOperationBindings &bindings) mutable {
          auto result = op;
          result.values = bindings.at("values").array();
          result.output = bindings.at("output").array();
          if constexpr (is_metric<Op>) {
            if (const auto it = bindings.find("other"); it != bindings.end())
              result.other = it->second.array();
          }
          return result;
        };
      },
      "",
      is_metric<Op> ? std::vector<std::string>{"other"}
                    : std::vector<std::string>{});
#ifdef TI_WITH_LLVM
  registry.register_context_factory<Op>(host_arch(), NativeOperationMode::cpu,
                                        prepare_cpu_diagnostic<Op>);
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<Op>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const Op &op) {
        auto &context = CUDAContext::get_instance();
        auto lock = context.get_submission_lock_guard();
        auto guard = context.get_guard();
        return std::make_shared<CudaDiagnostic<Op>>(device, op);
      },
      [] {
        return NativeOperationAvailability{cuda::driver_check_count_available(),
                                           0};
      });
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<Op>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch arch, Device &device, const Op &op) {
        validate_diagnostic_storage(arch, op);
        const size_t width = op.value_type < 3 ? 4 : 8;
        const gfx::VulkanDiagnosticInput values{op.values.pointer, width};
        if constexpr (is_metric<Op>) {
          gfx::VulkanDiagnosticInput other;
          if (op.other)
            other = {op.other->pointer, width};
          return gfx::prepare_vulkan_metric_operation(
              device, values, other, op.output.pointer, op.size, op.value_type,
              op.op, op.vulkan_items_per_group);
        } else {
          return gfx::prepare_vulkan_check_operation(
              device, values, op.output.pointer, op.size, op.value_type, op.op,
              op.lower, op.upper, op.vulkan_items_per_group);
        }
      });
#endif
}
}  // namespace

void register_diagnostic_operations(NativeOperationRegistry &registry) {
  register_diagnostic<CheckCountOperation>(registry);
  register_diagnostic<MetricReduceOperation>(registry);
}
}  // namespace taichi::lang
