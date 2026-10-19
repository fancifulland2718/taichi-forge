#include "taichi/runtime/native_transform.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cpu/transform.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/linear_ptx.h"
#endif

namespace taichi::lang {
namespace {
const std::array<DataType, 6> &transform_types() {
  static const std::array<DataType, 6> types{
      PrimitiveType::i32, PrimitiveType::f32, PrimitiveType::u32,
      PrimitiveType::u64, PrimitiveType::i64, PrimitiveType::f64};
  return types;
}

void validate(Device &device, const TransformOperation &op) {
  const auto &types = transform_types();
  if (!op.size || op.value_type < 0 || op.value_type >= types.size())
    throw std::invalid_argument(
        "Transform requires nonempty supported storage");
  const std::uint64_t width = op.value_type < 3 ? 4 : 8;
  if (op.value_type != 1 && op.value_type != 5) {
    const bool is_signed = op.value_type == 0 || op.value_type == 4;
    const double bound = std::ldexp(1., width * 8 - int(is_signed));
    for (const double value : {op.scale, op.bias}) {
      if (!std::isfinite(value) || std::trunc(value) != value ||
          value < (is_signed ? -bound : 0.) || value >= bound)
        throw std::invalid_argument(
            "Transform integer coefficients must fit dtype");
    }
  }
  for (const auto *array : {&op.src, &op.dst}) {
    if (array->pointer.device != &device)
      throw std::invalid_argument("Transform requires same-device arrays");
    validate_native_array_layout(*array, {{op.size},
                                          {},
                                          width,
                                          width,
                                          false,
                                          types[op.value_type]->to_string()});
    if (array->shape[0] > std::numeric_limits<std::size_t>::max() / width ||
        array->bytes != array->shape[0] * width)
      throw std::invalid_argument(
          "Transform requires exact packed storage extents");
  }
}

class TransformBase : public NativeOperation {
 public:
  TransformBase(Arch arch, Device &device, const TransformOperation &op)
      : arch_(arch), device_(device), size_(op.size), type_(op.value_type) {
    for (const auto *array : {&op.src, &op.dst}) {
      validate_native_shader_storage(arch, array->pointer);
      resolve_native_storage_range(arch, array->pointer, array->bytes);
    }
    const auto bytes = size_ * (type_ < 3 ? 4 : 8);
    validate_native_disjoint_writes(
        {resolve_native_storage_range(arch, op.src.pointer, bytes)},
        {{resolve_native_storage_range(arch, op.dst.pointer, bytes), 0}});
  }
  Arch arch() const override {
    return arch_;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }

 protected:
  Arch arch_;
  Device &device_;
  std::size_t size_;
  int type_;
  void *src_{nullptr}, *dst_{nullptr};
};

#ifdef TI_WITH_LLVM
template <typename T>
class CpuTransform final : public TransformBase {
 public:
  CpuTransform(NativePreparationContext &context,
               const TransformOperation &op,
               T scale,
               T bias)
      : TransformBase(context.arch, context.device, op),
        scale_(scale),
        bias_(bias),
        pool_(context.cpu_thread_pool),
        partitions_(
            cpu::simple_loop_partitions(size_, context.cpu_max_threads)) {
    auto *cpu = dynamic_cast<cpu::CpuDevice *>(&device_);
    if (!cpu ||
        cpu->map_range_for_cpu_native(op.src.pointer, op.src.bytes, &src_) !=
            RhiResult::success ||
        cpu->map_range_for_cpu_native(op.dst.pointer, op.dst.bytes, &dst_) !=
            RhiResult::success ||
        (partitions_ > 1 && !pool_))
      throw std::invalid_argument(
          "Transform CPU storage or scheduler is invalid");
  }
  void run_cpu() override {
    cpu::transform(static_cast<const T *>(src_), static_cast<T *>(dst_), size_,
                   scale_, bias_, partitions_, pool_);
  }

 private:
  T scale_, bias_;
  ThreadPool *pool_;
  int partitions_;
};

std::shared_ptr<NativeOperation> prepare_cpu(NativePreparationContext &context,
                                             const TransformOperation &op) {
  switch (op.value_type) {
    case 0:
      return std::make_shared<CpuTransform<std::uint32_t>>(
          context, op,
          static_cast<std::uint32_t>(static_cast<std::int32_t>(op.scale)),
          static_cast<std::uint32_t>(static_cast<std::int32_t>(op.bias)));
    case 1:
      return std::make_shared<CpuTransform<float>>(context, op, op.scale,
                                                   op.bias);
    case 2:
      return std::make_shared<CpuTransform<std::uint32_t>>(context, op,
                                                           op.scale, op.bias);
    case 3:
      return std::make_shared<CpuTransform<std::uint64_t>>(context, op,
                                                           op.scale, op.bias);
    case 4:
      return std::make_shared<CpuTransform<std::uint64_t>>(
          context, op,
          static_cast<std::uint64_t>(static_cast<std::int64_t>(op.scale)),
          static_cast<std::uint64_t>(static_cast<std::int64_t>(op.bias)));
    case 5:
      return std::make_shared<CpuTransform<double>>(context, op, op.scale,
                                                    op.bias);
    default:
      throw std::invalid_argument("Unsupported CPU transform dtype");
  }
}
#endif

#ifdef TI_WITH_CUDA
class CudaTransform final : public TransformBase {
 public:
  CudaTransform(Device &device, const TransformOperation &op)
      : TransformBase(Arch::cuda, device, op),
        scale_(op.scale),
        bias_(op.bias) {
    if (size_ > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument(
          "Transform exceeds the CUDA uint32 shader count ABI");
    auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device);
    if (!cuda)
      throw std::invalid_argument("Transform requires a CUDA device");
    src_ = static_cast<char *>(cuda->get_memory_addr(op.src.pointer)) +
           op.src.pointer.offset;
    dst_ = static_cast<char *>(cuda->get_memory_addr(op.dst.pointer)) +
           op.dst.pointer.offset;
    cuda::driver_prepare_transform(
        static_cast<cuda::CudaTransformValueType>(type_));
  }
  void record_cuda(void *stream) override {
    cuda::driver_transform_affine_on_stream(
        src_, dst_, static_cast<std::uint32_t>(size_),
        static_cast<cuda::CudaTransformValueType>(type_), scale_, bias_,
        stream);
  }

 private:
  double scale_, bias_;
};
#endif

double coefficient(const operation_json::JsonValue &value) {
  const auto bits = operation_json::scalar_bits(value, 16);
  double result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
}  // namespace

void validate_indexed_transform_operation(
    Device &device, const IndexedTransformOperation &op) {
  if (op.value_type < 0 || op.value_type > 2 || !op.index_size || !op.dst_size)
    throw std::invalid_argument(
        "Indexed transform requires nonempty i32/u32/f32 arrays");
  validate(device, TransformOperation{op.src_size, op.value_type, op.scale,
                                      op.bias, op.src, op.transformed});
  for (const auto *array : {&op.indices, &op.gathered, &op.dst}) {
    const auto size = array == &op.dst ? op.dst_size : op.index_size;
    const auto type = array == &op.indices ? PrimitiveType::i32
                                         : transform_types()[op.value_type];
    if (array->pointer.device != &device ||
        size > std::numeric_limits<std::size_t>::max() / 4 ||
        array->bytes != size * 4)
      throw std::invalid_argument("Indexed transform requires exact same-device storage");
    validate_native_array_layout(*array, {{size}, {}, 4, 4, true,
                                          type->to_string()});
  }
  if (op.src.shape[0] != op.src_size ||
      op.transformed.shape[0] != op.src_size)
    throw std::invalid_argument("Indexed transform requires exact source shapes");
}

void register_transform_operation(NativeOperationRegistry &registry) {
  registry.define<IndexedTransformOperation>(validate_indexed_transform_operation);
  registry.register_schema<IndexedTransformOperation>(
      "builtin", "indexed_transform",
      {"src", "transformed", "indices", "gathered", "dst"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<IndexedTransformOperation> {
        operation_json::fields(parameters,
                               {"src_size", "index_size", "dst_size", "dtype",
                                "scale_bits", "bias_bits"});
        std::array<std::size_t, 3> sizes;
        const char *names[] = {"src_size", "index_size", "dst_size"};
        for (std::size_t i = 0; i < sizes.size(); ++i) {
          const auto size = operation_json::integer(parameters[names[i]]);
          if (!size || size > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument(
                "Indexed transform sizes must be nonempty and addressable");
          sizes[i] = static_cast<std::size_t>(size);
        }
        const auto dtype = operation_json::name(parameters["dtype"]);
        int type = 0;
        while (type < 3 && transform_types()[type]->to_string() != dtype)
          ++type;
        if (type == 3)
          throw std::invalid_argument("Indexed transform requires i32/u32/f32");
        const auto scale = coefficient(parameters["scale_bits"]);
        const auto bias = coefficient(parameters["bias_bits"]);
        return [sizes, type, scale, bias](const NativeOperationBindings &bindings) {
          return IndexedTransformOperation{
              sizes[0], sizes[1], sizes[2], type, scale, bias,
              bindings.at("src").array(), bindings.at("transformed").array(),
              bindings.at("indices").array(), bindings.at("gathered").array(),
              bindings.at("dst").array()};
        };
      });
  registry.define<TransformOperation>(validate);
  registry.register_schema<TransformOperation>(
      "builtin", "transform", {"src", "dst"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<TransformOperation> {
        operation_json::fields(parameters,
                               {"size", "dtype", "scale_bits", "bias_bits"});
        const auto size = operation_json::integer(parameters["size"]);
        if (!size || size > std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument(
              "Transform size must be nonempty and addressable");
        const auto dtype = operation_json::name(parameters["dtype"]);
        const auto &types = transform_types();
        int type = 0;
        while (type < types.size() && types[type]->to_string() != dtype)
          ++type;
        if (type == types.size())
          throw std::invalid_argument(
              "Transform requires a supported numeric type");
        const auto scale = coefficient(parameters["scale_bits"]);
        const auto bias = coefficient(parameters["bias_bits"]);
        return
            [size, type, scale, bias](const NativeOperationBindings &bindings) {
              return TransformOperation{static_cast<std::size_t>(size),
                                        type,
                                        scale,
                                        bias,
                                        bindings.at("src").array(),
                                        bindings.at("dst").array()};
            };
      });
#ifdef TI_WITH_LLVM
  registry.register_context_factory<TransformOperation>(
      host_arch(), NativeOperationMode::cpu, prepare_cpu);
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<TransformOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const TransformOperation &op) {
        auto &context = CUDAContext::get_instance();
        auto lock = context.get_submission_lock_guard();
        auto guard = context.get_guard();
        return std::make_shared<CudaTransform>(device, op);
      },
      [] {
        return NativeOperationAvailability{cuda::driver_transform_available(),
                                           0};
      });
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<IndexedTransformOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const IndexedTransformOperation &op) {
        return gfx::prepare_vulkan_indexed_transform_operation(device, op);
      });
  registry.register_factory<TransformOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const TransformOperation &op) {
        return gfx::prepare_vulkan_transform_operation(device, op);
      });
#endif
}
}  // namespace taichi::lang
