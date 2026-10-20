#include "taichi/runtime/native_reduce.h"

#include <array>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cpu/reduce.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

namespace taichi::lang {
namespace {
const std::array<DataType, 6> &reduce_types() {
  static const std::array<DataType, 6> types{
      PrimitiveType::i32, PrimitiveType::f32, PrimitiveType::u32,
      PrimitiveType::u64, PrimitiveType::i64, PrimitiveType::f64};
  return types;
}

void validate(Device &device, const ReduceOperation &op) {
  const auto &types = reduce_types();
  if (!op.size || op.op < 0 || op.op > 2 || op.cpu_max_threads < 0 ||
      op.value_type < 0 || op.value_type >= types.size() ||
      (op.strategy != ReduceStrategy::packed_array &&
       op.strategy != ReduceStrategy::scalar_field &&
       op.strategy != ReduceStrategy::strided_scalar &&
       op.strategy != ReduceStrategy::packed_field))
    throw std::invalid_argument(
        "Reduce requires nonempty supported sum/min/max");
  const std::uint64_t width = op.value_type < 3 ? 4 : 8;
  const auto record = reduce_item_bytes(op);
  for (const auto *array : {&op.values, &op.output}) {
    if (array->pointer.device != &device)
      throw std::invalid_argument("Reduce requires same-device arrays");
    if (op.strategy == ReduceStrategy::scalar_field ||
        op.strategy == ReduceStrategy::strided_scalar) {
      native_scalar_prefix(*array, array == &op.values ? op.size : 1,
                            types[op.value_type]);
      continue;
    }
    const bool scalar = array == &op.output && array->shape.empty();
    validate_native_array_layout(
        *array,
        {scalar ? std::vector<std::uint64_t>{}
                : std::vector<std::uint64_t>{array == &op.values ? op.size : 1},
         op.element_shape,
         record,
         width,
         false,
         types[op.value_type]->to_string()});
    const auto count = scalar ? 1 : array->shape[0];
    if (count > std::numeric_limits<std::size_t>::max() / record ||
        array->bytes != count * record)
      throw std::invalid_argument(
          "Reduce requires exact packed storage extents");
  }
}

class ReduceBase : public NativeOperation {
 public:
  ReduceBase(Arch arch, Device &device, const ReduceOperation &op)
      : arch_(arch),
        device_(device),
        size_(op.size),
        type_(op.value_type),
        op_(op.op) {
    addressing_ = qualify_reduce_addressing(arch, device, op);
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

 protected:
  Arch arch_;
  Device &device_;
  std::size_t size_;
  int type_, op_;
  std::size_t bytes_{0};
  ReduceAddressing addressing_;
  void *values_{nullptr}, *output_{nullptr};
};

#ifdef TI_WITH_LLVM
template <typename T>
class CpuReduce final : public ReduceBase {
 public:
  CpuReduce(NativePreparationContext &context, const ReduceOperation &op)
      : ReduceBase(context.arch, context.device, op),
        strided_(op.strategy == ReduceStrategy::strided_scalar ||
                 op.strategy == ReduceStrategy::packed_field ||
                 (op.strategy == ReduceStrategy::scalar_field &&
                  addressing_.values_stride != sizeof(T))),
        pool_(context.cpu_thread_pool),
        partitions_(cpu::reduce_partitions(op.size,
                                           op.cpu_max_threads
                                               ? op.cpu_max_threads
                                               : context.cpu_max_threads)) {
    auto *cpu = dynamic_cast<cpu::CpuDevice *>(&device_);
    if (!cpu ||
        cpu->map_range_for_cpu_native(op.values.pointer, op.values.bytes,
                                      &values_) != RhiResult::success ||
        cpu->map_range_for_cpu_native(op.output.pointer, op.output.bytes,
                                      &output_) != RhiResult::success)
      throw std::invalid_argument("Reduce CPU storage range is invalid");
    if (partitions_ > 1) {
      if (!pool_)
        throw std::invalid_argument(
            "Parallel reduce requires its owner's CPU scheduler");
      partial_.resize(partitions_);
      bytes_ = partial_.size() * sizeof(T);
    }
  }
  void run_cpu() override {
    if (strided_) {
      for (std::size_t lane = 0; lane < addressing_.lanes; ++lane) {
        const auto lane_offset = lane * sizeof(T);
        cpu::reduce_strided<T>(
            static_cast<const std::uint8_t *>(values_),
            reinterpret_cast<T *>(static_cast<std::uint8_t *>(output_) +
                                   addressing_.output_offset + lane_offset),
            op_, size_, addressing_.values_offset + lane_offset,
            addressing_.values_stride, partitions_, partial_.data(), pool_);
      }
      return;
    }
    cpu::reduce(reinterpret_cast<const T *>(
                    static_cast<const std::uint8_t *>(values_) + addressing_.values_offset),
                reinterpret_cast<T *>(
                    static_cast<std::uint8_t *>(output_) + addressing_.output_offset), op_,
                size_, partitions_, partial_.data(), pool_);
  }

 private:
  bool strided_;
  ThreadPool *pool_;
  int partitions_;
  std::vector<T> partial_;
};

std::shared_ptr<NativeResource> prepare_cpu(NativePreparationContext &context,
                                            const ReduceOperation &op) {
  switch (op.value_type) {
    case 0:
      return std::make_shared<CpuReduce<std::int32_t>>(context, op);
    case 1:
      return std::make_shared<CpuReduce<float>>(context, op);
    case 2:
      return std::make_shared<CpuReduce<std::uint32_t>>(context, op);
    case 3:
      return std::make_shared<CpuReduce<std::uint64_t>>(context, op);
    case 4:
      return std::make_shared<CpuReduce<std::int64_t>>(context, op);
    case 5:
      return std::make_shared<CpuReduce<double>>(context, op);
    default:
      throw std::invalid_argument("Unsupported CPU reduce dtype");
  }
}
#endif

#ifdef TI_WITH_CUDA
class CudaReduce final : public ReduceBase {
 public:
  CudaReduce(Device &device, const ReduceOperation &op)
      : ReduceBase(Arch::cuda, device, op) {
    if (size_ > std::size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument(
          "Reduce exceeds the CUDA primitive count ABI");
    auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device);
    if (!cuda)
      throw std::invalid_argument("Reduce requires a CUDA device");
    values_ = static_cast<char *>(cuda->get_memory_addr(op.values.pointer)) +
              op.values.pointer.offset;
    output_ = static_cast<char *>(cuda->get_memory_addr(op.output.pointer)) +
              op.output.pointer.offset;
    bytes_ = cuda::driver_reduce_workspace_bytes(
        static_cast<int>(size_),
        static_cast<cuda::CudaTransformValueType>(type_));
    if (bytes_) {
      Device::AllocParams params{};
      params.size = bytes_;
      auto memory = device.allocate_memory_unique(params);
      if (memory.second != RhiResult::success)
        throw std::runtime_error("Failed to allocate fixed reduce workspace");
      workspace_ = std::move(memory.first);
      scratch_ = cuda->get_memory_addr(workspace_->get_ptr());
    }
    cuda::driver_prepare_reduce();
  }
  void record_cuda(void *stream) override {
    const auto width = type_ < 3 ? 4 : 8;
    for (std::size_t lane = 0; lane < addressing_.lanes; ++lane) {
      const auto lane_offset = lane * width;
      cuda::driver_reduce_strided_with_workspace(
          values_, output_, static_cast<int>(size_),
          static_cast<cuda::CudaTransformValueType>(type_),
          addressing_.values_offset + lane_offset, addressing_.values_stride,
          addressing_.output_offset + lane_offset, addressing_.output_stride,
          static_cast<cuda::CudaHierarchicalReduceOp>(op_), scratch_, stream);
    }
  }

 private:
  DeviceAllocationUnique workspace_;
  void *scratch_{nullptr};
};
#endif
}  // namespace

std::size_t reduce_item_bytes(const ReduceOperation &op) {
  if (op.value_type < 0 || op.value_type >= reduce_types().size())
    throw std::invalid_argument("Reduce requires a supported dtype");
  if ((op.strategy == ReduceStrategy::packed_field) != !op.element_shape.empty())
    throw std::invalid_argument(
        "Packed Field reduce requires tensor shape; scalar strategies do not");
  std::size_t bytes = op.value_type < 3 ? 4 : 8;
  for (const auto axis : op.element_shape) {
    if (!axis || axis > std::numeric_limits<std::size_t>::max() / bytes)
      throw std::invalid_argument("Reduce payload exceeds address space");
    bytes *= axis;
  }
  if (op.size > std::numeric_limits<std::size_t>::max() / bytes)
    throw std::invalid_argument("Reduce prefix exceeds address space");
  return bytes;
}

ReduceAddressing qualify_reduce_addressing(
    Arch arch, Device &device, const ReduceOperation &op) {
  validate(device, op);
  for (const auto *array : {&op.values, &op.output}) {
    validate_native_shader_storage(arch, array->pointer);
    resolve_native_storage_range(arch, array->pointer, array->bytes);
  }
  const auto width = op.value_type < 3 ? 4 : 8;
  if (op.strategy == ReduceStrategy::packed_array ||
      op.strategy == ReduceStrategy::packed_field) {
    const auto record = reduce_item_bytes(op);
    validate_native_read_write_disjoint(
        {resolve_native_storage_range(arch, op.values.pointer, op.size * record)},
        {resolve_native_storage_range(arch, op.output.pointer, record)});
    return {0, record, 0, record, record / width};
  }
  const auto type = reduce_types()[op.value_type];
  const auto src = native_scalar_prefix(op.values, op.size, type);
  const auto dst = native_scalar_prefix(op.output, 1, type);
  if (!resolve_native_dense_array(arch, src).proven_disjoint(
          resolve_native_dense_array(arch, dst)))
    throw std::invalid_argument("Reduce input and output members overlap");
  return {std::uint64_t(src.strides->byte_offset),
          std::uint64_t(src.strides->index_bytes[0]),
          std::uint64_t(dst.strides->byte_offset),
          std::uint64_t(dst.strides->index_bytes[0])};
}

void register_reduce_operation(NativeOperationRegistry &registry) {
  registry.define<ReduceOperation>(validate);
  registry.register_schema<ReduceOperation>(
      "builtin", "reduce", {"values", "output"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<ReduceOperation> {
        operation_json::fields(parameters,
                               {"size", "dtype", "op", "cpu_max_threads", "strategy", "element_shape"});
        const auto size = operation_json::integer(parameters["size"]);
        if (!size || size > std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument(
              "Reduce size must be nonempty and addressable");
        const auto dtype = operation_json::name(parameters["dtype"]);
        const auto &types = reduce_types();
        int type = 0;
        while (type < types.size() && types[type]->to_string() != dtype)
          ++type;
        if (type == types.size())
          throw std::invalid_argument(
              "Reduce requires a supported numeric type");
        const auto name = operation_json::name(parameters["op"]);
        const int op = name == "sum"   ? 0
                       : name == "min" ? 1
                       : name == "max" ? 2
                                       : -1;
        if (op < 0)
          throw std::invalid_argument("Reduce supports sum/min/max");
        const auto threads =
            operation_json::optional_integer(parameters, "cpu_max_threads");
        if (threads &&
            (!*threads || *threads > std::numeric_limits<int>::max()))
          throw std::invalid_argument(
              "Reduce CPU configuration must fit its positive int ABI");
        auto strategy = ReduceStrategy::packed_array;
        if (const auto selected = parameters.obj.inner.find("strategy");
            selected != parameters.obj.inner.end()) {
          const auto name = operation_json::name(selected->second);
          if (name == "strided_scalar")
            strategy = ReduceStrategy::strided_scalar;
          else if (name == "scalar_field")
            strategy = ReduceStrategy::scalar_field;
          else if (name == "packed_field")
            strategy = ReduceStrategy::packed_field;
          else if (name != "packed_array")
            throw std::invalid_argument("Reduce strategy is invalid");
        }
        std::vector<std::uint64_t> element_shape;
        const auto shape = parameters.obj.inner.find("element_shape");
        if (shape != parameters.obj.inner.end()) {
          if (!shape->second.is_arr())
            throw std::invalid_argument("Reduce element_shape must be an array");
          for (const auto &axis : shape->second.arr.inner)
            element_shape.push_back(operation_json::integer(axis));
        }
        ReduceOperation descriptor;
        descriptor.size = size;
        descriptor.value_type = type;
        descriptor.strategy = strategy;
        descriptor.element_shape = element_shape;
        reduce_item_bytes(descriptor);  // Reject arithmetic before any binding.
        const bool affine = strategy == ReduceStrategy::scalar_field ||
                            strategy == ReduceStrategy::strided_scalar;
        return
            [size, type, op, threads, strategy, element_shape, affine](const NativeOperationBindings &bindings) {
              return ReduceOperation{static_cast<std::size_t>(size),
                                     type,
                                     op,
                                     static_cast<int>(threads.value_or(0)),
                                     affine
                                         ? bindings.at("values").dense_array()
                                         : bindings.at("values").array(),
                                     affine
                                         ? bindings.at("output").dense_array()
                                         : bindings.at("output").array(),
                                     strategy, element_shape};
            };
      });
#ifdef TI_WITH_LLVM
  registry.register_context_factory<ReduceOperation>(
      host_arch(), NativeOperationMode::cpu, prepare_cpu);
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<ReduceOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const ReduceOperation &op) {
        auto &context = CUDAContext::get_instance();
        auto lock = context.get_submission_lock_guard();
        auto guard = context.get_guard();
        return std::make_shared<CudaReduce>(device, op);
      },
      [] {
        return NativeOperationAvailability{
            cuda::driver_hierarchical_available(), 0};
      });
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<ReduceOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const ReduceOperation &op) {
        return gfx::prepare_vulkan_reduce_operation(device, op);
      });
#endif
}
}  // namespace taichi::lang
