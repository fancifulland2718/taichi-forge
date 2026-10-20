#include "taichi/runtime/native_scatter_add.h"
#include "taichi/runtime/native_grouped_reduce.h"
#include "taichi/runtime/vulkan_grouped_reduce.h"

#include <array>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cpu/aggregation.h"
#include "taichi/rhi/cpu/scatter_add.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

namespace taichi::lang {
namespace {
const std::array<DataType, 6> &indexed_sum_types() {
  static const std::array<DataType, 6> types{
      PrimitiveType::i32, PrimitiveType::f32, PrimitiveType::u32,
      PrimitiveType::u64, PrimitiveType::i64, PrimitiveType::f64};
  return types;
}

const ScatterAddOperation &indexed_description(const ScatterAddOperation &op) {
  return op;
}

ScatterAddOperation indexed_description(const GroupedReduceOperation &op) {
  return {op.size,   op.num_groups, op.value_type, op.cpu_max_threads,
          op.values, op.keys,       op.output};
}

template <bool accumulate>
void validate(Device &device, const ScatterAddOperation &op) {
  const auto &types = indexed_sum_types();
  if ((accumulate && !op.size) || !op.dst_items || op.cpu_max_threads < 0 ||
      op.value_type < 0 || op.value_type >= types.size())
    throw std::invalid_argument(
        "Indexed sum requires numeric prefixes and nonempty output");
  const auto record = scatter_add_item_bytes(op);
  for (const auto *a : {&op.src, &op.indices, &op.dst}) {
    const auto type =
        a == &op.indices ? DataType(PrimitiveType::i32) : types[op.value_type];
    const std::uint64_t width = data_type_size(type);
    const auto item = a == &op.indices ? width : record;
    const auto shape = a == &op.indices ? std::vector<std::uint64_t>{}
                                        : op.element_shape;
    const auto count = a == &op.dst ? op.dst_items : op.size;
    if (a->pointer.device != &device)
      throw std::invalid_argument("Scatter-add requires same-device arrays");
    validate_native_array_layout(
        *a, {{count}, shape, item, width, false, type->to_string()});
    if (a->shape[0] > std::numeric_limits<std::size_t>::max() / item ||
        a->bytes != a->shape[0] * item)
      throw std::invalid_argument(
          "Scatter-add requires exact packed storage");
  }
}

class IndexedSumBase : public NativeOperation {
 public:
  IndexedSumBase(Arch arch, Device &device, const ScatterAddOperation &op)
      : arch_(arch),
        device_(device),
        size_(op.size),
        bound_(op.dst_items),
        record_(scatter_add_item_bytes(op)),
        lanes_(record_ / (op.value_type < 3 ? 4 : 8)),
        packed_(op.strategy == ScatterAddStrategy::packed_field),
        type_(op.value_type) {
    validate_scatter_add_storage(arch, op);
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
  std::size_t size_, bound_, record_, lanes_, bytes_{0};
  bool packed_;
  int type_;
  void *src_{nullptr}, *indices_{nullptr}, *dst_{nullptr};
};

#ifdef TI_WITH_LLVM
template <typename T, bool accumulate>
class CpuIndexedSum final : public IndexedSumBase {
 public:
  CpuIndexedSum(NativePreparationContext &context,
                const ScatterAddOperation &op)
      : IndexedSumBase(context.arch, context.device, op),
        pool_(context.cpu_thread_pool),
        partitions_(packed_
                        ? cpu::packed_scatter_add_partitions<T>(
                              size_, bound_, lanes_,
                              op.cpu_max_threads ? op.cpu_max_threads
                                                 : context.cpu_max_threads)
                        : cpu::scatter_add_partitions<T>(
                              size_, bound_,
                              op.cpu_max_threads ? op.cpu_max_threads
                                                 : context.cpu_max_threads)) {
    auto *cpu = dynamic_cast<cpu::CpuDevice *>(&device_);
    if (!cpu ||
        cpu->map_range_for_cpu_native(op.src.pointer, op.src.bytes, &src_) !=
            RhiResult::success ||
        cpu->map_range_for_cpu_native(op.indices.pointer, op.indices.bytes,
                                      &indices_) != RhiResult::success ||
        cpu->map_range_for_cpu_native(op.dst.pointer, op.dst.bytes, &dst_) !=
            RhiResult::success ||
        (partitions_ > 1 && !pool_))
      throw std::invalid_argument(
          "Indexed sum CPU storage or scheduler is invalid");
    bytes_ = cpu::scatter_add_workspace_bytes<T>(bound_ * lanes_, partitions_);
    partial_.resize(bytes_ / sizeof(T));
  }
  void run_cpu() override {
    std::fill(partial_.begin(), partial_.end(), T{});
    if (packed_) {
      cpu::packed_scatter_add(
          static_cast<const uint8_t *>(src_), 0, record_,
          static_cast<const int32_t *>(indices_),
          static_cast<uint8_t *>(dst_), 0, record_, size_, bound_, lanes_,
          partitions_, partial_.data(), pool_);
      return;
    }
    cpu::indexed_sum<T, accumulate>(static_cast<const T *>(src_),
                                    static_cast<const int32_t *>(indices_),
                                    static_cast<T *>(dst_), size_, bound_,
                                    partitions_, partial_.data(), pool_);
  }

 private:
  ThreadPool *pool_;
  int partitions_;
  std::vector<T> partial_;
};

template <bool accumulate>
std::shared_ptr<NativeResource> prepare_cpu(NativePreparationContext &c,
                                            const ScatterAddOperation &op) {
  switch (op.value_type) {
    case 0:
      return std::make_shared<CpuIndexedSum<int32_t, accumulate>>(c, op);
    case 1:
      return std::make_shared<CpuIndexedSum<float, accumulate>>(c, op);
    case 2:
      return std::make_shared<CpuIndexedSum<uint32_t, accumulate>>(c, op);
    case 3:
      return std::make_shared<CpuIndexedSum<uint64_t, accumulate>>(c, op);
    case 4:
      return std::make_shared<CpuIndexedSum<int64_t, accumulate>>(c, op);
    case 5:
      return std::make_shared<CpuIndexedSum<double, accumulate>>(c, op);
    default:
      throw std::invalid_argument("Unsupported CPU indexed sum dtype");
  }
}
#endif

#ifdef TI_WITH_CUDA
template <bool accumulate>
class CudaIndexedSum final : public IndexedSumBase {
 public:
  CudaIndexedSum(Device &device, const ScatterAddOperation &op)
      : IndexedSumBase(Arch::cuda, device, op) {
    if (size_ > UINT32_MAX || (!accumulate && bound_ > UINT32_MAX))
      throw std::invalid_argument(
          "Indexed sum exceeds its CUDA uint32 count ABI");
    auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device_);
    if (!cuda)
      throw std::invalid_argument("Indexed sum requires a CUDA device");
    const auto address = [&](const NativeArrayBinding &a) {
      return static_cast<char *>(cuda->get_memory_addr(a.pointer)) +
             a.pointer.offset;
    };
    src_ = address(op.src);
    indices_ = address(op.indices);
    dst_ = address(op.dst);
    const auto type = static_cast<cuda::CudaTransformValueType>(type_);
    if constexpr (accumulate)
      cuda::driver_prepare_scatter_add(type);
    else
      cuda::driver_prepare_grouped_reduce(type);
  }
  void record_cuda(void *stream) override {
    const auto type = static_cast<cuda::CudaTransformValueType>(type_);
    if constexpr (accumulate) {
      const auto bound = std::min<std::size_t>(bound_, std::size_t{1} << 31);
      const auto width = type_ < 3 ? 4 : 8;
      for (std::size_t lane = 0; lane < lanes_; ++lane)
        cuda::driver_scatter_add_strided_on_stream(
            src_, indices_, dst_, size_, bound, type, lane * width, record_,
            0, sizeof(int32_t), lane * width, record_, stream);
    } else {
      cuda::driver_grouped_reduce_on_stream(indices_, src_, dst_, size_, bound_,
                                            type, stream);
    }
  }
};
#endif
}  // namespace

std::size_t scatter_add_item_bytes(const ScatterAddOperation &op) {
  if (op.value_type < 0 || op.value_type >= indexed_sum_types().size() ||
      (op.strategy != ScatterAddStrategy::packed_array &&
       op.strategy != ScatterAddStrategy::packed_field))
    throw std::invalid_argument("Scatter-add requires a supported type and strategy");
  if ((op.strategy == ScatterAddStrategy::packed_field) != !op.element_shape.empty())
    throw std::invalid_argument("Packed Field scatter-add requires its tensor shape");
  std::size_t bytes = op.value_type < 3 ? 4 : 8;
  for (const auto axis : op.element_shape) {
    if (!axis || axis > std::numeric_limits<std::size_t>::max() / bytes)
      throw std::invalid_argument("Scatter-add payload exceeds address space");
    bytes *= axis;
  }
  if (std::max(op.size, op.dst_items) > std::numeric_limits<std::size_t>::max() / bytes)
    throw std::invalid_argument("Scatter-add prefix exceeds address space");
  return bytes;
}

void validate_scatter_add_storage(Arch arch, const ScatterAddOperation &op) {
  const auto record = scatter_add_item_bytes(op);
  for (const auto *a : {&op.src, &op.indices, &op.dst}) {
    validate_native_shader_storage(arch, a->pointer);
    resolve_native_storage_range(arch, a->pointer, a->bytes);
  }
  validate_native_read_write_disjoint(
      {resolve_native_storage_range(arch, op.src.pointer, op.size * record),
       resolve_native_storage_range(arch, op.indices.pointer,
                                    op.size * sizeof(int32_t))},
      {resolve_native_storage_range(arch, op.dst.pointer,
                                    op.dst_items * record)});
}

namespace {
template <typename Descriptor, bool accumulate>
void register_indexed_sum(NativeOperationRegistry &registry) {
  registry.define<Descriptor>([](Device &device, const Descriptor &op) {
    validate<accumulate>(device, indexed_description(op));
  });
  const char *input_port = accumulate ? "src" : "values";
  const char *index_port = accumulate ? "indices" : "keys";
  const char *output_port = accumulate ? "dst" : "output";
  registry.register_schema<Descriptor>(
      "builtin", Descriptor::kind, {input_port, index_port, output_port},
      [input_port, index_port, output_port](const liong::json::JsonValue &p) ->
      typename NativeOperationRegistry::Binder<Descriptor> {
        const char *bound_name = accumulate ? "dst_items" : "num_groups";
        if constexpr (accumulate)
          operation_json::fields(p, {"size", bound_name, "dtype", "cpu_max_threads",
                                     "strategy", "element_shape"});
        else
          operation_json::fields(
              p, {"size", bound_name, "dtype", "cpu_max_threads"});
        const auto size = operation_json::integer(p["size"]);
        const auto bound = operation_json::integer(p[bound_name]);
        if ((accumulate && !size) || !bound || size > SIZE_MAX ||
            bound > SIZE_MAX)
          throw std::invalid_argument(
              "Indexed sum extents must be addressable with nonempty output");
        const auto dtype = operation_json::name(p["dtype"]);
        const auto &types = indexed_sum_types();
        int type = 0;
        while (type < types.size() && types[type]->to_string() != dtype)
          ++type;
        if (type == types.size())
          throw std::invalid_argument("Unsupported indexed sum dtype");
        const auto threads =
            operation_json::optional_integer(p, "cpu_max_threads");
        if (threads && (!*threads || *threads > INT_MAX))
          throw std::invalid_argument(
              "Indexed sum CPU configuration must fit its positive int ABI");
        auto strategy = ScatterAddStrategy::packed_array;
        std::vector<std::uint64_t> element_shape;
        if constexpr (accumulate) {
          const auto requested = p.obj.inner.find("strategy");
          if (requested != p.obj.inner.end()) {
            const auto name = operation_json::name(requested->second);
            if (name == "packed_field")
              strategy = ScatterAddStrategy::packed_field;
            else if (name != "packed_array")
              throw std::invalid_argument("Scatter-add strategy is invalid");
          }
          const auto shape = p.obj.inner.find("element_shape");
          if (shape != p.obj.inner.end()) {
            if (!shape->second.is_arr())
              throw std::invalid_argument("Scatter-add element_shape must be an array");
            for (const auto &axis : shape->second.arr.inner)
              element_shape.push_back(operation_json::integer(axis));
          }
        }
        ScatterAddOperation description;
        description.size = size;
        description.dst_items = bound;
        description.value_type = type;
        description.strategy = strategy;
        description.element_shape = element_shape;
        scatter_add_item_bytes(description);  // Check arithmetic before binding.
        return [size, bound, type, threads, input_port, index_port,
                output_port, strategy, element_shape](const NativeOperationBindings &bindings) {
          const auto &source = bindings.at(input_port).array();
          const auto &index = bindings.at(index_port).array();
          const auto &output = bindings.at(output_port).array();
          if constexpr (accumulate)
            return Descriptor{
                size_t(size), size_t(bound),
                type,         static_cast<int>(threads.value_or(0)),
                source,       index,
                output, strategy, element_shape};
          else
            return Descriptor{
                size_t(size), size_t(bound),
                type,         static_cast<int>(threads.value_or(0)),
                index,        source,
                output};
        };
      });
#ifdef TI_WITH_LLVM
  registry.register_context_factory<Descriptor>(
      host_arch(), NativeOperationMode::cpu,
      [](NativePreparationContext &context, const Descriptor &op) {
        return prepare_cpu<accumulate>(context, indexed_description(op));
      });
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<Descriptor>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const Descriptor &op) {
        auto &context = CUDAContext::get_instance();
        auto lock = context.get_submission_lock_guard();
        auto guard = context.get_guard();
        return std::make_shared<CudaIndexedSum<accumulate>>(
            device, indexed_description(op));
      },
      [] {
        return NativeOperationAvailability{
            cuda::driver_hierarchical_available(), 0};
      });
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<Descriptor>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const Descriptor &op) {
        if constexpr (accumulate) {
          return gfx::prepare_vulkan_scatter_add_operation(device, op);
        } else {
          validate_scatter_add_storage(Arch::vulkan, indexed_description(op));
          const size_t width = op.value_type < 3 ? 4 : 8;
          return gfx::prepare_vulkan_grouped_reduce_operation(
              device, {op.keys.pointer, 4}, {op.values.pointer, width},
              {op.output.pointer, width}, op.size, op.num_groups,
              op.value_type);
        }
      });
#endif
}
}  // namespace

void register_scatter_add_operation(NativeOperationRegistry &registry) {
  register_indexed_sum<ScatterAddOperation, true>(registry);
}
void register_grouped_reduce_operation(NativeOperationRegistry &registry) {
  register_indexed_sum<GroupedReduceOperation, false>(registry);
}
}  // namespace taichi::lang
