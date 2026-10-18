#include "taichi/runtime/native_scatter_add.h"

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
const std::array<DataType, 6> &scatter_types() {
  static const std::array<DataType, 6> types{
      PrimitiveType::i32, PrimitiveType::f32, PrimitiveType::u32,
      PrimitiveType::u64, PrimitiveType::i64, PrimitiveType::f64};
  return types;
}

void validate(Device &device, const ScatterAddOperation &op) {
  const auto &types = scatter_types();
  if (!op.size || !op.dst_items || op.cpu_max_threads < 0 ||
      op.value_type < 0 || op.value_type >= types.size())
    throw std::invalid_argument(
        "Scatter-add requires nonempty numeric prefixes");
  for (const auto *a : {&op.src, &op.indices, &op.dst}) {
    const auto type =
        a == &op.indices ? DataType(PrimitiveType::i32) : types[op.value_type];
    const std::uint64_t width = data_type_size(type);
    const auto count = a == &op.dst ? op.dst_items : op.size;
    if (a->pointer.device != &device)
      throw std::invalid_argument("Scatter-add requires same-device arrays");
    validate_native_array_layout(
        *a, {{count}, {}, width, width, false, type->to_string()});
    if (a->shape[0] > std::numeric_limits<std::size_t>::max() / width ||
        a->bytes != a->shape[0] * width)
      throw std::invalid_argument(
          "Scatter-add requires exact packed scalar storage");
  }
}

class ScatterBase : public NativeOperation {
 public:
  ScatterBase(Arch arch, Device &device, const ScatterAddOperation &op)
      : arch_(arch),
        device_(device),
        size_(op.size),
        bound_(op.dst_items),
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
  std::size_t size_, bound_, bytes_{0};
  int type_;
  void *src_{nullptr}, *indices_{nullptr}, *dst_{nullptr};
};

#ifdef TI_WITH_LLVM
template <typename T>
class CpuScatter final : public ScatterBase {
 public:
  CpuScatter(NativePreparationContext &context, const ScatterAddOperation &op)
      : ScatterBase(context.arch, context.device, op),
        pool_(context.cpu_thread_pool),
        partitions_(cpu::scatter_add_partitions<T>(
            size_,
            bound_,
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
          "Scatter-add CPU storage or scheduler is invalid");
    bytes_ = cpu::scatter_add_workspace_bytes<T>(bound_, partitions_);
    partial_.resize(bytes_ / sizeof(T));
  }
  void run_cpu() override {
    std::fill(partial_.begin(), partial_.end(), T{});
    cpu::scatter_add(static_cast<const T *>(src_),
                     static_cast<const int32_t *>(indices_),
                     static_cast<T *>(dst_), size_, bound_, partitions_,
                     partial_.data(), pool_);
  }

 private:
  ThreadPool *pool_;
  int partitions_;
  std::vector<T> partial_;
};

std::shared_ptr<NativeResource> prepare_cpu(NativePreparationContext &c,
                                            const ScatterAddOperation &op) {
  switch (op.value_type) {
    case 0:
      return std::make_shared<CpuScatter<int32_t>>(c, op);
    case 1:
      return std::make_shared<CpuScatter<float>>(c, op);
    case 2:
      return std::make_shared<CpuScatter<uint32_t>>(c, op);
    case 3:
      return std::make_shared<CpuScatter<uint64_t>>(c, op);
    case 4:
      return std::make_shared<CpuScatter<int64_t>>(c, op);
    case 5:
      return std::make_shared<CpuScatter<double>>(c, op);
    default:
      throw std::invalid_argument("Unsupported CPU scatter-add dtype");
  }
}
#endif

#ifdef TI_WITH_CUDA
class CudaScatter final : public ScatterBase {
 public:
  CudaScatter(Device &device, const ScatterAddOperation &op)
      : ScatterBase(Arch::cuda, device, op) {
    if (size_ > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument(
          "Scatter-add exceeds its CUDA uint32 count ABI");
    auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device_);
    if (!cuda)
      throw std::invalid_argument("Scatter-add requires a CUDA device");
    const auto address = [&](const NativeArrayBinding &a) {
      return static_cast<char *>(cuda->get_memory_addr(a.pointer)) +
             a.pointer.offset;
    };
    src_ = address(op.src);
    indices_ = address(op.indices);
    dst_ = address(op.dst);
    cuda::driver_prepare_scatter_add(
        static_cast<cuda::CudaTransformValueType>(type_));
  }
  void record_cuda(void *stream) override {
    const auto bound = std::min<std::size_t>(bound_, std::size_t{1} << 31);
    cuda::driver_scatter_add_on_stream(
        src_, indices_, dst_, size_, bound,
        static_cast<cuda::CudaTransformValueType>(type_), stream);
  }
};
#endif
}  // namespace

void validate_scatter_add_storage(Arch arch, const ScatterAddOperation &op) {
  for (const auto *a : {&op.src, &op.indices, &op.dst}) {
    validate_native_shader_storage(arch, a->pointer);
    resolve_native_storage_range(arch, a->pointer, a->bytes);
  }
  const auto width = op.value_type < 3 ? 4 : 8;
  validate_native_read_write_disjoint(
      {resolve_native_storage_range(arch, op.src.pointer, op.size * width),
       resolve_native_storage_range(arch, op.indices.pointer,
                                    op.size * sizeof(int32_t))},
      {resolve_native_storage_range(arch, op.dst.pointer,
                                    op.dst_items * width)});
}

void register_scatter_add_operation(NativeOperationRegistry &registry) {
  registry.define<ScatterAddOperation>(validate);
  registry.register_schema<ScatterAddOperation>(
      "builtin", "scatter_add", {"src", "indices", "dst"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<ScatterAddOperation> {
        operation_json::fields(
            p, {"size", "dst_items", "dtype", "cpu_max_threads"});
        const auto size = operation_json::integer(p["size"]);
        const auto bound = operation_json::integer(p["dst_items"]);
        if (!size || !bound || size > std::numeric_limits<std::size_t>::max() ||
            bound > std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument(
              "Scatter-add extents must be nonempty and addressable");
        const auto dtype = operation_json::name(p["dtype"]);
        const auto &types = scatter_types();
        int type = 0;
        while (type < types.size() && types[type]->to_string() != dtype)
          ++type;
        if (type == types.size())
          throw std::invalid_argument("Unsupported scatter-add dtype");
        const auto threads =
            operation_json::optional_integer(p, "cpu_max_threads");
        if (threads &&
            (!*threads || *threads > std::numeric_limits<int>::max()))
          throw std::invalid_argument(
              "Scatter-add CPU configuration must fit its positive int ABI");
        return [size, bound, type,
                threads](const NativeOperationBindings &bindings) {
          return ScatterAddOperation{static_cast<std::size_t>(size),
                                     static_cast<std::size_t>(bound),
                                     type,
                                     static_cast<int>(threads.value_or(0)),
                                     bindings.at("src").array(),
                                     bindings.at("indices").array(),
                                     bindings.at("dst").array()};
        };
      });
#ifdef TI_WITH_LLVM
  registry.register_context_factory<ScatterAddOperation>(
      host_arch(), NativeOperationMode::cpu, prepare_cpu);
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<ScatterAddOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const ScatterAddOperation &op) {
        auto &context = CUDAContext::get_instance();
        auto lock = context.get_submission_lock_guard();
        auto guard = context.get_guard();
        return std::make_shared<CudaScatter>(device, op);
      },
      [] {
        return NativeOperationAvailability{
            cuda::driver_hierarchical_available(), 0};
      });
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<ScatterAddOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const ScatterAddOperation &op) {
        return gfx::prepare_vulkan_scatter_add_operation(device, op);
      });
#endif
}
}  // namespace taichi::lang
