#include "taichi/runtime/native_indexed_copy.h"

#include <array>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_LLVM
#include "taichi/rhi/cpu/cpu_device.h"
#include "taichi/rhi/cpu/indexed_copy.h"
#include "taichi/rhi/cpu/scatter_validation.h"
#endif
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/linear_ptx.h"
#endif

namespace taichi::lang {
namespace {
const std::array<DataType, 6> &copy_types() {
  static const std::array<DataType, 6> types{
      PrimitiveType::i32, PrimitiveType::u32, PrimitiveType::f32,
      PrimitiveType::i64, PrimitiveType::u64, PrimitiveType::f64};
  return types;
}

void validate(Device &device, const IndexedCopyOperation &op) {
  const auto bytes = indexed_copy_item_bytes(op);
  if (!op.size || !op.src_items || !op.dst_items ||
      op.size > (op.scatter ? op.src_items : op.dst_items))
    throw std::invalid_argument(
        "Indexed copy requires nonempty addressable prefixes");
  const auto check = [&](const NativeArrayBinding &a, std::size_t count,
                         const std::vector<std::uint64_t> &element_shape,
                         std::size_t record, DataType dtype) {
    if (a.pointer.device != &device)
      throw std::invalid_argument("Indexed copy requires same-device arrays");
    validate_native_array_layout(
        a, {{count},
            element_shape,
            record,
            static_cast<std::uint64_t>(data_type_size(dtype)),
            false,
            dtype->to_string()});
    if (a.shape[0] > std::numeric_limits<std::size_t>::max() / record ||
        a.bytes != a.shape[0] * record)
      throw std::invalid_argument(
          "Indexed copy requires exact packed storage extents");
  };
  check(op.src, op.src_items, op.element_shape, bytes, op.dtype);
  check(op.dst, op.dst_items, op.element_shape, bytes, op.dtype);
  check(op.indices, op.size, {}, sizeof(int32_t), PrimitiveType::i32);
}

class CopyBase : public NativeOperation {
 public:
  CopyBase(Arch arch, Device &device, const IndexedCopyOperation &op)
      : arch_(arch),
        device_(device),
        size_(op.size),
        bound_(op.scatter ? op.dst_items : op.src_items),
        bytes_(indexed_copy_item_bytes(op)),
        scatter_(op.scatter) {
    validate_indexed_copy_storage(arch, op);
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
  std::size_t size_, bound_, bytes_;
  bool scatter_;
  void *src_{nullptr}, *indices_{nullptr}, *dst_{nullptr};
};

#ifdef TI_WITH_LLVM
class CpuCopy final : public CopyBase {
 public:
  CpuCopy(NativePreparationContext &context, const IndexedCopyOperation &op)
      : CopyBase(context.arch, context.device, op),
        pool_(context.cpu_thread_pool),
        partitions_(cpu::indexed_copy_partitions(size_,
                                                 context.cpu_max_threads,
                                                 scatter_)),
        validator_(scatter_ ? size_ : 0, bound_) {
    auto *device = dynamic_cast<cpu::CpuDevice *>(&device_);
    if (!device ||
        device->map_range_for_cpu_native(op.src.pointer, op.src.bytes, &src_) !=
            RhiResult::success ||
        device->map_range_for_cpu_native(op.indices.pointer, op.indices.bytes,
                                         &indices_) != RhiResult::success ||
        device->map_range_for_cpu_native(op.dst.pointer, op.dst.bytes, &dst_) !=
            RhiResult::success ||
        (partitions_ > 1 && !pool_))
      throw std::invalid_argument(
          "Indexed-copy CPU storage or scheduler is invalid");
  }
  std::size_t workspace_bytes() const override {
    return validator_.workspace_bytes();
  }
  void run_cpu() override {
    if (!validator_.unique(static_cast<const int32_t *>(indices_)))
      throw std::invalid_argument(
          "CPU native plain scatter requires unique destination indices");
    cpu::indexed_copy(static_cast<const uint8_t *>(src_),
                      static_cast<const int32_t *>(indices_),
                      static_cast<uint8_t *>(dst_), size_, bound_, bytes_,
                      scatter_, partitions_, pool_);
  }

 private:
  ThreadPool *pool_;
  int partitions_;
  cpu::PlainScatterIndexValidator validator_;
};
#endif

#ifdef TI_WITH_CUDA
class CudaCopy final : public CopyBase {
 public:
  CudaCopy(Device &device, const IndexedCopyOperation &op)
      : CopyBase(Arch::cuda, device, op) {
    if (size_ > std::numeric_limits<uint32_t>::max() ||
        bytes_ / sizeof(uint32_t) > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument(
          "Indexed copy exceeds its CUDA uint32 count/payload ABI");
    auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device_);
    if (!cuda)
      throw std::invalid_argument("Indexed copy requires a CUDA device");
    const auto address = [&](const NativeArrayBinding &a) {
      return static_cast<char *>(cuda->get_memory_addr(a.pointer)) +
             a.pointer.offset;
    };
    src_ = address(op.src);
    indices_ = address(op.indices);
    dst_ = address(op.dst);
    cuda::driver_prepare_indexed_copy(operation(), bytes_ / sizeof(uint32_t));
  }
  void record_cuda(void *stream) override {
    // The index is signed i32; larger backing arrays need no artificial size
    // rejection, since positions beyond INT32_MAX cannot be named by this ABI.
    const auto bound = std::min<std::size_t>(bound_, std::size_t{1} << 31);
    cuda::driver_indexed_copy_on_stream(src_, indices_, dst_, size_, bound,
                                        bytes_ / sizeof(uint32_t), operation(),
                                        stream);
  }

 private:
  cuda::CudaIndexedCopyOp operation() const {
    return scatter_ ? cuda::CudaIndexedCopyOp::scatter
                    : cuda::CudaIndexedCopyOp::gather;
  }
};
#endif
}  // namespace

std::size_t indexed_copy_item_bytes(const IndexedCopyOperation &op) {
  const auto &types = copy_types();
  if (std::find(types.begin(), types.end(), op.dtype) == types.end())
    throw std::invalid_argument(
        "Indexed copy requires a supported numeric payload");
  std::size_t bytes = data_type_size(op.dtype);
  for (const auto axis : op.element_shape) {
    if (!axis || axis > std::numeric_limits<std::size_t>::max() / bytes)
      throw std::invalid_argument("Indexed-copy payload exceeds address space");
    bytes *= axis;
  }
  return bytes;
}

void validate_indexed_copy_storage(Arch arch, const IndexedCopyOperation &op) {
  for (const auto *a : {&op.src, &op.indices, &op.dst}) {
    validate_native_shader_storage(arch, a->pointer);
    resolve_native_storage_range(arch, a->pointer, a->bytes);
  }
  const auto bytes = indexed_copy_item_bytes(op);
  validate_native_disjoint_writes(
      {resolve_native_storage_range(
           arch, op.src.pointer, (op.scatter ? op.size : op.src_items) * bytes),
       resolve_native_storage_range(arch, op.indices.pointer,
                                    op.size * sizeof(int32_t))},
      {{resolve_native_storage_range(
            arch, op.dst.pointer,
            (op.scatter ? op.dst_items : op.size) * bytes),
        std::nullopt}});
}

void register_indexed_copy_operation(NativeOperationRegistry &registry) {
  registry.define<IndexedCopyOperation>(validate);
  registry.register_schema<IndexedCopyOperation>(
      "builtin", "indexed_copy", {"src", "indices", "dst"},
      [](const liong::json::JsonValue &p)
          -> NativeOperationRegistry::Binder<IndexedCopyOperation> {
        operation_json::fields(p, {"size", "src_items", "dst_items", "dtype",
                                   "element_shape", "op"});
        IndexedCopyOperation op;
        const auto count = [&](const char *name) {
          const auto n = operation_json::integer(p[name]);
          if (!n || n > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument(
                "Indexed-copy extents must be nonempty and addressable");
          return static_cast<std::size_t>(n);
        };
        op.size = count("size");
        op.src_items = count("src_items");
        op.dst_items = count("dst_items");
        const auto dtype = operation_json::name(p["dtype"]);
        bool found = false;
        for (const auto &type : copy_types())
          if (type->to_string() == dtype) {
            op.dtype = type;
            found = true;
          }
        if (!found || !p["element_shape"].is_arr())
          throw std::invalid_argument(
              "Indexed copy requires a numeric payload layout");
        for (const auto &axis : p["element_shape"].arr.inner)
          op.element_shape.push_back(operation_json::integer(axis));
        indexed_copy_item_bytes(op);
        const auto mode = operation_json::name(p["op"]);
        if (mode != "gather" && mode != "scatter")
          throw std::invalid_argument(
              "Indexed-copy operation must be gather or scatter");
        op.scatter = mode == "scatter";
        if (op.size > (op.scatter ? op.src_items : op.dst_items))
          throw std::invalid_argument(
              "Indexed-copy prefix exceeds its sequential side");
        return [op](const NativeOperationBindings &bindings) {
          auto bound = op;
          bound.src = bindings.at("src").array();
          bound.indices = bindings.at("indices").array();
          bound.dst = bindings.at("dst").array();
          return bound;
        };
      });
#ifdef TI_WITH_LLVM
  registry.register_context_factory<IndexedCopyOperation>(
      host_arch(), NativeOperationMode::cpu,
      [](NativePreparationContext &c, const IndexedCopyOperation &op) {
        return std::make_shared<CpuCopy>(c, op);
      });
#endif
#ifdef TI_WITH_CUDA
  registry.register_factory<IndexedCopyOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const IndexedCopyOperation &op) {
        auto &context = CUDAContext::get_instance();
        auto lock = context.get_submission_lock_guard();
        auto guard = context.get_guard();
        return std::make_shared<CudaCopy>(device, op);
      },
      [] {
        return NativeOperationAvailability{
            cuda::driver_indexed_copy_available(), 0};
      });
#endif
#ifdef TI_WITH_VULKAN
  registry.register_factory<IndexedCopyOperation>(
      Arch::vulkan, NativeOperationMode::vulkan_inline,
      [](Arch, Device &device, const IndexedCopyOperation &op) {
        return gfx::prepare_vulkan_indexed_copy_operation(device, op);
      });
#endif
}
}  // namespace taichi::lang
