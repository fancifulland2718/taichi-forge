#include "taichi/runtime/native_retained_scan.h"

#include <algorithm>
#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#include "taichi/runtime/native_operation_registry.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/hierarchical_ptx.h"
#endif

namespace taichi::lang {
namespace {
void validate(Device &device, const RetainedScanOperation &op) {
  if (op.size < 0 ||
      (op.dtype != PrimitiveType::i32 && op.dtype != PrimitiveType::u32) ||
      op.values.scalar_type != op.dtype ||
      op.scratch.scalar_type != PrimitiveType::u32 ||
      op.values.pointer.device != &device ||
      op.scratch.pointer.device != &device)
    throw std::invalid_argument(
        "Retained scan requires i32/u32 values and same-device u32 scratch");
  validate_native_array_layout(op.values, {{std::uint64_t(op.size)}, {}, 4, 4});
  validate_native_array_layout(op.scratch, {{1}, {}, 4, 4});
  for (const auto *array : {&op.values, &op.scratch})
    if (array->shape[0] > std::numeric_limits<std::uint64_t>::max() / 4 ||
        array->bytes != array->shape[0] * 4)
      throw std::invalid_argument(
          "Retained scan requires exact packed storage extents");
  if (!resolve_native_storage_range(Arch::cuda, op.values.pointer,
                                    op.values.bytes)
           .proven_disjoint(resolve_native_storage_range(
               Arch::cuda, op.scratch.pointer, op.scratch.bytes)))
    throw std::invalid_argument(
        "Retained scan values and scratch must be disjoint");
}

#ifdef TI_WITH_CUDA
class CudaRetainedScan final : public NativeOperation {
 public:
  CudaRetainedScan(Device &device, const RetainedScanOperation &op)
      : device_(device),
        size_(op.size),
        type_(op.dtype == PrimitiveType::i32
                  ? cuda::CudaTransformValueType::i32
                  : cuda::CudaTransformValueType::u32) {
    const auto required = std::max<std::size_t>(
        4, cuda::driver_scan_workspace_bytes(size_, type_));
    if (op.scratch.bytes != required)
      throw std::invalid_argument(
          "Retained scan requires its exact selected scratch layout");
    auto *cuda_device = dynamic_cast<cuda::CudaDevice *>(&device);
    if (!cuda_device)
      throw std::invalid_argument("Retained scan requires a CUDA device");
    auto address = [&](const NativeArrayBinding &a) -> void * {
      auto *base = static_cast<char *>(cuda_device->get_memory_addr(a.pointer));
      return base ? base + a.pointer.offset : nullptr;
    };
    values_ = address(op.values);
    scratch_ = address(op.scratch);
    cuda::driver_prepare_scan();
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    return 0;
  }
  void record_cuda(void *stream) override {
    cuda::driver_inclusive_scan_with_workspace(values_, size_, type_, scratch_,
                                               stream);
  }

 private:
  Device &device_;
  int size_;
  cuda::CudaTransformValueType type_;
  void *values_{nullptr}, *scratch_{nullptr};
};
#endif
}  // namespace

void register_retained_scan_operation(NativeOperationRegistry &registry) {
  registry.define<RetainedScanOperation>(validate);
  registry.register_schema<RetainedScanOperation>(
      "cuda_driver", "retained_scan", {"values", "scratch"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<RetainedScanOperation> {
        operation_json::fields(parameters, {"size", "dtype"});
        const auto size = operation_json::integer(parameters["size"]);
        // The shared CUDA primitive has an int count ABI. No application
        // threshold.
        if (size > std::uint64_t(std::numeric_limits<int>::max()))
          throw std::invalid_argument(
              "Retained scan size exceeds the shared primitive count ABI");
        const auto dtype = operation_json::name(parameters["dtype"]);
        if (dtype != "i32" && dtype != "u32")
          throw std::invalid_argument("Retained scan requires i32/u32 values");
        return [size, dtype](const NativeOperationBindings &bindings) {
          return RetainedScanOperation{
              static_cast<int>(size),
              dtype == "i32" ? PrimitiveType::i32 : PrimitiveType::u32,
              bindings.at("values").array(), bindings.at("scratch").array()};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_factory<RetainedScanOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const RetainedScanOperation &op) {
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaRetainedScan>(device, op);
      },
      [] {
        return NativeOperationAvailability{
            cuda::driver_hierarchical_available(), 0};
      });
#endif
}
}  // namespace taichi::lang
