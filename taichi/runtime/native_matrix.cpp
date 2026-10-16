#include "taichi/runtime/native_operation_registry.h"

#include <limits>
#include "taichi/common/operation_json.h"
#include "taichi/runtime/native_array_contract.h"
#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/rhi/cuda/primitives/matrix_ptx.h"
#endif

namespace taichi::lang {
namespace {
struct MatrixMmaOperation {
  static constexpr const char *kind = "cuda_matrix_mma_f16_f32";
  static constexpr std::uint32_t version = 1;
  int batch_count;
  NativeArrayBinding a, b, output;
};

void validate(Device &device, const MatrixMmaOperation &op) {
  if (op.batch_count <= 0)
    throw std::invalid_argument(
        "CUDA matrix MMA requires a positive batch count");
  const auto count = static_cast<std::uint64_t>(op.batch_count);
  const std::vector<std::uint64_t> shape =
      count == 1 ? std::vector<std::uint64_t>{16, 16}
                 : std::vector<std::uint64_t>{count, 16, 16};
  for (const auto *array : {&op.a, &op.b, &op.output}) {
    const auto dtype =
        array == &op.output ? PrimitiveType::f32 : PrimitiveType::f16;
    const auto item_bytes = array == &op.output ? 4u : 2u;
    if (array->pointer.device != &device || array->scalar_type != dtype)
      throw std::invalid_argument(
          "CUDA matrix MMA needs same-device f16 inputs and f32 output");
    validate_native_array_layout(*array, {shape, {}, item_bytes, 32, true});
    if (count > std::numeric_limits<std::size_t>::max() / (256 * item_bytes) ||
        array->bytes != count * 256 * item_bytes)
      throw std::invalid_argument(
          "CUDA matrix MMA requires exact packed matrix storage");
  }
  const auto output = resolve_native_storage_range(
      Arch::cuda, op.output.pointer, op.output.bytes);
  for (const auto *input : {&op.a, &op.b})
    if (!output.proven_disjoint(resolve_native_storage_range(
            Arch::cuda, input->pointer, input->bytes)))
      throw std::invalid_argument(
          "CUDA matrix MMA output must not overlap its inputs");
}

#ifdef TI_WITH_CUDA
class CudaMatrixMma final : public NativeOperation {
 public:
  CudaMatrixMma(Device &device, const MatrixMmaOperation &op)
      : device_(device), batch_count_(op.batch_count) {
    auto *cuda = dynamic_cast<cuda::CudaDevice *>(&device);
    if (!cuda)
      throw std::invalid_argument("CUDA matrix MMA requires a CUDA device");
    auto pointer = [&](const NativeArrayBinding &array) -> void * {
      auto *base = static_cast<char *>(cuda->get_memory_addr(array.pointer));
      auto *address = base ? base + array.pointer.offset : nullptr;
      if (!address || reinterpret_cast<std::uintptr_t>(address) % 32)
        throw std::invalid_argument(
            "CUDA matrix MMA buffers require 32-byte alignment");
      return address;
    };
    a_ = pointer(op.a);
    b_ = pointer(op.b);
    output_ = pointer(op.output);
    cuda::driver_prepare_matrix_mma_f16_f32();
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
    cuda::driver_matrix_mma_f16_f32(a_, b_, output_, batch_count_, stream);
  }

 private:
  Device &device_;
  int batch_count_;
  void *a_{nullptr}, *b_{nullptr}, *output_{nullptr};
};
#endif
}  // namespace

void register_matrix_operations(NativeOperationRegistry &registry) {
  registry.define<MatrixMmaOperation>(validate);
  registry.register_schema<MatrixMmaOperation>(
      "cuda_driver", "matrix_mma_f16_f32", {"a", "b", "output"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<MatrixMmaOperation> {
        operation_json::fields(parameters, {"batch_count"});
        const auto batch_count =
            operation_json::integer(parameters["batch_count"]);
        // The existing driver kernel uses a signed int batch/grid-count ABI.
        if (!batch_count ||
            batch_count > std::uint64_t(std::numeric_limits<int>::max()))
          throw std::invalid_argument(
              "CUDA matrix MMA batch count must fit its positive int ABI");
        return [batch_count](const NativeOperationBindings &bindings) {
          return MatrixMmaOperation{
              static_cast<int>(batch_count), bindings.at("a").array(),
              bindings.at("b").array(), bindings.at("output").array()};
        };
      });
#ifdef TI_WITH_CUDA
  registry.register_factory<MatrixMmaOperation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      [](Arch, Device &device, const MatrixMmaOperation &op) {
        auto &cuda = CUDAContext::get_instance();
        auto lock = cuda.get_submission_lock_guard();
        auto guard = cuda.get_guard();
        return std::make_shared<CudaMatrixMma>(device, op);
      },
      [] {
        return NativeOperationAvailability{
            cuda::driver_matrix_mma_f16_f32_available(), 0};
      });
#endif
}
}  // namespace taichi::lang
