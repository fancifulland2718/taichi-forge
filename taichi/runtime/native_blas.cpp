#include "taichi/runtime/native_operation_registry.h"

#include <cmath>
#include <limits>

namespace taichi::lang {
#ifdef TI_WITH_CUDA
std::shared_ptr<NativeOperation> prepare_cuda_gemm_f32_operation(
    Arch arch,
    Device &device,
    const GemmF32Operation &operation);
NativeOperationAvailability probe_cuda_gemm_f32_operation();
#endif

std::array<std::size_t, 3> gemm_f32_element_counts(const GemmF32Operation &op) {
  if (!op.rows || !op.columns || !op.inner || !std::isfinite(op.alpha) ||
      !std::isfinite(op.beta))
    throw std::invalid_argument(
        "GEMM requires positive dimensions and finite f32 scalars");
  const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(float);
  auto product = [&](std::size_t a, std::size_t b) {
    if (a > limit / b)
      throw std::invalid_argument("GEMM matrix extent overflows storage size");
    return a * b;
  };
  return {product(op.rows, op.inner), product(op.inner, op.columns),
          product(op.rows, op.columns)};
}

void register_gemm_operation(NativeOperationRegistry &registry) {
  registry.define<GemmF32Operation>(
      [](Device &device, const GemmF32Operation &op) {
        gemm_f32_element_counts(op);
        if (op.a.device != &device || op.b.device != &device ||
            op.output.device != &device)
          throw std::invalid_argument("GEMM storage belongs to another device");
      });
#ifdef TI_WITH_CUDA
  registry.register_factory<GemmF32Operation>(
      Arch::cuda, NativeOperationMode::cuda_capture,
      prepare_cuda_gemm_f32_operation, probe_cuda_gemm_f32_operation);
#endif
}

std::shared_ptr<NativeOperation> prepare_gemm_f32_operation(
    Arch arch,
    Device &device,
    const GemmF32Operation &operation) {
  return native_operation_registry().prepare(arch, device, operation);
}
}  // namespace taichi::lang
