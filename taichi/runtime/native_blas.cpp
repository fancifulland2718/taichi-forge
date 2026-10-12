#include "taichi/runtime/native_operation_registry.h"
#include "taichi/common/operation_json.h"

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
  registry.register_schema<GemmF32Operation>(
      "cublas", "gemm_f32", {"a", "b", "output"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<GemmF32Operation> {
        using namespace operation_json;
        fields(parameters, {"rows", "columns", "inner", "alpha", "beta",
                            "transpose_a", "transpose_b"});
        auto dimension = [&](const char *key) {
          const auto value = integer(parameters[key]);
          if (value > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument(
                "GEMM dimension overflows address space");
          return static_cast<std::size_t>(value);
        };
        GemmF32Operation descriptor{{},
                                    {},
                                    {},
                                    dimension("rows"),
                                    dimension("columns"),
                                    dimension("inner"),
                                    optional_f32(parameters, "alpha", 1),
                                    optional_f32(parameters, "beta", 0),
                                    optional_bool(parameters, "transpose_a"),
                                    optional_bool(parameters, "transpose_b")};
        gemm_f32_element_counts(descriptor);
        return [descriptor](const NativeOperationBindings &bindings) mutable {
          auto matrix = [&](const char *port, std::size_t rows,
                            std::size_t cols) {
            const auto &value = bindings.at(port);
            if (value.scalar_type != PrimitiveType::f32 ||
                !value.element_shape.empty() ||
                value.shape != std::vector<std::uint64_t>{rows, cols} ||
                value.bytes != rows * cols * sizeof(float) ||
                value.base_offset % sizeof(float) ||
                value.pointer.offset % sizeof(float))
              throw std::invalid_argument(
                  "GEMM requires aligned compact scalar f32 matrices matching "
                  "the physical shape and transpose flags");
            return value.pointer;
          };
          auto result = descriptor;
          result.a =
              matrix("a", result.transpose_a ? result.inner : result.rows,
                     result.transpose_a ? result.rows : result.inner);
          result.b =
              matrix("b", result.transpose_b ? result.columns : result.inner,
                     result.transpose_b ? result.inner : result.columns);
          result.output = matrix("output", result.rows, result.columns);
          return result;
        };
      },
      "gemm_f32");
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
