#include "taichi/runtime/native_operation_registry.h"
#include "taichi/common/operation_json.h"

#include <limits>

namespace taichi::lang {
#ifdef TI_WITH_CUDA
// The existing cuFFT implementation owns both JIT and AOT vendor plans.
std::shared_ptr<NativeOperation> prepare_cuda_fft_operation(
    Arch arch,
    Device &device,
    const FftC2COperation &operation);
NativeOperationAvailability probe_cuda_fft_operation();
#endif

std::size_t fft_c2c_element_count(const FftC2COperation &operation) {
  if (operation.dimensions.empty() || !operation.batch_count)
    throw std::invalid_argument(
        "FFT requires positive dimensions and batch count");
  const auto limit =
      std::numeric_limits<std::size_t>::max() / (2 * sizeof(float));
  std::size_t count = operation.batch_count;
  for (auto dimension : operation.dimensions) {
    if (!dimension || count > limit / dimension)
      throw std::invalid_argument(
          "FFT extent is zero or overflows storage size");
    count *= dimension;
  }
  return count;
}

void register_fft_operation(NativeOperationRegistry &registry) {
  registry.define<FftC2COperation>(
      [](Device &device, const FftC2COperation &op) {
        fft_c2c_element_count(op);
        if (op.input.device != &device || op.output.device != &device)
          throw std::invalid_argument("FFT storage belongs to another device");
      });
  registry.register_schema<FftC2COperation>(
      "cufft", "fft_c2c_f32", {"input", "output"},
      [](const liong::json::JsonValue &parameters)
          -> NativeOperationRegistry::Binder<FftC2COperation> {
        using namespace operation_json;
        fields(parameters, {"dimensions", "batch_count", "direction"});
        FftC2COperation descriptor{{}, {}, {}, 1, false};
        const auto &dimensions = parameters["dimensions"];
        if (!dimensions.is_arr() || dimensions.arr.inner.empty())
          throw std::invalid_argument(
              "FFT dimensions must be a nonempty array");
        for (const auto &axis : dimensions.arr.inner) {
          const auto value = integer(axis);
          if (value > std::numeric_limits<std::size_t>::max())
            throw std::invalid_argument(
                "FFT dimension overflows address space");
          descriptor.dimensions.push_back(static_cast<std::size_t>(value));
        }
        const auto batch =
            optional_integer(parameters, "batch_count").value_or(1);
        if (batch > std::numeric_limits<std::size_t>::max())
          throw std::invalid_argument(
              "FFT batch count overflows address space");
        descriptor.batch_count = static_cast<std::size_t>(batch);
        const auto direction = name(parameters["direction"]);
        if (direction != "forward" && direction != "inverse")
          throw std::invalid_argument(
              "FFT direction must be forward or inverse");
        descriptor.inverse = direction == "inverse";
        const auto bytes =
            fft_c2c_element_count(descriptor) * 2 * sizeof(float);
        return [descriptor, bytes](const NativeOperationBindings &bindings) {
          auto buffer = [&](const char *port) {
            const auto &value = bindings.at(port);
            if (value.scalar_type != PrimitiveType::f32 ||
                value.element_shape != std::vector<std::uint64_t>{2} ||
                value.base_offset % 8 || value.pointer.offset % 8)
              throw std::invalid_argument(
                  "FFT requires aligned interleaved complex f32 vector(2) "
                  "ndarrays");
            if (value.bytes != bytes)
              throw std::invalid_argument(
                  "FFT dimensions/batch do not match ndarray storage");
            return value.pointer;
          };
          auto result = descriptor;
          result.input = buffer("input");
          result.output = buffer("output");
          return result;
        };
      },
      "fft_c2c_f32");
#ifdef TI_WITH_CUDA
  registry.register_factory<FftC2COperation>(
      Arch::cuda, NativeOperationMode::cuda_capture, prepare_cuda_fft_operation,
      probe_cuda_fft_operation);
#endif
}

std::shared_ptr<NativeOperation> prepare_fft_operation(
    Arch arch,
    Device &device,
    const FftC2COperation &operation) {
  return native_operation_registry().prepare(arch, device, operation);
}
}  // namespace taichi::lang
