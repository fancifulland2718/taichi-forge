#include "taichi/runtime/native_operation_registry.h"

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
