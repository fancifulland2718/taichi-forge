#include "taichi/program/program.h"

#include <limits>

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/native_operation_registry.h"
#include "taichi/runtime/native_blas.h"

namespace taichi::lang {
namespace {

int cublas_dimension(std::size_t value, const char *name) {
  TI_ERROR_IF(value == 0 ||
                  value > static_cast<std::size_t>(
                              (std::numeric_limits<int>::max)()),
              "CUDA cuBLAS {} must be in [1, INT_MAX].", name);
  return static_cast<int>(value);
}

std::size_t matrix_elements(std::size_t rows,
                            std::size_t columns,
                            const char *name) {
  TI_ERROR_IF(rows > (std::numeric_limits<std::size_t>::max)() / columns,
              "CUDA cuBLAS {} element count overflow.", name);
  return rows * columns;
}

void validate_matrix(Ndarray *array,
                     std::size_t expected_elements,
                     const char *name,
                     Program *program) {
  TI_ERROR_IF(!array, "CUDA cuBLAS {} received a null ndarray.", name);
  TI_ERROR_IF(!array->get_element_shape().empty() ||
                  array->get_element_data_type() != PrimitiveType::f32 ||
                  array->get_nelement() != expected_elements ||
                  array->get_element_size() != sizeof(float32),
              "CUDA cuBLAS {} must be a compact scalar f32 ndarray with {} "
              "entries.",
              name, expected_elements);
  TI_ERROR_IF(array->owning_program() != program,
              "CUDA cuBLAS {} must belong to the active runtime.", name);
}

cublasHandle_t create_blas_handle(
    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST) {
  auto &driver = CUBLASDriver::get_instance();
  TI_ERROR_IF(!driver.load_cublas(),
              "CUDA cuBLAS could not load a compatible shared library.");
  cublasHandle_t handle = nullptr;
  const auto created = driver.cubCreate.call(&handle);
  TI_ERROR_IF(created != CUBLAS_STATUS_SUCCESS || !handle,
              "CUDA cuBLAS failed to create a handle (status {}).", created);
  const auto stream = driver.cubSetStream.call(handle, nullptr);
  const auto pointers =
      stream == CUBLAS_STATUS_SUCCESS
          ? driver.cubSetPointerMode.call(handle, pointer_mode)
          : stream;
  if (stream != CUBLAS_STATUS_SUCCESS || pointers != CUBLAS_STATUS_SUCCESS) {
    driver.cubDestroy.call(handle);
    TI_ERROR("CUDA cuBLAS handle setup failed (stream {}, pointer mode {}).",
             stream, pointers);
  }
  return handle;
}

void execute_gemm(cublasHandle_t handle,
                  int m,
                  int n,
                  int k,
                  const float *a,
                  const float *b,
                  float *output,
                  float alpha,
                  float beta,
                  bool transpose_a = false,
                  bool transpose_b = false) {
  // Row-major C = op(A) * op(B) is column-major C^T = op(B)^T * op(A)^T
  // over the same compact allocations. Leading dimensions are physical widths.
  const auto status = CUBLASDriver::get_instance().cubSgemm.call(
      handle, transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
      transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N, n, m, k, &alpha, b,
      transpose_b ? k : n, a, transpose_a ? m : k, &beta, output, n);
  TI_ERROR_IF(status != CUBLAS_STATUS_SUCCESS,
              "CUDA cuBLAS SGEMM failed (status {}).", status);
}

class CudaBlasOperation : public NativeOperation {
 public:
  CudaBlasOperation(Device &device, cublasPointerMode_t pointer_mode)
      : device_(device), handle_(create_blas_handle(pointer_mode)) {
  }
  ~CudaBlasOperation() override {
    // FixedExecution retires the captured graph before this handle. cuBLAS may
    // retain opaque allocations associated with its handle/capture.
    auto &cuda = CUDAContext::get_instance();
    auto lock = cuda.get_submission_lock_guard();
    auto guard = cuda.get_guard();
    const auto status = CUBLASDriver::get_instance().cubDestroy.call(handle_);
    TI_WARN_IF(status != CUBLAS_STATUS_SUCCESS,
               "CUDA cuBLAS handle destruction returned status {}.", status);
  }
  Arch arch() const override {
    return Arch::cuda;
  }
  Device &device() const override {
    return device_;
  }
  std::size_t workspace_bytes() const override {
    // Known Forge-owned scratch only; vendor-private handle/capture storage
    // cannot be queried through the basic cuBLAS API.
    return 0;
  }

 protected:
  void bind_stream(void *stream) {
    const auto status = CUBLASDriver::get_instance().cubSetStream.call(
        handle_, static_cast<CUstream>(stream));
    TI_ERROR_IF(status != CUBLAS_STATUS_SUCCESS,
                "CUDA cuBLAS capture stream setup failed (status {}).", status);
  }
  Device &device_;
  cublasHandle_t handle_;
};

class CudaGemmOperation final : public CudaBlasOperation {
 public:
  CudaGemmOperation(Device &device,
                    GemmF32Operation op,
                    const float *a,
                    const float *b,
                    float *output)
      : CudaBlasOperation(device, CUBLAS_POINTER_MODE_HOST),
        op_(op),
        a_(a),
        b_(b),
        output_(output) {
  }
  void record_cuda(void *stream) override {
    bind_stream(stream);
    execute_gemm(handle_, static_cast<int>(op_.rows),
                 static_cast<int>(op_.columns), static_cast<int>(op_.inner), a_,
                 b_, output_, op_.alpha, op_.beta, op_.transpose_a,
                 op_.transpose_b);
  }

 private:
  GemmF32Operation op_;
  const float *a_;
  const float *b_;
  float *output_;
};

class CudaDotOperation final : public CudaBlasOperation {
 public:
  CudaDotOperation(Device &device,
                   int size,
                   const float *x,
                   const float *y,
                   float *output)
      : CudaBlasOperation(device, CUBLAS_POINTER_MODE_DEVICE),
        size_(size),
        x_(x),
        y_(y),
        output_(output) {
  }
  void record_cuda(void *stream) override {
    bind_stream(stream);
    const auto status = CUBLASDriver::get_instance().cubSdot.call(
        handle_, size_, x_, 1, y_, 1, output_);
    TI_ERROR_IF(status != CUBLAS_STATUS_SUCCESS,
                "CUDA cuBLAS SDOT failed (status {}).", status);
  }

 private:
  int size_;
  const float *x_;
  const float *y_;
  float *output_;
};

NativeOperationAvailability probe_cuda_blas() {
  if (!CUDADriver::get_instance_without_context().nvidia_extensions_available())
    return {};
  auto &driver = CUBLASDriver::get_instance();
  if (!driver.load_cublas())
    return {};
  const auto caps = driver.capabilities();
  // Encode the public major/minor/patch properties without creating a handle.
  const auto version =
      caps.library_version_major >= 0 && caps.library_version_minor >= 0 &&
              caps.library_version_patch >= 0
          ? caps.library_version_major * 10000 +
                caps.library_version_minor * 100 + caps.library_version_patch
          : 0;
  return {driver.cubCreate.available() && driver.cubDestroy.available() &&
              driver.cubSetStream.available() &&
              driver.cubSetPointerMode.available(),
          static_cast<std::uint32_t>(version)};
}

}  // namespace

NativeOperationAvailability probe_cuda_gemm_f32_operation() {
  auto result = probe_cuda_blas();
  if (result.available)
    result.available = CUBLASDriver::get_instance().cubSgemm.available();
  return result;
}

NativeOperationAvailability probe_cuda_dot_f32_operation() {
  auto result = probe_cuda_blas();
  if (result.available) {
    auto &driver = CUBLASDriver::get_instance();
    result.available = driver.cubSdot.available();
  }
  return result;
}

std::shared_ptr<NativeOperation> prepare_cuda_gemm_f32_operation(
    Arch,
    Device &device,
    const GemmF32Operation &op) {
  auto *cuda_device = dynamic_cast<cuda::CudaDevice *>(&device);
  if (!cuda_device)
    throw std::invalid_argument("cuBLAS requires a CUDA device");
  for (auto extent : {op.rows, op.columns, op.inner})
    if (extent > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
      throw std::invalid_argument("cuBLAS SGEMM dimensions must fit int32");
  const auto counts = gemm_f32_element_counts(op);
  auto map = [&](DevicePtr ptr) {
    auto *base = static_cast<char *>(cuda_device->get_memory_addr(ptr));
    return reinterpret_cast<float *>(base ? base + ptr.offset : nullptr);
  };
  auto *a = map(op.a);
  auto *b = map(op.b);
  auto *output = map(op.output);
  const auto dst = reinterpret_cast<std::uintptr_t>(output);
  for (const auto &[pointer, count] :
       {std::pair{a, counts[0]}, std::pair{b, counts[1]}}) {
    const auto src = reinterpret_cast<std::uintptr_t>(pointer);
    if (!src || !dst || src % sizeof(float) || dst % sizeof(float) ||
        (src <= dst ? dst - src < count * sizeof(float)
                    : src - dst < counts[2] * sizeof(float)))
      throw std::invalid_argument(
          "GEMM requires aligned storage and output disjoint from inputs");
  }
  auto &cuda = CUDAContext::get_instance();
  auto lock = cuda.get_submission_lock_guard();
  auto guard = cuda.get_guard();
  return std::make_shared<CudaGemmOperation>(device, op, a, b, output);
}

std::shared_ptr<NativeOperation> prepare_cuda_dot_f32_operation(
    Arch,
    Device &device,
    const DotF32Operation &op) {
  auto *cuda_device = dynamic_cast<cuda::CudaDevice *>(&device);
  if (!cuda_device)
    throw std::invalid_argument("cuBLAS requires a CUDA device");
  if (op.size > static_cast<std::size_t>((std::numeric_limits<int>::max)()))
    throw std::invalid_argument("cuBLAS SDOT size must fit int32");
  auto map = [&](DevicePtr ptr) {
    auto *base = static_cast<char *>(cuda_device->get_memory_addr(ptr));
    return reinterpret_cast<float *>(base ? base + ptr.offset : nullptr);
  };
  auto *x = map(op.x);
  auto *y = map(op.y);
  auto *output = map(op.output);
  const auto dst = reinterpret_cast<std::uintptr_t>(output);
  if (!dst || dst % sizeof(float))
    throw std::invalid_argument("Dot requires aligned output storage");
  for (const auto *pointer : {x, y}) {
    const auto src = reinterpret_cast<std::uintptr_t>(pointer);
    if ((op.size && !src) || src % sizeof(float) ||
        (op.size && (src <= dst ? dst - src < op.size * sizeof(float)
                                : src - dst < sizeof(float))))
      throw std::invalid_argument(
          "Dot requires aligned storage and output disjoint from inputs");
  }
  auto &cuda = CUDAContext::get_instance();
  auto lock = cuda.get_submission_lock_guard();
  auto guard = cuda.get_guard();
  return std::make_shared<CudaDotOperation>(device, static_cast<int>(op.size),
                                            x, y, output);
}

std::size_t Program::cuda_cublas_gemm_f32(Ndarray *a,
                                          Ndarray *b,
                                          Ndarray *output,
                                          std::size_t rows,
                                          std::size_t columns,
                                          std::size_t inner,
                                          float alpha,
                                          float beta) {
  auto submission_guard = acquire_runtime_resource_submission_guard();
  TI_ERROR_IF(compile_config().arch != Arch::cuda,
              "CUDA cuBLAS GEMM requires the CUDA backend.");
  const int m = cublas_dimension(rows, "rows");
  const int n = cublas_dimension(columns, "columns");
  const int k = cublas_dimension(inner, "inner dimension");
  validate_matrix(a, matrix_elements(rows, inner, "A"), "A", this);
  validate_matrix(b, matrix_elements(inner, columns, "B"), "B", this);
  validate_matrix(output, matrix_elements(rows, columns, "output"), "output",
                  this);
  const auto a_allocation = a->get_device_allocation();
  const auto b_allocation = b->get_device_allocation();
  const auto output_allocation = output->get_device_allocation();
  TI_ERROR_IF(output_allocation == a_allocation ||
                  output_allocation == b_allocation,
              "CUDA cuBLAS GEMM output must not alias either input.");
  TI_ERROR_IF(!CUDADriver::get_instance_without_context()
                   .nvidia_extensions_available(),
              "CUDA cuBLAS GEMM requires the NVIDIA CUDA provider.");

  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  std::lock_guard<std::mutex> provider_lock(cuda_cublas_gemm_mutex_);
  auto handle = static_cast<cublasHandle_t>(cuda_cublas_gemm_handle_);
  if (!handle) {
    handle = create_blas_handle();
    cuda_cublas_gemm_handle_ = handle;
  }

  auto *a_ptr = reinterpret_cast<const float *>(get_ndarray_data_ptr_as_int(a));
  auto *b_ptr = reinterpret_cast<const float *>(get_ndarray_data_ptr_as_int(b));
  auto *output_ptr =
      reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output));
  TI_ERROR_IF(!a_ptr || !b_ptr || !output_ptr,
              "CUDA cuBLAS GEMM received a null device pointer.");

  execute_gemm(handle, m, n, k, a_ptr, b_ptr, output_ptr, alpha, beta);
  auto leases = acquire_ndarray_leases({a, b, output});
  pin_ndarray_launch_leases(leases);
  mark_runtime_submission_pending();
  return 0;
}

void Program::cuda_clear_cublas_gemm() {
  std::lock_guard<std::mutex> provider_lock(cuda_cublas_gemm_mutex_);
  auto handle = static_cast<cublasHandle_t>(cuda_cublas_gemm_handle_);
  cuda_cublas_gemm_handle_ = nullptr;
  if (!handle || runtime_has_fatal_fault() ||
      !CUBLASDriver::get_instance().is_loaded()) {
    return;
  }
  auto cuda_submission_guard =
      CUDAContext::get_instance().get_submission_lock_guard();
  auto context_guard = CUDAContext::get_instance().get_guard();
  const auto status = CUBLASDriver::get_instance().cubDestroy.call(handle);
  TI_WARN_IF(status != CUBLAS_STATUS_SUCCESS,
             "CUDA cuBLAS GEMM handle destruction returned status {}.",
             status);
}

}  // namespace taichi::lang

#else

namespace taichi::lang {

std::size_t Program::cuda_cublas_gemm_f32(Ndarray *,
                                          Ndarray *,
                                          Ndarray *,
                                          std::size_t,
                                          std::size_t,
                                          std::size_t,
                                          float,
                                          float) {
  TI_ERROR("CUDA cuBLAS GEMM requires TI_WITH_CUDA=ON.");
}

void Program::cuda_clear_cublas_gemm() {
}

}  // namespace taichi::lang

#endif
