#include "taichi/program/program.h"

#include <limits>

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/rhi/cuda/cuda_driver.h"

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

}  // namespace

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
  auto &cublas = CUBLASDriver::get_instance();
  TI_ERROR_IF(!cublas.is_loaded() && !cublas.load_cublas(),
              "CUDA cuBLAS could not load a compatible shared library.");

  auto handle = static_cast<cublasHandle_t>(cuda_cublas_gemm_handle_);
  if (!handle) {
    const auto create_status = cublas.cubCreate.call(&handle);
    TI_ERROR_IF(create_status != CUBLAS_STATUS_SUCCESS || !handle,
                "CUDA cuBLAS failed to create a handle (status {}).",
                create_status);
    const auto stream_status = cublas.cubSetStream.call(handle, nullptr);
    const auto pointer_status =
        stream_status == CUBLAS_STATUS_SUCCESS
            ? cublas.cubSetPointerMode.call(handle, CUBLAS_POINTER_MODE_HOST)
            : stream_status;
    if (stream_status != CUBLAS_STATUS_SUCCESS ||
        pointer_status != CUBLAS_STATUS_SUCCESS) {
      const auto destroy_status = cublas.cubDestroy.call(handle);
      TI_WARN_IF(destroy_status != CUBLAS_STATUS_SUCCESS,
                 "CUDA cuBLAS cleanup after handle setup returned status {}.",
                 destroy_status);
      TI_ERROR("CUDA cuBLAS failed to bind the runtime stream or host scalar "
               "mode (stream status {}, pointer status {}).",
               stream_status, pointer_status);
    }
    cuda_cublas_gemm_handle_ = handle;
  }

  auto *a_ptr = reinterpret_cast<const float *>(get_ndarray_data_ptr_as_int(a));
  auto *b_ptr = reinterpret_cast<const float *>(get_ndarray_data_ptr_as_int(b));
  auto *output_ptr =
      reinterpret_cast<float *>(get_ndarray_data_ptr_as_int(output));
  TI_ERROR_IF(!a_ptr || !b_ptr || !output_ptr,
              "CUDA cuBLAS GEMM received a null device pointer.");

  // cuBLAS is column-major. Row-major C = A * B is the equivalent
  // column-major C^T = B^T * A^T over the same compact allocations.
  const auto status = cublas.cubSgemm.call(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b_ptr, n, a_ptr, k,
      &beta, output_ptr, n);
  TI_ERROR_IF(status != CUBLAS_STATUS_SUCCESS,
              "CUDA cuBLAS SGEMM failed (status {}).", status);
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
