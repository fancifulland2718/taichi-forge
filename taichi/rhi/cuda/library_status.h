#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace taichi::lang::cuda::detail {

struct LibraryStatus {
  const char *library;
  const char *name;
};

template <std::size_t N>
inline const char *status_name(const char *const (&names)[N],
                               std::uint32_t status) {
  return status < N && names[status] ? names[status] : "UNKNOWN_STATUS";
}

// The dynamic wrappers retain the actual vendor symbol, including optional
// entry points. Select its status domain only on the error path; do not load a
// vendor library, query the CUDA driver, or add state to every wrapper.
// Numeric ABI values: NVIDIA Fortran CUDA Interfaces 26.9 and cuFFT 13.4.
inline LibraryStatus library_status(std::string_view symbol,
                                    std::uint32_t status) {
  if (symbol.substr(0, 8) == "cusolver") {
    static constexpr const char *names[] = {
        "CUSOLVER_STATUS_SUCCESS",
        "CUSOLVER_STATUS_NOT_INITIALIZED",
        "CUSOLVER_STATUS_ALLOC_FAILED",
        "CUSOLVER_STATUS_INVALID_VALUE",
        "CUSOLVER_STATUS_ARCH_MISMATCH",
        "CUSOLVER_STATUS_MAPPING_ERROR",
        "CUSOLVER_STATUS_EXECUTION_FAILED",
        "CUSOLVER_STATUS_INTERNAL_ERROR",
        "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
        "CUSOLVER_STATUS_NOT_SUPPORTED",
        "CUSOLVER_STATUS_ZERO_PIVOT",
        "CUSOLVER_STATUS_INVALID_LICENSE",
        "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED",
        "CUSOLVER_STATUS_IRS_PARAMS_INVALID",
        "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC",
        "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE",
        "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER",
        nullptr,
        nullptr,
        nullptr,
        "CUSOLVER_STATUS_IRS_INTERNAL_ERROR",
        "CUSOLVER_STATUS_IRS_NOT_SUPPORTED",
        "CUSOLVER_STATUS_IRS_OUT_OF_RANGE",
        "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES",
        nullptr,
        "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED",
        "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED",
        nullptr,
        nullptr,
        nullptr,
        "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR",
        "CUSOLVER_STATUS_INVALID_WORKSPACE"};
    return {"cuSOLVER", status_name(names, status)};
  }
  if (symbol.substr(0, 8) == "cusparse") {
    static constexpr const char *names[] = {
        "CUSPARSE_STATUS_SUCCESS",
        "CUSPARSE_STATUS_NOT_INITIALIZED",
        "CUSPARSE_STATUS_ALLOC_FAILED",
        "CUSPARSE_STATUS_INVALID_VALUE",
        "CUSPARSE_STATUS_ARCH_MISMATCH",
        "CUSPARSE_STATUS_MAPPING_ERROR",
        "CUSPARSE_STATUS_EXECUTION_FAILED",
        "CUSPARSE_STATUS_INTERNAL_ERROR",
        "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
        "CUSPARSE_STATUS_ZERO_PIVOT",
        "CUSPARSE_STATUS_NOT_SUPPORTED",
        "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES"};
    return {"cuSPARSE", status_name(names, status)};
  }
  if (symbol.substr(0, 6) == "cublas") {
    static constexpr const char *names[] = {"CUBLAS_STATUS_SUCCESS",
                                            "CUBLAS_STATUS_NOT_INITIALIZED",
                                            nullptr,
                                            "CUBLAS_STATUS_ALLOC_FAILED",
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            "CUBLAS_STATUS_INVALID_VALUE",
                                            "CUBLAS_STATUS_ARCH_MISMATCH",
                                            nullptr,
                                            nullptr,
                                            "CUBLAS_STATUS_MAPPING_ERROR",
                                            nullptr,
                                            "CUBLAS_STATUS_EXECUTION_FAILED",
                                            "CUBLAS_STATUS_INTERNAL_ERROR",
                                            "CUBLAS_STATUS_NOT_SUPPORTED",
                                            "CUBLAS_STATUS_LICENSE_ERROR"};
    return {"cuBLAS", status_name(names, status)};
  }
  if (symbol.substr(0, 5) == "cufft") {
    static constexpr const char *names[] = {"CUFFT_SUCCESS",
                                            "CUFFT_INVALID_PLAN",
                                            "CUFFT_ALLOC_FAILED",
                                            "CUFFT_INVALID_TYPE",
                                            "CUFFT_INVALID_VALUE",
                                            "CUFFT_INTERNAL_ERROR",
                                            "CUFFT_EXEC_FAILED",
                                            "CUFFT_SETUP_FAILED",
                                            "CUFFT_INVALID_SIZE",
                                            "CUFFT_UNALIGNED_DATA",
                                            "CUFFT_INCOMPLETE_PARAMETER_LIST",
                                            "CUFFT_INVALID_DEVICE",
                                            "CUFFT_PARSE_ERROR",
                                            "CUFFT_NO_WORKSPACE",
                                            "CUFFT_NOT_IMPLEMENTED",
                                            "CUFFT_LICENSE_ERROR",
                                            "CUFFT_NOT_SUPPORTED",
                                            "CUFFT_MISSING_DEPENDENCY",
                                            "CUFFT_NVRTC_FAILURE",
                                            "CUFFT_NVJITLINK_FAILURE",
                                            "CUFFT_NVSHMEM_FAILURE"};
    return {"cuFFT", status_name(names, status)};
  }
  return {nullptr, nullptr};
}

}  // namespace taichi::lang::cuda::detail
