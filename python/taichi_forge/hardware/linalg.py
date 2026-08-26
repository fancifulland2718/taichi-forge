"""Public optional hardware linear-algebra provider API."""

from taichi_forge.hardware._amgx import AmgxProvider, AmgxSolver
from taichi_forge.hardware._linalg import (
    CudssPlan,
    CudssRefactorSolveRecording,
    CudssSolveRecording,
    CublasGemmRecording,
    CusparseSpmvRecording,
    cublas_is_available,
    cusparse_is_available,
    cudss_is_available,
    gemm_f32,
    is_available,
    spmv_f32,
)

__all__ = [
    "AmgxProvider",
    "AmgxSolver",
    "CudssPlan",
    "CudssRefactorSolveRecording",
    "CudssSolveRecording",
    "CublasGemmRecording",
    "CusparseSpmvRecording",
    "cublas_is_available",
    "cusparse_is_available",
    "cudss_is_available",
    "gemm_f32",
    "is_available",
    "spmv_f32",
]
