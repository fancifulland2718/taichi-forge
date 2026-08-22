"""Public optional hardware linear-algebra provider API."""

from taichi_forge.hardware._linalg import (
    CublasGemmRecording,
    gemm_f32,
    is_available,
)

__all__ = ["CublasGemmRecording", "gemm_f32", "is_available"]
