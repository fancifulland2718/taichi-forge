"""Cold cuBLAS definitions shared by direct recordings and native lowering."""

from dataclasses import dataclass
import math
from numbers import Real

from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding


@dataclass(frozen=True)
class GemmDefinition:
    rows: int
    columns: int
    inner: int
    alpha: float
    beta: float

    def __post_init__(self):
        for name, value in (("rows", self.rows), ("columns", self.columns), ("inner dimension", self.inner)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 0x7FFFFFFF:
                raise ValueError(f"CUDA cuBLAS {name} must be in [1, INT_MAX]")
        for name in ("alpha", "beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"CUDA cuBLAS {name} must be a real scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"CUDA cuBLAS {name} must be finite")
            object.__setattr__(self, name, value)

    def bind(self, a, b, output):
        return NativeOperationBinding(
            NativeDefinition(
                "cublas",
                "gemm_f32",
                dict(
                    rows=self.rows,
                    columns=self.columns,
                    inner=self.inner,
                    alpha=self.alpha,
                    beta=self.beta,
                ),
            ),
            (
                NativeArrayBinding("a", a, GraphAccess.READ),
                NativeArrayBinding("b", b, GraphAccess.READ),
                NativeArrayBinding("output", output, GraphAccess.WRITE if self.beta == 0.0 else GraphAccess.READ_WRITE),
            ),
        )
