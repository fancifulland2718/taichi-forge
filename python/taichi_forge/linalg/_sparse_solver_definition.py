"""Sparse factor configuration shared by direct creation and reconstruction."""

from dataclasses import dataclass

from taichi_forge.graph._native_definition import NativeDefinition
from taichi_forge.lang.exception import TaichiRuntimeError


@dataclass(frozen=True)
class SparseSolverDefinition:
    dtype: object
    method: str
    ordering: str

    def resource(self, implementation, rows, nonzeros, dtype):
        if dtype != str(self.dtype):
            raise TaichiRuntimeError("SparseSolver snapshot dtype differs from its creation definition")
        return NativeDefinition(
            "sparse_direct",
            "plan",
            dict(
                rows=rows,
                nonzeros=nonzeros,
                dtype=dtype,
                implementation=implementation,
                method=self.method,
                ordering=self.ordering,
                format="csc" if implementation == "eigen" else "csr",
                phase="solve",
            ),
        )

    def solve(self, implementation):
        return NativeDefinition(
            "sparse_direct",
            "solve" if implementation == "eigen" or self.method == "LLT" else "solve_ordered",
            dict(rhs_count=1),
        )
