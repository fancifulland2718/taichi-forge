"""Frozen sparse operation parameters; matrix snapshots remain deployment state."""

from dataclasses import dataclass

from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

SPMM_ALGORITHMS = {
    "row_major": 0,
    "deterministic": 1,  # Legacy optional preprocessing, not a bitwise guarantee.
    "csr3_direct": 2,
    "csr3_preprocessed": 3,
}


def bind_sparse_operation(definition, input_name, output_name):
    return NativeOperationBinding(
        definition,
        sparse_array_ports(input_name, output_name),
    )


def sparse_array_ports(input_name, output_name):
    return (
        NativeArrayBinding("input", input_name, GraphAccess.READ),
        NativeArrayBinding("output", output_name, GraphAccess.WRITE),
    )


@dataclass(frozen=True)
class SpmmDefinition:
    rhs_count: int
    algorithm: str

    def __post_init__(self):
        if isinstance(self.rhs_count, bool) or not isinstance(self.rhs_count, int):
            raise TypeError("CUDA cuSPARSE SpMM rhs_count must be an integer")
        if self.rhs_count < 2 or self.rhs_count > 0x7FFFFFFF:
            raise ValueError("CUDA cuSPARSE SpMM rhs_count must be in [2, INT_MAX]")
        if self.algorithm not in SPMM_ALGORITHMS:
            raise ValueError(
                "CUDA cuSPARSE SpMM algorithm must be row_major, deterministic, csr3_direct, or csr3_preprocessed"
            )

    @property
    def algorithm_code(self):
        return SPMM_ALGORITHMS[self.algorithm]

    def operation(self, prepared=None):
        code = self.algorithm_code
        parameters = dict(
            rhs_count=self.rhs_count,
            algorithm="csr_alg2" if code == 0 else "csr_alg3",
            preprocess="optional" if code == 1 else "required" if code == 3 else "none",
        )
        if prepared is not None and prepared["prepared"]:
            parameters["expected_workspace_bytes"] = prepared["workspace_bytes"]
            # Persist an already selected optional outcome without preparing a
            # new plan or rerunning the provider's choice during export.
            if code == 1:
                parameters["preprocess"] = "required" if prepared["preprocessed"] else "none"
        return NativeDefinition("cusparse", "spmm_f32", parameters)


@dataclass(frozen=True)
class TriangularDefinition:
    rhs_count: int
    fill_mode: str
    unit_diagonal: bool
    transpose: bool
    algorithm: str = "default"

    def __post_init__(self):
        if isinstance(self.rhs_count, bool) or not isinstance(self.rhs_count, int):
            raise TypeError("CUDA cuSPARSE triangular rhs_count must be an integer")
        if self.rhs_count < 1 or self.rhs_count > 0x7FFFFFFF:
            raise ValueError("CUDA cuSPARSE triangular rhs_count must be in [1, INT_MAX]")
        if self.fill_mode not in ("lower", "upper"):
            raise ValueError("CUDA cuSPARSE fill_mode must be lower or upper")
        if not isinstance(self.unit_diagonal, bool):
            raise TypeError("CUDA cuSPARSE unit_diagonal must be a bool")
        if not isinstance(self.transpose, bool):
            raise TypeError("CUDA cuSPARSE transpose must be a bool")
        if self.algorithm != "default":
            raise ValueError("CUDA cuSPARSE triangular algorithm must be default")

    @property
    def fill_mode_code(self):
        return 0 if self.fill_mode == "lower" else 1

    def resource(self, *, mutable_values):
        return NativeDefinition(
            "cusparse",
            "triangular_plan_f32",
            dict(
                rhs_count=self.rhs_count,
                fill_mode=self.fill_mode,
                unit_diagonal=self.unit_diagonal,
                transpose=self.transpose,
                mutable_values=mutable_values,
            ),
        )
