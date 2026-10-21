"""Cold cuDSS configuration and operation roles, separate from numerical seeds."""

from dataclasses import dataclass, field

from taichi_forge.graph._ir import GraphAccess
from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

MATRIX_TYPES = {"general": 0, "symmetric": 1, "spd": 3}
MATRIX_VIEWS = {"full": 0, "lower": 1, "upper": 2}
PHASES = {"solve": 0, "factor_solve": 4, "refactor_solve": 8}
REORDERINGS = ("default", "amd", "nested_dissection", "natural")
SOLVES = ("default", "general")


def plan_definition(rows, nonzeros, matrix_type, matrix_view, configuration):
    reordering, solve = (0, 0) if configuration is None else configuration
    if reordering not in range(len(REORDERINGS)) or solve not in range(len(SOLVES)):
        raise ValueError("Unsupported cuDSS encoded configuration")
    return NativeDefinition(
        "cudss",
        "plan_f32",
        dict(
            rows=rows,
            nonzeros=nonzeros,
            matrix_type=matrix_type,
            matrix_view=matrix_view,
            reordering=REORDERINGS[reordering],
            solve=SOLVES[solve],
        ),
    )


def resource_definition(plan, phase, snapshot):
    """Validate native creation facts; retain the native owner's actual seed."""
    rows, nonzeros, matrix_type, matrix_view, numeric_phase, reordering, solve, *_ = snapshot
    actual = plan_definition(
        rows,
        nonzeros,
        {v: k for k, v in MATRIX_TYPES.items()}[matrix_type],
        tuple(MATRIX_VIEWS)[matrix_view],
        (reordering, solve),
    )
    if actual != plan._definition:
        from taichi_forge.lang.exception import TaichiRuntimeError

        raise TaichiRuntimeError("cuDSS native resource differs from its frozen creation definition")
    expected = -1 if phase == "transactional" else PHASES[phase]
    if numeric_phase not in (-1, expected):
        from taichi_forge.lang.exception import TaichiRuntimeError

        raise TaichiRuntimeError("cuDSS AOT recording differs from its frozen numerical phase")
    if phase == "transactional":
        return NativeDefinition("cudss", "transaction_plan_f32", plan._definition.parameters)
    return plan._definition


@dataclass(frozen=True)
class CudssCallDefinition:
    phase: str
    rhs_count: int = 1
    disjoint_bindings: bool = False
    native: NativeDefinition = field(init=False)

    def __post_init__(self):
        if self.phase == "transactional":
            operation, parameters = "refactor_solve_f32", {}
        else:
            if self.phase not in PHASES:
                raise ValueError("Unsupported cuDSS numerical phase")
            operation = "solve_group_f32"
            parameters = dict(rhs_count=self.rhs_count)
            if self.disjoint_bindings:
                parameters["disjoint_bindings"] = True
        object.__setattr__(self, "native", NativeDefinition("cudss", operation, parameters))

    def bind(self, pairs, *, values=None, resource=None):
        arrays = [] if self.phase == "solve" else [NativeArrayBinding("values", values, GraphAccess.READ)]
        for index, (rhs, output) in enumerate(pairs):
            suffix = "" if self.phase == "transactional" else f"_{index}"
            arrays += [
                NativeArrayBinding("rhs" + suffix, rhs, GraphAccess.READ),
                NativeArrayBinding("output" + suffix, output, GraphAccess.WRITE),
            ]
        return NativeOperationBinding(self.native, arrays, () if resource is None else (("plan", resource),))
