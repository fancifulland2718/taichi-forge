"""Cold native algorithm semantics, independent of execution and export owners."""

from dataclasses import dataclass

from taichi_forge.types.primitive_types import f32, f64, i32, i64, u32, u64

SCAN_VALUE_TYPE = {i32: 0, f32: 1, u32: 2, u64: 3, i64: 4, f64: 5}


@dataclass(frozen=True)
class ScanDefinition:
    """The inclusive scalar scan performed by a selected native call.

    Storage addressing and backend admission remain with the caller. In
    particular, a JIT member-view call does not imply AOT member-view support.
    """

    size: int
    dtype: object

    def __post_init__(self):
        if self.dtype not in SCAN_VALUE_TYPE:
            raise RuntimeError("unsupported PrefixSumExecutor ndarray dtype.")

    @property
    def value_type(self):
        return SCAN_VALUE_TYPE[self.dtype]

    def bind(self, values):
        from taichi_forge.graph._ir import GraphAccess
        from taichi_forge.graph._native_definition import NativeArrayBinding, NativeDefinition, NativeOperationBinding

        return NativeOperationBinding(
            NativeDefinition("builtin", "scan", dict(size=self.size, dtype=str(self.dtype))),
            (NativeArrayBinding("values", values, GraphAccess.READ_WRITE),),
        )
