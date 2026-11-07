"""Model-sized local tensors over the existing scalar-array call ABI.

Only immutable captured integers and declared dimensions carry extent
provenance. Mutable locals and device data cannot size Host allocations.
"""

from contextlib import contextmanager, nullcontext

from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiCompilationError, TaichiIndexError
from taichi_forge.lang.expr import Expr


def extent(value):
    if type(value) is int and 0 <= value <= 2**31 - 1:
        return ("const", value)
    return getattr(value, "_uniform_extent", None) if isinstance(value, Expr) else None


def binary_extent(result, operation, left, right):
    # Retain only operations whose complete immutable inputs are known.
    if not hasattr(left, "_uniform_extent") and not hasattr(right, "_uniform_extent"):
        return result
    left, right = extent(left), extent(right)
    if left is not None and right is not None and operation in ("add", "sub", "mul", "max", "min", "floordiv"):
        result._uniform_extent = (operation, left, right)
    return result


@contextmanager
def _scope(deployment, index, count):
    previous = deployment.local_tensor_scope
    if previous is None:
        deployment.local_tensor_scope = (index, count)
    try:
        yield
    finally:
        deployment.local_tensor_scope = previous


def iteration_scope(ctx, index=None, begin=None, end=None):
    if ctx.deployment is None:
        return nullcontext()
    if ctx.deployment.local_tensor_scope is not None:
        return nullcontext()
    left, right = extent(begin), extent(end)
    count = None if left is None or right is None else ("sub", ("max", left, right), left)
    if index is not None and begin is not None:
        index = index - begin
    return _scope(ctx.deployment, index, count)


class LocalTensor:
    def __init__(self, deployment, array, index, shape, dtype):
        self.deployment, self.array, self.index = deployment, array, index
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim = len(shape)
        self.n = shape[0]
        self.m = shape[1] if self.ndim == 2 else 1
        self._temporary = True

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) != self.ndim or any(isinstance(i, slice) for i in indices):
            raise TaichiIndexError("Model local tensor requires one scalar index per axis")
        return impl.subscript(None, self.array, self.index, *indices)

    def _visit(self, value):
        from taichi_forge._lib import core
        from taichi_forge.lang import ops
        from taichi_forge.types import primitive_types

        builder = impl.get_runtime().compiling_callable.ast_builder()
        debug = core.DebugInfo(impl.get_runtime().get_current_src_info())
        count = 1
        for size in self.shape:
            count = count * size
        index = Expr(builder.make_id_expr(""))
        builder.begin_frontend_range_for(index.ptr, Expr(0).ptr, ops.cast(count, primitive_types.i32).ptr, debug)
        indices = (index,) if self.ndim == 1 else (index // self.m, index % self.m)
        self[indices]._assign(value(indices) if callable(value) else value)
        builder.end_frontend_range_for()

    def _ti_expr_init(self):
        if self._temporary:
            self._temporary = False
            return self
        copied = self.deployment.local_tensor(self.dtype, self.shape)
        copied._visit(lambda indices: self[indices])
        copied._temporary = False
        return copied

    def _assign(self, other):
        if (
            not isinstance(other, LocalTensor)
            or self.dtype != other.dtype
            or (tuple(map(extent, self.shape)) != tuple(map(extent, other.shape)))
        ):
            raise TaichiCompilationError(
                "Model local tensor assignment requires matching element type and extent relations"
            )
        self._visit(lambda indices: other[indices])
        return self

    def __bool__(self):
        raise TaichiCompilationError("A model local tensor is not a scalar condition")


def filled(dtype, shape, value):
    deployment = getattr(impl.get_runtime().current_kernel, "_deployment_context", None)
    if deployment is None:
        raise TaichiCompilationError("Runtime-sized local tensors require declared model dimensions in AOT capture")
    tensor = deployment.local_tensor(dtype, shape)
    tensor._visit(value)
    return tensor
