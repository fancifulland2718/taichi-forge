"""Model-sized local tensors over the existing scalar-array call ABI.

Only immutable captured integers and declared dimensions carry extent
provenance. Mutable locals and device data cannot size Host allocations.
"""

from contextlib import contextmanager, nullcontext
from numbers import Integral, Real

from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiCompilationError, TaichiIndexError
from taichi_forge.lang.expr import Expr


def extent(value):
    if isinstance(value, Integral) and not isinstance(value, bool) and 0 <= value <= 2**31 - 1:
        return ("const", int(value))
    return getattr(value, "_uniform_extent", None) if isinstance(value, Expr) else None


def upper_extent(value):
    """An allocation bound, never a proof that the device value is uniform."""
    exact = extent(value)
    return exact if exact is not None else getattr(value, "_allocation_upper_extent", None)


def is_model_value(value):
    """An immutable scalar expression depending on a declared model axis."""
    return isinstance(value, Expr) and getattr(value, "_model_uniform", False)


def model_expression(result, *inputs):
    # Device loads and mutable locals never inherit this proof. Constants can
    # participate, but cannot on their own relax an explicit static operation.
    if any(is_model_value(value) for value in inputs) and all(
        is_model_value(value) or isinstance(value, Real)
        or (isinstance(value, Expr) and (extent(value) or (None,))[0] == "const")
        for value in inputs
    ):
        result._model_uniform = True
    return result


def binary_extent(result, operation, left, right):
    # A clamped device range can retain its actual trip count while prepare
    # allocates scratch from an immutable upper bound. Do not propagate this
    # as an exact extent or through arithmetic which may overflow or wrap.
    if operation == "min":
        from taichi_forge.types import primitive_types

        if result.ptr.get_rvalue_type() == primitive_types.i32:
            bounds = [bound for value in (left, right) if (bound := upper_extent(value)) is not None]
            if bounds:
                result._allocation_upper_extent = bounds[0] if len(bounds) == 1 else ("min", *bounds)
    # Retain only operations whose complete immutable inputs are known.
    if not hasattr(left, "_uniform_extent") and not hasattr(right, "_uniform_extent"):
        return result
    boolean = operation in ("logical_and", "logical_or")
    comparison = operation in ("cmp_lt", "cmp_le", "cmp_gt", "cmp_ge", "cmp_eq", "cmp_ne")
    # Comparisons produce a nonnegative boolean even when an integer policy
    # uses a negative sentinel. Keep that guard without admitting negative
    # standalone allocation extents.
    def operand(value):
        if comparison and isinstance(value, Integral) and -(2**31) <= value <= 2**31 - 1:
            value = int(value)
            if value < 0:
                return ("sub", ("sub", ("const", 0), ("const", -(value + 1))), ("const", 1))
            return ("const", value)
        return ("const", int(value)) if boolean and isinstance(value, bool) else extent(value)

    left, right = operand(left), operand(right)
    if left is not None and right is not None and operation in (
        "add", "sub", "mul", "max", "min", "floordiv", "cmp_lt", "cmp_le", "cmp_gt", "cmp_ge", "cmp_eq", "cmp_ne",
        "logical_and", "logical_or"
    ):
        result._uniform_extent = (operation, left, right)
    return result


def select_extent(result, condition, yes, no):
    # Static and/or lower to operand selection. A fixed backend comparison
    # can leave a bool arm; retain its proof without allowing bool dimensions
    # in the general extent API or admitting mutable/device predicates.
    dimensions = tuple(
        ("const", int(value)) if isinstance(value, bool) else extent(value) for value in (condition, yes, no)
    )
    if all(value is not None for value in dimensions):
        result._uniform_extent = ("select", *dimensions)
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
    left, right = extent(begin), upper_extent(end)
    count = None if left is None or right is None else ("sub", ("max", left, right), left)
    if index is not None and begin is not None:
        index = index - begin
    return _scope(ctx.deployment, index, count)


def field_iteration_scope(ctx, indices, field):
    """Use declared Dense logical axes, never sparse activation or sample sizes."""
    if ctx.deployment is None or ctx.deployment.local_tensor_scope is not None:
        return nullcontext()
    if not hasattr(field, "_component_index_extents"):
        return iteration_scope(ctx)
    linear, count = 0, 1
    for index, size in zip(indices, field.shape):
        linear = linear * size + index
        count = count * size
    return iteration_scope(ctx, linear, 0, count)


class ModelTensor:
    """A scalar-indexed model tensor, either a Field element or local value.

    Whole-value copies use the same invocation workspace as local constructors.
    Field references are materialized before assignment, including chained and
    overlapping assignments, so later writes cannot change an earlier value.
    """

    def _copy(self):
        deployment = getattr(impl.get_runtime().current_kernel, "_deployment_context", None)
        if deployment is None:
            raise TaichiCompilationError("Model tensor copies require an AOT capture context")
        copied = deployment.local_tensor(self.dtype, self.shape)
        copied._visit(lambda indices: self[indices])
        return copied

    def _shape_key(self):
        from taichi_forge.aot._component import Dimension

        deployment = impl.get_runtime().current_kernel._deployment_context

        def resolve(expression, visiting=()):
            op, *args = expression
            if op == "dim":
                declaration = deployment.declarations.get(args[0])
                if isinstance(declaration, Dimension) and declaration.expression:
                    if args[0] in visiting:
                        raise TaichiCompilationError("Cyclic model tensor dimension relation")
                    return resolve(declaration.expression, (*visiting, args[0]))
            return (op, *args) if op in ("dim", "const") else (op, *(resolve(arg, visiting) for arg in args))

        return tuple(resolve(extent(size)) for size in self.shape)

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

    def _assign(self, other):
        if isinstance(other, ModelTensor):
            if self.dtype != other.dtype or self._shape_key() != other._shape_key():
                raise TaichiCompilationError(
                    "Model tensor assignment requires matching element type and extent relations"
                )
            # Ordinary AST assignments already snapshot the RHS. Also retain
            # value semantics when a compiler helper passes a Field reference.
            if not isinstance(other, LocalTensor):
                other = other._copy()
            self._visit(lambda indices: other[indices])
        elif isinstance(other, Real) or (isinstance(other, Expr) and not other.ptr.is_tensor()):
            self._visit(other)
        else:
            raise TaichiCompilationError("Model tensor assignment requires a compatible tensor or scalar")
        return self


class LocalTensor(ModelTensor):
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

    def _ti_expr_init(self):
        if self._temporary:
            self._temporary = False
            return self
        copied = self._copy()
        copied._temporary = False
        return copied

    def __bool__(self):
        raise TaichiCompilationError("A model local tensor is not a scalar condition")


def filled(dtype, shape, value):
    deployment = getattr(impl.get_runtime().current_kernel, "_deployment_context", None)
    if deployment is None:
        raise TaichiCompilationError("Runtime-sized local tensors require declared model dimensions in AOT capture")
    tensor = deployment.local_tensor(dtype, shape)
    tensor._visit(value)
    return tensor
