"""Opt-in declarations at normal Field construction, independent of export.

These are dimension relations, not an evaluator for application Python code.
Without an explicit builder, ordinary Field construction is unchanged. An
explicit builder preserves the caller's normal SNode ownership boundary.
"""

import inspect
import operator
from dataclasses import dataclass


from taichi_forge_runtime._dimensions import evaluate_extent, extent_dependencies, freeze_extent, validate_extent


@dataclass(frozen=True)
class Extent:
    expression: tuple

    def __post_init__(self):
        object.__setattr__(self, "expression", freeze_extent(self.expression))

    def _binary(self, op, other):
        if isinstance(other, Extent):
            expression = other.expression
        elif type(other) is int:
            expression = ("const", other)
        else:
            raise TypeError("Dimension expressions accept declared dimensions or integer constants")
        return Extent((op, self.expression, expression))

    def __add__(self, other):
        return self._binary("add", other)

    __radd__ = __add__

    def __mul__(self, other):
        return self._binary("mul", other)

    __rmul__ = __mul__

    def __floordiv__(self, other):
        return self._binary("floordiv", other)

    def maximum(self, other):
        return self._binary("max", other)

    def minimum(self, other):
        return self._binary("min", other)

    def __bool__(self):
        raise TypeError("A declared extent is not a Python condition; use the normal model value")


class ResourceScope:
    """Declare normal owner attributes while constructing ordinary Fields.

    ``scope = ResourceScope(owner)`` forwards attributes to that same owner.
    ``scope.x = scope.field(ti.f32, shape=scope.dim('n').maximum(1))`` retains
    both the declaration and the normal Field. No process hook or kernel list.
    The API is experimental; only simple dense Field construction is qualified.
    """

    def __init__(self, owner, *, builder=None):
        if isinstance(owner, ResourceScope):
            owner = owner._owner
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_builder", builder)
        if not hasattr(owner, "_forge_resource_declarations"):
            owner._forge_resource_declarations = {"fields": {}, "dimensions": {}}

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def __setattr__(self, name, value):
        from taichi_forge.lang.field import Field

        setattr(self._owner, name, value)
        if isinstance(value, Field):
            declaration = getattr(value, "_logical_resource_declaration", None)
            if declaration is not None:
                if declaration["identity"] is None:
                    declaration["identity"] = name
                self._owner._forge_resource_declarations["fields"][name] = dict(declaration)

    def dim(self, name):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("A dimension requires an explicit owner attribute name")
        # Counts may be zero; Field extents are independently required positive.
        # Accessing the owner here also catches misspelled declarations early.
        value = operator.index(getattr(self._owner, name))
        if not 0 <= value <= 2**31 - 1:
            raise ValueError(f"Dimension {name!r} is outside the i32 count range")
        caller = inspect.currentframe().f_back
        self._owner._forge_resource_declarations["dimensions"].setdefault(
            name, (caller.f_code.co_filename, caller.f_lineno)
        )
        del caller
        return Extent(("dim", name))

    def _create(self, factory, args, kwargs):
        shape = kwargs.get("shape")
        if shape is None:
            raise ValueError("ResourceScope Field factories require an explicit shape")
        shape = shape if isinstance(shape, tuple) else (shape,)
        if any(not isinstance(item, Extent) and type(item) is not int for item in shape):
            raise TypeError("Resource extents require declared dimensions or explicit integer constants")
        concrete = tuple(
            (
                evaluate_extent(item.expression, lambda name: getattr(self._owner, name))
                if isinstance(item, Extent)
                else item
            )
            for item in shape
        )
        if any(not 0 < value <= 2**31 - 1 for value in concrete):
            raise ValueError("Resource Field extents must be positive i32 values")
        if self._builder is None:
            field = factory(*args, **{**kwargs, "shape": concrete})
        else:
            from taichi_forge.lang import impl
            from taichi_forge.lang.enums import Layout

            if (
                any(kwargs.get(key) for key in ("offset", "needs_grad", "needs_dual"))
                or kwargs.get("layout", Layout.AOS) != Layout.AOS
            ):
                raise ValueError("ResourceScope explicit groups currently require plain AOS Fields")
            field = factory(*args, **{**kwargs, "shape": None})
            self._builder.dense(impl.index_nd(len(concrete)), concrete).place(field)
        caller = inspect.currentframe().f_back.f_back
        field._logical_resource_declaration = dict(
            identity=None, shape=shape, source=(caller.f_code.co_filename, caller.f_lineno)
        )
        del caller
        return field

    def field(self, *args, **kwargs):
        from taichi_forge.lang.impl import field

        return self._create(field, args, kwargs)

    def vector_field(self, *args, **kwargs):
        from taichi_forge.lang.matrix import Vector

        return self._create(Vector.field, args, kwargs)

    def matrix_field(self, *args, **kwargs):
        from taichi_forge.lang.matrix import Matrix

        return self._create(Matrix.field, args, kwargs)
