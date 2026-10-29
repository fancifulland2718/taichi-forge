"""Opt-in declarations at normal Field construction, independent of export.

These are dimension relations, not an evaluator for application Python code.
Without an explicit builder, ordinary Field construction is unchanged. An
explicit builder preserves the caller's normal SNode ownership boundary.
"""

import inspect
import operator
from dataclasses import dataclass
from types import MemberDescriptorType


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


def _source_members(field):
    """Read declaration provenance, never infer layout from sample extents."""
    group = getattr(field, "_logical_resource_group", None)
    if group is None:
        raise ValueError("Borrowed resources require an explicit declared FieldsBuilder source group")
    members = {}
    for member in group:
        declaration = member._logical_resource_declaration
        identity = declaration["identity"]
        if identity is None or identity in members:
            raise ValueError("Borrowed source group members require distinct normal declaration identities")
        members[identity] = member
    return members


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
        object.__setattr__(self, "_read_only", False)
        object.__setattr__(self, "_parent", None)
        if not hasattr(owner, "_forge_resource_declarations"):
            owner._forge_resource_declarations = {"fields": {}, "dimensions": {}, "records": {}}
        object.__setattr__(self, "_registry", owner._forge_resource_declarations)
        if builder is not None:
            namespace = self._registry
            previous = getattr(builder, "_resource_declaration_namespace", namespace)
            if previous is not namespace:
                raise ValueError("A declared FieldsBuilder group requires one dimension declaration namespace")
            builder._resource_declaration_namespace = namespace
            if not hasattr(builder, "_resource_declaration_members"):
                builder._resource_declaration_members = []

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def __setattr__(self, name, value):
        from taichi_forge.lang.field import Field

        if self._read_only:
            raise TypeError("A declared record scope cannot modify its existing members")
        setattr(self._owner, name, value)
        if isinstance(value, Field):
            declaration = getattr(value, "_logical_resource_declaration", None)
            if declaration is not None:
                if declaration["identity"] is None:
                    declaration["identity"] = name
                self._registry["fields"][name] = dict(declaration)
                self._registry.get("borrowed", {}).pop(name, None)
        elif inspect.getattr_static(value, "_forge_resource_declarations", None) is not None:
            # Assignment through an explicit declaration scope exposes a
            # normal child resource owner. No attribute-dictionary scan.
            caller = inspect.currentframe().f_back
            self._registry.setdefault("records", {})[name] = (
                caller.f_code.co_filename,
                caller.f_lineno,
            )
            del caller

    def dim(self, name):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("A dimension requires an explicit owner attribute name")
        # Counts may be zero; Field extents are independently required positive.
        # Accessing the owner here also catches misspelled declarations early.
        value = operator.index(getattr(self._owner, name))
        if not 0 <= value <= 2**31 - 1:
            raise ValueError(f"Dimension {name!r} is outside the i32 count range")
        caller = inspect.currentframe().f_back
        self._registry["dimensions"].setdefault(name, (caller.f_code.co_filename, caller.f_lineno))
        del caller
        return Extent(("dim", name))

    def record(self, name, value, *, optional=False):
        """Declare an existing plain record without mutating or wrapping it.

        The parent owns declaration metadata; the child scope only describes
        existing values. Frozen dataclasses therefore keep their usual contract.
        """
        if not isinstance(name, str) or not name.isidentifier() or type(optional) is not bool:
            raise TypeError("Resource records require an attribute name and a bool optional flag")
        if value is None and not optional:
            raise ValueError("A required resource record cannot be None")
        if self._read_only:
            if getattr(self._owner, name) is not value:
                raise ValueError("A record declaration must refer to its existing member")
        else:
            setattr(self._owner, name, value)
        caller = inspect.currentframe().f_back
        registry = {"fields": {}, "dimensions": {}, "records": {}}
        self._registry.setdefault("records", {})[name] = (caller.f_code.co_filename, caller.f_lineno)
        self._registry.setdefault("record_declarations", {})[name] = registry
        self._registry.setdefault("record_options", {})[name] = optional
        del caller
        child = object.__new__(ResourceScope)
        for key, item in dict(_owner=value, _builder=None, _registry=registry, _read_only=True, _parent=self).items():
            object.__setattr__(child, key, item)
        return child

    def parent_dim(self, name):
        """Explicitly relate a read-only record's layout to its parent count."""
        if self._parent is None:
            raise ValueError("parent_dim requires a declared record scope")
        self._parent.dim(name)
        alias = "__parent_" + name
        self._registry["dimensions"][alias] = self._parent._registry["dimensions"][name]
        self._registry.setdefault("inherited_dimensions", {})[alias] = ("dim", name)
        return Extent(("dim", alias))

    def borrow(self, name, field, *, dimensions):
        """Bind an existing Field and explicitly relate its whole source group.

        The target retains the original Field and storage ownership. This is a
        normal binding declaration, independent of kernels/export. Source
        dimensions include layout-only siblings, not just this Field's shape.
        """
        from taichi_forge.lang.field import Field

        if not isinstance(name, str) or not name.isidentifier() or not isinstance(field, Field):
            raise TypeError("Resource borrow requires an attribute name and a Field")
        _source_members(field)
        mapping = {}
        for source, target in dimensions.items():
            if not isinstance(source, str) or not source.isidentifier() or not isinstance(target, Extent):
                raise TypeError("Borrowed dimension mappings require source names and declared target extents")
            validate_extent(target.expression)
            if not set(extent_dependencies(target.expression)) <= self._registry["dimensions"].keys():
                raise ValueError("Borrowed dimensions must reference declared target dimensions")
            mapping[source] = target.expression
        registry = self._registry
        # The first normal binding names this source group in the target
        # namespace. Later aliases do not rename it or invalidate unrelated
        # entry code. Object identity is used only to recognize the existing
        # source group during binding; it never enters a serialized identity.
        group = name
        for alias, previous in registry.get("borrowed", {}).items():
            candidate = inspect.getattr_static(self._owner, alias, None)
            if isinstance(candidate, MemberDescriptorType):
                candidate = candidate.__get__(self._owner, type(self._owner))
            if getattr(candidate, "_logical_resource_group", None) is field._logical_resource_group:
                group = previous["group"]
                break
        caller = inspect.currentframe().f_back
        declaration = dict(
            identity=field._logical_resource_declaration["identity"],
            group=group,
            dimensions=mapping,
            source=(caller.f_code.co_filename, caller.f_lineno),
        )
        del caller
        if self._read_only:
            if getattr(self._owner, name) is not field:
                raise ValueError("A borrowed record declaration must refer to its existing Field")
        else:
            setattr(self._owner, name, field)
        registry["fields"].pop(name, None)
        registry.get("records", {}).pop(name, None)
        registry.setdefault("borrowed", {})[name] = declaration
        return field

    def _create(self, factory, args, kwargs):
        if self._read_only:
            raise TypeError("A declared record scope describes existing resources; it cannot allocate Fields")
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
        if self._builder is not None:
            # Keep the complete declared layout unit, without retaining its
            # Python owner. Native SNode capture still checks actual placement.
            group = self._builder._resource_declaration_members
            group.append(field)
            field._logical_resource_group = group
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
