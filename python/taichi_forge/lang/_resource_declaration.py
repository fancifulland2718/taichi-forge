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
    The API is experimental; simple dense Fields and whole managed arrays use
    their existing storage contracts.
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
        from taichi_forge.lang._ndarray import Ndarray
        from taichi_forge.lang.struct import StructField

        if self._read_only:
            raise TypeError("A declared record scope cannot modify its existing members")
        if name in self._registry.get("values", {}) and (
            isinstance(value, (Field, Ndarray)) or inspect.getattr_static(value, "_forge_resource_declarations", None) is not None
        ):
            raise ValueError(f"Normal attribute {name!r} already has a value role")
        setattr(self._owner, name, value)
        if isinstance(value, Ndarray):
            declaration = getattr(value, "_logical_resource_declaration", None)
            if declaration is not None:
                if declaration["identity"] is None:
                    declaration["identity"] = name
                self._registry.setdefault("arrays", {})[name] = dict(declaration)
        elif isinstance(value, Field) and not isinstance(value, StructField):
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

    def _declare_value(self, name, declaration):
        from taichi_forge.aot._component import StaticValue, _member, _static_value

        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("A value declaration requires an explicit owner attribute name")
        if any(name in self._registry.get(kind, {}) for kind in ("fields", "arrays", "dimensions", "records", "borrowed")):
            raise ValueError(f"Normal attribute {name!r} already has a resource or dimension role")
        # Validate the existing contract without wrapping/replacing normal JIT
        # values or evaluating arbitrary Python descriptors. An absent optional
        # record can still declare its leaf types for a later present instance.
        declaration.schema
        if self._owner is not None:
            value = _member(self._owner, name)
            if isinstance(declaration, StaticValue):
                declaration.validate(value, name, "normal declaration")
        values = self._registry.get("values", {})
        if name in values:
            previous = values[name]["declaration"]
            if previous.schema != declaration.schema or (
                isinstance(declaration, StaticValue)
                and (
                    (None if previous.choices is None else tuple(map(_static_value, previous.choices)))
                    != (None if declaration.choices is None else tuple(map(_static_value, declaration.choices)))
                    or previous.kernel_only != declaration.kernel_only
                )
            ):
                raise ValueError(f"Normal attribute {name!r} already has a different value declaration")
            return
        caller = inspect.currentframe().f_back.f_back
        self._registry.setdefault("values", {})[name] = dict(
            declaration=declaration, source=(caller.f_code.co_filename, caller.f_lineno)
        )
        del caller

    def value(self, name, dtype):
        """Declare an existing scalar/vector/matrix as a runtime input.

        The original owner value remains unchanged. Only entries that read it
        acquire an ABI input; dtype and fixed element shape come from this role.
        """
        from taichi_forge.aot._component import RuntimeValue

        self._declare_value(name, RuntimeValue(dtype))

    def static(self, name, *, choices, kernel_only=False):
        """Declare finite configuration; kernel-only values cannot control Host recording.

        A kernel-only declaration enters root admission only when an exported
        numerical entry actually captures it. Recording choices always remain.
        """
        from taichi_forge.aot._component import StaticValue

        if choices is None:
            raise ValueError("Normal static declarations require finite choices")
        self._declare_value(name, StaticValue(choices=choices, kernel_only=kernel_only))

    def constant(self, name):
        """Declare a compile-time definition, not a consumer configuration input.

        Actual numerical users retain the value in their specialization and
        cache guards. Changing it requires rebuilding code. Model-dependent
        choices must instead use static(..., choices=...) or runtime values.
        """
        from taichi_forge.aot._component import StaticValue

        self._declare_value(name, StaticValue())

    def static_dtype(self, name, *, choices, kernel_only=False):
        """Declare a primitive DSL type specialization at its normal selection site."""
        from taichi_forge.aot._component import StaticValue, _static_value

        choices = tuple(choices)
        if any(_static_value(value)[0] != "dtype" for value in choices):
            raise ValueError("Static dtype choices require actual qualified DSL primitive types")
        self._declare_value(name, StaticValue(choices=choices, kernel_only=kernel_only))

    def dim(self, name):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("A dimension requires an explicit owner attribute name")
        if name in self._registry.get("values", {}):
            raise ValueError(f"Normal attribute {name!r} already has a value role")
        # Counts may be zero; Field extents are independently required positive.
        # Accessing the owner here also catches misspelled declarations early.
        value = operator.index(getattr(self._owner, name))
        if not 0 <= value <= 2**31 - 1:
            raise ValueError(f"Dimension {name!r} is outside the i32 count range")
        caller = inspect.currentframe().f_back
        self._registry["dimensions"].setdefault(name, (caller.f_code.co_filename, caller.f_lineno))
        del caller
        return Extent(("dim", name))

    def require_le(self, name, left, right, *, when):
        """Declare left <= right for entries that actually use one named role.

        Expressions reuse the dimension algebra. Bounds come from the normal
        algorithm/layout policy, not inferred sample capacities. This neither
        changes numerical code nor adds runtime arguments to unrelated entries.
        """
        from taichi_forge_runtime._dimensions import check_constraints, validate_constraints

        def expression(value):
            return value.expression if isinstance(value, Extent) else freeze_extent(("const", value))

        spec = dict(left=expression(left), right=expression(right), when=when)
        roles = set().union(*(self._registry.get(kind, {}) for kind in ("fields", "arrays", "dimensions", "values", "borrowed")))
        validate_constraints({name: spec}, self._registry["dimensions"].keys(), roles)
        if self._owner is not None:
            check_constraints({name: spec}, self._normal_dimension)
        constraints = self._registry.setdefault("constraints", {})
        if name in constraints and constraints[name]["spec"] != spec:
            raise ValueError(f"Dimension constraint {name!r} already has another declaration")
        caller = inspect.currentframe().f_back
        constraints.setdefault(name, dict(spec=spec, source=(caller.f_code.co_filename, caller.f_lineno)))
        del caller

    def _normal_dimension(self, name):
        inherited = self._registry.get("inherited_dimensions", {}).get(name)
        if inherited is not None:
            return self._parent._normal_dimension(inherited[1])
        return operator.index(getattr(self._owner, name))

    def derived_dim(self, name, expression):
        """Relate an existing count to declared counts without changing it.

        The normal constructor still computes its integer attribute. The same
        bounded expression is retained for deployment layout and loop bounds;
        it is not another independently supplied consumer dimension.
        """
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(expression, Extent):
            raise TypeError("A derived dimension requires an attribute name and an Extent expression")
        relations = self._registry.get("derived_dimensions", {})
        if name in relations and relations[name] != expression.expression:
            raise ValueError(f"Derived dimension {name!r} was declared with another expression")
        unknown = extent_dependencies(expression.expression) - self._registry["dimensions"].keys()
        if unknown:
            raise ValueError(f"Derived dimension {name!r} references undeclared dimensions {sorted(unknown)!r}")
        pending = {**relations, name: expression.expression}

        def resolve(key, visiting=()):
            if key in visiting:
                raise ValueError(f"Cyclic derived dimension {key!r}")
            if key in pending:
                value = evaluate_extent(pending[key], lambda dependency: resolve(dependency, (*visiting, key)))
            elif key in self._registry.get("inherited_dimensions", {}):
                return self._normal_dimension(key)
            else:
                return operator.index(getattr(self._owner, key))
            if value != operator.index(getattr(self._owner, key)):
                raise ValueError(f"Derived dimension {key!r} does not match its normal attribute")
            return value

        value = resolve(name)
        if not 0 <= value <= 2**31 - 1:
            raise ValueError(f"Derived dimension {name!r} is outside the i32 count range")
        caller = inspect.currentframe().f_back
        self._registry["dimensions"].setdefault(name, (caller.f_code.co_filename, caller.f_lineno))
        self._registry.setdefault("derived_dimensions", {})[name] = expression.expression
        del caller
        return Extent(("dim", name))

    def power_of_two_dim(self, name, minimum):
        """Relate an existing capacity to the smallest power of two >= minimum.

        Zero rounds to one. The constructor still computes the ordinary value;
        deployment reuses the same checked relation, not a consumer override.
        """
        if not isinstance(minimum, Extent):
            raise TypeError("A power-of-two dimension requires a declared Extent minimum")
        result = self.derived_dim(name, Extent(("ceil_pow2", minimum.expression)))
        caller = inspect.currentframe().f_back
        self._registry["dimensions"][name] = (caller.f_code.co_filename, caller.f_lineno)
        del caller
        return result

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

    def borrow(self, name, field, *, dimensions=None, source_scope=False):
        """Bind an existing Field and explicitly relate its whole source group.

        The target retains the original Field and storage ownership. This is a
        normal binding declaration, independent of kernels/export. Source
        dimensions include layout-only siblings, not just this Field's shape.
        source_scope=True instead requires the original allocation scope in
        the component root. Its layout remains authoritative; dimensions then
        supplies only the explicit target relations that need checking.
        """
        from taichi_forge.lang.field import Field

        if not isinstance(name, str) or not name.isidentifier() or not isinstance(field, Field):
            raise TypeError("Resource borrow requires an attribute name and a Field")
        if type(source_scope) is not bool or (dimensions is None and not source_scope):
            raise TypeError("Resource borrow requires explicit dimensions or source_scope=True")
        if source_scope:
            declaration = getattr(field, "_logical_resource_declaration", None)
            if declaration is None or declaration["identity"] is None:
                raise ValueError("Source-scope borrows require an existing normal Field declaration")
        else:
            _source_members(field)
        mapping = {}
        for source, target in ({} if dimensions is None else dimensions).items():
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
            source_group = getattr(field, "_logical_resource_group", None)
            if source_group is not None and getattr(candidate, "_logical_resource_group", None) is source_group:
                group = previous["group"]
                break
        caller = inspect.currentframe().f_back
        declaration = dict(
            identity=field._logical_resource_declaration["identity"],
            group=group,
            dimensions=mapping,
            source=(caller.f_code.co_filename, caller.f_lineno),
        )
        if source_scope:
            declaration["source_scope"] = True
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
        kwargs = dict(kwargs)
        placement = kwargs.pop("placement", None)
        shape = kwargs.get("shape")
        if shape is None:
            raise ValueError("ResourceScope Field factories require an explicit shape")
        shape = shape if isinstance(shape, tuple) else (shape,)
        if any(not isinstance(item, Extent) and type(item) is not int for item in shape):
            raise TypeError("Resource extents require declared dimensions or explicit integer constants")
        if placement is not None:
            from taichi_forge.lang import impl
            from taichi_forge.lang.snode import SNode

            if not isinstance(placement, SNode):
                raise ValueError("Fixed SNode placement requires a normal SNode")
            builder = self._builder if self._builder is not None else impl._root_fb
            root = placement.ptr
            while root.parent is not None:
                root = root.parent
            if root.id != builder.ptr.id:
                raise ValueError("Fixed SNode placement belongs to another FieldsBuilder")
            if any(isinstance(item, Extent) for item in shape):
                raise ValueError("Fixed SNode schema cannot use dynamic extents; declare active counts separately")
            if tuple(placement.shape) != shape:
                raise ValueError("Fixed SNode placement shape differs from the declared Field shape")
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
        if self._builder is None and placement is None:
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
            if placement is None:
                self._builder.dense(impl.index_nd(len(concrete)), concrete).place(field)
            else:
                placement.place(field)
        caller = inspect.currentframe().f_back.f_back
        source = (caller.f_code.co_filename, caller.f_lineno)
        from taichi_forge.lang.struct import StructField

        if isinstance(field, StructField):
            # The normal StructField remains a record namespace; its actual
            # scalar/vector/matrix members enter the existing Field contract.
            field._forge_resource_declarations = {"fields": {}, "dimensions": {}, "records": {}}
            members = field.field_dict.items()
        else:
            members = ((None, field),)
        for identity, member in members:
            member._logical_resource_declaration = dict(identity=identity, shape=shape, source=source)
            # A Field alias in another declared child keeps the allocation
            # site's dimension namespace, just like a StructField member.
            member._logical_shape_namespace = self._registry
            if placement is not None:
                member._logical_resource_declaration["storage"] = "fixed_snode"
            if isinstance(field, StructField):
                field._forge_resource_declarations["fields"][identity] = dict(member._logical_resource_declaration)
                # The record is a storage namespace, not a new dimension
                # declaration site. Aliases retain this lexical provenance.
            if self._builder is not None:
                # Keep the complete declared layout unit, without retaining its
                # Python owner. Native SNode capture still checks actual placement.
                group = self._builder._resource_declaration_members
                group.append(member)
                member._logical_resource_group = group
        del caller
        return field

    def field(self, *args, **kwargs):
        from taichi_forge.lang.impl import field

        return self._create(field, args, kwargs)

    def ndarray(self, dtype, *, shape):
        """Declare a normal independent array, retaining its logical extents.

        StructNdarray keeps the existing fixed AoS element layout. A surrounding
        FieldsBuilder does not make an independent array part of an SNode root.
        """
        from taichi_forge.lang.impl import ndarray

        if self._read_only:
            raise TypeError("A declared record scope cannot allocate arrays")
        shape = shape if isinstance(shape, tuple) else (shape,)
        if any(not isinstance(item, Extent) and type(item) is not int for item in shape):
            raise TypeError("Array extents require declared dimensions or explicit integer constants")
        concrete = tuple(evaluate_extent(item.expression, lambda name: getattr(self._owner, name))
                         if isinstance(item, Extent) else item for item in shape)
        if any(not 0 <= value <= 2**31 - 1 for value in concrete):
            raise ValueError("Array extents must fit the nonnegative i32 indexing domain")
        array = ndarray(dtype, shape=concrete)
        caller = inspect.currentframe().f_back
        array._logical_resource_declaration = dict(
            identity=None, shape=shape, source=(caller.f_code.co_filename, caller.f_lineno))
        array._logical_shape_namespace = self._registry
        del caller
        return array

    def vector_field(self, *args, **kwargs):
        from taichi_forge.lang.matrix import Vector

        return self._create(Vector.field, args, kwargs)

    def matrix_field(self, *args, **kwargs):
        from taichi_forge.lang.matrix import Matrix

        return self._create(Matrix.field, args, kwargs)

    def struct_field(self, members, **kwargs):
        """Declare a flat record at its normal allocation site.

        Member Field access and aliases retain the original shared placement.
        Implicit-root SOA uses Struct.field's ordinary per-member placement;
        explicit groups/placement currently require AOS. Whole Struct element
        indexing requires fixed-SNode capture; nested records are unqualified.
        """
        from taichi_forge.lang.enums import Layout
        from taichi_forge.lang.struct import Struct, StructType

        if (
            kwargs.get("shape") is None
            or kwargs.get("layout", Layout.AOS) not in (Layout.AOS, Layout.SOA)
            or any(kwargs.get(key) for key in ("offset", "needs_grad", "needs_dual"))
            or any(isinstance(dtype, StructType) for dtype in members.values())
        ):
            raise ValueError("ResourceScope struct_field requires a flat plain record with an explicit shape")
        return self._create(Struct.field, (members,), kwargs)
