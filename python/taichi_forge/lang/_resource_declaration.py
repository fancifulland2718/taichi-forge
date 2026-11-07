"""Opt-in declarations at normal Field construction, independent of export.

These are dimension relations, not an evaluator for application Python code.
Without an explicit builder, ordinary Field construction is unchanged. An
explicit builder preserves the caller's normal SNode ownership boundary.
"""

import inspect
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from types import MemberDescriptorType


from taichi_forge_runtime._dimensions import Extent, evaluate_extent, extent_dependencies, freeze_extent, validate_extent


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


@dataclass(frozen=True)
class _DeclaredDensePlacement:
    node: object
    shape: tuple
    namespace: object


class ResourceScope:
    """Declare normal owner attributes while constructing ordinary Fields.

    ``scope = ResourceScope(owner)`` forwards attributes to that same owner.
    ``scope.x = scope.field(ti.f32, shape=scope.dim('n').maximum(1))`` retains
    both the declaration and the normal Field. No process hook or kernel list.
    The API is experimental; simple dense Fields and whole managed arrays use
    their existing storage contracts.
    """

    struct_member_initialization = True
    optional_owner_references = True

    def __init__(self, owner, *, builder=None, initialization="consumer_explicit"):
        if initialization not in ("consumer_explicit", "zero"):
            raise ValueError("Resource initialization must be 'consumer_explicit' or 'zero'")
        if isinstance(owner, ResourceScope):
            owner = owner._owner
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_builder", builder)
        object.__setattr__(self, "_initialization", initialization)
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
        if any(name in self._registry.get(kind, {}) for kind in ("fields", "arrays", "dimensions", "records", "borrowed", "owner_sequences")):
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
            if (
                previous.schema != declaration.schema
                or getattr(previous, "affects_plan", False) != getattr(declaration, "affects_plan", False)
            ) or (
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

    def value(self, name, dtype, *, affects_plan=False):
        """Declare an existing scalar/vector/matrix as a runtime input.

        The original owner value remains unchanged. Only entries that read it
        acquire an ABI input; dtype and fixed element shape come from this role.
        A scalar affecting Host graph construction can also guard the recorded
        plan. Its value remains a runtime kernel input, never a specialization.
        """
        from taichi_forge.aot._component import RuntimeValue

        self._declare_value(name, RuntimeValue(dtype, affects_plan=affects_plan))

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

    def _normal_dimension(self, name, visiting=()):
        key = (id(self._registry), name)
        if key in visiting:
            raise ValueError(f"Cyclic owner dimension {name!r}")
        visiting = (*visiting, key)
        inherited = self._registry.get("inherited_dimensions", {}).get(name)
        if inherited is not None:
            return self._parent._normal_dimension(inherited[1], visiting)
        referenced = self._registry.get("referenced_dimensions", {}).get(name)
        if referenced is not None:
            owner, dimension = referenced
            target = getattr(self._owner, owner)
            registry = inspect.getattr_static(target, "_forge_resource_declarations", None)
            if registry is None or dimension not in registry["dimensions"]:
                raise ValueError(f"Owner dimension {owner}.{dimension} no longer has its declaration")
            return ResourceScope(target)._normal_dimension(dimension, visiting)
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
            elif (key in self._registry.get("inherited_dimensions", {})
                  or key in self._registry.get("referenced_dimensions", {})):
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

    def owner_reference(self, name, value, *, optional=False):
        """Declare a non-owning link to an owner in the same component root.

        Its unique ownership path is resolved when the root is collected. The
        normal object can therefore refer back to a parent under construction;
        this does not recursively declare that parent's resources again.
        Optional references may be absent during normal construction. Presence
        is a captured structural choice, not a mutable device pointer.
        """
        if (not isinstance(name, str) or not name.isidentifier() or type(optional) is not bool
                or (value is None and not optional)):
            raise TypeError("An owner reference requires an attribute name and a non-null owner unless optional=True")
        if any(name in self._registry.get(kind, {}) for kind in
               ("fields", "arrays", "dimensions", "records", "borrowed", "values", "owner_sequences")):
            raise ValueError(f"Normal attribute {name!r} already has another declaration role")
        if self._read_only:
            if getattr(self._owner, name) is not value:
                raise ValueError("An owner reference must refer to its existing member")
        else:
            setattr(self._owner, name, value)
        caller = inspect.currentframe().f_back
        self._registry.setdefault("owner_references", {})[name] = (
            caller.f_code.co_filename, caller.f_lineno)
        self._registry.setdefault("owner_reference_options", {})[name] = optional
        del caller

    def owner_sequence(self, name, values):
        """Declare an existing list/tuple of normal resource owners.

        The sequence is a Host composition namespace, not a device array or a
        new storage owner. Re-declare after changing its composition. Entries
        keep their own declarations and ordinary resource lifetimes.
        """
        self._declare_owner_sequence(name, values, owning=True)

    def owner_sequence_reference(self, name, values):
        """Declare a fixed sequence borrowing owners declared elsewhere.

        Every member must have exactly one owning path in the component root.
        Order, length and member identity are guarded like ordinary owner links;
        no state or allocation declaration is copied into this namespace.
        """
        self._declare_owner_sequence(name, values, owning=False)

    def _declare_owner_sequence(self, name, values, *, owning):
        if not isinstance(name, str) or not name.isidentifier() or type(values) not in (list, tuple):
            raise TypeError("Owner sequences require an attribute name and a plain list/tuple")
        for value in values:
            if inspect.getattr_static(value, "_forge_resource_declarations", None) is None:
                raise ValueError("Owner sequence entries require normal resource declarations")
        if any(name in self._registry.get(kind, {}) for kind in
               ("fields", "arrays", "dimensions", "records", "borrowed", "values", "owner_references")):
            raise ValueError(f"Normal attribute {name!r} already has another declaration role")
        if self._read_only:
            if getattr(self._owner, name) is not values:
                raise ValueError("An owner sequence must refer to its existing member")
        else:
            setattr(self._owner, name, values)
        caller = inspect.currentframe().f_back
        self._registry.setdefault("owner_sequences", {})[name] = dict(
            length=len(values), owning=owning,
            source=(caller.f_back.f_code.co_filename, caller.f_back.f_lineno)
        )
        del caller

    def child(self, name, owner, *, builder=None, initialization="consumer_explicit"):
        """Construct an owned resource scope with explicit parent dimensions.

        Unlike ``record``, this scope allocates the child's ordinary Fields and
        uses its own declaration namespace. It never copies the parent's layout.
        """
        if self._read_only or not isinstance(name, str) or not name.isidentifier() or owner is None:
            raise TypeError("A child resource scope requires a writable parent, a name and an owner")
        if owner is self._owner:
            raise ValueError("A child resource scope cannot own its parent")
        occupied = set().union(*(self._registry.get(kind, {}) for kind in
            ("fields", "arrays", "dimensions", "values", "borrowed", "owner_sequences", "owner_references")))
        if name in occupied or name in self._registry.get("record_declarations", {}):
            raise ValueError(f"Normal attribute {name!r} already has another declaration role")
        if name in self._registry["records"] and getattr(self._owner, name) is not owner:
            raise ValueError("An existing child resource owner cannot be replaced")
        child = ResourceScope(owner, builder=builder, initialization=initialization)
        object.__setattr__(child, "_parent", self)
        setattr(self, name, owner)
        return child

    def parent_dim(self, name):
        """Explicitly relate a record or child layout to its parent count."""
        if self._parent is None:
            raise ValueError("parent_dim requires a declared record or child scope")
        self._parent.dim(name)
        alias = "__parent_" + name
        self._registry["dimensions"][alias] = self._parent._registry["dimensions"][name]
        self._registry.setdefault("inherited_dimensions", {})[alias] = ("dim", name)
        return Extent(("dim", alias))

    def owner_dim(self, reference, name):
        """Use a declared non-owning owner's dimension in this owner's layout.

        The referenced owner must have one ownership path in the component.
        This creates a dimension relation, never another consumer input, Field
        alias, or copied layout. The target's dimension declaration stays authoritative.
        """
        if (not isinstance(reference, str) or reference not in self._registry.get("owner_references", {})
                or not isinstance(name, str) or not name.isidentifier()):
            raise ValueError("owner_dim requires a declared owner_reference and dimension name")
        target = getattr(self._owner, reference)
        registry = inspect.getattr_static(target, "_forge_resource_declarations", None)
        if registry is None or name not in registry["dimensions"]:
            raise ValueError(f"Referenced owner dimension {reference}.{name} is not declared")
        if (name in registry.get("inherited_dimensions", {})
                or name in registry.get("referenced_dimensions", {})):
            raise ValueError("owner_dim requires a named normal dimension, not an internal dimension alias")
        value = ResourceScope(target)._normal_dimension(name)
        if not 0 <= value <= 2**31 - 1:
            raise ValueError(f"Referenced owner dimension {reference}.{name} is outside the i32 count range")
        alias = f"__owner_{len(reference)}_{reference}_{name}"
        relation = (reference, name)
        previous = self._registry.get("referenced_dimensions", {}).get(alias)
        if alias in self._registry["dimensions"] and previous != relation:
            raise ValueError(f"Owner dimension alias {alias!r} conflicts with another declaration")
        caller = inspect.currentframe().f_back
        self._registry["dimensions"].setdefault(alias, (caller.f_code.co_filename, caller.f_lineno))
        self._registry.setdefault("referenced_dimensions", {})[alias] = relation
        del caller
        return Extent(("dim", alias))

    def borrow(self, name, field, *, dimensions=None, source_scope=False, external=False):
        """Bind an existing Field and explicitly relate its whole source group.

        The target retains the original Field and storage ownership. This is a
        normal binding declaration, independent of kernels/export. Source
        dimensions include layout-only siblings, not just this Field's shape.
        source_scope=True instead requires the original allocation scope in
        the component root. Its layout remains authoritative; dimensions then
        supplies only the explicit target relations that need checking.
        external=True instead declares a read-only component input. Only this
        member's dimensions are mapped; its original group stays owned by the
        supplying component. The native member-view ABI still qualifies access.
        """
        from taichi_forge.lang.field import Field

        if not isinstance(name, str) or not name.isidentifier() or not isinstance(field, Field):
            raise TypeError("Resource borrow requires an attribute name and a Field")
        if type(external) is not bool or external and source_scope:
            raise TypeError("External Field inputs cannot also require a source scope")
        if type(source_scope) is not bool or (dimensions is None and not source_scope):
            raise TypeError("Resource borrow requires explicit dimensions or source_scope=True")
        if source_scope or external:
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
            same_source = (candidate is field if external else
                           source_group is not None and getattr(candidate, "_logical_resource_group", None) is source_group)
            if previous.get("external", False) == external and same_source:
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
        if external:
            declaration["external"] = True
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

    def dense(self, indices, shape):
        """Declare one normal Dense cell shared by multiple Field members.

        Keep extent provenance before placing members. Native SNode capture and
        StructCompiler remain authoritative for offsets, strides and allocation.
        Raw ``placement=`` nodes retain their fixed-tree contract.
        """
        if self._read_only:
            raise TypeError("A record scope cannot allocate Dense placements")
        from taichi_forge.lang import impl

        shape = shape if isinstance(shape, tuple) else (shape,)
        if not shape or any(not isinstance(item, Extent) and type(item) is not int for item in shape):
            raise TypeError("Dense placement extents require declared dimensions or integer constants")
        concrete = tuple(evaluate_extent(item.expression, self._normal_dimension)
                         if isinstance(item, Extent) else item for item in shape)
        if any(not 0 < value <= 2**31 - 1 for value in concrete):
            raise ValueError("Dense placement extents must be positive i32 values")
        builder = self._builder if self._builder is not None else impl._root_fb
        return _DeclaredDensePlacement(builder.dense(indices, concrete), shape, self._registry)

    def _create(self, factory, args, kwargs, *, member_initialization=None):
        if self._read_only:
            raise TypeError("A declared record scope describes existing resources; it cannot allocate Fields")
        kwargs = dict(kwargs)
        initialization = kwargs.pop("initialization", self._initialization)
        if initialization not in ("consumer_explicit", "zero"):
            raise ValueError("Resource initialization must be 'consumer_explicit' or 'zero'")
        placement = kwargs.pop("placement", None)
        declared_dense = isinstance(placement, _DeclaredDensePlacement)
        dense_declaration = placement if declared_dense else None
        if declared_dense:
            if placement.namespace is not self._registry:
                raise ValueError("Dense placement belongs to another declaration namespace")
            placement = placement.node
        if placement is not None and not declared_dense and (
            initialization != "consumer_explicit"
            or any(value != "consumer_explicit" for value in (member_initialization or {}).values())
        ):
            raise ValueError("Fixed SNode placement retains its empty activation initialization contract")
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
            if declared_dense and shape != dense_declaration.shape:
                raise ValueError("Dense placement requires the same declared extent relations")
            if not declared_dense and any(isinstance(item, Extent) for item in shape):
                raise ValueError("Fixed SNode schema cannot use dynamic extents; declare active counts separately")
            if not declared_dense and tuple(placement.shape) != shape:
                raise ValueError("Fixed SNode placement shape differs from the declared Field shape")
        concrete = tuple(
            (
                evaluate_extent(item.expression, self._normal_dimension)
                if isinstance(item, Extent)
                else item
            )
            for item in shape
        )
        if any(not 0 < value <= 2**31 - 1 for value in concrete):
            raise ValueError("Resource Field extents must be positive i32 values")
        if declared_dense and tuple(placement.shape) != concrete:
            raise ValueError("Dense placement shape changed after its declaration")
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
            order = kwargs.pop("order", None)
            if placement is not None and order is not None:
                raise ValueError("Explicit placement already determines the Field index order")
            axis_seq, shape_seq, same_level = impl._field_layout_axes(concrete, order)
            field = factory(*args, **{**kwargs, "shape": None})
            if placement is None:
                impl._create_snode(axis_seq, shape_seq, same_level, root_node=self._builder).place(field)
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
            member._logical_resource_declaration = dict(
                identity=identity, shape=shape, source=source,
                initialization=(member_initialization or {}).get(identity, initialization))
            # A Field alias in another declared child keeps the allocation
            # site's dimension namespace, just like a StructField member.
            member._logical_shape_namespace = self._registry
            if placement is not None and not declared_dense:
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

    def field_like(self, source, *, initialization=None, layout=None):
        """Allocate independent dense storage with a declared source's shape.

        The new Field retains the source dimension namespace, not its concrete
        producer capacity. Its placement and ownership are independent. The
        source allocation must be visible in the captured component closure.
        """
        from taichi_forge.lang import impl
        from taichi_forge.lang.enums import Layout
        from taichi_forge.lang.field import Field
        from taichi_forge.lang.matrix import MatrixField, Vector, Matrix
        from taichi_forge.lang.struct import Struct, StructField

        if self._read_only:
            raise ValueError("field_like requires a writable dense allocation scope")
        if not isinstance(source, Field):
            raise TypeError("field_like requires an ordinary declared Field")
        layout = Layout.AOS if layout is None else layout
        if layout not in (Layout.AOS, Layout.SOA):
            raise ValueError("field_like requires an AOS or SOA dense layout")
        initialization = self._initialization if initialization is None else initialization
        if initialization not in ("consumer_explicit", "zero"):
            raise ValueError("Resource initialization must be 'consumer_explicit' or 'zero'")
        sources = source.field_dict if isinstance(source, StructField) else {None: source}
        for member in sources.values():
            declaration = getattr(member, "_logical_resource_declaration", None)
            if declaration is None or declaration.get("storage") is not None:
                raise ValueError("field_like requires declared dense members, not a fixed or undeclared SNode")
            if getattr(member, "_logical_shape_namespace", None) is None:
                raise ValueError("field_like requires the original dimension declaration namespace")

        if isinstance(source, StructField):
            from taichi_forge.types import vector, matrix
            types = {name: (vector(member.n, member.dtype) if member.ndim == 1 else
                           matrix(member.n, member.m, member.dtype)) if isinstance(member, MatrixField)
                     else member.dtype for name, member in sources.items()}
            field = self._create(Struct.field, (types,), dict(shape=source.shape, layout=layout, initialization=initialization))
            field._forge_resource_declarations = {"fields": {}, "dimensions": {}, "records": {}}
            targets = field.field_dict
        elif isinstance(source, MatrixField):
            factory, args = ((Vector.field, (source.n, source.dtype)) if source.ndim == 1 else
                             (Matrix.field, (source.n, source.m, source.dtype)))
            field = self._create(factory, args, dict(shape=source.shape, layout=layout, initialization=initialization))
            targets = {None: field}
        else:
            field = self._create(impl.field, (source.dtype,), dict(shape=source.shape, initialization=initialization))
            targets = {None: field}
        caller = inspect.currentframe().f_back
        location = (caller.f_code.co_filename, caller.f_lineno)
        del caller
        for name, target in targets.items():
            original = sources[name]
            target._logical_resource_declaration = dict(identity=name,
                shape=original._logical_resource_declaration["shape"], source=location,
                initialization=initialization, require_shape_source=True)
            target._logical_shape_namespace = original._logical_shape_namespace
            if isinstance(field, StructField):
                field._forge_resource_declarations["fields"][name] = dict(target._logical_resource_declaration)
        return field

    def ndarray(self, dtype, *, shape, initialization=None):
        """Declare a normal independent array, retaining its logical extents.

        StructNdarray keeps the existing fixed AoS element layout. A surrounding
        FieldsBuilder does not make an independent array part of an SNode root.
        """
        from taichi_forge.lang.impl import ndarray

        if self._read_only:
            raise TypeError("A declared record scope cannot allocate arrays")
        if initialization is None:
            initialization = self._initialization
        if initialization not in ("consumer_explicit", "zero"):
            raise ValueError("Resource initialization must be 'consumer_explicit' or 'zero'")
        shape = shape if isinstance(shape, tuple) else (shape,)
        if any(not isinstance(item, Extent) and type(item) is not int for item in shape):
            raise TypeError("Array extents require declared dimensions or explicit integer constants")
        concrete = tuple(evaluate_extent(item.expression, self._normal_dimension)
                         if isinstance(item, Extent) else item for item in shape)
        if any(not 0 <= value <= 2**31 - 1 for value in concrete):
            raise ValueError("Array extents must fit the nonnegative i32 indexing domain")
        array = ndarray(dtype, shape=concrete)
        caller = inspect.currentframe().f_back
        array._logical_resource_declaration = dict(
            identity=None, shape=shape, source=(caller.f_code.co_filename, caller.f_lineno),
            initialization=initialization)
        array._logical_shape_namespace = self._registry
        del caller
        return array

    def vector_field(self, *args, **kwargs):
        from taichi_forge.lang.matrix import Vector

        return self._create(Vector.field, args, kwargs)

    def matrix_field(self, *args, **kwargs):
        from taichi_forge.lang.matrix import Matrix

        return self._create(Matrix.field, args, kwargs)

    def struct_field(self, members, *, member_initialization=None, **kwargs):
        """Declare a flat record at its normal allocation site.

        Member Field access and aliases retain the original shared placement.
        Implicit-root SOA uses Struct.field's ordinary per-member placement;
        explicit groups/placement currently require AOS. Whole Struct element
        indexing requires fixed-SNode capture; nested records are unqualified.
        ``member_initialization`` overrides the scope/record policy for named
        members without changing their normal placement or allocation.
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
        if member_initialization is not None:
            if not isinstance(member_initialization, Mapping):
                raise TypeError("Struct member initialization requires a mapping")
            member_initialization = dict(member_initialization)
            if any(name not in members for name in member_initialization):
                raise ValueError("Struct member initialization names an unknown member")
            if any(value not in ("consumer_explicit", "zero") for value in member_initialization.values()):
                raise ValueError("Struct member initialization must be 'consumer_explicit' or 'zero'")
        return self._create(Struct.field, (members,), kwargs, member_initialization=member_initialization)
