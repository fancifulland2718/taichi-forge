"""Component capture implementation behind the public ``ti.aot.Component``.

The declaration environment describes a component, not a kernel signature.
The normal frontend appends only captures that an entry actually references.
Dense SNode groups require the shared layout/instantiation contract and are
deliberately not converted to unrelated ndarrays here.
"""

import hashlib
import inspect
import json
import operator
import sys
from dataclasses import dataclass
from types import CodeType, MappingProxyType, MemberDescriptorType, ModuleType
from typing import Final, get_origin

from taichi_forge.lang import impl, kernel_arguments
from taichi_forge.lang._resource_declaration import (
    Extent,
    _source_members,
    evaluate_extent,
    extent_dependencies,
    freeze_extent,
    validate_extent,
)
from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge.lang.matrix import Matrix, MatrixType, Vector
from taichi_forge.lang.util import cook_dtype
from taichi_forge.types import primitive_types
from taichi_forge.types.ndarray_type import NdarrayType
from taichi_forge.types._argument_descriptor import describe_element_type
from taichi_forge_runtime._static_values import static_key


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _static_value(value):
    # No repr(), object IDs, getters or user-defined serialization in a key.
    try:
        return static_key(value)
    except ValueError as error:
        raise TaichiCompilationError(str(error)) from error


def _final_global(function, name):
    """Read an explicit module Final role without evaluating annotations/getters.

    A lexical closure cannot borrow a same-named module declaration. Deferred
    string annotations require separate qualification; never eval them here.
    """
    annotations = function.__globals__.get("__annotations__", {})
    if name in function.__code__.co_freevars or type(annotations) is not dict:
        return False
    annotation = annotations.get(name)
    return annotation is Final or get_origin(annotation) is Final


def _function(value):
    value = getattr(value, "__func__", value)
    if getattr(value, "_is_taichi_function", False):
        return value.func.func
    return value


def _dsl_function(value):
    return getattr(value, "_is_taichi_function", False) or (
        inspect.isfunction(value) and value.__module__.startswith("taichi_forge.")
    )


def _definition(value):
    value = _function(value)
    return (value.__module__, value.__qualname__, _digest(_code_definition(value.__code__)))


def _code_definition(code):
    # marshal's reference/intern encoding can change when inspect/AST parsing
    # retains constants, even though the immutable code object did not change.
    # Hash semantic contents, not serializer reference bookkeeping or locations.
    def constant(value):
        if isinstance(value, CodeType):
            return ("code", _code_definition(value))
        if isinstance(value, tuple):
            return ("tuple", tuple(constant(item) for item in value))
        if isinstance(value, frozenset):
            return ("frozenset", sorted((constant(item) for item in value), key=lambda x: json.dumps(x)))
        if isinstance(value, bytes):
            return ("bytes", value.hex())
        if isinstance(value, complex):
            return ("complex", value.real.hex(), value.imag.hex())
        if value is Ellipsis:
            return ("ellipsis",)
        return _static_value(value)

    return (
        sys.implementation.cache_tag,
        code.co_code.hex(),
        tuple(constant(value) for value in code.co_consts),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        getattr(code, "co_exceptiontable", b"").hex(),
    )


def _member(owner, name):
    # Deployment compilation must not execute arbitrary properties/descriptors.
    descriptor = inspect.getattr_static(owner, name)
    if isinstance(descriptor, MemberDescriptorType):
        # Built-in slots read stored state; properties/custom descriptors can
        # execute arbitrary Python and remain outside the capture contract.
        return descriptor.__get__(owner, type(owner))
    if hasattr(descriptor, "__get__"):
        raise TaichiCompilationError(f"Component capture {name!r} uses a descriptor; declare plain state instead")
    return descriptor


@dataclass(frozen=True)
class RuntimeValue:
    dtype: object

    @property
    def schema(self):
        if isinstance(self.dtype, MatrixType):
            return ("value", str(self.dtype.dtype), tuple(self.dtype.get_shape()))
        return ("value", str(cook_dtype(self.dtype)), ())

    def declare(self, name):
        if isinstance(self.dtype, MatrixType):
            return kernel_arguments.decl_matrix_arg(self.dtype, name, 0)
        return kernel_arguments.decl_scalar_arg(self.dtype, name, 0)

    @property
    def annotation(self):
        return self.dtype


@dataclass(frozen=True)
class Array:
    dtype: object
    ndim: int
    shape: tuple = None
    resource_id: str = ""

    @property
    def schema(self):
        if type(self.ndim) is not int or self.ndim < 0:
            raise ValueError("Component array ndim must be a nonnegative integer")
        if self.shape is not None and (len(self.shape) != self.ndim or any(
            not (isinstance(x, str) and x) and not (type(x) is int and x >= 0) for x in self.shape
        )):
            raise ValueError("Component array shape requires declared dimensions or nonnegative extents")
        return ("ndarray", describe_element_type(self.dtype).structural_key(), self.ndim, self.shape, self.resource_id)

    def declare(self, name):
        from taichi_forge._lib import core

        return kernel_arguments.decl_ndarray_arg(
            describe_element_type(self.dtype).logical_type, self.ndim, name, False, core.BoundaryMode.UNSAFE,
            logical_element_type=self.dtype,
        )

    def validate(self, value, component=None, owner=None):
        from taichi_forge.lang._ndarray import Ndarray

        if not isinstance(value, Ndarray):
            raise TaichiCompilationError("Component Array requires a whole managed Ndarray; Fields require the SNode layout contract")
        if not describe_element_type(self.dtype).matches(value.get_type().element_type) or len(value.shape) != self.ndim:
            raise TaichiCompilationError("Component array dtype/rank does not match its declaration")
        if value.arr is None or value._runtime_prog is not impl.get_runtime().prog:
            raise TaichiCompilationError("Component array belongs to a retired or different Runtime")
        if getattr(value, "grad", None) is not None:
            raise TaichiCompilationError("Component array gradients require separate qualification")
        if component is not None and self.shape is not None:
            expected = tuple(_dimension_value(component, owner, dim) if isinstance(dim, str) else dim for dim in self.shape)
            if tuple(value.shape) != expected:
                raise TaichiCompilationError("Component array shape differs from its declared extent relation")

    @property
    def annotation(self):
        return NdarrayType(dtype=self.dtype, ndim=self.ndim)


@dataclass(frozen=True)
class StaticValue:
    """Static code input, optionally with an explicit finite deployment domain.

    Choices are component admission policy, not additional numerical code
    dependencies. The frontend still specializes only actual captured values.
    kernel_only limits admission to exported numerical users; such a value
    must not control Python recording or a resource layout on its own.
    """

    choices: tuple = None
    kernel_only: bool = False

    def __post_init__(self):
        if type(self.kernel_only) is not bool:
            raise TypeError("Component static kernel_only must be a bool")
        if self.choices is not None:
            values = tuple(self.choices)
            keys = tuple(_static_value(value) for value in values)
            if not keys or len(set(keys)) != len(keys):
                raise ValueError("Component static choices must be nonempty and distinct typed values")
            object.__setattr__(self, "choices", values)

    def validate(self, value, name, entry):
        key = _static_value(value)
        if self.choices is not None and key not in tuple(_static_value(item) for item in self.choices):
            raise TaichiCompilationError(
                f"Component entry {entry}: static configuration {self._label(name)}={value!r} "
                f"is outside its declared domain {self.choices!r}"
            )
        return key

    def _label(self, name):
        return f"self.{name}"

    @property
    def schema(self):
        return ("static",)


@dataclass(frozen=True)
class RootOption(StaticValue):
    """Finite value of a normal recording argument, not an owner capture."""

    def __post_init__(self):
        super().__post_init__()
        if self.kernel_only:
            raise ValueError("Recording arguments cannot be kernel-only specializations")
        if self.choices is None:
            raise ValueError("Component recording options require finite choices")

    @property
    def schema(self):
        return ("root_option",)

    def _label(self, name):
        return f"recording argument {name!r}"


@dataclass(frozen=True)
class Dimension:
    """An explicitly named extent or active count on the normal owner."""

    kind: str = "extent"
    capacity: str = ""
    expression: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "expression", freeze_extent(self.expression) if self.expression else ())

    @property
    def schema(self):
        if self.kind not in ("extent", "active") or (self.kind == "extent" and self.capacity):
            raise ValueError("Dimension requires extent, or active with an optional capacity reference")
        base = ("dimension", "i32", self.kind, self.capacity)
        if self.expression:
            validate_extent(self.expression)
        return (*base, self.expression) if self.expression else base

    def declare(self, name):
        return kernel_arguments.decl_scalar_arg(primitive_types.i32, name, 0)

    @property
    def annotation(self):
        return primitive_types.i32


def _dimension_value(component, owner, name, visiting=()):
    declaration = component.declarations.get(name)
    if not isinstance(declaration, Dimension) or name in visiting:
        raise TaichiCompilationError(f"Invalid or cyclic dimension declaration {name!r}")
    value = (
        evaluate_extent(
            declaration.expression,
            lambda dependency: _dimension_value(component, owner, dependency, (*visiting, name)),
        )
        if declaration.expression
        else operator.index(component.read(owner, name))
    )
    minimum = 1 if declaration.kind == "extent" else 0
    if not minimum <= value <= 2**31 - 1:
        raise TaichiCompilationError(f"Dimension {name!r} is outside its declared i32 {declaration.kind} range")
    if name in component._derived_attributes and operator.index(component.read(owner, name)) != value:
        path = ".".join(component._paths[name])
        raise TaichiCompilationError(
            f"Derived dimension self.{path} does not match its normal attribute; "
            f"expected {value} from {declaration.expression!r}; declared at {component._sources.get(name)!r}"
        )
    if declaration.capacity:
        capacity = component.declarations.get(declaration.capacity)
        if not isinstance(capacity, Dimension) or capacity.kind != "extent":
            raise TaichiCompilationError(f"Active dimension {name!r} requires an extent capacity")
        if value > _dimension_value(component, owner, declaration.capacity, (*visiting, name)):
            raise TaichiCompilationError(f"Active dimension {name!r} exceeds capacity {declaration.capacity!r}")
    return value


@dataclass(frozen=True)
class DenseField:
    dtype: object
    shape: tuple
    group: str

    @property
    def schema(self):
        if not self.group or not isinstance(self.shape, tuple):
            raise ValueError("DenseField requires a group ID and a tuple of extent declarations")
        if any(not (isinstance(x, str) and x) and not (type(x) is int and x > 0) for x in self.shape):
            raise ValueError("DenseField extents must be dimension IDs or positive static integers")
        return ("dense_field", RuntimeValue(self.dtype).schema, self.shape, self.group)

    def declare(self, name):
        from taichi_forge._lib import core

        element = self.dtype.tensor_type if isinstance(self.dtype, MatrixType) else cook_dtype(self.dtype)
        reference = kernel_arguments.decl_ndarray_arg(element, len(self.shape), name, False, core.BoundaryMode.UNSAFE)
        # Preserve Field.dtype (the scalar component type, also for vector and
        # matrix Fields) on its lowered reference. The declaration/ABI already
        # guards this static metadata; shape and storage stay runtime inputs.
        reference.dtype = cook_dtype(self.dtype.dtype if isinstance(self.dtype, MatrixType) else self.dtype)
        return reference

    @property
    def annotation(self):
        return NdarrayType(dtype=self.dtype, ndim=len(self.shape))

    def validate(self, field):
        from taichi_forge.lang._storage_view import describe_storage

        description = describe_storage(field)
        if not description.supported or not (
            description.properties["ndarray_abi_compatible"]
            or description.properties["single_record_stride_compatible"]
        ):
            raise TaichiCompilationError("Dense component Field requires a qualified uniform-record SNode member")
        return description


@dataclass(frozen=True)
class FixedSNodeField:
    """Declared fixed tree schema; values and activation belong to an instance.

    Keep the original Field in frontend lowering. This is not an ndarray ABI.
    """

    dtype: object
    shape: tuple
    group: str

    @property
    def schema(self):
        if not self.group or not isinstance(self.shape, tuple) or any(type(x) is not int or x <= 0 for x in self.shape):
            raise ValueError("Fixed SNode fields require a group and positive static extents")
        return ("fixed_snode_field", RuntimeValue(self.dtype).schema, self.shape, self.group)

    def validate(self, field):
        from taichi_forge.lang.field import Field

        scalar = cook_dtype(self.dtype.dtype if isinstance(self.dtype, MatrixType) else self.dtype)
        element = tuple(self.dtype.get_shape()) if isinstance(self.dtype, MatrixType) else ()
        actual = ((field.n,) if field.ndim == 1 else (field.n, field.m)) if hasattr(field, "n") else ()
        if not isinstance(field, Field) or field.dtype != scalar or actual != element or tuple(field.shape) != self.shape:
            raise TaichiCompilationError("Fixed SNode Field type/shape differs from its normal declaration")


@dataclass(frozen=True, init=False)
class Record:
    """A declared plain record namespace, not a static Python object.

    Member dimension references are local to this record. Group identifiers
    remain component-wide so aliases across records can retain one SNode root.
    Only presence is static; actual leaves keep their own capture contracts.
    """

    optional: bool
    members: object

    def __init__(self, *, optional=False, **members):
        if type(optional) is not bool:
            raise TypeError("Record optional must be a bool")
        object.__setattr__(self, "optional", optional)
        object.__setattr__(self, "members", MappingProxyType(dict(members)))

    @property
    def schema(self):
        # Unused members must not become code dependencies.
        return ("record", self.optional)


@dataclass(frozen=True)
class _RecordReference:
    path: tuple

    def __bool__(self):
        raise TaichiCompilationError("A declared record is a namespace; test its presence with 'is not None'")


@dataclass(frozen=True)
class _ResourceTemplate:
    slot: int
    name: str
    snode_depth: int = None


def _capture_id(path):
    return path[0] if len(path) == 1 else "__record_" + "_".join(f"{len(part)}_{part}" for part in path)


def _dense_group(component, owner, group):
    """Project a real SNode group, using declared provenance for every axis.

    No byte layout is computed here. The native schema retains placements and
    calls the original SNode constructors/StructCompiler on instantiation.
    """
    from taichi_forge._lib import core
    from taichi_forge.lang.field import Field

    members = {}
    root = None
    axes = {}
    dimensions = {}
    for name, declaration in component.declarations.items():
        if not isinstance(declaration, DenseField) or declaration.group != group:
            continue
        field = component.read(owner, name)
        if not isinstance(field, Field):
            raise TaichiCompilationError(f"Dense resource {name!r} must be a Field")
        scalar_dtype = (
            declaration.dtype.dtype if isinstance(declaration.dtype, MatrixType) else cook_dtype(declaration.dtype)
        )
        element_shape = tuple(declaration.dtype.get_shape()) if isinstance(declaration.dtype, MatrixType) else ()
        actual_element = ((field.n,) if field.ndim == 1 else (field.n, field.m)) if hasattr(field, "n") else ()
        if field.dtype != scalar_dtype or actual_element != element_shape:
            raise TaichiCompilationError(f"Dense resource {name!r} element type differs from its declaration")
        expected_shape = []
        for extent in declaration.shape:
            if isinstance(extent, str):
                dim = component.declarations.get(extent)
                if not isinstance(dim, Dimension) or dim.kind != "extent":
                    raise TaichiCompilationError(f"Dense resource {name!r} requires extent declaration {extent!r}")
                dimensions[extent] = dim.schema
                expected_shape.append(_dimension_value(component, owner, extent))
            else:
                expected_shape.append(extent)
        if tuple(field.shape) != tuple(expected_shape):
            raise TaichiCompilationError(f"Dense resource {name!r} shape differs from its declared extent relation")
        nodes = tuple(var.ptr.snode() for var in field._get_field_members())
        members[name] = nodes
        for node in nodes:
            current_root = impl.get_runtime().prog.get_snode_root(node.get_snode_tree_id())
            if root is None:
                root = current_root
            if current_root.id != root.id:
                raise TaichiCompilationError(f"Resource group {group!r} must preserve one actual SNode root")
            parent = node.parent
            if declaration.shape and parent.type != core.SNodeType.dense:
                raise TaichiCompilationError(f"Resource {name!r} is not a simple dense member")
            physical_axes = node.get_physical_index_position()[: len(declaration.shape)]
            for axis, extent in zip(physical_axes, declaration.shape):
                key = (parent.id, axis)
                if key in axes and axes[key][1] != extent:
                    raise TaichiCompilationError(f"Conflicting dimension provenance in group {group!r}")
                axes[key] = (parent, extent)
    paths = {}

    def visit(node, path):
        paths[node.id] = path
        if node.type == core.SNodeType.dense:
            for axis in node.get_physical_index_position()[: node.num_active_indices()]:
                if (node.id, axis) not in axes:
                    raise TaichiCompilationError(f"Group {group!r} has an undeclared sibling extent at {path}")
        for index in range(node.get_num_ch()):
            visit(node.get_ch(index), (*path, index))

    visit(root, ())
    parameters = [(node, axis, extent) for (_, axis), (node, extent) in axes.items() if isinstance(extent, str)]
    schema = core.DenseSNodeSchema.capture(root, parameters).to_json()
    return (
        schema,
        {name: tuple(paths[node.id] for node in nodes) for name, nodes in members.items()},
        dimensions,
        root.id,
    )


def _fixed_snode_group(component, owner, group):
    """Query native construction/layout facts; never calculate sparse storage."""
    from taichi_forge._lib import core

    program = impl.get_runtime().prog
    root, tree_id, members = None, None, {}
    for name, declaration in component.declarations.items():
        if not isinstance(declaration, FixedSNodeField) or declaration.group != group:
            continue
        field = component.read(owner, name)
        declaration.validate(field)
        nodes = tuple(member.ptr.snode() for member in field._get_field_members())
        for node in nodes:
            current = node.get_snode_tree_id()
            if tree_id is None:
                tree_id, root = current, program.get_snode_root(current)
            if current != tree_id:
                raise TaichiCompilationError("Fixed SNode group must preserve one original root")
        members[name] = nodes
    paths = {}

    def visit(node, path):
        if node.type not in (core.SNodeType.root, core.SNodeType.dense, core.SNodeType.pointer,
                             core.SNodeType.bitmasked, core.SNodeType.place):
            raise TaichiCompilationError(f"Fixed component SNode type {node.type} is not qualified at {path}")
        paths[node.id] = path
        for index in range(node.get_num_ch()):
            visit(node.get_ch(index), (*path, index))

    visit(root, ())
    schema = dict(kind="fixed_snode", version=1,
                  allocation=json.loads(program._fixed_snode_tree_layout(tree_id)),
                  access_fingerprint=program._snode_tree_layout_fingerprint(tree_id))
    dependency = tuple(program._snapshot_snode_tree_dependencies([tree_id])[0])
    return (schema, {name: tuple(paths[node.id] for node in nodes) for name, nodes in members.items()}, dependency)


class Component:
    """Immutable, component-wide declarations; currently an internal API.

    Attribute names are explicit declaration IDs in this initial flat namespace.
    ``compile(owner.entry, ...)`` uses the original decorated numerical entry.
    No owner is retained by the resulting code, and no per-entry binding table
    is supplied by the application. Graph export follows separately.
    """

    __slots__ = ("_declarations", "_sources", "_paths", "_borrowed_reads", "_derived_attributes", "_constraints")

    def __init__(self, **declarations):
        flattened, paths = {}, {}

        def qualify(expression, prefix):
            op, *args = expression
            if op == "dim":
                return (op, _capture_id((*prefix, args[0])))
            return expression if op == "const" else (op, *(qualify(arg, prefix) for arg in args))

        def add(members, prefix=()):
            if len(prefix) > 32:
                raise ValueError("Component record nesting exceeds the declaration depth limit")
            for name, declaration in members.items():
                if (
                    not name.isidentifier()
                    or name.startswith("__record_")
                    or name.startswith("__borrowed_")
                    or not isinstance(declaration, (RuntimeValue, Array, StaticValue, Dimension, DenseField, FixedSNodeField, Record))
                ):
                    raise TypeError("Component declarations require attribute IDs and supported declaration types")
                path = (*prefix, name)
                identity = _capture_id(path)
                if prefix:
                    if isinstance(declaration, RootOption):
                        raise TypeError("Recording options belong to the component root, not a record")
                    if isinstance(declaration, Dimension):
                        declaration = Dimension(
                            declaration.kind,
                            _capture_id((*prefix, declaration.capacity)) if declaration.capacity else "",
                            qualify(declaration.expression, prefix) if declaration.expression else (),
                        )
                    elif isinstance(declaration, DenseField):
                        declaration = DenseField(
                            declaration.dtype,
                            tuple(
                                _capture_id((*prefix, value)) if isinstance(value, str) else value
                                for value in declaration.shape
                            ),
                            declaration.group,
                        )
                    elif isinstance(declaration, Array):
                        declaration = Array(declaration.dtype, declaration.ndim,
                            None if declaration.shape is None else tuple(
                                _capture_id((*prefix, value)) if isinstance(value, str) else value
                                for value in declaration.shape),
                            _capture_id((*prefix, declaration.resource_id)) if declaration.resource_id else "")
                declaration.schema
                flattened[identity], paths[identity] = declaration, path
                if isinstance(declaration, Record):
                    add(declaration.members, path)

        add(declarations)
        self._declarations = MappingProxyType(flattened)
        self._paths = MappingProxyType(paths)
        self._sources = MappingProxyType({})
        self._borrowed_reads = MappingProxyType({})
        self._derived_attributes = frozenset()
        self._constraints = MappingProxyType({})

    def read(self, owner, name):
        borrowed = self._borrowed_reads.get(name)
        if borrowed is not None:
            anchor_path, anchor_identity, member_identity, expected = borrowed
            for part in anchor_path:
                owner = _member(owner, part)
            try:
                members = _source_members(owner)
                if owner._logical_resource_declaration["identity"] != anchor_identity:
                    raise ValueError("source anchor identity changed")
                field = members[member_identity]
                if field._logical_resource_declaration != expected:
                    raise ValueError("source member declaration changed")
            except (AttributeError, KeyError, ValueError) as error:
                raise TaichiCompilationError(f"Borrowed resource {'.'.join(anchor_path)!r}: {error}") from error
            return field
        for part in self._paths.get(name, (name,)):
            owner = _member(owner, part)
        return owner

    def available(self, owner, name):
        for part in self._paths.get(name, (name,))[:-1]:
            owner = _member(owner, part)
            if owner is None:
                return False
        return True

    @property
    def declarations(self):
        return self._declarations

    @classmethod
    def from_owner(cls, owner, **values):
        """Project opt-in normal construction declarations, never a kernel list.

        Actual roots determine groups. Values come from normal explicit roles
        or caller declarations; no attribute dictionary is scanned for captures.
        """
        from taichi_forge.lang.matrix import MatrixType, VectorType

        impl.get_runtime().materialize(for_aot=True)
        declarations, sources, roots, fields, borrowed, arrays = {}, {}, {}, [], [], []
        inherited_dimensions, record_options, constraints = {}, {}, {}
        derived_attributes = set()
        namespaces, record_shapes, record_extents, array_declarations = {}, {}, {}, {}

        def root_of(field, name):
            nodes = tuple(member.ptr.snode() for member in field._get_field_members())
            root_ids = {node.get_snode_tree_id() for node in nodes}
            if len(root_ids) != 1:
                raise TaichiCompilationError(f"Normal resource {name!r} spans multiple SNode groups")
            return next(iter(root_ids))

        def collect(current, prefix=(), ancestors=(), registry=None):
            if any(current is ancestor for ancestor in ancestors) or len(prefix) > 32:
                raise TaichiCompilationError("Normal resource records must form a bounded acyclic declaration tree")
            if registry is None:
                registry = _member(current, "_forge_resource_declarations")
            namespaces[id(registry)] = prefix

            def qualify(expression):
                op, *args = expression
                if op == "dim":
                    return (op, _capture_id((*prefix, args[0])))
                return tuple(expression) if op == "const" else (op, *(qualify(arg) for arg in args))

            for name, constraint in registry.get("constraints", {}).items():
                spec = constraint["spec"]
                constraints[_capture_id((*prefix, name))] = dict(
                    spec=dict(left=qualify(spec["left"]), right=qualify(spec["right"]),
                              when=_capture_id((*prefix, spec["when"]))),
                    source=constraint["source"],
                )
            occupied = set()
            for kind in ("fields", "arrays", "dimensions", "records", "borrowed", "values"):
                names = registry.get(kind, {}).keys()
                if occupied.intersection(names):
                    raise ValueError(f"Conflicting normal declaration roles: {sorted(occupied.intersection(names))!r}")
                occupied.update(names)
            members = {name: Dimension("active") for name in registry["dimensions"]}
            for name, declaration in registry.get("values", {}).items():
                members[name] = declaration["declaration"]
                sources[_capture_id((*prefix, name))] = declaration["source"]
            for name, expression in registry.get("derived_dimensions", {}).items():
                members[name] = Dimension("active", expression=expression)
                derived_attributes.add(_capture_id((*prefix, name)))
            for name, source in registry["dimensions"].items():
                sources[_capture_id((*prefix, name))] = source
            for name, expression in registry.get("inherited_dimensions", {}).items():
                inherited_dimensions[_capture_id((*prefix, name))] = ("dim", _capture_id((*prefix[:-1], expression[1])))
            for name, declaration in registry.get("arrays", {}).items():
                array = _member(current, name)
                if getattr(array, "_logical_resource_declaration", None) != declaration:
                    raise TaichiCompilationError(f"Normal array {name!r} was replaced without redeclaration")
                arrays.append((members, prefix, name, array, declaration))
                sources[_capture_id((*prefix, name))] = declaration["source"]
            for name, declaration in registry["fields"].items():
                field = _member(current, name)
                if getattr(field, "_logical_resource_declaration", None) != declaration:
                    raise TaichiCompilationError(
                        f"Normal resource declaration {'.'.join((*prefix, name))!r} was replaced without redeclaration"
                    )
                root_id = root_of(field, name)
                roots.setdefault(root_id, set()).add(".".join((*prefix, declaration["identity"])))
                fields.append((members, prefix, name, field, declaration, root_id))
            for name, declaration in registry.get("borrowed", {}).items():
                field = _member(current, name)
                path = (*prefix, name)
                root_id = root_of(field, name)
                borrowed.append((path, field, declaration, root_id))
                roots.setdefault(root_id, set()).add("borrow:" + ".".join((*prefix, declaration["group"])))
            for name, source in registry.get("records", {}).items():
                sources[_capture_id((*prefix, name))] = source
                record_options[_capture_id((*prefix, name))] = registry.get("record_options", {}).get(name, False)
                members[name] = collect(
                    _member(current, name),
                    (*prefix, name),
                    (*ancestors, current),
                    registry.get("record_declarations", {}).get(name),
                )
            return members

        declarations = collect(owner)
        for members, prefix, name, array, declaration in arrays:
            namespace = getattr(array, "_logical_shape_namespace", None)
            if id(namespace) not in namespaces:
                raise TaichiCompilationError("Array dimensions require their original normal declaration scope")
            source_prefix = namespaces[id(namespace)]
            def qualify_array(expression):
                op, *args = expression
                if op == "dim":
                    return (op, _capture_id((*source_prefix, args[0])))
                return expression if op == "const" else (op, *(qualify_array(arg) for arg in args))
            shape = []
            for extent in declaration["shape"]:
                if isinstance(extent, Extent):
                    expression = qualify_array(extent.expression)
                    identifier = "__extent_" + _digest(expression)
                    record_extents[identifier] = Dimension(expression=expression)
                    sources[identifier] = declaration["source"]
                    shape.append(identifier)
                else:
                    shape.append(extent)
            resource = Array(array.get_type().element_type, len(shape), tuple(shape),
                             _capture_id((*source_prefix, declaration["identity"])))
            members[name] = resource
            array_declarations[_capture_id((*prefix, name))] = resource
        fixed_roots = {item[5] for item in fields if item[4].get("storage") == "fixed_snode"}
        for members, prefix, name, field, declaration, root_id in fields:
            if name.startswith("__extent_"):
                raise ValueError("Normal resource IDs cannot use the reserved __extent_ prefix")
            sources[_capture_id((*prefix, name))] = declaration["source"]
            shape = []
            namespace = getattr(field, "_logical_shape_namespace", None)
            shape_prefix = prefix
            if namespace is not None:
                if id(namespace) not in namespaces:
                    raise TaichiCompilationError("Record Field dimensions require their original declaration scope")
                shape_prefix = namespaces[id(namespace)]

            def qualify_shape(expression):
                op, *args = expression
                if op == "dim":
                    return (op, _capture_id((*shape_prefix, args[0])))
                return expression if op == "const" else (op, *(qualify_shape(arg) for arg in args))

            for extent in declaration["shape"]:
                if isinstance(extent, Extent):
                    expression = qualify_shape(extent.expression) if namespace is not None else extent.expression
                    identifier = "__extent_" + _digest(expression)
                    if namespace is not None:
                        record_extents[identifier] = Dimension(expression=expression)
                        sources[identifier] = declaration["source"]
                    else:
                        members[identifier] = Dimension(expression=expression)
                        sources[_capture_id((*prefix, identifier))] = declaration["source"]
                    shape.append(identifier)
                else:
                    shape.append(extent)
            dtype = field.dtype
            if hasattr(field, "n"):
                dtype = VectorType(field.n, dtype) if field.ndim == 1 else MatrixType(field.n, field.m, 2, dtype)
            # The normal declaration identity, not a tree/object/instance ID.
            group = "snode:" + min(roots[root_id])
            field_type = FixedSNodeField if root_id in fixed_roots else DenseField
            members[name] = field_type(dtype, tuple(shape), group)
            if namespace is not None:
                record_shapes[_capture_id((*prefix, name))] = tuple(shape)

        def complete(members, additions, prefix=()):
            result = {}
            for name, declaration in members.items():
                if isinstance(declaration, dict):
                    overlay = additions.get(
                        name, Record(optional=record_options.get(_capture_id((*prefix, name)), False))
                    )
                    if not isinstance(overlay, Record):
                        raise TypeError(f"Normal resource record {name!r} requires Record value declarations")
                    result[name] = Record(
                        optional=overlay.optional, **complete(declaration, overlay.members, (*prefix, name))
                    )
                else:
                    if name in additions:
                        selected = additions[name]
                        if type(declaration) is StaticValue and type(selected) is StaticValue:
                            # Normal construction states the available policy;
                            # a component may qualify a smaller algorithm domain.
                            # This does not change a role or add code dependencies.
                            allowed = {_static_value(value) for value in declaration.choices}
                            if (
                                selected.choices is None
                                or not {_static_value(value) for value in selected.choices} <= allowed
                                or selected.kernel_only != declaration.kernel_only
                            ):
                                raise ValueError(f"Component static domain {name!r} must narrow its normal declaration")
                            declaration = selected
                        else:
                            raise ValueError(f"Duplicate normal component declaration {name!r}")
                    result[name] = declaration
            result.update((name, declaration) for name, declaration in additions.items() if name not in members)
            return result

        result = cls(**complete(declarations, values))
        flattened = dict(result.declarations)
        flattened.update(record_extents)
        flattened.update(array_declarations)
        result._paths = MappingProxyType({**result._paths, **{name: (name,) for name in record_extents}})
        for name, shape in record_shapes.items():
            field = flattened[name]
            flattened[name] = type(field)(field.dtype, shape, field.group)
        for name, expression in inherited_dimensions.items():
            flattened[name] = Dimension("active", expression=expression)
        result._declarations = MappingProxyType(flattened)
        result._project_borrowed(borrowed, roots, sources)
        result._sources = MappingProxyType(sources)
        result._derived_attributes = frozenset(derived_attributes)
        result._constraints = MappingProxyType(constraints)
        return result

    def constraint_facts(self, owner, used, *, entry=None):
        """Project admission policy from actual uses without changing Kernel ABI."""
        from taichi_forge_runtime._dimensions import check_constraints, validate_constraints

        constraints = {name: item["spec"] for name, item in self._constraints.items() if item["spec"]["when"] in used}
        sources = {name: self._constraints[name]["source"] for name in constraints}
        dimensions = {}

        def depend(name):
            if name in dimensions:
                return
            declaration = self.declarations.get(name)
            if not isinstance(declaration, Dimension):
                raise TaichiCompilationError(f"Constraint requires declared dimension {name!r}")
            _dimension_value(self, owner, name)
            dimensions[name] = declaration.schema
            if declaration.capacity:
                depend(declaration.capacity)
            for child in extent_dependencies(declaration.expression) if declaration.expression else ():
                depend(child)

        for spec in constraints.values():
            for name in extent_dependencies(spec["left"]) | extent_dependencies(spec["right"]):
                depend(name)
        validate_constraints(constraints, dimensions.keys(), used)
        check_constraints(constraints, lambda name: _dimension_value(self, owner, name), sources, entry=entry)
        return constraints, dimensions, sources

    def _project_borrowed(self, borrowed, roots, sources):
        """Complete layout dependencies from source declarations, not a new ABI.

        Virtual members are used only by the existing native group capture.
        Numerical captures keep their normal target attribute/record paths.
        """
        from taichi_forge.lang.matrix import VectorType

        declarations, paths, reads, mappings = dict(self.declarations), dict(self._paths), {}, {}

        def rewrite(expression, mapping):
            op, *args = expression
            if op == "dim":
                return mapping[args[0]]
            return expression if op == "const" else (op, *(rewrite(arg, mapping) for arg in args))

        for path, anchor, binding, root_id in sorted(borrowed, key=lambda item: item[0]):
            try:
                members = _source_members(anchor)
                identity = anchor._logical_resource_declaration["identity"]
                if identity != binding["identity"]:
                    raise ValueError("source Field was replaced without redeclaration")
                dependencies = {
                    dim
                    for field in members.values()
                    for extent in field._logical_resource_declaration["shape"]
                    if isinstance(extent, Extent)
                    for dim in extent_dependencies(extent.expression)
                }
                if dependencies != binding["dimensions"].keys():
                    raise ValueError(
                        f"whole source group dimensions require {sorted(dependencies)!r}; "
                        f"received {sorted(binding['dimensions'])!r}"
                    )
                target_dimensions = {
                    name: ("dim", _capture_id((*path[:-1], name)))
                    for expression in binding["dimensions"].values()
                    for name in extent_dependencies(expression)
                }
                mapping = {
                    name: rewrite(expression, target_dimensions) for name, expression in binding["dimensions"].items()
                }
                if root_id in mappings and mappings[root_id] != mapping:
                    raise ValueError("aliases of one source group have conflicting dimension relations")
                first = root_id not in mappings
                mappings[root_id] = mapping
            except (AttributeError, KeyError, ValueError) as error:
                raise TaichiCompilationError(f"Borrowed resource {'.'.join(path)!r}: {error}") from error

            group = "snode:" + min(roots[root_id])
            for member_identity, field in members.items():
                source = field._logical_resource_declaration
                shape = []
                for extent in source["shape"]:
                    if isinstance(extent, Extent):
                        expression = rewrite(extent.expression, mapping)
                        dimension = "__extent_" + _digest(expression)
                        declarations[dimension] = Dimension(expression=expression)
                        paths[dimension] = (dimension,)
                        sources[dimension] = binding["source"]
                        shape.append(dimension)
                    else:
                        shape.append(extent)
                dtype = field.dtype
                if hasattr(field, "n"):
                    dtype = VectorType(field.n, dtype) if field.ndim == 1 else MatrixType(field.n, field.m, 2, dtype)
                declaration = DenseField(dtype, tuple(shape), group)
                accessor = (path, identity, member_identity, dict(source))
                if first:
                    name = "__borrowed_" + _digest((group, member_identity))
                    declarations[name], paths[name], reads[name] = declaration, path, accessor
                if field is anchor:
                    name = _capture_id(path)
                    if name in declarations:
                        raise ValueError(f"Duplicate borrowed component declaration {'.'.join(path)!r}")
                    declarations[name], paths[name], reads[name] = declaration, path, accessor
                    sources[name] = binding["source"]
        self._declarations = MappingProxyType(declarations)
        self._paths = MappingProxyType(paths)
        self._borrowed_reads = MappingProxyType(reads)

    def compile(self, entry, *args, owner=None, template_args=None):
        from taichi_forge.lang.kernel_impl import _process_args, TaichiCallableTemplateMapper
        from taichi_forge.lang._ndarray import Ndarray
        from taichi_forge.lang.field import Field
        from taichi_forge.lang.snode import SNode

        kernel = getattr(entry, "_primal", None)
        bound_owner = getattr(entry, "_kernel_owner", None)
        owner = bound_owner if owner is None else owner
        if kernel is None or owner is None or bound_owner is not None and bound_owner is not owner:
            raise TypeError("Component.compile requires a decorated kernel and its declared owner")
        if kernel.autodiff_mode.name != "NONE":
            raise TaichiCompilationError("Component capture autodiff is not qualified")
        templates = {} if template_args is None else dict(template_args)
        slots = set(kernel.template_slot_locations)
        if bound_owner is not None:
            name = kernel.arguments[0].name
            if name in templates and templates[name] is not owner:
                raise TaichiCompilationError("Component template owner differs from its bound method")
            templates[name] = owner
        expected = {kernel.arguments[i].name for i in slots}
        if set(templates) != expected:
            raise TaichiCompilationError("Component template_args must exactly name the kernel's template parameters")
        explicit = tuple(arg for i, arg in enumerate(kernel.arguments) if i not in slots)
        if len(args) > len(explicit):
            raise TypeError("Too many explicit component arguments")
        call_args = list(_process_args(kernel, (), {**templates, **dict(zip((a.name for a in explicit), args))}))
        features, abi, template_facts = [], [], []
        resource_templates = {}
        for i, (argument, value) in enumerate(zip(kernel.arguments, call_args)):
            if i in slots:
                if value is not owner and isinstance(value, (Field, SNode)):
                    if isinstance(value, SNode):
                        name, depth = self._template_snode(owner, value, argument.name)
                    else:
                        name, depth = self._template_resource(owner, value, argument.name), None
                    resource_templates[i] = name if depth is None else (name, depth)
                    call_args[i] = _ResourceTemplate(i, name, depth)
                    # Only actual uses select resource bindings below. An unused
                    # declared Field template must not specialize the entry.
                    template_facts.append((i, "resource", None))
                else:
                    template_facts.append(
                        (i, "owner" if value is owner else "static", None if value is owner else _static_value(value))
                    )
                features.append(None)
            elif isinstance(argument.annotation, NdarrayType):
                if not isinstance(value, Ndarray):
                    raise TaichiCompilationError("Component explicit arrays require whole managed JIT ndarrays")
                feature = TaichiCallableTemplateMapper.extract_arg(value, argument.annotation, argument.name)
                if feature[2]:
                    raise TaichiCompilationError("Component explicit ndarray gradients are not qualified")
                features.append(feature)
                abi.append(("ndarray", str(feature[0]), feature[1], feature[2], str(feature[3]), feature[4]))
            elif id(argument.annotation) in primitive_types.type_ids or isinstance(argument.annotation, MatrixType):
                features.append(None)
                abi.append(("value", str(argument.annotation)))
            else:
                raise TaichiCompilationError(
                    "Component explicit arguments require scalar/vector/matrix values or ndarrays"
                )
        signature = (tuple(template_facts), tuple(abi))
        runtime = impl.get_runtime()
        runtime.materialize(for_aot=True)
        with runtime._kernel_compilation_lock:
            for code in kernel._component_candidates:
                if (
                    code.call_signature == signature
                    and all(resource_templates.get(slot) == name for slot, name in code.resource_templates.items())
                    and code.matches(self, owner)
                ):
                    code.validate_bindings(self, owner)
                    return code
            context = _Capture(self, owner, kernel, signature)
            # No self identity or runtime value enters the template mapper.
            key = (kernel.func, "component_" + str(len(kernel._component_candidates)), kernel.autodiff_mode)
            kernel.materialize(key=key, args=call_args, arg_features=tuple(features), for_aot=True, deployment=context)
            code = _ComponentCode(kernel, key, context)
            kernel._component_candidates.append(code)
            return code

    def _template_resource(self, owner, value, argument):
        # Search only explicit declarations, never arbitrary owner attributes.
        # Identity is used to associate this call with a declared binding; no
        # Python object identity is retained in the code key or artifact.
        for name, declaration in sorted(self.declarations.items()):
            if not isinstance(declaration, (DenseField, FixedSNodeField)):
                continue
            try:
                if self.available(owner, name) and self.read(owner, name) is value:
                    return name
            except (AttributeError, TaichiCompilationError):
                # An unused optional/missing declaration is not a candidate.
                # _member never executes a user property while resolving it.
                continue
        raise TaichiCompilationError(
            f"Component Field template {argument!r} must refer to a declared resource of this owner"
        )

    def _template_snode(self, owner, value, argument):
        # A SNode template is an actual ancestor of a declared fixed Field,
        # never a Python object address or a newly interpreted tree schema.
        for name, declaration in sorted(self.declarations.items()):
            if not isinstance(declaration, FixedSNodeField) or not self.available(owner, name):
                continue
            node = self.read(owner, name).snode
            depth = 0
            while node is not None:
                if node.ptr.id == value.ptr.id:
                    return name, depth
                node, depth = node.parent(), depth + 1
        raise TaichiCompilationError(
            f"Component SNode template {argument!r} must descend from a declared fixed resource group"
        )

    def compile_graph(self, record, *args, **kwargs):
        """Compile a normal Graph recording with automatic captured inputs.

        initializer= registers another normal recording on the same owner.
        Its capture joins this root; numerical initialization belongs to the
        consumer instance, never to each newly prepared replay call.
        """
        from taichi_forge.aot._component_graph import compile_component_graph

        return compile_component_graph(self, record, *args, **kwargs)


class _Capture:
    def __init__(self, component, owner, kernel, call_signature):
        self.component = component
        self.declarations = dict(component.declarations)
        self.owner = owner
        self.kernel = kernel
        self.call_signature = call_signature
        self.used = {}
        self.expressions = {}
        self.dynamic = []
        self.static = {}
        self.static_objects = {}
        self.objects = {}
        self.methods = {}
        self.functions = {}
        self.constants = {}
        self.entry_definition = _definition(kernel.func)
        self.contract = None
        self.groups = {}
        self.member_paths = {}
        self.resource_aliases = {}
        self.bound_group_roots = {}
        self.fixed_groups = {}
        self.fixed_members = {}
        self.fixed_dependencies = {}
        self.resource_templates = {}

    def resolve_template(self, ctx, value):
        if not isinstance(value, _ResourceTemplate):
            return value
        self.resource_templates[value.slot] = value.name if value.snode_depth is None else (value.name, value.snode_depth)
        path = self.component._paths[value.name]
        # Record namespace presence just as the normal owner.record.field path
        # does, including optional records and borrowed source namespaces.
        for depth in range(1, len(path)):
            self.attribute(ctx, path[depth - 1], prefix=path[: depth - 1])
        field = self.attribute(ctx, value.name)
        if value.snode_depth is None:
            return field
        node = field.snode
        for _ in range(value.snode_depth):
            node = node.parent()
        return node

    def dimension_dependency(self, name):
        declaration = self.declarations[name]
        _dimension_value(self.component, self.owner, name)
        self.used[name] = declaration.schema
        if declaration.capacity:
            self.dimension_dependency(declaration.capacity)
        for dependency in extent_dependencies(declaration.expression) if declaration.expression else ():
            self.dimension_dependency(dependency)

    def attribute(self, ctx, name, *, prefix=()):
        path = (*prefix, name)
        name = _capture_id(path)
        declaration = self.declarations.get(name)
        if isinstance(declaration, RootOption):
            raise TaichiCompilationError(f"Component {name!r} is a recording option, not a numerical owner capture")
        if declaration is None:
            value = inspect.getattr_static(self.owner, name, None) if not prefix else None
            if not prefix and getattr(value, "_is_taichi_function", False):
                self.methods[name] = _definition(value)
                return value.__get__(self.owner, type(self.owner))
            raise TaichiCompilationError(
                f"Component entry {self.kernel.func.__qualname__}: undeclared capture self.{'.'.join(path)}; "
                "declare it as a runtime value, resource or static configuration"
            )
        self.used[name] = declaration.schema
        if isinstance(declaration, Record):
            present = self.component.read(self.owner, name) is not None
            if not present and not declaration.optional:
                raise TaichiCompilationError(f"Required component record self.{'.'.join(path)} is absent")
            previous = self.objects.setdefault(name, present)
            if previous != present:
                raise TaichiCompilationError("Component record presence changed during compilation; retry")
            return _RecordReference(path) if present else None
        if isinstance(declaration, Dimension):
            self.dimension_dependency(name)
        if isinstance(declaration, StaticValue):
            value = self.component.read(self.owner, name)
            fact = declaration.validate(value, name, self.kernel.func.__qualname__)
            if name in self.static and self.static[name] != fact:
                raise TaichiCompilationError(f"Component static dependency {name!r} changed during compilation; retry")
            self.static[name] = fact
            self.static_objects[name] = value
            return self.static_objects[name]
        # A resource reference can be selected/aliased at compile time while
        # its storage, shape and contents remain runtime inputs. This matches
        # ordinary ti.static(Field/AnyArray); scalar values still cannot escape
        # their declared runtime role through a static selection.
        if ctx.is_in_static_scope() and not isinstance(declaration, (Array, DenseField, FixedSNodeField)):
            raise TaichiCompilationError(
                f"Component entry {self.kernel.func.__qualname__}: runtime capture self.{name} "
                "cannot be used in ti.static; its declaration is dynamic"
            )
        if name not in self.expressions:
            value = self.component.read(self.owner, name)
            if isinstance(declaration, FixedSNodeField):
                schema, members, dependency = _fixed_snode_group(self.component, self.owner, declaration.group)
                previous = self.fixed_groups.setdefault(declaration.group, schema)
                if previous != schema or self.fixed_dependencies.setdefault(declaration.group, dependency) != dependency:
                    raise TaichiCompilationError("Fixed SNode group changed during compilation; retry")
                if any(other != declaration.group and bound[0] == dependency[0]
                       for other, bound in self.fixed_dependencies.items()):
                    raise TaichiCompilationError("Distinct component groups cannot split the same SNode root")
                self.fixed_members[name] = (declaration.group, members[name])
                self.expressions[name] = value
                return value
            if isinstance(declaration, Array):
                declaration.validate(value, self.component, self.owner)
                for dim in declaration.shape or ():
                    if isinstance(dim, str):
                        self.dimension_dependency(dim)
                if declaration.resource_id:
                    canonical = self.resource_aliases.setdefault(("ndarray", declaration.resource_id), name)
                    if canonical != name:
                        if self.component.read(self.owner, canonical) is not value:
                            raise TaichiCompilationError("Component array aliases must refer to the same allocation")
                        self.expressions[name] = self.expressions[canonical]
                        return self.expressions[name]
            elif isinstance(declaration, DenseField):
                declaration.validate(value)
            if isinstance(declaration, DenseField):
                schema, members, dimensions, root_id = _dense_group(self.component, self.owner, declaration.group)
                other_group = self.bound_group_roots.setdefault(root_id, declaration.group)
                if other_group != declaration.group:
                    raise TaichiCompilationError("Distinct component groups cannot split the same SNode root")
                previous = self.groups.setdefault(declaration.group, schema)
                if previous != schema:
                    raise TaichiCompilationError("Resource group changed during compilation; retry")
                for dim in dimensions:
                    self.dimension_dependency(dim)
                self.member_paths[name] = (declaration.group, members[name])
                canonical = self.resource_aliases.setdefault(self.member_paths[name], name)
                if canonical != name:
                    self.expressions[name] = self.expressions[canonical]
                    return self.expressions[name]
            self.dynamic.append(name)
            self.expressions[name] = declaration.declare("__capture_" + name)
        return self.expressions[name]

    def field_operand(self, ctx, value):
        """Resolve a declared fixed StructField into the ordinary Field lowering.

        Member-first access remains lazy. Whole-record indexing builds all
        member references in impl.subscript, so those declarations must be
        captured before handing the original StructField to that same path.
        """
        if not isinstance(value, _RecordReference):
            return value
        from taichi_forge.lang.struct import StructField

        field = self.component.read(self.owner, _capture_id(value.path))
        if not isinstance(field, StructField):
            raise TaichiCompilationError("A declared record namespace is not an indexable or iterable Field")
        names = tuple(field.field_dict)
        if not names or any(
            not isinstance(self.declarations.get(_capture_id((*value.path, name))), FixedSNodeField)
            for name in names
        ):
            raise TaichiCompilationError("Whole StructField indexing/traversal requires declared fixed SNode members")
        for name in names:
            self.attribute(ctx, name, prefix=value.path)
        return field

    def named_attribute(self, ctx, args, keywords):
        """Resolve builtin getattr through the same declared attribute path."""
        if keywords or len(args) not in (2, 3) or type(args[1]) is not str:
            raise TaichiCompilationError(
                "Component getattr requires a declared owner/record and a compile-time string name"
            )
        # A default is unused for an existing declared attribute. Resolve via
        # the normal capture path so neither undeclared values nor descriptors
        # can be hidden by a fallback. Optional absence is not implicit static
        # configuration; applications must declare that role explicitly.
        owner, name = args[:2]
        if owner is self.owner:
            return self.attribute(ctx, name)
        if isinstance(owner, _RecordReference):
            return self.attribute(ctx, name, prefix=owner.path)
        raise TaichiCompilationError("Component getattr requires a declared owner or record")

    def check_name(self, ctx, node):
        if node.id not in ctx.global_vars:
            return  # Builtins are checked at their call site.
        value = ctx.global_vars[node.id]
        if isinstance(value, ModuleType):
            if value.__name__ == "taichi_forge" or value.__name__.startswith("taichi_forge."):
                return
            # Some consumers load this same package under the legacy public
            # import name. Qualify its actual package and frontend, not an
            # arbitrary module merely named "taichi" or re-exporting cast.
            frontend = sys.modules["taichi_forge"]
            if (
                value.__name__ == "taichi"
                and vars(value).get("__file__") == frontend.__file__
                and vars(value).get("lang") is frontend.lang
            ):
                return
        if _dsl_function(value):
            self.functions[(ctx.func.func, node.id)] = _definition(value)
            return
        if id(value) in primitive_types.type_ids or isinstance(value, MatrixType):
            return
        if value is Matrix or value is Vector:
            return  # Built-in DSL constructors, also available through ti.Matrix/Vector.
        function = ctx.func.func
        if _final_global(function, node.id):
            key = _static_value(value)
            binding = (function, node.id)
            if self.constants.setdefault(binding, key) != key:
                raise TaichiCompilationError(f"Static global {node.id!r} changed during compilation; retry")
            return
        raise TaichiCompilationError(
            f"Component entry {self.kernel.func.__qualname__}: undeclared global/closure capture {node.id!r}; "
            "use a declared component value, an immutable module typing.Final, or a literal"
        )

    def check_call(self, value):
        if getattr(value, "_is_taichi_function", False):
            return
        # The AST routes builtin print to ti_print/PrintStmt, never to a host
        # Python callback. Preserve diagnostics in original deployed kernels.
        if any(value is item for item in (range, len, min, max, abs, int, float, bool, enumerate, zip, print)):
            return
        if getattr(value, "__module__", "").startswith("taichi_forge"):
            return
        raise TaichiCompilationError("Uncontrolled Python calls are not supported in component captures")

    def check_record_comparison(self, left, right, operation):
        if isinstance(left, _RecordReference) or isinstance(right, _RecordReference):
            if operation not in ("Is", "IsNot") or left is not None and right is not None:
                raise TaichiCompilationError("Declared records only support presence comparisons with None")

    def finish(self):
        self.component.constraint_facts(self.owner, self.used, entry=self.entry_definition[1])
        mismatch = self.dependency_mismatch(self.component, self.owner)
        if mismatch:
            raise TaichiCompilationError(f"Component dependency changed during compilation: {mismatch}; retry")
        self.contract = {
            "version": 1,
            "entry": self.entry_definition,
            "explicit_abi": self.call_signature[1],
            "templates": self.call_signature[0],
            "captures": [(name, self.used[name]) for name in self.dynamic],
            "static": sorted(self.static.items()),
            "groups": self.groups,
            "members": self.member_paths,
            "dimensions": {name: schema for name, schema in self.used.items() if schema[0] == "dimension"},
            "methods": sorted(self.methods.items()),
            "functions": sorted((name, definition) for (_, name), definition in self.functions.items()),
            "declaration_sources": {
                name: self.component._sources[name] for name in self.used if name in self.component._sources
            },
        }
        if self.objects:
            self.contract.update(
                objects=sorted(self.objects.items()),
                access_paths={name: self.component._paths[name] for name in self.used},
            )
        if self.constants:
            self.contract["constants"] = sorted(
                (function.__module__, function.__qualname__, name, value)
                for (function, name), value in self.constants.items())
        if self.resource_templates:
            self.contract["resource_templates"] = sorted(self.resource_templates.items())
        if self.fixed_groups:
            self.contract.update(fixed_groups=self.fixed_groups, fixed_members=self.fixed_members,
                                 fixed_captures=[(name, self.used[name]) for name in self.fixed_members])

    def matches(self, component, owner):
        return self.dependency_mismatch(component, owner) is None

    def dependency_mismatch(self, component, owner):
        if _definition(self.kernel.func) != self.entry_definition:
            return "entry definition"
        for name, present in self.objects.items():
            if (component.read(owner, name) is not None) != present:
                return f"record presence {name!r}"
        arrays = {}
        for name, schema in self.used.items():
            declaration = component.declarations.get(name)
            if declaration is None or declaration.schema != schema:
                return f"declaration {name!r}"
            if isinstance(declaration, Dimension):
                _dimension_value(component, owner, name)
            if isinstance(declaration, Array):
                value = component.read(owner, name)
                declaration.validate(value, component, owner)
                if declaration.resource_id and arrays.setdefault(declaration.resource_id, value) is not value:
                    return f"array alias {name!r}"
        roots = set()
        for group, schema in self.fixed_groups.items():
            actual, members, dependency = _fixed_snode_group(component, owner, group)
            if schema != actual:
                return f"fixed layout of group {group!r}"
            # Producer definitions retain physical Field bindings. They cannot
            # run on another owner just because the deployable schema matches.
            # This binding guard never enters the semantic code identity.
            if self.fixed_dependencies[group] != dependency:
                return f"source binding of fixed group {group!r}"
            for name, (member_group, paths) in self.fixed_members.items():
                if member_group == group and members.get(name) != paths:
                    return f"fixed member {name!r} in group {group!r}"
        for group, schema in self.groups.items():
            actual, members, _, root_id = _dense_group(component, owner, group)
            if root_id in roots:
                raise TaichiCompilationError("Distinct component groups cannot split the same SNode root")
            roots.add(root_id)
            if actual != schema:
                return f"layout of group {group!r}"
            for name, (member_group, paths) in self.member_paths.items():
                if member_group == group and members.get(name) != paths:
                    return f"member {name!r} in group {group!r}"
        for name, value in self.static.items():
            if _static_value(component.read(owner, name)) != value:
                return f"static value {name!r}"
        for (function, name), value in self.constants.items():
            if not _final_global(function, name) or name not in function.__globals__:
                return f"static global declaration {name!r} in {function.__qualname__}"
            try:
                current = _static_value(function.__globals__[name])
            except TaichiCompilationError:
                return f"static global value {name!r} in {function.__qualname__}"
            if current != value:
                return f"static global value {name!r} in {function.__qualname__}"
        for name, definition in self.methods.items():
            value = inspect.getattr_static(owner, name, None)
            if not getattr(value, "_is_taichi_function", False) or _definition(value) != definition:
                return f"method {name!r}"
        for (function, name), definition in self.functions.items():
            # getclosurevars only examines the outer code's co_names. A
            # comprehension can read a global helper solely from its nested
            # code object (e.g. matrix inverse's E), which the AST did capture.
            # Resolve that recorded name in the real lexical environment.
            cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
            try:
                value = cells[name].cell_contents if name in cells else function.__globals__.get(name)
            except ValueError:  # An emptied nonlocal cell invalidates the guard.
                return f"function {name!r} in {function.__qualname__}"
            if not _dsl_function(value) or _definition(value) != definition:
                return f"function {name!r} in {function.__qualname__}"
        return None


class _ComponentCode:
    def __init__(self, kernel, key, context):
        self.kernel = kernel
        self.key = key
        self.call_signature = context.call_signature
        self.runtime = kernel.runtime
        self.semantic_key = _digest(
            {key: value for key, value in context.contract.items() if key != "declaration_sources"}
        )
        # Copy plain dependency facts, never the lowering context/owner/Exprs.
        self.entry_definition = context.entry_definition
        self.used = dict(context.used)
        self.static = dict(context.static)
        self.objects = dict(context.objects)
        self.methods = dict(context.methods)
        self.functions = dict(context.functions)
        self.constants = dict(context.constants)
        self.groups = dict(context.groups)
        self.member_paths = dict(context.member_paths)
        self.fixed_groups = dict(context.fixed_groups)
        self.fixed_members = dict(context.fixed_members)
        self.fixed_dependencies = dict(context.fixed_dependencies)
        self.dynamic = tuple(context.dynamic)
        self.resource_templates = dict(context.resource_templates)
        self._contract = json.dumps(context.contract)
        self.explicit_arguments = tuple(
            arg for i, arg in enumerate(kernel.arguments) if i not in kernel.template_slot_locations
        )
        self.arguments = self.explicit_arguments + tuple(
            kernel_arguments.KernelArgument(context.component.declarations[name].annotation, "__capture_" + name)
            for name in self.dynamic
        )

    matches = _Capture.matches
    dependency_mismatch = _Capture.dependency_mismatch

    @property
    def contract(self):
        return json.loads(self._contract)

    def validate_bindings(self, component, owner):
        component.constraint_facts(owner, self.used, entry=self.entry_definition[1])
        for name in self.static:
            component.declarations[name].validate(component.read(owner, name), name, self.kernel.func.__qualname__)
        for name in self.dynamic:
            value = component.read(owner, name)
            declaration = component.declarations[name]
            if isinstance(declaration, (Array, DenseField)):
                declaration.validate(value)

    def export(self, module, name):
        """Export this compiled entry and its capture projection, without state.

        This low-level entry hook will also serve component-root/Graph closure
        export; it does not require a decorated wrapper or recompiling an owner.
        """
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("Component entry requires a nonempty name without NUL")
        if impl.get_runtime() is not self.runtime or self.key not in self.kernel.compiled_kernels:
            raise RuntimeError("Component code belongs to a retired Runtime")
        if module._runtime is not self.runtime:
            raise ValueError("Component entry and export Module belong to different Runtimes")
        if module._arch != impl.current_cfg().arch:
            raise ValueError("Component entry target must match its compilation Runtime")
        compiled = self.kernel.compiled_kernels[self.key]
        if not module._check_kernel_export(name, compiled):
            if module._component_entries.get(name, {}).get("semantic_key") != self.semantic_key:
                raise ValueError("Existing entry was not exported with this component contract")
            return
        try:
            module._aot_builder.add(name, compiled)
            actual = set(module._aot_builder._used_snode_tree_ids([name]))
            declared = {dependency[0] for dependency in self.fixed_dependencies.values()}
            if not actual <= declared:
                raise TaichiCompilationError("Generic component entry retains an undeclared fixed Field dependency")
            if self.fixed_dependencies:
                live = {tuple(dependency) for dependency in self.runtime.prog._snapshot_snode_tree_dependencies(sorted(actual))}
                if not live <= set(self.fixed_dependencies.values()):
                    raise TaichiCompilationError("Component fixed Field lifetime changed before export")
        except BaseException:
            # Native builders cannot roll back partially added code. Preserve
            # the same fail-closed publication rule as existing Graph export.
            module._graph_export_failed = True
            raise
        module._component_entries[name] = {
            "version": 1,
            "entry": name,
            "target": module._arch.name,
            "semantic_key": self.semantic_key,
            "contract": self.contract,
            "initialization": {"group_storage": "zero", "model_data": "consumer",
                               **({"fixed_snode": "empty_activation_and_zero_values"} if self.fixed_groups else {})},
        }
        module._exported_kernels[name] = compiled
        module._kernels.append(self.kernel)
        module._content.extend(("kernel:" + name, "component_entry:" + name))

    def run(self, component, owner, *args):
        if impl.get_runtime() is not self.runtime or self.key not in self.kernel.compiled_kernels:
            raise RuntimeError("Component code belongs to a retired Runtime")
        if not self.matches(component, owner):
            raise ValueError("Component bindings do not match the compiled declaration/static dependencies")
        self.validate_bindings(component, owner)
        if len(args) != len(self.explicit_arguments):
            raise TypeError("Wrong number of explicit component arguments")
        from taichi_forge.lang._storage_view import ndarray_view

        values = tuple(
            (
                ndarray_view(component.read(owner, name))
                if isinstance(component.declarations[name], DenseField)
                else component.read(owner, name)
            )
            for name in self.dynamic
        )
        return self.kernel.launch_kernel(
            self.kernel.compiled_kernels[self.key], *args, *values, _arguments=self.arguments
        )
