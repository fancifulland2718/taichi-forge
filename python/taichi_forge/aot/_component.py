"""Internal component capture compiler (public SDK API is not frozen yet).

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
from types import CodeType, MappingProxyType, ModuleType

from taichi_forge.lang import impl, kernel_arguments
from taichi_forge.lang._resource_declaration import (
    Extent,
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
from taichi_forge_runtime._static_values import static_key


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _static_value(value):
    # No repr(), object IDs, getters or user-defined serialization in a key.
    try:
        return static_key(value)
    except ValueError as error:
        raise TaichiCompilationError(str(error)) from error


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

    @property
    def schema(self):
        if type(self.ndim) is not int or self.ndim < 0:
            raise ValueError("Component array ndim must be a nonnegative integer")
        return ("ndarray", str(cook_dtype(self.dtype)), self.ndim)

    def declare(self, name):
        from taichi_forge._lib import core

        return kernel_arguments.decl_ndarray_arg(
            cook_dtype(self.dtype), self.ndim, name, False, core.BoundaryMode.UNSAFE
        )

    def validate(self, value):
        from taichi_forge.lang._ndarray import ScalarNdarray

        if not isinstance(value, ScalarNdarray):
            raise TaichiCompilationError(
                "Component Array currently requires a scalar Ndarray; Dense Field groups require the SNode layout contract"
            )
        if value.dtype != cook_dtype(self.dtype) or len(value.shape) != self.ndim:
            raise TaichiCompilationError("Component array dtype/rank does not match its declaration")

    @property
    def annotation(self):
        return NdarrayType(dtype=self.dtype, ndim=self.ndim)


@dataclass(frozen=True)
class StaticValue:
    """Static code input, optionally with an explicit finite deployment domain.

    Choices are component admission policy, not additional numerical code
    dependencies. The frontend still specializes only actual captured values.
    """

    choices: tuple = None

    def __post_init__(self):
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
        else operator.index(_member(owner, name))
    )
    minimum = 1 if declaration.kind == "extent" else 0
    if not minimum <= value <= 2**31 - 1:
        raise TaichiCompilationError(f"Dimension {name!r} is outside its declared i32 {declaration.kind} range")
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
        return kernel_arguments.decl_ndarray_arg(element, len(self.shape), name, False, core.BoundaryMode.UNSAFE)

    @property
    def annotation(self):
        return NdarrayType(dtype=self.dtype, ndim=len(self.shape))

    def validate(self, field):
        from taichi_forge.lang._storage_view import describe_storage

        description = describe_storage(field)
        if not description.supported or not description.properties["ndarray_abi_compatible"]:
            raise TaichiCompilationError("Dense component Field requires a qualified compact SNode member")
        return description


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
        field = _member(owner, name)
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


class Component:
    """Immutable, component-wide declarations; currently an internal API.

    Attribute names are explicit declaration IDs in this initial flat namespace.
    ``compile(owner.entry, ...)`` uses the original decorated numerical entry.
    No owner is retained by the resulting code, and no per-entry binding table
    is supplied by the application. Graph export follows separately.
    """

    __slots__ = ("_declarations", "_sources")

    def __init__(self, **declarations):
        for name, declaration in declarations.items():
            if not name.isidentifier() or not isinstance(
                declaration, (RuntimeValue, Array, StaticValue, Dimension, DenseField)
            ):
                raise TypeError("Component declarations require attribute IDs and supported declaration types")
            declaration.schema  # Validate without reading an instance.
        self._declarations = MappingProxyType(dict(declarations))
        self._sources = MappingProxyType({})

    @property
    def declarations(self):
        return self._declarations

    @classmethod
    def from_owner(cls, owner, **values):
        """Project opt-in normal construction declarations, never a kernel list.

        Actual roots determine groups. Values remain explicit static/runtime
        declarations; no Python attribute dictionary is scanned for captures.
        """
        from taichi_forge.lang.matrix import MatrixType, VectorType

        registry = _member(owner, "_forge_resource_declarations")
        impl.get_runtime().materialize(for_aot=True)
        declarations = {name: Dimension("active") for name in registry["dimensions"]}
        sources = dict(registry["dimensions"])
        roots = {}
        fields = {}
        for name, declaration in registry["fields"].items():
            field = _member(owner, name)
            if getattr(field, "_logical_resource_declaration", None) != declaration:
                raise TaichiCompilationError(f"Normal resource declaration {name!r} was replaced without redeclaration")
            nodes = tuple(member.ptr.snode() for member in field._get_field_members())
            root_ids = {node.get_snode_tree_id() for node in nodes}
            if len(root_ids) != 1:
                raise TaichiCompilationError(f"Normal resource {name!r} spans multiple SNode groups")
            root_id = next(iter(root_ids))
            roots.setdefault(root_id, set()).add(declaration["identity"])
            fields[name] = (field, declaration, root_id)
        for name, (field, declaration, root_id) in fields.items():
            if name.startswith("__extent_"):
                raise ValueError("Normal resource IDs cannot use the reserved __extent_ prefix")
            sources[name] = declaration["source"]
            shape = []
            for extent in declaration["shape"]:
                if isinstance(extent, Extent):
                    identifier = "__extent_" + _digest(extent.expression)
                    declarations[identifier] = Dimension(expression=extent.expression)
                    sources[identifier] = declaration["source"]
                    shape.append(identifier)
                else:
                    shape.append(extent)
            dtype = field.dtype
            if hasattr(field, "n"):
                dtype = VectorType(field.n, dtype) if field.ndim == 1 else MatrixType(field.n, field.m, 2, dtype)
            # The normal declaration identity, not a tree/object/instance ID.
            group = "snode:" + min(roots[root_id])
            declarations[name] = DenseField(dtype, tuple(shape), group)
        for name, declaration in values.items():
            if name in declarations:
                raise ValueError(f"Duplicate normal component declaration {name!r}")
            declarations[name] = declaration
        result = cls(**declarations)
        result._sources = MappingProxyType(sources)
        return result

    def compile(self, entry, *args):
        from taichi_forge.lang.kernel_impl import _process_args

        kernel = getattr(entry, "_primal", None)
        owner = getattr(entry, "_kernel_owner", None)
        if kernel is None or owner is None:
            raise TypeError("Component.compile requires a bound, decorated kernel method")
        if kernel.template_slot_locations != [0] or kernel.autodiff_mode.name != "NONE":
            raise TaichiCompilationError("Component capture currently requires only the self template and no autodiff")
        if any(
            id(arg.annotation) not in primitive_types.type_ids and not isinstance(arg.annotation, MatrixType)
            for arg in kernel.arguments[1:]
        ):
            raise TaichiCompilationError(
                "Component capture currently supports only explicit scalar/vector/matrix values"
            )
        call_args = _process_args(kernel, (owner, *args), {})
        runtime = impl.get_runtime()
        runtime.materialize(for_aot=True)
        with runtime._kernel_compilation_lock:
            for code in kernel._component_candidates:
                if code.matches(self, owner):
                    code.validate_bindings(self, owner)
                    return code
            context = _Capture(self, owner, kernel)
            # No self identity or runtime value enters the template mapper.
            # Explicit arguments in this slice are value types only.
            features = tuple(None for _ in call_args)
            key = (kernel.func, "component_" + str(len(kernel._component_candidates)), kernel.autodiff_mode)
            kernel.materialize(key=key, args=call_args, arg_features=features, for_aot=True, deployment=context)
            code = _ComponentCode(kernel, key, context)
            kernel._component_candidates.append(code)
            return code

    def compile_graph(self, record, *args, **kwargs):
        """Compile a normal Graph recording with automatic captured inputs."""
        from taichi_forge.aot._component_graph import compile_component_graph

        return compile_component_graph(self, record, *args, **kwargs)


class _Capture:
    def __init__(self, component, owner, kernel):
        self.component = component
        self.declarations = dict(component.declarations)
        self.owner = owner
        self.kernel = kernel
        self.used = {}
        self.expressions = {}
        self.dynamic = []
        self.static = {}
        self.static_objects = {}
        self.methods = {}
        self.functions = {}
        self.entry_definition = _definition(kernel.func)
        self.contract = None
        self.groups = {}
        self.member_paths = {}
        self.resource_aliases = {}
        self.bound_group_roots = {}

    def dimension_dependency(self, name):
        declaration = self.declarations[name]
        _dimension_value(self.component, self.owner, name)
        self.used[name] = declaration.schema
        if declaration.capacity:
            self.dimension_dependency(declaration.capacity)
        for dependency in extent_dependencies(declaration.expression) if declaration.expression else ():
            self.dimension_dependency(dependency)

    def attribute(self, ctx, name):
        declaration = self.declarations.get(name)
        if isinstance(declaration, RootOption):
            raise TaichiCompilationError(f"Component {name!r} is a recording option, not a numerical owner capture")
        if declaration is None:
            value = inspect.getattr_static(self.owner, name)
            if getattr(value, "_is_taichi_function", False):
                self.methods[name] = _definition(value)
                return value.__get__(self.owner, type(self.owner))
            raise TaichiCompilationError(
                f"Component entry {self.kernel.func.__qualname__}: undeclared capture self.{name}; "
                "declare it as a runtime value, resource or static configuration"
            )
        self.used[name] = declaration.schema
        if isinstance(declaration, Dimension):
            self.dimension_dependency(name)
        if isinstance(declaration, StaticValue):
            value = _member(self.owner, name)
            fact = declaration.validate(value, name, self.kernel.func.__qualname__)
            if name in self.static and self.static[name] != fact:
                raise TaichiCompilationError(f"Component static dependency {name!r} changed during compilation; retry")
            self.static[name] = fact
            self.static_objects[name] = value
            return self.static_objects[name]
        if ctx.is_in_static_scope():
            raise TaichiCompilationError(
                f"Component entry {self.kernel.func.__qualname__}: runtime capture self.{name} "
                "cannot be used in ti.static; its declaration is dynamic"
            )
        if name not in self.expressions:
            value = _member(self.owner, name)
            if isinstance(declaration, (Array, DenseField)):
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

    def check_name(self, ctx, node):
        if node.id not in ctx.global_vars:
            return  # Builtins are checked at their call site.
        value = ctx.global_vars[node.id]
        if isinstance(value, ModuleType) and value.__name__.startswith("taichi_forge"):
            return
        if _dsl_function(value):
            self.functions[(ctx.func.func, node.id)] = _definition(value)
            return
        if id(value) in primitive_types.type_ids or isinstance(value, MatrixType):
            return
        if value is Matrix or value is Vector:
            return  # Built-in DSL constructors, also available through ti.Matrix/Vector.
        raise TaichiCompilationError(
            f"Component entry {self.kernel.func.__qualname__}: undeclared global/closure capture {node.id!r}; "
            "use a declared component value or a literal"
        )

    def check_call(self, value):
        if getattr(value, "_is_taichi_function", False):
            return
        if any(value is item for item in (range, len, min, max, abs, int, float, bool, enumerate, zip)):
            return
        if getattr(value, "__module__", "").startswith("taichi_forge"):
            return
        raise TaichiCompilationError("Uncontrolled Python calls are not supported in component captures")

    def finish(self):
        mismatch = self.dependency_mismatch(self.component, self.owner)
        if mismatch:
            raise TaichiCompilationError(f"Component dependency changed during compilation: {mismatch}; retry")
        self.contract = {
            "version": 1,
            "entry": self.entry_definition,
            "explicit_abi": [str(arg.annotation) for arg in self.kernel.arguments[1:]],
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

    def matches(self, component, owner):
        return self.dependency_mismatch(component, owner) is None

    def dependency_mismatch(self, component, owner):
        if _definition(self.kernel.func) != self.entry_definition:
            return "entry definition"
        for name, schema in self.used.items():
            declaration = component.declarations.get(name)
            if declaration is None or declaration.schema != schema:
                return f"declaration {name!r}"
            if isinstance(declaration, Dimension):
                _dimension_value(component, owner, name)
        roots = set()
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
            if _static_value(_member(owner, name)) != value:
                return f"static value {name!r}"
        for name, definition in self.methods.items():
            value = inspect.getattr_static(owner, name, None)
            if not getattr(value, "_is_taichi_function", False) or _definition(value) != definition:
                return f"method {name!r}"
        for (function, name), definition in self.functions.items():
            variables = inspect.getclosurevars(function)
            value = {**variables.globals, **variables.nonlocals}.get(name)
            if not _dsl_function(value) or _definition(value) != definition:
                return f"function {name!r} in {function.__qualname__}"
        return None


class _ComponentCode:
    def __init__(self, kernel, key, context):
        self.kernel = kernel
        self.key = key
        self.runtime = kernel.runtime
        self.semantic_key = _digest(
            {key: value for key, value in context.contract.items() if key != "declaration_sources"}
        )
        # Copy plain dependency facts, never the lowering context/owner/Exprs.
        self.entry_definition = context.entry_definition
        self.used = dict(context.used)
        self.static = dict(context.static)
        self.methods = dict(context.methods)
        self.functions = dict(context.functions)
        self.groups = dict(context.groups)
        self.member_paths = dict(context.member_paths)
        self.dynamic = tuple(context.dynamic)
        self._contract = json.dumps(context.contract)
        self.arguments = tuple(kernel.arguments[1:]) + tuple(
            kernel_arguments.KernelArgument(context.component.declarations[name].annotation, "__capture_" + name)
            for name in self.dynamic
        )

    matches = _Capture.matches
    dependency_mismatch = _Capture.dependency_mismatch

    @property
    def contract(self):
        return json.loads(self._contract)

    def validate_bindings(self, component, owner):
        for name in self.static:
            component.declarations[name].validate(_member(owner, name), name, self.kernel.func.__qualname__)
        for name in self.dynamic:
            value = _member(owner, name)
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
            if module._aot_builder._used_snode_tree_ids([name]):
                raise TaichiCompilationError("Generic component entry retains an undeclared fixed Field dependency")
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
            "initialization": {"group_storage": "zero", "model_data": "consumer"},
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
        if len(args) != len(self.kernel.arguments) - 1:
            raise TypeError("Wrong number of explicit component arguments")
        from taichi_forge.lang._storage_view import ndarray_view

        values = tuple(
            (
                ndarray_view(_member(owner, name))
                if isinstance(component.declarations[name], DenseField)
                else _member(owner, name)
            )
            for name in self.dynamic
        )
        return self.kernel.launch_kernel(
            self.kernel.compiled_kernels[self.key], *args, *values, _arguments=self.arguments
        )
