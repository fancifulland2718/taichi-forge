"""Consumer of the experimental definition-driven package projection.

No frontend import, source compilation, numeric AST or kernel interpreter.
Bindings use the ordinary Runtime Kernel API and its per-call resource leases.
"""

from contextvars import ContextVar
from dataclasses import dataclass
import functools
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import json
import operator
from pathlib import Path
import sys
from types import SimpleNamespace

from ._runtime import NdArray, Scalar
from ._package_values import StructField, capture_value, path_value, dimension_value, check_admission
from .._dimensions import Extent, evaluate_extent
from .._static_values import read_static_key, static_key

_construction = ContextVar("forge_definition_owner", default=None)
_ids = itertools.count()
_DTYPES = ("f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64")


@dataclass(frozen=True)
class _Backend:
    name: str


_BACKENDS = {name: _Backend(name) for name in ("x64", "arm64", "cuda", "vulkan")}


@dataclass(frozen=True)
class _NdarrayAnnotation:
    """Logical Host annotation; allocation and call validation stay in Runtime."""

    dtype: object
    ndim: int
    needs_grad: object
    boundary: str


class _ResourceType(type):
    def __new__(metaclass, name, bases, namespace):
        if any(isinstance(base, _ResourceType) for base in bases):
            raise TypeError("Host resource subclasses require further projection support")
        return super().__new__(metaclass, name, bases, namespace)

    def __instancecheck__(cls, value):
        # Dense Field member views use the ndarray ABI, but remain Fields in
        # ordinary Host dispatch. ABI compatibility is not nominal subtyping.
        array = isinstance(value, NdArray)
        field = array and hasattr(value, "_group_owner")
        if cls.kind == "ndarray":
            return array and not field
        if cls.kind == "StructField":
            return isinstance(value, StructField)
        if cls.kind == "Field":
            return field or isinstance(value, StructField)
        return field and bool(value.element_shape) == (cls.kind == "MatrixField")

    def __subclasscheck__(cls, value):
        if value is cls:
            return True
        raise TypeError("Host resource projection supports instance checks, not implementation subclass inspection")

    def __call__(cls, *args, **kwargs):
        raise TypeError("Host resource type is not an allocation API; use the retained resource constructor")


_RESOURCE_TYPES = {
    name: _ResourceType(name, (), {"kind": name})
    for name in ("ndarray", "Field", "ScalarField", "MatrixField", "StructField")
}


@dataclass(frozen=True)
class _Layout:
    name: str


_LAYOUTS = {name: _Layout(name) for name in ("AOS", "SOA")}


class _HostAPI(SimpleNamespace):
    def __getattr__(self, name):
        # An unavailable frontend capability is not proof that the producer
        # lacked it. In particular, do not select an old JIT allocation fallback
        # through getattr(..., None) after removing the numerical frontend.
        raise RuntimeError(f"Host frontend capability ti.{name} has no Runtime projection")


@dataclass(frozen=True)
class _ProducerType:
    name: str

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.name} is a producer-only API; build from numerical source before installing the package. "
            "Use retained Graph recordings with loaded entries for runtime scheduling."
        )

    def from_owner(self, *args, **kwargs):
        return self(*args, **kwargs)


class _ElementType:
    def __init__(self, package, spec):
        self.package, self.spec = package, spec

    def __call__(self, *args, **kwargs):
        from ._package_values import HostTensor, typed_tensor

        if self.spec["kind"] == "struct":
            result = {}
            for index, (name, member) in enumerate(self.spec["members"].items()):
                value = args[index] if index < len(args) else kwargs.get(name, 0)
                if member["kind"] != "scalar" and not isinstance(value, (dict, list, tuple, HostTensor)):
                    value = _ElementType(self.package, member)(value)
                result[name] = value
            return result
        shape = self.spec["element_shape"]
        if kwargs:
            raise TypeError("Host tensor values require positional components")
        return typed_tensor(shape, self.spec["dtype"], args)


class _ModuleBinding:
    def __init__(self, package):
        from ._package_graph import Arg, ArgKind, GraphBuilder

        self.package = package
        self.target = _BACKENDS[package.runtime.arch]
        self.importlib = _Importlib(self)
        self.host = _HostAPI(
            ndarray=self.ndarray,
            sync=package.runtime.wait,
            graph=SimpleNamespace(Arg=Arg, ArgKind=ArgKind, GraphBuilder=lambda: GraphBuilder(package)),
            **{dtype: dtype for dtype in _DTYPES},
        )

    def ndarray(self, dtype, shape):
        self.package._check()
        element_shape = ()
        if isinstance(dtype, _ElementType):
            if dtype.package is not self.package:
                raise ValueError("Element type belongs to another loaded package")
            if dtype.spec["kind"] == "struct":
                dtype = self.package._module.ndarray_layout(dtype.spec["layout"])
            else:
                element_shape, dtype = dtype.spec["element_shape"], dtype.spec["dtype"]
        value = self.package.runtime.ndarray(dtype, shape, element_shape=element_shape)
        self.package._arrays.append(value)
        construction = _construction.get()
        if construction is not None and construction[0] is self.package:
            construction[1].append(value)
        return value

    def arch(self, name):
        return _BACKENDS[name]

    def tensor(self, values, dt=None):
        from ._package_values import HostTensor

        return HostTensor(values, dt)

    def producer_type(self, name):
        return _ProducerType(name)

    def resource_type(self, name):
        if name not in _RESOURCE_TYPES:
            raise ValueError(f"Unsupported Host resource type: {name}")
        return _RESOURCE_TYPES[name]

    def layout(self, name):
        return _LAYOUTS[name]

    def producer_allocation(self, operation, location):
        raise RuntimeError(
            f"ti.{operation} at {location} requires producer resource declaration lowering; "
            "this ordinary Host path has no exported allocation contract. "
            "This operation did not allocate device resources; private JIT is unavailable."
        )

    @property
    def program(self):
        self.package._check()
        return self.package.runtime

    def vector_type(self, n, dtype):
        return self.matrix_type(n, None, dtype)

    def matrix_type(self, n, m, dtype):
        dimensions = [operator.index(n)] + ([] if m is None else [operator.index(m)])
        if any(n <= 0 for n in dimensions) or dtype not in _DTYPES:
            raise ValueError("Host tensor type requires positive dimensions and a supported scalar dtype")
        return self.type(dict(kind="tensor", dtype=dtype, element_shape=dimensions))

    def check_target_branch(self, condition, expected, location):
        self.package._check()
        if bool(condition) != expected:
            raise ValueError(f"Host backend condition changed after export at {location}; layout selection rejected")

    def entry(self, symbol):
        return entry(self.package, symbol)

    def import_module(self, name):
        self.package._check()
        modules = self.package.manifest.get("application_modules")
        if modules is not None:
            if name not in modules:
                raise ImportError("Module is outside this definition package: " + name)
            return importlib.import_module(self.package._name + "." + name)
        root = self.package.manifest["module_root"]
        if name != root and not name.startswith(root + "."):
            raise ImportError("Module is outside this definition package: " + name)
        return importlib.import_module(self.package._name + name[len(root) :])

    def import_from(self, origin, name):
        module = self.import_module(origin)
        try:
            return getattr(module, name)
        except AttributeError:
            if not hasattr(module, "__path__"):
                raise ImportError(f"Cannot import {name} from {origin}") from None
        return self.import_module(origin + "." + name)

    def dynamic_import(self, name, package=None):
        self.package._check()
        resolved = importlib.util.resolve_name(name, package) if name.startswith(".") else name
        if resolved == self.package._name or resolved.startswith(self.package._name + "."):
            return importlib.import_module(resolved)
        modules = self.package.manifest.get("application_modules")
        if modules is not None:
            if resolved in modules:
                return self.import_module(resolved)
            if resolved.split(".")[0] in self.package.manifest["dependencies"]["local_roots"]:
                raise ImportError("Dynamic local dependency was not exported: " + resolved)
        else:
            root = self.package.manifest["module_root"]
            if resolved == root or resolved.startswith(root + "."):
                return self.import_module(resolved)
        if resolved.split(".")[0] in ("taichi", "taichi_forge"):
            raise ImportError("Definition packages cannot dynamically import the numerical frontend")
        return importlib.import_module(resolved)

    def field(self, owner_key, name, dimensions, *, placement=None):
        self.package._check()
        construction = _construction.get()
        if construction is None or construction[0] is not self.package:
            raise ValueError("Field allocation requires its managed constructor")
        _, allocated, owner = construction
        member = self.package.manifest["owners"][owner_key][name]
        group_key = member.get("field_group", owner_key)
        contract = self.package.manifest["field_groups"][group_key]
        if dimensions.keys() != set(contract["dimensions"]):
            raise ValueError("Field group dimensions differ from its native schema")
        declared_members = contract["members"]
        from ._package_layout import layout_selection, select_layout

        selection = layout_selection(contract, owner)
        selected = owner._forge_package_layouts.get(group_key)
        if selected is None:
            selected = (selection, select_layout(contract, selection))
            owner._forge_package_layouts[group_key] = selected
        elif selected[0] != selection:
            raise ValueError("Conditional layout configuration changed during construction")
        contract = selected[1]
        member_ids = member.get("member_ids", {name: name})
        if any(identity not in contract["members"] for identity in member_ids.values()):
            raise ValueError("Field allocation contradicts its declared static layout condition")
        dimensions = {key: value for key, value in dimensions.items() if key in contract["dimensions"]}
        if dimensions.keys() != set(contract["dimensions"]):
            raise ValueError("Field group dimensions differ from its native schema")
        expected_shape = tuple(dimensions[n] if isinstance(n, str) else n for n in member["shape"])
        if placement is not None and tuple(placement) != expected_shape:
            raise ValueError("Declared Dense placement changed before its Field allocation")
        previous = owner._forge_package_field_groups.get(group_key)
        if previous is None:
            for _, bindings in owner._forge_package_field_groups.values():
                if any(bindings[key] != dimensions[key] for key in bindings.keys() & dimensions.keys()):
                    raise ValueError("Shared dimension changed between resource groups")
            group = self.package.runtime.dense_snode_group(contract["schema"], dimensions)
            self.package._arrays.append(group)
            allocated.append(group)
            owner._forge_package_field_groups[group_key] = (group, dict(dimensions))
            for key, value in dimensions.items():
                setattr(owner, key, value)
        else:
            group, original = previous
            if original != dimensions:
                raise ValueError("Field group dimensions changed during construction")
        values = {}
        for attribute, identity in member_ids.items():
            element = member["element"]["members"][attribute] if member.get("struct_field") else member["element"]
            value = group.member(
                contract["members"][identity], element["dtype"], element_shape=element["element_shape"]
            )
            self.package._arrays.append(value)
            allocated.append(value)
            values[attribute] = value
        for attribute in declared_members:
            owner._forge_package_member_layouts[attribute] = selection
        return StructField(values) if member.get("struct_field") else values[name]

    def owner_reference(self, owner_key, scope, name, value):
        self.package._check()
        construction = _construction.get()
        if (
            not isinstance(scope, _ResourceScope)
            or scope._parent is not None
            or construction is None
            or construction[0] is not self.package
            or construction[2] is not scope._root
        ):
            raise ValueError("Owner input requires its managed constructor scope")
        declaration = self.package.manifest["owners"][owner_key].get(name, {})
        if declaration.get("kind") != "owner_input":
            raise ValueError("Owner input differs from its declared reference")
        from ._package_values import check_owner_bindings

        setattr(scope._root, name, value)
        check_owner_bindings(scope._root, (dict(path=[name], owner_type=declaration["owner_type"]),))

    def borrow(self, owner_key, scope, name, field, *, dimensions=None, source_scope=False, external=False):
        self.package._check()
        construction = _construction.get()
        if (
            not isinstance(scope, _ResourceScope)
            or scope._parent is not None
            or construction is None
            or construction[0] is not self.package
            or construction[2] is not scope._root
            or source_scope is not True
            or external is not False
        ):
            raise ValueError("Borrowed Field requires its managed source-scope construction")
        contract = self.package.manifest["owners"][owner_key].get(name, {})
        if contract.get("kind") != "borrowed_field" or path_value(scope._root, contract["source_path"]) is not field:
            raise ValueError("Borrowed Field differs from its declared allocation")
        from .._dimensions import freeze_extent, check_constraints

        dimensions = {} if dimensions is None else dimensions
        if not isinstance(dimensions, dict) or dimensions.keys() != contract["dimensions"].keys():
            raise ValueError("Borrowed dimensions differ from their source declaration")
        for key, value in dimensions.items():
            if not isinstance(value, Extent) or value.expression != freeze_extent(contract["dimensions"][key]):
                raise ValueError("Borrowed target relation differs from its declared expression")
        check_constraints(
            {key: value["spec"] for key, value in contract.get("constraints", {}).items()},
            lambda key: (
                path_value(scope._root, contract["dimension_paths"][key])
                if key in contract.get("dimension_paths", {})
                else dimension_value(scope._root, key)
            ),
        )
        setattr(scope._owner, name, field)
        return field

    def placement(self, owner, shape):
        shape = shape if isinstance(shape, tuple) else (shape,)
        dimensions = tuple(
            (
                evaluate_extent(n.expression, lambda key: dimension_value(owner, key))
                if isinstance(n, Extent)
                else operator.index(n)
            )
            for n in shape
        )
        if any(not 0 < n <= 2**31 - 1 for n in dimensions):
            raise ValueError("Declared Dense placement extents must be positive i32 values")
        return dimensions

    def group_builder(self, owner, owner_key, group):
        self.package._check()
        if getattr(owner, "_forge_package_owner", None) is not self.package:
            raise ValueError("Resource builder belongs to another loaded package")
        return _GroupBuilder(owner, owner_key + ":" + group)

    def context_builder(self, owner, key):
        self.package._check()
        construction = _construction.get()
        if (
            getattr(owner, "_forge_package_owner", None) is not self.package
            or key not in self.package.manifest["field_groups"]
            or construction is None
            or construction[0] is not self.package
            or construction[2] is not owner
        ):
            raise ValueError("Context builder requires its declared owner and resource group")
        return _GroupBuilder(owner, key, allow_empty=True)

    def context_function(self, module, name):
        self.package._check()
        spec = self.package.manifest.get("host_contexts", {}).get(module)
        if spec is None or name not in spec["definitions"]:
            raise ValueError("Unknown retained Host context definition")
        module = (
            self.import_module(spec["module"])
            if spec.get("application_module")
            else importlib.import_module(self.package._name + "." + spec["module"])
        )
        return getattr(module, name)

    def resource_context(self, module, name, *args, **kwargs):
        return self.context_function(module, name)(*args, **kwargs)

    def type(self, spec):
        if spec["kind"] == "ndarray_annotation":
            element = spec["element"]
            dtype = element["dtype"] if element["kind"] == "scalar" else self.type(element)
            return _NdarrayAnnotation(dtype, spec["ndim"], spec["needs_grad"], spec["boundary"])
        return _ElementType(self.package, spec)

    def extent(self, expression, owner):
        return evaluate_extent(expression, lambda key: dimension_value(owner, key))

    def resource_scope(self, owner):
        self.package._check()
        if getattr(owner, "_forge_package_owner", None) is not self.package:
            raise ValueError("Resource scope belongs to another loaded package")
        return _ResourceScope(owner)

    def declare_value(self, owner, name, spec, path=None):
        self.package._check()
        if getattr(owner, "_forge_package_owner", None) is not self.package:
            raise ValueError("Value declaration belongs to another loaded package")
        # Match ResourceScope.value's presence/type-role check. Packing and
        # per-call ABI validation remain in the native Kernel API.
        path_value(owner, path or (name,))
        if spec["kind"] not in ("scalar", "tensor"):
            raise ValueError("Runtime value requires a scalar/vector/matrix type")

    def managed_class(self, cls):
        return managed_class(self.package, cls)

    def unbound_entry(self, definition):
        return _UnboundEntry(definition)


class _GroupBuilder:
    def __init__(self, owner, key, *, allow_empty=False):
        self.owner, self.key, self.finalized = owner, key, False
        self.allow_empty = allow_empty

    def finalize(self):
        if self.finalized:
            raise RuntimeError("FieldsBuilder already finalized")
        allocation = self.owner._forge_package_field_groups.get(self.key)
        if allocation is None and not self.allow_empty:
            raise ValueError("Declared group finalized without its resource allocation")
        self.finalized = True
        return _FieldTree(allocation[0] if allocation is not None else None)


@dataclass(frozen=True)
class _FieldTree:
    group: object

    def destroy(self):
        # The existing native family retirement revokes all member views.
        if self.group is not None:
            self.group.close()


class _ResourceScope:
    """Forward original Host attributes and reuse the shared dimension algebra."""

    def __init__(self, owner, *, parent=None, path=()):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_root", parent._root if parent is not None else owner)
        if parent is None and not hasattr(owner, "_forge_package_dimension_paths"):
            owner._forge_package_dimension_paths = {}

    def _identity(self, name):
        path = (*self._path, name)
        identity = name if len(path) == 1 else "__record_" + "_".join(f"{len(part)}_{part}" for part in path)
        self._root._forge_package_dimension_paths[identity] = path
        return identity

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def __setattr__(self, name, value):
        if self._parent is not None:
            raise TypeError("A resource record describes existing members and is read-only")
        setattr(self._owner, name, value)

    def static(self, name, *, choices, kernel_only=False):
        if type(name) is not str or not name.isidentifier() or type(kernel_only) is not bool:
            raise ValueError("Invalid static resource declaration")
        keys = tuple(static_key(value) for value in choices)
        if not keys or len(set(keys)) != len(keys) or static_key(getattr(self._owner, name)) not in keys:
            raise ValueError(f"Static configuration {name!r} is outside its declared domain")

    def dim(self, name):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("A dimension requires an explicit owner attribute name")
        value = operator.index(getattr(self._owner, name))
        if not 0 <= value <= 2**31 - 1:
            raise ValueError(f"Dimension {name!r} is outside the i32 count range")
        return Extent(("dim", self._identity(name)))

    def parent_dim(self, name):
        if self._parent is None:
            raise ValueError("parent_dim requires a declared record scope")
        return self._parent.dim(name)

    def derived_dim(self, name, expression):
        if not isinstance(expression, Extent):
            raise TypeError("A derived dimension requires an Extent expression")
        result = self.dim(name)
        value = evaluate_extent(expression.expression, lambda key: operator.index(dimension_value(self._root, key)))
        if value != operator.index(getattr(self._owner, name)):
            raise ValueError(f"Derived dimension {name!r} does not match its normal attribute")
        return result

    def power_of_two_dim(self, name, minimum):
        if not isinstance(minimum, Extent):
            raise TypeError("A power-of-two dimension requires an Extent minimum")
        return self.derived_dim(name, Extent(("ceil_pow2", minimum.expression)))

    def record(self, name, value, *, optional=False):
        if (
            type(name) is not str
            or not name.isidentifier()
            or type(optional) is not bool
            or (value is None and not optional)
        ):
            raise ValueError("Resource record requires a named existing value or an explicit optional absence")
        if self._parent is not None:
            if getattr(self._owner, name) is not value:
                raise ValueError("A record declaration must refer to its existing member")
        else:
            setattr(self._owner, name, value)
        return _ResourceScope(value, parent=self, path=(*self._path, name))

    def require_le(self, name, left, right, *, when):
        from .._dimensions import check_constraints, freeze_extent

        if type(name) is not str or not name.isidentifier() or type(when) is not str or not when.isidentifier():
            raise ValueError("Dimension constraint requires a name and declared trigger")
        spec = dict(
            left=left.expression if isinstance(left, Extent) else freeze_extent(("const", left)),
            right=right.expression if isinstance(right, Extent) else freeze_extent(("const", right)),
            when=self._identity(when),
        )
        check_constraints({self._identity(name): spec}, lambda key: dimension_value(self._root, key))


class _Importlib:
    """Local importlib binding; never install a process-global import hook."""

    def __init__(self, binding):
        self.import_module = binding.dynamic_import

    def __getattr__(self, name):
        return getattr(importlib, name)


def bind_module(name):
    # Each loaded package has its own Python namespace and permanent code owner.
    # Lazy submodule imports cannot select another package's currently active context.
    root = sys.modules[name.partition(".")[0]]
    return _ModuleBinding(root._forge_package_owner)


def managed_class(package, cls):
    """Bind a projected Host instance once; never switch module pointers."""
    original = cls.__init__
    relative = cls.__module__[len(package._name) :]
    module = (
        relative.lstrip(".")
        if "application_modules" in package.manifest
        else package.manifest["module_root"] + relative
    )
    key = module + "." + cls.__name__
    parameters = {
        name: item["presence_parameter"]
        for name, item in package.manifest["owners"].get(key, {}).items()
        if item.get("presence_owner") == key
    }
    signature = inspect.signature(original) if parameters else None

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        package._check()
        if parameters:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            for attribute, parameter in parameters.items():
                value = bound.arguments[parameter]
                if type(value) is not bool:
                    raise ValueError(f"Optional resource parameter {parameter!r} requires a boolean")
                setattr(self, attribute, value)
        construction = _construction.get()
        if construction is not None and construction[0] is package and construction[2] is self:
            # Cooperative base construction belongs to this same invocation.
            # Do not reset groups or lose allocations needed for failure rollback.
            return original(self, *args, **kwargs)
        self._forge_package_owner = package
        self._forge_package_field_groups = {}
        self._forge_package_layouts = {}
        self._forge_package_member_layouts = {}
        allocated = []
        token = _construction.set((package, allocated, self))
        try:
            original(self, *args, **kwargs)
            from ._package_values import check_owner_bindings

            check_owner_bindings(
                self,
                (
                    dict(path=item.get("path", [name]), owner_type=item["owner_type"])
                    for name, item in package.manifest["owners"].get(key, {}).items()
                    if item["kind"] in ("owner", "owner_input")
                ),
            )
        except BaseException:
            for value in reversed(allocated):
                value.close()
                package._arrays.remove(value)
            raise
        finally:
            _construction.reset(token)
        if construction is not None and construction[0] is package:
            # A successful child is part of the enclosing constructor transaction.
            # The parent must also release these resources if its construction fails.
            construction[1].extend(allocated)

    cls.__init__ = initialize
    cls._forge_package_type = key
    return cls


class _UnboundEntry:
    def __init__(self, definition):
        self.definition = definition

    def __get__(self, owner, cls=None):
        raise TypeError("Numerical mixin entry requires its exported concrete receiver: " + self.definition)

    def __call__(self, *args, **kwargs):
        raise TypeError("Numerical mixin entry requires its exported concrete receiver: " + self.definition)


class entry:
    def __init__(self, package, symbol):
        self.symbol = symbol
        self.package = package
        self.family = package.manifest.get("entry_families", {}).get(symbol)
        self.contract = self.family if self.family is not None else package.manifest["entries"][symbol]
        self._variants = []
        self._domain = {}
        self._selected = {}
        if self.family is not None:
            self._domain = {
                name: {static_key(read_static_key(key)) for key in values}
                for name, values in self.family["domain"].items()
            }
            for variant in self.family["variants"]:
                if variant["symbol"] not in package.manifest["entries"]:
                    raise ValueError("Static entry family references missing numerical code")
                selection = {name: static_key(read_static_key(key)) for name, key in variant["selection"].items()}
                if not selection.keys() <= self._domain.keys():
                    raise ValueError("Static entry selection differs from its declared domain")
                self._variants.append((selection, variant["symbol"]))
        self.signature = inspect.Signature(
            [
                inspect.Parameter(item["name"], inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for item in self.contract["arguments"]
            ]
        )

    def __get__(self, owner, cls=None):
        if owner is None:
            return self
        if owner._forge_package_owner is not self.package:
            raise ValueError("Numerical method belongs to a different loaded package")
        return _BoundEntry(self, owner)

    def __call__(self, *args, **kwargs):
        return self._call(self.package, None, *args, **kwargs)

    def _call(self, package, owner, *args, **kwargs):
        package._check()
        selected = self._select(owner)
        if selected is not self:
            return selected._call(package, owner, *args, **kwargs)
        contract = self.contract
        check_admission(owner, contract)
        self._check_layout(owner)
        bound = self.signature.bind(*args, **kwargs)
        arguments = [
            (
                Scalar(item["dtype"], bound.arguments[item["name"]])
                if item["kind"] == "value"
                else bound.arguments[item["name"]]
            )
            for item in contract["arguments"]
        ]
        for item in contract["captures"]:
            value = capture_value(owner, item)
            arguments.append(
                Scalar(item["dtype"], value)
                if item["kind"] in ("value", "dimension") and not item.get("element_shape")
                else value
            )
        kernel = package._kernels.get(self.symbol)
        if kernel is None:
            kernel = package._kernels[self.symbol] = package._module.kernel(self.symbol)
        from ._package_buffers import HostBuffers

        with HostBuffers(package.runtime, contract["arguments"], arguments) as prepared:
            return kernel.run(*prepared)

    def _check_layout(self, owner):
        if owner is None:
            return
        for item in self.contract["captures"]:
            allocated_owner = path_value(owner, item.get("layout_owner_path", ()))
            for name, value in allocated_owner._forge_package_member_layouts.get(
                item.get("layout_member", item["name"]), {}
            ).items():
                if getattr(allocated_owner, name) is not value:
                    raise ValueError("Conditional layout configuration changed; construct a new instance")

    def _select(self, owner):
        if self.family is None:
            return self
        if owner is None:
            raise ValueError("Static package entries require their bound instance")
        values = {
            name: static_key(path_value(owner, self.family.get("paths", {}).get(name, (name,))))
            for name in self._domain
        }
        if any(value not in self._domain[name] for name, value in values.items()):
            raise ValueError("Static configuration is outside its declared package domain")
        matches = {
            symbol
            for selection, symbol in self._variants
            if all(values[name] == value for name, value in selection.items())
        }
        if len(matches) != 1:
            raise ValueError("Static configuration has missing or ambiguous package code")
        symbol = matches.pop()
        if symbol not in self._selected:
            self._selected[symbol] = entry(self.package, symbol)
        return self._selected[symbol]


@dataclass(frozen=True)
class _BoundEntry:
    entry: entry
    owner: object

    def __call__(self, *args, **kwargs):
        return self.entry._call(self.entry.package, self.owner, *args, **kwargs)


class LoadedPackage:
    """Own code and allocations while preserving independent Host instances.

    The prototype's allocation lifetime is the package context. It does not
    claim general component composition, independent object retirement or Plan
    capture. Existing Host calls keep their original scheduling semantics.
    """

    def __init__(self, runtime, directory):
        self.runtime, self.directory = runtime, Path(directory).resolve(strict=True)
        self.manifest = json.loads((self.directory / "package.json").read_text(encoding="utf8"))
        if (
            self.manifest.get("version")
            not in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
            or self.manifest.get("kind") != "forge_definition_package_prototype"
        ):
            raise ValueError("Unsupported definition package")
        boundary = {"application_modules", "entry_module", "dependencies"}
        if (self.manifest["version"] == 2 or boundary & self.manifest.keys()) and not boundary <= self.manifest.keys():
            raise ValueError("Application source package is missing its import boundary")
        if self.manifest["target"] != runtime.arch:
            raise ValueError("Package target differs from Runtime")
        files = self.manifest["files"]
        if not {"code.tcm", "host/__init__.py"} <= files.keys():
            raise ValueError("Package is missing code or Host entry")
        for relative, digest in files.items():
            path = (self.directory / relative).resolve(strict=True)
            if self.directory not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Package payload changed: " + relative)
        actual = {p.relative_to(self.directory).as_posix() for p in (self.directory / "host").rglob("*.py")}
        if actual != {p for p in files if p.startswith("host/") and p.endswith(".py")}:
            raise ValueError("Host source inventory differs from package manifest")
        self._module = runtime.load_module(self.directory / "code.tcm")
        self._arrays, self._kernels = [], {}
        self._graph_ids = itertools.count()
        self._closed, self._entered = False, False
        self._name = "_forge_package_loaded_" + str(next(_ids))
        self.host = None

    def _check(self):
        if self._closed:
            raise RuntimeError("Loaded package is closed")

    def module(self, name):
        """Resolve a retained module inside this package's declared source boundary."""
        self._check()
        if not self._entered:
            raise RuntimeError("Enter the package context before resolving Host modules")
        return _ModuleBinding(self).import_module(name)

    def graph_builder(self):
        """Record retained Host methods against this package's loaded entries."""
        from ._package_graph import GraphBuilder

        self._check()
        return GraphBuilder(self)

    def __enter__(self):
        self._check()
        if self._entered:
            raise RuntimeError("Package context is already active")
        self._entered = True
        try:
            if self.host is None:
                path = self.directory / "host/__init__.py"
                spec = importlib.util.spec_from_file_location(
                    self._name, path, submodule_search_locations=[str(path.parent)]
                )
                self.host = importlib.util.module_from_spec(spec)
                self.host._forge_package_owner = self
                sys.modules[self._name] = self.host
                spec.loader.exec_module(self.host)
                if self.manifest.get("entry_module"):
                    self.host = importlib.import_module(self._name + "." + self.manifest["entry_module"])
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def close(self):
        if self._closed:
            return
        self._module.close()
        for array in reversed(self._arrays):
            array.close()
        self._arrays.clear()
        self._closed = True
        for name in tuple(sys.modules):
            if name == self._name or name.startswith(self._name + "."):
                del sys.modules[name]

    def __exit__(self, *_):
        try:
            self.close()
        finally:
            self._entered = False


def load_package(runtime, directory):
    return LoadedPackage(runtime, directory)
