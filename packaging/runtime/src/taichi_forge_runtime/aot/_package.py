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

from ._runtime import Scalar
from .._dimensions import Extent, evaluate_extent

_construction = ContextVar("forge_definition_owner", default=None)
_ids = itertools.count()
_DTYPES = ("f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64")


@dataclass(frozen=True)
class _Backend:
    name: str


_BACKENDS = {name: _Backend(name) for name in ("x64", "arm64", "cuda", "vulkan")}


class _ElementType:
    def __init__(self, package, spec):
        self.package, self.spec = package, spec

    def __call__(self, *args, **kwargs):
        if self.spec["kind"] == "struct":
            result = {}
            for index, (name, member) in enumerate(self.spec["members"].items()):
                value = args[index] if index < len(args) else kwargs.get(name, 0)
                if member["kind"] != "scalar" and not isinstance(value, (dict, list, tuple)):
                    value = _ElementType(self.package, member)(value)
                result[name] = value
            return result
        shape = self.spec["element_shape"]
        if kwargs:
            raise TypeError("Host tensor values require positional components")
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return args[0]
        if len(args) == 1:

            def fill(dimensions):
                return [fill(dimensions[1:]) for _ in range(dimensions[0])] if dimensions else args[0]

            return fill(shape)
        return list(args)


class _ModuleBinding:
    def __init__(self, package):
        from ._package_graph import Arg, ArgKind, GraphBuilder

        self.package = package
        self.target = _BACKENDS[package.runtime.arch]
        self.importlib = _Importlib(self)
        self.host = SimpleNamespace(
            ndarray=self.ndarray,
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

    def field(self, owner_key, name, dimensions):
        self.package._check()
        construction = _construction.get()
        if construction is None or construction[0] is not self.package:
            raise ValueError("Field allocation requires its managed constructor")
        _, allocated, owner = construction
        contract = self.package.manifest["field_groups"][owner_key]
        if dimensions.keys() != set(contract["dimensions"]):
            raise ValueError("Field group dimensions differ from its native schema")
        previous = owner._forge_package_field_groups.get(owner_key)
        if previous is None:
            group = self.package.runtime.dense_snode_group(contract["schema"], dimensions)
            self.package._arrays.append(group)
            allocated.append(group)
            owner._forge_package_field_groups[owner_key] = (group, dict(dimensions))
            for key, value in dimensions.items():
                setattr(owner, key, value)
        else:
            group, original = previous
            if original != dimensions:
                raise ValueError("Field group dimensions changed during construction")
        element = self.package.manifest["owners"][owner_key][name]["element"]
        value = group.member(contract["members"][name], element["dtype"], element_shape=element["element_shape"])
        self.package._arrays.append(value)
        allocated.append(value)
        return value

    def type(self, spec):
        return _ElementType(self.package, spec)

    def extent(self, expression, owner):
        return evaluate_extent(expression, lambda key: getattr(owner, key))

    def resource_scope(self, owner):
        self.package._check()
        if getattr(owner, "_forge_package_owner", None) is not self.package:
            raise ValueError("Resource scope belongs to another loaded package")
        return _ResourceScope(owner)

    def declare_value(self, owner, name, spec):
        self.package._check()
        if getattr(owner, "_forge_package_owner", None) is not self.package:
            raise ValueError("Value declaration belongs to another loaded package")
        # Match ResourceScope.value's presence/type-role check. Packing and
        # per-call ABI validation remain in the native Kernel API.
        getattr(owner, name)
        if spec["kind"] not in ("scalar", "tensor"):
            raise ValueError("Runtime value requires a scalar/vector/matrix type")

    def managed_class(self, cls):
        return managed_class(self.package, cls)

    def unbound_entry(self, definition):
        return _UnboundEntry(definition)


class _ResourceScope:
    """Forward original Host attributes and reuse the shared dimension algebra."""

    def __init__(self, owner):
        object.__setattr__(self, "_owner", owner)

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def __setattr__(self, name, value):
        setattr(self._owner, name, value)

    def dim(self, name):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("A dimension requires an explicit owner attribute name")
        value = operator.index(getattr(self._owner, name))
        if not 0 <= value <= 2**31 - 1:
            raise ValueError(f"Dimension {name!r} is outside the i32 count range")
        return Extent(("dim", name))


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

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        package._check()
        construction = _construction.get()
        if construction is not None and construction[0] is package and construction[2] is self:
            # Cooperative base construction belongs to this same invocation.
            # Do not reset groups or lose allocations needed for failure rollback.
            return original(self, *args, **kwargs)
        self._forge_package_owner = package
        self._forge_package_field_groups = {}
        allocated = []
        token = _construction.set((package, allocated, self))
        try:
            original(self, *args, **kwargs)
        except BaseException:
            for value in reversed(allocated):
                value.close()
                package._arrays.remove(value)
            raise
        finally:
            _construction.reset(token)

    cls.__init__ = initialize
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
        self.contract = package.manifest["entries"][symbol]
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
        return functools.partial(self._call, owner._forge_package_owner, owner)

    def __call__(self, *args, **kwargs):
        return self._call(self.package, None, *args, **kwargs)

    def _call(self, package, owner, *args, **kwargs):
        package._check()
        contract = self.contract
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
            value = getattr(owner, item["name"])
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
            self.manifest.get("version") not in (1, 2, 3, 4, 5, 6)
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
        self._closed, self._entered = False, False
        self._name = "_forge_package_loaded_" + str(next(_ids))
        self.host = None

    def _check(self):
        if self._closed:
            raise RuntimeError("Loaded package is closed")

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
