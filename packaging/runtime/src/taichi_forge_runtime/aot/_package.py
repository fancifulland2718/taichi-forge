"""Consumer of the experimental definition-driven package projection.

No frontend import, source compilation, numeric AST or kernel interpreter.
Bindings use the ordinary Runtime Kernel API and its per-call resource leases.
"""

from contextvars import ContextVar
import functools
import hashlib
import importlib.util
import inspect
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from ._runtime import Scalar

_construction = ContextVar("forge_definition_owner", default=None)
_ids = itertools.count()
_DTYPES = ("f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64")


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

    def entry(self, symbol):
        return entry(self.package, symbol)

    def type(self, spec):
        return _ElementType(self.package, spec)

    def managed_class(self, cls):
        return managed_class(self.package, cls)


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
        self._forge_package_owner = package
        allocated = []
        token = _construction.set((package, allocated))
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
            arguments.append(Scalar(item["dtype"], value) if item["kind"] == "value" else value)
        kernel = package._kernels.get(self.symbol)
        if kernel is None:
            kernel = package._kernels[self.symbol] = package._module.kernel(self.symbol)
        return kernel.run(*arguments)


class LoadedPackage:
    """Own code and allocations while preserving independent Host instances.

    The prototype's allocation lifetime is the package context. It does not
    claim general component composition, independent object retirement or Plan
    capture. Existing Host calls keep their original scheduling semantics.
    """

    def __init__(self, runtime, directory):
        self.runtime, self.directory = runtime, Path(directory).resolve(strict=True)
        self.manifest = json.loads((self.directory / "package.json").read_text(encoding="utf8"))
        if self.manifest.get("version") != 1 or self.manifest.get("kind") != "forge_definition_package_prototype":
            raise ValueError("Unsupported definition package")
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
