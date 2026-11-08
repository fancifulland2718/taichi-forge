"""Select source JIT or discovered definition packages at application startup.

Discovery reads manifests and installed distribution RECORDs only. It does not
import providers, replace existing modules, compile source, or select sample models.
"""

from contextlib import ExitStack
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import threading
import sys
from types import ModuleType

_import_lock = threading.RLock()
_import_session = None


class _PackageImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, session):
        self.session = session
        self.modules = {}

    def find_spec(self, fullname, path=None, target=None):
        if self.session.mode == "hybrid":
            if fullname not in self.session._module_names:
                return None
        elif fullname.split(".")[0] not in self.session._roots:
            return None
        spec = importlib.util.spec_from_loader(fullname, self, is_package=self.session._is_package(fullname))
        if self.session.mode == "hybrid" and spec.submodule_search_locations is not None:
            source = importlib.machinery.PathFinder.find_spec(fullname, path)
            if source is not None and source.submodule_search_locations is not None:
                spec.submodule_search_locations = list(source.submodule_search_locations)
        return spec

    def create_module(self, spec):
        return ModuleType(spec.name)

    def exec_module(self, module):
        name = module.__name__
        projected = self.session._import_projection(name)
        if projected is not None:
            # Functions/classes retain their private package globals and owner.
            # Never mutate the loaded projection's __name__/spec/import boundary.
            aliases = {}
            for key, value in vars(projected).items():
                if key.startswith("__") and key.endswith("__"):
                    continue
                public_name = self.session._public_module_name(value)
                if public_name is None:
                    module.__dict__[key] = value
                else:
                    aliases[key] = public_name
            module.__doc__ = projected.__doc__
            module.__file__ = projected.__file__
            if hasattr(projected, "__all__"):
                module.__all__ = projected.__all__
            original_getattr = vars(projected).get("__getattr__")
            original_dir = vars(projected).get("__dir__")

            def get_attribute(key):
                if key in aliases:
                    value = importlib.import_module(aliases[key])
                elif original_getattr is not None:
                    value = original_getattr(key)
                    public_name = self.session._public_module_name(value)
                    if public_name is None:
                        return value
                    value = importlib.import_module(public_name)
                else:
                    raise AttributeError(f"module {name!r} has no attribute {key!r}")
                setattr(module, key, value)
                return value

            # A retained parent may already hold private child-module objects.
            # Resolve them lazily through normal import to preserve public
            # identity for both `from package import child` and direct imports.
            module.__getattr__ = get_attribute
            module.__dir__ = lambda: sorted(
                set(vars(module)) | aliases.keys() | (set(original_dir()) if original_dir is not None else set())
            )
        self.modules[name] = module

    def close(self):
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        for name, module in reversed(tuple(self.modules.items())):
            if sys.modules.get(name) is module:
                del sys.modules[name]
            parent, _, child = name.rpartition(".")
            namespace = sys.modules.get(parent)
            if namespace is not None and vars(namespace).get(child) is module:
                delattr(namespace, child)
        self.modules.clear()


def _arch(name):
    if name == "cpu":
        return "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if name not in ("x64", "arm64", "cuda", "vulkan"):
        raise ValueError(f"Unsupported execution backend: {name!r}")
    return name


def _paths(value):
    if isinstance(value, (str, os.PathLike)):
        value = (value,)
    return tuple(Path(path).expanduser().resolve() for path in value)


class ExecutionSession:
    """Resolve the same module API through JIT or AOT, once per session.

    Use ordinary imports inside the session. ``mode`` defaults to
    FORGE_EXECUTION_MODE, then jit. AOT discovers packages
    below explicit ``package_paths`` / FORGE_AOT_PACKAGE_PATH and in installed
    distributions. Use separate sessions for separate modes; existing objects
    never change implementation. Close instances/calls before closing a session.
    A supplied AOT Runtime remains owned by the caller (including borrowed JIT
    runtimes used to prepare script-side inputs).
    ``hybrid`` shares the frontend Program and adapts qualified input storage;
    applications normally select it with ``ti.init(execution_mode='hybrid')``.
    """

    def __init__(
        self, *, mode=None, arch="cpu", package_paths=(), runtime=None, jit_options=None, runtime_options=None
    ):
        self.mode = mode if mode is not None else os.environ.get("FORGE_EXECUTION_MODE", "jit")
        if self.mode not in ("jit", "aot", "hybrid"):
            raise ValueError("execution mode must be 'jit', 'aot' or 'hybrid'")
        self.arch = _arch(arch)
        self.package_paths = _paths(package_paths) + _paths(
            path for path in os.environ.get("FORGE_AOT_PACKAGE_PATH", "").split(os.pathsep) if path
        )
        self.jit_options, self.runtime_options = dict(jit_options or {}), dict(runtime_options or {})
        if {"arch", "enable_fallback"} & self.jit_options.keys():
            raise ValueError("Select arch on ExecutionSession; backend fallback is disabled")
        if "arch" in self.runtime_options:
            raise ValueError("Select arch on ExecutionSession")
        if self.mode == "jit" and (runtime is not None or self.runtime_options):
            raise ValueError("runtime and runtime_options apply only to AOT")
        if self.mode == "aot" and self.jit_options:
            raise ValueError("jit_options apply only to JIT")
        if self.mode == "hybrid" and self.runtime_options:
            raise ValueError("Hybrid execution uses the initialized JIT Program")
        if runtime is not None and (runtime.arch != self.arch or self.runtime_options):
            raise ValueError("Supplied Runtime must match arch and cannot be reconfigured")
        self.runtime = runtime
        self._stack = ExitStack()
        self._lock = threading.RLock()
        self._modules, self._packages = {}, {}
        self._index = None
        self._roots = set()
        self._module_names = set()
        self._active = self._closed = False

    @property
    def loaded_packages(self):
        """Resolved package directories; querying does not initialize a device."""
        return tuple(self._packages)

    def __enter__(self):
        try:
            return self._enter()
        except BaseException:
            if not self._active:
                self.close()
            raise

    def _enter(self):
        global _import_session
        with self._lock:
            if self._closed or self._active:
                raise RuntimeError("Execution session is closed or already active")
            if self.mode in ("jit", "hybrid"):
                if _import_session is not None:
                    raise RuntimeError("Cannot select JIT imports inside an active AOT import session")
                ti = importlib.import_module("taichi_forge")
                if ti.lang.impl.get_runtime().prog is not None:
                    if str(ti.lang.impl.current_cfg().arch).split(".")[-1] != self.arch or self.jit_options:
                        raise ValueError("Existing JIT runtime differs or would require reconfiguration")
                else:
                    ti.init(
                        arch=getattr(ti, self.arch), enable_fallback=False, execution_mode="jit", **self.jit_options
                    )
                    self._stack.callback(ti.reset)
            if self.mode != "jit":
                with _import_lock:
                    if _import_session is not None:
                        raise RuntimeError("An AOT import session is already active")
                    self._discover()
                    if not self._index:
                        raise ModuleNotFoundError(
                            "No definition packages found; install the SDK or set FORGE_AOT_PACKAGE_PATH"
                        )
                    self._roots = {name.split(".")[0] for item in self._index for name in item[3]}
                    self._module_names = {name for item in self._index for name in item[3]}
                    self._module_names.update(
                        name.rsplit(".", count)[0]
                        for name in tuple(self._module_names)
                        for count in range(1, name.count(".") + 1)
                    )
                    if self.mode == "hybrid":
                        exported = {item[2] for item in self._index}
                        existing = sorted(
                            name
                            for name in sys.modules
                            if name in self._module_names and (name in exported or not self._is_package(name))
                        )
                    else:
                        existing = sorted(name for name in sys.modules if name.split(".")[0] in self._roots)
                    if existing:
                        raise RuntimeError(f"Select AOT before importing application packages: {existing}")
                    if self.mode == "hybrid":
                        from taichi_forge.aot._hybrid import PackageArrays

                        if self.runtime is None:
                            self.runtime = self._stack.enter_context(ti.aot.borrow_runtime())
                        if getattr(self.runtime, "_frontend_program", None) is not ti.lang.impl.get_runtime().prog:
                            raise ValueError("Hybrid packages must borrow the active JIT Program")
                        arrays = PackageArrays(self.runtime)
                        self.runtime._package_array_adapter = arrays
                        self._stack.callback(arrays.close)
                    self._finder = _PackageImports(self)
                    sys.meta_path.insert(0, self._finder)
                    _import_session = self
            self._active = True
            return self

    def _discover(self):
        candidates = set()
        for root in self.package_paths:
            if not root.exists():
                raise FileNotFoundError(f"AOT package search path does not exist: {root}")
            if root.is_file():
                if root.name != "package.json":
                    raise ValueError("Use a package directory, catalog directory, or package.json (unzip ZIPs first)")
                candidates.add(root)
            else:
                candidates.update(path.resolve() for path in root.rglob("package.json"))
        # A conventional payload directory on Python's import path also works
        # before installation, without a new registry or importable factory.
        for location in sys.path:
            location = Path(location or os.curdir)
            if (location / "package.json").is_file():
                candidates.add((location / "package.json").resolve())
            root = location / "forge_aot"
            if root.is_dir():
                candidates.update(path.resolve() for path in root.rglob("package.json"))
        # RECORD locations avoid walking site-packages or executing entry-point
        # factories. Wheels keep the existing package.json beside their payload.
        for distribution in metadata.distributions():
            for file in distribution.files or ():
                if file.name == "package.json":
                    candidates.add(Path(distribution.locate_file(file)).resolve())
        index = []
        for path in sorted(candidates):
            try:
                manifest = json.loads(path.read_text(encoding="utf8"))
            except (OSError, ValueError) as error:
                raise ValueError(f"Cannot read package manifest: {path}") from error
            if not isinstance(manifest, dict) or manifest.get("kind") != "forge_definition_package_prototype":
                continue
            root = manifest.get("entry_module", manifest.get("module_root"))
            modules = manifest.get("application_modules", ())
            if (
                not isinstance(root, str)
                or not isinstance(modules, list)
                or not all(isinstance(x, str) for x in modules)
            ):
                raise ValueError(f"Package has no valid module import boundary: {path}")
            module_names = frozenset(modules) | {root}
            packages = frozenset(
                name
                for name in module_names
                if "host/" + name.replace(".", "/") + "/__init__.py" in manifest.get("files", {})
            )
            index.append((path.parent, manifest.get("target"), root, module_names, packages))
        self._index = tuple(index)

    def _is_package(self, name):
        return any(name in item[4] or any(m.startswith(name + ".") for m in item[3]) for item in self._index)

    def _public_module_name(self, value):
        if isinstance(value, ModuleType):
            for package in self._packages.values():
                prefix = package._name + "."
                if value.__name__.startswith(prefix):
                    return value.__name__[len(prefix) :]
        return None

    def _import_projection(self, name):
        matches = [item for item in self._index if name in item[3] and item[1] == self.arch]
        # Shared ancestor namespaces can contain independent library roots.
        # Do not pick one library's __init__ as authority for the others.
        descendants = [item for item in self._index if item[2].startswith(name + ".") and item[1] == self.arch]
        if len(descendants) > 1 and not any(item[2] == name for item in matches):
            return None
        if not matches and descendants:
            return None
        return self.module(name)

    def _resolve(self, name):
        if self._index is None:
            self._discover()
        matches = [item for item in self._index if name in item[3] and item[1] == self.arch]
        # An explicit exported module wins over another package's retained Host
        # dependency. Multiple exports of the same root/target remain ambiguous.
        roots = [item for item in matches if item[2] == name]
        matches = roots or matches
        if not matches:
            available = sorted({str(item[1]) for item in self._index if name in item[3]})
            raise ModuleNotFoundError(
                f"No AOT package for {name!r} on {self.arch}; available targets: {available}. "
                "Install the backend SDK or set package_paths / FORGE_AOT_PACKAGE_PATH. No JIT fallback."
            )
        if len(matches) != 1:
            raise ValueError(f"Ambiguous AOT package for {name!r}: {[str(item[0]) for item in matches]}")
        return matches[0][0]

    def module(self, name):
        """Return the normal module interface, with numerical code selected by mode."""
        if not isinstance(name, str) or not name or not all(part.isidentifier() for part in name.split(".")):
            raise ValueError("Expected an absolute Python module name")
        with self._lock:
            if not self._active or self._closed:
                raise RuntimeError("Enter an active execution session before resolving modules")
            if name not in self._modules:
                if self.mode == "jit":
                    module = importlib.import_module(name)
                else:
                    directory = self._resolve(name)
                    from .aot import Runtime, load_package

                    if self.runtime is None:
                        self.runtime = self._stack.enter_context(Runtime(self.arch, **self.runtime_options))
                    if directory not in self._packages:
                        package = self._stack.enter_context(load_package(self.runtime, directory))
                        self._packages[directory] = package
                    module = self._packages[directory].module(name)
                self._modules[name] = module
            return self._modules[name]

    def close(self):
        global _import_session
        with self._lock:
            if self._closed:
                return
            try:
                self._stack.close()
            finally:
                with _import_lock:
                    if _import_session is self:
                        self._finder.close()
                        _import_session = None
                self._closed, self._active = True, False
                self._modules.clear()
                self._packages.clear()

    def __exit__(self, *_):
        self.close()
