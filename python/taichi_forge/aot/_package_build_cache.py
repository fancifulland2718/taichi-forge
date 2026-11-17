"""Optional producer warm start; completed packages remain the artifact owners.

The first stage reuses an unchanged complete export before witnesses/capture.
It does not pretend to reuse a changed entry's frontend or to compose independently
compiled resource contracts. Cache misses follow the ordinary compiler pipeline.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Final
import warnings

from taichi_forge.lang._build_cache import BuildIdentity, file_digest, unchanged_file_digest, value_digest


class _UntrackedInput(Exception):
    pass


def _namespace_inputs(modules):
    """Snapshot source-defined bindings, including live globals and closures.

    Opaque instances and undeclared external Python modules are cache misses,
    not repr/object-address identities. Ordinary export still supports them.
    """
    from taichi_forge.aot._component import _code_definition
    from taichi_forge._lib import core
    from taichi_forge.lang.matrix import MatrixType
    from taichi_forge.lang.struct import StructType
    from taichi_forge.types.ndarray_type import NdarrayType
    from taichi_forge.types._argument_descriptor import describe_annotation

    active = set()

    def framework(name):
        return any(name == root or name.startswith(root + ".")
                   for root in ("taichi_forge", "taichi_forge_runtime"))

    def encode(value):
        if value is Final:
            return ("declaration", "typing.Final")
        if value is None or type(value) in (bool, int, str):
            return (type(value).__name__, value)
        if type(value) is float:
            return ("float", value.hex())
        if type(value) is bytes:
            return ("bytes", value.hex())
        if isinstance(value, core.DataType):
            return ("dtype", str(value))
        if isinstance(value, MatrixType):
            return ("matrix", value.n, value.m, encode(value.dtype))
        if isinstance(value, StructType):
            return ("struct", encode(value.members))
        if isinstance(value, NdarrayType):
            return ("ndarray", describe_annotation(value).structural_key(), str(value.boundary), str(value.layout))
        if isinstance(value, ModuleType):
            name = value.__name__
            if name in modules or framework(name):
                return ("module", name)
            # These modules supply immutable language/numeric declarations.
            # Environment/I/O/provider modules need a later explicit input contract.
            if name in ("math", "typing", "enum", "dataclasses", "builtins"):
                return ("module", name, file_digest(value.__file__) if getattr(value, "__file__", None) else None)
            raise _UntrackedInput(name)
        if isinstance(value, (staticmethod, classmethod)):
            return (type(value).__name__, encode(value.__func__))
        if isinstance(value, property):
            return ("property", encode(value.fget), encode(value.fset), encode(value.fdel))
        origin = getattr(value, "__module__", "") or ""
        if framework(origin) or origin in ("builtins", "typing", "enum", "dataclasses", "math"):
            if inspect.isclass(value) or inspect.isroutine(value):
                return ("external", origin, value.__qualname__)
        if id(value) in active:
            if inspect.isclass(value) or inspect.isfunction(value):
                return ("reference", origin, value.__qualname__)
            raise _UntrackedInput(type(value).__name__)
        active.add(id(value))
        try:
            if type(value) in (list, tuple):
                return (type(value).__name__, [encode(item) for item in value])
            if type(value) is dict:
                return ("dict", sorted(((encode(k), encode(v)) for k, v in value.items()), key=repr))
            if type(value) in (set, frozenset):
                return (type(value).__name__, sorted((encode(item) for item in value), key=repr))
            if inspect.isclass(value) and origin in modules:
                return ("class", origin, value.__qualname__, [encode(base) for base in value.__bases__],
                        {k: encode(v) for k, v in vars(value).items()
                         if k not in ("__dict__", "__weakref__", "__doc__", "__module__")})
            if inspect.isfunction(value):
                function, options = value, None
                if getattr(value, "_is_wrapped_kernel", False):
                    function = value._primal.func
                    options = value._primal.opt_level
                elif getattr(value, "_is_taichi_function", False):
                    function = value.func.func
                if function.__module__ not in modules:
                    raise _UntrackedInput(function.__module__ + "." + function.__qualname__)
                return ("function", function.__module__, function.__qualname__,
                        _code_definition(function.__code__), options,
                        encode(function.__annotations__),
                        encode(function.__defaults__), encode(function.__kwdefaults__),
                        [encode(cell.cell_contents) for cell in function.__closure__ or ()])
            raise _UntrackedInput(type(value).__name__)
        finally:
            active.remove(id(value))

    return {name: value_digest({key: encode(value) for key, value in vars(module).items()
                               if not (key.startswith("__") and key.endswith("__"))})
            for name, module in modules.items()}


def _producer_inputs(previous, current):
    import taichi_forge
    import taichi_forge_runtime
    from taichi_forge._lib import core, utils

    result = {}
    for module in (taichi_forge, taichi_forge_runtime):
        root = Path(module.__file__).resolve().parent
        result[module.__name__] = value_digest({path.relative_to(root).as_posix(): unchanged_file_digest(path, previous, current)
                                               for path in sorted(root.rglob("*.py"))})
    native = getattr(utils, "_loaded_native_runtime_path", None)
    if not native:
        # Some checkout launchers preload the same explicit split runtime.
        directory = os.environ.get("TAICHI_NATIVE_RUNTIME_DIR")
        native = Path(directory) / utils._native_runtime_library_name() if directory else None
    if not native or not Path(native).is_file():
        raise _UntrackedInput("native Runtime provenance")
    result["native"] = unchanged_file_digest(native, previous, current)
    result["shim"] = unchanged_file_digest(core.__file__, previous, current)
    runtime_dir = Path(utils._runtime_bitcode_dir())
    result["runtime_bitcode"] = value_digest({path.name: unchanged_file_digest(path, previous, current)
                                              for path in sorted(runtime_dir.glob("*.bc"))})
    result["python"] = sys.implementation.cache_tag + ":" + sys.version
    return result


class PackageBuildCache:
    def __init__(self, directory, sources, modules, packages, caps, artifact_format):
        from taichi_forge._lib import core
        from taichi_forge.lang import impl

        self.directory = Path(directory).resolve()
        self.identity = None
        self.inputs = None
        self.modules = modules
        self.toolchain_files = {}
        try:
            self.previous = json.loads((self.directory / "latest.json").read_text(encoding="utf8"))
            if not isinstance(self.previous, dict) or self.previous.get("version") != 1:
                self.previous = {}
        except (OSError, ValueError):
            self.previous = {}
        self.reason = "miss"
        if artifact_format != "bc-library" or impl.current_cfg().arch != core.cuda:
            raise ValueError("build_cache currently requires CUDA bc-library export")
        try:
            self.inputs = dict(
                sources={name: (str(path), str(relative), hashlib.sha256(text.encode("utf8")).hexdigest())
                         for name, path, relative, text, *_ in sources},
                namespaces=_namespace_inputs(modules),
                producer=_producer_inputs(self.previous.get("toolchain_files", {}), self.toolchain_files),
                config=core._build_cache_config_key(impl.current_cfg(), impl.get_runtime().prog.get_device_caps(), True),
                option_sources=core._get_compile_option_sources(impl.current_cfg()),
                roots=[item.__name__ for item in packages],
                caps=sorted(caps or ()), artifact_format=artifact_format,
            )
            self.identity = BuildIdentity.create("aot-package", 1,
                                                 {key: value_digest(value) for key, value in self.inputs.items()})
        except _UntrackedInput as error:
            self.reason = "untracked input: " + str(error)

    def reuse(self, output, publish):
        if self.identity is None:
            return None
        try:
            record = self.previous
            if record["key"] != self.identity.key:
                self.reason = "inputs changed"
                return None
            previous = Path(record["output"])
            raw = (previous / "package.json").read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["manifest_sha256"]:
                return None
            manifest = json.loads(raw)
            if manifest["sources"] != {name: facts[2] for name, facts in self.inputs["sources"].items()}:
                return None
        except (OSError, ValueError, KeyError, TypeError):
            return None
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".forge-package-", dir=output.parent))
        complete = False
        try:
            # Verify while copying once. No archive extraction, LLVM parsing,
            # re-finalization, hardlinks to mutable outputs, or second hash pass.
            for relative, expected in manifest["files"].items():
                target = (staging / relative).resolve()
                source = (previous / relative).resolve()
                if not target.is_relative_to(staging) or not source.is_relative_to(previous):
                    return None
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                try:
                    reader = source.open("rb")
                except OSError:
                    return None
                with reader, target.open("wb") as writer:
                    while True:
                        try:
                            block = reader.read(1024 * 1024)
                        except OSError:
                            return None
                        if not block:
                            break
                        digest.update(block)
                        writer.write(block)
                if digest.hexdigest() != expected:
                    self.reason = "artifact changed"
                    return None
            if any(hashlib.sha256(Path(facts[0]).read_text(encoding="utf8").encode("utf8")).hexdigest() != facts[2]
                   for facts in self.inputs["sources"].values()):
                return None
            (staging / "package.json").write_bytes(raw)
            complete = True
            publish(staging, output)
            self.reason = "hit"
            return manifest
        finally:
            if not complete:
                shutil.rmtree(staging)

    def record(self, output):
        if self.identity is None:
            return
        try:
            if _namespace_inputs(self.modules) != self.inputs["namespaces"]:
                return  # Construction changed an input; do not cache that run.
            self._record(output)
        except _UntrackedInput:
            return
        except OSError as error:
            warnings.warn(f"Package was published at {output}, but its optional build cache could not be updated: {error}",
                          RuntimeWarning, stacklevel=2)

    def _record(self, output):
        self.directory.mkdir(parents=True, exist_ok=True)
        # One pointer, published only after a complete package. No duplicate
        # artifact store or mutable frontend object survives the transaction.
        record = dict(version=1, key=self.identity.key, output=str(output),
                      inputs=dict(self.identity.inputs), toolchain_files=self.toolchain_files,
                      manifest_sha256=file_digest(output / "package.json"))
        fd, temporary = tempfile.mkstemp(prefix=".latest-", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf8") as stream:
                json.dump(record, stream, sort_keys=True)
            os.replace(temporary, self.directory / "latest.json")
        finally:
            Path(temporary).unlink(missing_ok=True)
