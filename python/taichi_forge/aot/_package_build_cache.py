"""Producer warm start and dependency-based reuse of native package objects."""

import ast
import base64
import builtins
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
from types import ModuleType
from typing import Final, get_args, get_origin
import warnings
import zlib

from taichi_forge.lang._build_cache import BuildIdentity, file_digest, unchanged_file_digest, value_digest


class _UntrackedInput(Exception):
    pass


def _scalar_kernel_inputs(kernel, modules):
    """Dependency identity for free, lexically closed scalar numerical code.

    Inspect syntax as well as live code: Python bytecode can fold a branch
    that the Taichi frontend still reads. A Host callable, arbitrary attribute
    lookup or closure needs a richer dependency contract and uses capture.
    """
    from taichi_forge._lib import core
    from taichi_forge.aot._component import _code_definition, _final_global
    from taichi_forge.lang.exception import TaichiCompilationError

    seen = {}

    def visit(function, annotations):
        if (function.__module__ not in modules or function.__closure__
                or function.__globals__ is not vars(modules[function.__module__])
                or function.__qualname__ != function.__name__
                or function.__defaults__ or function.__kwdefaults__):
            raise _UntrackedInput()
        if function in seen:
            return
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        if not isinstance(node, ast.FunctionDef) or node.name != function.__name__:
            raise _UntrackedInput()
        if not all(isinstance(item, core.DataType) for item in annotations):
            raise _UntrackedInput()
        facts = {}
        seen[function] = (function.__module__, function.__qualname__, ast.dump(node),
                          _code_definition(function.__code__), list(map(str, annotations)), facts)
        local = set(function.__code__.co_varnames)
        for statement in node.body:
            for use in ast.walk(statement):
                if isinstance(use, (ast.Attribute, ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
                    raise _UntrackedInput()
                if not isinstance(use, ast.Name) or not isinstance(use.ctx, ast.Load) or use.id in local:
                    continue
                name = use.id
                value = function.__globals__.get(name, function.__builtins__.get(name))
                if inspect.isfunction(value) and getattr(value, "_is_taichi_function", False):
                    helper = value.func
                    if helper.is_real_function or helper.pyfunc:
                        raise _UntrackedInput()
                    target = helper.func
                    facts[name] = (target.__module__, target.__qualname__)
                    visit(target, [arg.annotation for arg in helper.arguments] + list(helper.return_type or ()))
                elif type(value) in (int, bool, str, float) and name in function.__globals__:
                    facts[name] = (type(value).__name__, repr(value), _final_global(function, name))
                elif name in ("range", "int", "float", "abs", "min", "max") and value is getattr(builtins, name):
                    facts[name] = ("builtin", name)
                else:
                    raise _UntrackedInput()

    try:
        if kernel.template_slot_locations:
            return None
        visit(kernel.func, [arg.annotation for arg in kernel.arguments] + list(kernel.return_type or ()))
        return value_digest([kernel.opt_level, list(seen.values())])
    except (_UntrackedInput, TaichiCompilationError, OSError, TypeError, SyntaxError, ValueError):
        return None


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
        if get_origin(value) is Final:
            return ("final", encode(get_args(value)))
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
                               if key == "__annotations__" or not (key.startswith("__") and key.endswith("__"))})
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
    def __init__(self, directory, sources, modules, packages, caps, artifact_format, *, discovery=None, assets=None):
        from taichi_forge._lib import core
        from taichi_forge.lang import impl

        self.directory = Path(directory).resolve()
        self.identity = None
        self.inputs = None
        self.modules = modules
        self.objects = {}
        self.captures = {}
        self.method_inputs = {}
        self.capture_syntax = {}
        self.object_context = None
        self.toolchain_files = {}
        try:
            self.previous = json.loads((self.directory / "latest.json").read_text(encoding="utf8"))
            if not isinstance(self.previous, dict) or self.previous.get("version") != 1:
                self.previous = {}
        except (OSError, ValueError):
            self.previous = {}
        self.reason = "miss"
        arch = impl.current_cfg().arch
        supported = (
            (arch == core.cuda and artifact_format == "bc-library")
            or (arch in (core.x64, core.arm64) and artifact_format == "bc")
            or (arch == core.vulkan and artifact_format is None)
        )
        if not supported:
            raise ValueError("build_cache requires CUDA bc-library, CPU bc, or Vulkan default export")
        self.objects_enabled = arch in (core.x64, core.arm64, core.vulkan) and impl.current_cfg().offline_cache
        self.build_settings = dict(
            config=core._build_cache_config_key(impl.current_cfg(), impl.get_runtime().prog.get_device_caps(), True),
            option_sources=core._get_compile_option_sources(impl.current_cfg()),
            caps=sorted(caps or ()), artifact_format=artifact_format,
        )
        try:
            namespaces = _namespace_inputs(modules)
            self.inputs = dict(
                **self._compiler_inputs(),
                sources={name: (str(path), str(relative), hashlib.sha256(text.encode("utf8")).hexdigest())
                         for name, path, relative, text, *_ in sources},
                namespaces=namespaces,
                roots=[item.__name__ for item in packages],
                discovery=discovery,
            )
            if assets:
                self.inputs["assets"] = {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()}
            self.identity = BuildIdentity.create("aot-package", 1,
                                                 {key: value_digest(value) for key, value in self.inputs.items()})
        except _UntrackedInput as error:
            self.reason = "untracked input: " + str(error)

    def _compiler_inputs(self):
        producer = _producer_inputs(self.previous.get("toolchain_files", {}), self.toolchain_files)
        context = dict(producer=producer, **self.build_settings)
        self.object_context = value_digest(context)
        return context

    def scalar_object_key(self, kernel):
        if not self.objects_enabled:
            return None
        dependencies = _scalar_kernel_inputs(kernel, self.modules)
        if dependencies is None:
            return None
        if self.object_context is None:
            try:
                self._compiler_inputs()
            except _UntrackedInput:
                self.objects_enabled = False
                return None
        return value_digest((self.object_context, dependencies))

    def restore_object(self, key, module):
        objects = self.previous.get("objects")
        if not isinstance(objects, dict) or key not in objects:
            return None
        record = objects[key]
        try:
            native, entry, component = record["payload"]
            symbol = entry["symbol"]
            if record["digest"] != value_digest((key, record["payload"])):
                return None
            if entry["definitions"] != [entry["definition"]]:
                return None  # Recompute aliases against this build's declarations.
        except (KeyError, TypeError, ValueError):
            return None
        if not module._aot_builder._restore_build_object(symbol, native):
            return None
        module._component_entries[symbol] = component
        module._exported_kernels[symbol] = key
        module._content.extend(("kernel:" + symbol, "component_entry:" + symbol))
        self.objects[key] = record
        return entry

    def capture_scope(self, kernel, schema, environment):
        # A schema change may introduce a new branch, resource or receiver.
        # Traverse the current declaration domain even on a warm build.
        from ._package_capture_cache import _body, _callable

        fn = kernel.func
        if (not self.objects_enabled or fn.__module__ not in self.modules
                or environment._lexical_bindings):
            return None
        try:
            if _body(_callable(fn.__module__, fn.__qualname__)) is not fn:
                return None
            if self.object_context is None:
                self._compiler_inputs()
            def syntax(node):
                if isinstance(node, ast.AST):
                    return ast.dump(node)
                raise TypeError("Untracked producer declaration")

            declaration = json.dumps(schema, sort_keys=True, default=syntax)
            return value_digest((self.object_context, fn.__module__, fn.__qualname__, kernel.opt_level,
                                 declaration, environment._symbolic_dimensions, dict(environment._source_scope_paths)))
        except (TypeError, ValueError, AttributeError, _UntrackedInput):
            return None

    def capture_facts(self, code, component, owner):
        from taichi_forge.lang.exception import TaichiCompilationError
        from ._package_capture_cache import freeze

        try:
            return freeze(code, self.capture_syntax, component, owner)
        except (KeyError, AttributeError, OSError, ValueError, TypeError, TaichiCompilationError):
            return None

    def remember_capture(self, scope, code, module, symbol):
        facts = vars(code).pop("_cache_facts", None)
        if facts is None:
            return
        native = module._aot_builder._build_object_key(symbol, code.kernel.compiled_kernels[code.key])
        if not native:
            return
        payload = json.dumps((native, module._component_entries[symbol], facts), separators=(",", ":"))
        # Per-object compression leaves unrelated owners opaque during lookup
        # and preserves the original declaration traversal order on recovery.
        packed = base64.b64encode(zlib.compress(payload.encode("utf8"), 1)).decode("ascii")
        self.captures.setdefault(scope, {})[symbol] = dict(object=packed, digest=value_digest((scope, symbol, packed)))

    def restore_capture(self, compilation, component, owner, kernel, signature, resources, layouts):
        from taichi_forge.lang.exception import TaichiCompilationError
        from ._package_capture_cache import matches, thaw

        scope = compilation.capture_scope
        history = self.previous.get("captures")
        records = history.get(scope) if isinstance(history, dict) else None
        if not isinstance(records, dict):
            return None
        module = compilation.module
        for symbol, record in records.items():
            if symbol in module._exported_kernels:
                continue
            try:
                packed = record["object"]
                if record["digest"] != value_digest((scope, symbol, packed)):
                    continue
                native, entry, facts = json.loads(zlib.decompress(base64.b64decode(packed, validate=True)))
                key = kernel.func, "component_" + str(len(kernel._component_candidates)), kernel.autodiff_mode
                code = thaw(facts, kernel, key, entry)
                if not matches(code, facts, self.capture_syntax, component, owner, signature, resources, layouts):
                    continue
                code.validate_bindings(component, owner)
                code._bind_arguments(component.declarations)
            except (KeyError, AttributeError, OSError, ValueError, TypeError, zlib.error, TaichiCompilationError):
                continue
            if not module._aot_builder._restore_build_object(symbol, native):
                continue
            identity = kernel, key
            module._component_entries[symbol] = entry
            module._exported_kernels[symbol] = identity
            module._content.extend(("kernel:" + symbol, "component_entry:" + symbol))
            compilation.exported[identity] = symbol
            compilation.resource_effects[identity] = module._aot_builder._kernel_resource_effects(symbol)
            self.captures.setdefault(scope, {})[symbol] = record
            return code
        return None

    def record_object(self, key, native, entry, module):
        payload = (native, entry, module._component_entries[entry["symbol"]])
        self.objects[key] = dict(payload=payload, digest=value_digest((key, payload)))

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
            from taichi_forge_runtime.aot._package_manifest import decode_manifest, materialize_manifest

            manifest = materialize_manifest(decode_manifest(json.loads(raw)))
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
            self.objects = record.get("objects", {})
            self.captures = record.get("captures", {})
            self.method_inputs = record.get("method_inputs", {})
            return manifest
        finally:
            if not complete:
                shutil.rmtree(staging)

    def record(self, output):
        if self.identity is None and not (self.objects or self.captures or self.method_inputs):
            return
        try:
            if self.identity is not None and _namespace_inputs(self.modules) != self.inputs["namespaces"]:
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
        record = dict(version=1, key=self.identity.key if self.identity else None, output=str(output),
                      inputs=dict(self.identity.inputs) if self.identity else {}, toolchain_files=self.toolchain_files,
                      objects=self.objects,
                      captures=self.captures, method_inputs=self.method_inputs,
                      manifest_sha256=file_digest(output / "package.json"))
        fd, temporary = tempfile.mkstemp(prefix=".latest-", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf8") as stream:
                # Warm candidate lookup follows the producer's discovery order.
                json.dump(record, stream)
            os.replace(temporary, self.directory / "latest.json")
        finally:
            Path(temporary).unlink(missing_ok=True)
