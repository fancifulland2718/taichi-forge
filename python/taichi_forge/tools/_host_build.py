"""Private producer for source-hidden application hosts, usable as a script.

Build inputs and caches stay outside the immutable output layout. This tool does
not discover application reachability, install dependencies, or import the DSL.
"""

import argparse
import ast
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import io
import json
import keyword
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tokenize

if __package__:
    from ._compiled_host import lower_module
    from ._native_host import NativeConstructors
else:
    from _compiled_host import lower_module
    from _native_host import NativeConstructors


DIRECTIVES = {
    "language_level": 3,
    "binding": True,
    "annotation_typing": False,
    "infer_types": False,
    "emit_code_comments": False,
    "embedsignature": False,
}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _name(value):
    if not isinstance(value, str) or not all(p.isidentifier() and not keyword.iskeyword(p) for p in value.split(".")):
        raise ValueError(f"Invalid host module name: {value!r}")
    return value


def _relative(value):
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or any(p in ("..", ".") for p in value.split("/"))
        or ":" in value
    ):
        raise ValueError(f"Expected a relative POSIX path: {value!r}")
    return path


def read_spec(filename):
    filename = Path(filename).resolve()
    spec = json.loads(filename.read_text(encoding="utf-8"))
    if set(spec) - {"version", "source_root", "modules", "aliases", "toolchain"} or spec.get("version") != 1:
        raise ValueError("Unsupported host build specification")
    source_root = (filename.parent / spec["source_root"]).resolve()
    modules, paths, names = {}, {}, set()
    for name, relative in spec["modules"].items():
        _name(name)
        if os.path.normcase(name) in names:
            raise ValueError(f"Conflicting host module names on this platform: {name}")
        names.add(os.path.normcase(name))
        if name.endswith(".__init__"):
            raise ValueError("Use the canonical package name, not its __init__ spelling")
        relative = _relative(relative)
        source = (source_root / relative).resolve()
        if not source.is_relative_to(source_root) or source.suffix != ".py" or not source.is_file():
            raise ValueError(f"Missing or out-of-root Python source for {name}: {relative}")
        if source in paths:
            raise ValueError(f"Duplicate source for {paths[source]} and {name}; describe a registered alias instead")
        paths[source] = name
        modules[name] = {
            "source": str(source),
            "relative": relative.as_posix(),
            "package": source.name == "__init__.py",
        }
    if not modules:
        raise ValueError("A host build requires explicit modules")
    # An initializer can rewrite __file__ or replace its module. A post-import
    # inventory alone must not silently drop those real startup dependencies.
    for name, item in modules.items():
        for parent in Path(item["source"]).parents:
            if parent == source_root:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file() and initializer.resolve() not in paths:
                raise ValueError(
                    f"Missing package initializer needed by {name}: {initializer.relative_to(source_root)}"
                )
    aliases = spec.get("aliases", {})
    for alias, contract in aliases.items():
        _name(alias)
        if os.path.normcase(alias) in names or set(contract) != {"target", "registered_by"}:
            raise ValueError(f"Invalid host alias contract: {alias}")
        names.add(os.path.normcase(alias))
        if contract["target"] not in modules or contract["registered_by"] not in modules:
            raise ValueError(f"Host alias {alias} needs a built target and registrar")
    return source_root, modules, aliases, spec["toolchain"]


def prepare_source(source, name):
    emitter = NativeConstructors(name)
    lowered, entries = lower_module(source, name, _constructor=emitter.constructor)

    class PythonTupleBranches(ast.NodeTransformer):
        used = False

        def visit_IfExp(self, node):
            self.generic_visit(node)
            for field in ("body", "orelse"):
                value = getattr(node, field)
                if isinstance(value, ast.Tuple):
                    self.used = True
                    cast = ast.Call(
                        ast.Attribute(ast.Name("_forge_host_cython", ast.Load()), "cast", ast.Load()),
                        [ast.Attribute(ast.Name("_forge_host_cython", ast.Load()), "object", ast.Load()), value],
                        [],
                    )
                    setattr(node, field, ast.copy_location(cast, value))
            return node

    transform = PythonTupleBranches()
    tree = transform.visit(ast.parse(lowered))
    declaration, cpp, header = emitter.emit("templates.h")
    imports = []
    if transform.used:
        imports.append(ast.Import([ast.alias("cython", "_forge_host_cython")]))
    if declaration:
        imports.append(
            ast.ImportFrom("cython.cimports._forge_host_templates", [ast.alias("_forge_host_native")], level=0)
        )
    if imports:
        position = 0
        for index, node in enumerate(tree.body):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                position = index + 1
            elif (
                index == 0
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                position = 1
        tree.body[position:position] = imports
        lowered = ast.unparse(ast.fix_missing_locations(tree))
    # Ordinary modules can retain their original encoding cookie. Our build
    # inputs are UTF-8, so retaining a Latin-1 cookie would corrupt constants.
    lines = lowered.splitlines(keepends=True)
    for index in range(min(2, len(lines))):
        if re.match(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+", lines[index]):
            lines[index] = "# coding: utf-8\n"
    lowered = "".join(lines)
    # Compile application code as Python: names such as include/cdef/DEF are
    # legal Python identifiers but reserved in .pyx syntax. Native declarations
    # live in a producer-only .pxd; the cimport adds no runtime dependency.
    return {
        "host.py": lowered,
        "_forge_host_templates.pxd": declaration,
        "templates.cpp": cpp,
        "templates.h": header,
    }, entries


def toolchain(lock):
    observed = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "cython": importlib.metadata.version("Cython"),
        "setuptools": importlib.metadata.version("setuptools"),
    }
    if set(lock) != set(observed) or observed != lock:
        raise RuntimeError(f"Host toolchain does not match its lock: expected {lock}, found {observed}")
    compiler = shutil.which("cl" if os.name == "nt" else os.environ.get("CXX", "c++"))
    if compiler is None:
        raise RuntimeError("Initialize the matching C++ compiler environment before building hosts")
    producer = Path(__file__).resolve().parent
    headers = Path(sysconfig.get_path("include"))
    return {
        **observed,
        "python_build": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "platform": sysconfig.get_platform(),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "compiler_sha256": _digest(Path(compiler).read_bytes()),
        "producer_sha256": {
            name: _digest((producer / name).read_bytes())
            for name in ("_host_build.py", "_compiled_host.py", "_native_host.py")
        },
        "python_headers": {
            name: _digest((headers / name).read_bytes()) for name in ("Python.h", "pyconfig.h", "patchlevel.h")
        },
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "CC",
                "CXX",
                "CL",
                "_CL_",
                "CFLAGS",
                "CXXFLAGS",
                "CPPFLAGS",
                "LDFLAGS",
                "INCLUDE",
                "LIB",
                "VCToolsVersion",
                "WindowsSDKVersion",
            )
        },
        "directives": DIRECTIVES,
        "docstrings": False,
        "error_on_unknown_names": False,
        "compile_args": ["/EHsc", "/utf-8"] if os.name == "nt" else ["-std=c++17"],
    }


def verified_artifact(directory, recipe):
    receipt = directory / "receipt.json"
    if not receipt.is_file():
        return None
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        relative = _relative(record["file"])
        artifact = (directory / relative).resolve()
        if record["recipe"] != recipe or not artifact.is_relative_to(directory.resolve()):
            return None
        if not artifact.name.endswith(sysconfig.get_config_var("EXT_SUFFIX")) or not isinstance(
            record["dsl_entries"], list
        ):
            return None
        if artifact.stat().st_size != record["bytes"] or _digest(artifact.read_bytes()) != record["sha256"]:
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return record


def _worker(filename):
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext
    from Cython.Build import cythonize
    from Cython.Compiler import Options

    job = json.loads(Path(filename).read_text(encoding="utf-8"))
    directory = Path(filename).resolve().parent
    os.chdir(directory)
    Options.docstrings = False
    Options.error_on_unknown_names = False
    sources = ["host.py"] + (["templates.cpp"] if (directory / "templates.cpp").stat().st_size else [])
    extension = Extension(
        job["extension"], sources, language="c++", include_dirs=["."], extra_compile_args=job["compile_args"]
    )
    extensions = cythonize([extension], compiler_directives=DIRECTIVES, quiet=True, force=True)
    command = build_ext(Distribution({"name": "forge-compiled-host", "ext_modules": extensions}))
    command.build_lib, command.build_temp, command.force = "lib", "objects", True
    command.ensure_finalized()
    command.run()
    artifact = Path(command.get_ext_fullpath(extension.name)).resolve()
    record = {
        "recipe": job["recipe"],
        "file": artifact.relative_to(directory).as_posix(),
        "sha256": _digest(artifact.read_bytes()),
        "bytes": artifact.stat().st_size,
        "dsl_entries": job["dsl_entries"],
    }
    pending = directory / "receipt.pending.json"
    pending.write_text(_json(record), encoding="utf-8")
    pending.replace(directory / "receipt.json")


@contextmanager
def _exclusive_cache(directory):
    """A producer cache has one writer; individual module workers are parallel."""
    lock = directory / "producer.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(f"Host cache is locked: {lock}. Check its producer before removing a stale lock.") from error
    try:
        with handle:
            handle.write(str(os.getpid()))
            handle.flush()
            yield
    finally:
        lock.unlink()


def build(spec, work, output, jobs=1):
    source_root, modules, aliases, lock = read_spec(spec)
    work, output = Path(work).resolve(), Path(output).resolve()
    if work == output or work.is_relative_to(output) or output.is_relative_to(work):
        raise ValueError("Build cache and immutable output must be separate directories")
    if (
        work.is_relative_to(source_root)
        or source_root.is_relative_to(work)
        or output.is_relative_to(source_root)
        or source_root.is_relative_to(output)
    ):
        raise ValueError("Build/output directories must not contain or overwrite private sources")
    identity = toolchain(lock)
    work.mkdir(parents=True, exist_ok=True)
    with _exclusive_cache(work):
        return _build_locked(modules, aliases, lock, identity, work, output, jobs)


def _build_locked(modules, aliases, lock, identity, work, output, jobs):
    requests = []
    for name, item in sorted(modules.items()):
        raw = Path(item["source"]).read_bytes()
        source_hash = _digest(raw)
        recipe = _digest(
            _json(
                {
                    "name": name,
                    "package": item["package"],
                    "source": source_hash,
                    "toolchain": identity,
                }
            ).encode()
        )
        requests.append((name, item, raw, recipe, source_hash))

    def compile_one(request):
        name, item, raw, recipe, source_hash = request
        directory = work / recipe
        record = verified_artifact(directory, recipe)
        cached = record is not None
        if record is None:
            # Source bytes plus the exact producer/toolchain determine all
            # generated input. A hit needs neither AST reconstruction nor Cython.
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
            with io.TextIOWrapper(io.BytesIO(raw), encoding=encoding, newline=None) as stream:
                inputs, entries = prepare_source(stream.read(), name)
            directory.mkdir(parents=True, exist_ok=True)
            for filename, content in inputs.items():
                (directory / filename).write_text(content, encoding="utf-8", newline="\n")
            job = {
                "extension": name + (".__init__" if item["package"] else ""),
                "recipe": recipe,
                "compile_args": identity["compile_args"],
                "dsl_entries": list(entries),
            }
            (directory / "job.json").write_text(_json(job), encoding="utf-8")
            with (directory / "build.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(directory / "job.json")],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            record = verified_artifact(directory, recipe)
            if process.returncode or record is None:
                raise RuntimeError(f"Host build failed for {name}; see {directory / 'build.log'}")
        relative = Path(*name.split("."))
        if item["package"]:
            relative /= "__init__"
        return (
            {
                "name": name,
                "file": relative.as_posix() + identity["extension_suffix"],
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "source_sha256": source_hash,
                "recipe": recipe,
                "dsl_entries": record["dsl_entries"],
            },
            directory / record["file"],
            cached,
        )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(compile_one, requests))
    manifest = {
        "version": 1,
        "kind": "forge_compiled_host",
        "toolchain": lock,
        "platform": identity["platform"],
        "cache_tag": identity["cache_tag"],
        "producer_sha256": identity["producer_sha256"],
        "modules": [item for item, _, _ in results],
        "aliases": aliases,
    }
    if output.exists():
        existing = output / "host-manifest.json"
        if not existing.is_file() or json.loads(existing.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Output is immutable; choose a new output directory for a different build")
        expected = {item["file"] for item, _, _ in results} | {"host-manifest.json"}
        actual = set()
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Unexpected output symlink: {path.relative_to(output)}")
            if path.is_file():
                actual.add(path.relative_to(output).as_posix())
        if actual != expected:
            raise ValueError("Unexpected output contents; a host layout may contain only its declared artifacts")
        for item, _, _ in results:
            path = output / item["file"]
            if not path.is_file() or _digest(path.read_bytes()) != item["sha256"]:
                raise ValueError(f"Published host artifact changed: {item['name']}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=output.name + ".pending-", dir=output.parent)).resolve()
        for item, source, _ in results:
            target = stage / item["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (stage / "host-manifest.json").write_text(_json(manifest), encoding="utf-8")
        if stage.parent != output.parent or stage == output:
            raise ValueError("Invalid host publication directory")
        stage.rename(output)
    return {
        "modules": len(results),
        "cached": sum(cached for _, _, cached in results),
        "bytes": sum(item["bytes"] for item, _, _ in results),
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        if not all((args.spec, args.work, args.output)) or args.jobs <= 0:
            parser.error("--spec, --work, --output and a positive --jobs are required")
        print(_json(build(args.spec, args.work, args.output, args.jobs)))


if __name__ == "__main__":
    main()
