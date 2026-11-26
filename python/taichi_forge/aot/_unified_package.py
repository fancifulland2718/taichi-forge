"""Produce one LLVM body library from the engine's device-dependent declarations."""

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote
import warnings
import zipfile


def _bytes(value, *, sort_keys=False):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, sort_keys=sort_keys).encode()


class _Projection:
    def __init__(self, directory, staging):
        self.directory = directory
        self.staging = staging

    def finish(self, module, manifest, payloads):
        # All targets use the same LLVM emitter. Expose loose compiler outputs
        # to the single library transaction; there is no target archive here.
        module.save(str(self.directory / "module"), shared_component_contracts=True)
        absent_inputs = {}
        for symbol, entry in manifest["entries"].items():
            for index, spec in enumerate(entry["captures"], start=len(entry["arguments"])):
                for target in spec.get("absent_devices", ()):
                    absent_inputs.setdefault(target, {}).setdefault(symbol, []).append(index)
        if absent_inputs:
            (self.directory / "module" / "unified_absent_inputs.json").write_bytes(_bytes(absent_inputs, sort_keys=True))
        for relative in sorted(payloads, key=Path):
            data = payloads[relative]
            path = self.staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            manifest["files"][relative] = hashlib.sha256(data).hexdigest()
        (self.directory / "view.json").write_bytes(_bytes(manifest))


def _worker(request):
    import importlib
    import taichi_forge as ti
    from taichi_forge._lib import core
    from ._package import _export_package, _package_source_inputs, _preflight_package
    from ._package_build_cache import PackageBuildCache
    from ._package_compilation import PackageCompilation
    from ._package_sources import shader_assets

    if not Path(core.__file__).samefile(request["compiler"]):
        raise RuntimeError("Unified production worker loaded a different compiler")
    start = time.perf_counter()
    directory = Path(request["directory"]) / "portable"
    directory.mkdir()
    options = request["options"].copy()
    for name in ("default_fp", "default_ip"):
        if name in options:
            options[name] = getattr(ti, options[name])
    cache_dir = request["cache"]
    if cache_dir:
        options.update(offline_cache=True, offline_cache_file_path=str(Path(cache_dir) / "portable" / "native"))
    # The host Program owns frontend objects. Device predicates in numerical
    # definitions are emitted as IR and resolved by the prepare compiler.
    ti.init(arch=ti.cpu, enable_fallback=False, **options)
    roots = []
    for name, filename in request["roots"]:
        module = importlib.import_module(name)
        if not Path(module.__file__).samefile(filename):
            raise ValueError(f"Unified export root changed origin: {name}")
        roots.append(module)
    source_root = request["source_root"]
    inputs = _package_source_inputs(roots, ti.cpu, source_root)
    files, _, discovery, _, modules, _, packages, syntax = inputs
    assets = shader_assets(files, modules, syntax)
    cache = None
    if cache_dir:
        cache = PackageBuildCache(Path(cache_dir) / "portable" / "declarations", files, modules,
                                  packages, (), "unified-bc-library", discovery=discovery, assets=assets)
        _restore_leaf_cache(Path(cache_dir) / "unified", directory.parent, cache.object_context)
    declarations = _preflight_package(roots, inputs=inputs, build_cache=cache, portable=True)
    preflight = time.perf_counter()
    del inputs, files, modules
    with PackageCompilation(portable=True) as compilation:
        projection = _Projection(directory, Path(request["staging"]))
        _export_package(directory, declarations, source_root, (), "unified-bc-library", compilation,
                        cache, assets, projection=projection)
        if cache is not None and cache.object_context is not None:
            counts = compilation.module._aot_builder._streaming_export_statistics()
            counts.update(preflight_seconds=preflight - start, compile_and_seal_seconds=time.perf_counter() - preflight)
            (directory.parent / "portable-statistics.json").write_bytes(
                _bytes(dict(version=3, context=cache.object_context, produce=counts)))
            (directory.parent / "portable-declarations.json").write_bytes(_bytes(cache.declaration_record()))


def _run_worker(request_path):
    from ._package import PackageExportError

    try:
        _worker(json.loads(request_path.read_bytes()))
    except PackageExportError as error:
        (request_path.parent / "export-error.txt").write_text(str(error), encoding="utf8")
        raise SystemExit(1) from None


_COMPRESSION_LEVEL = 6


def _write_member(archive, name, data):
    info = zipfile.ZipInfo(name, date_time=(2000, 12, 1, 0, 0, 0))
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=_COMPRESSION_LEVEL)


def _module_members(module, manifest):
    from taichi_forge_runtime.aot._entry_blocks import (
        BLOCK_PREFIX, ENTRY_PREFIX, RUNTIME_REQUIREMENT, pack_entry_blocks,
    )

    entries = manifest["entries"]
    names = [ENTRY_PREFIX + quote(name, safe="-_.") + ".json"
             for name in sorted(entries, key=lambda name: (entries[name]["definition"], name))]
    paths = {path.relative_to(module).as_posix(): path for path in module.rglob("*") if path.is_file()}
    ordered = [name for name in names if name in paths]
    has_blocks = False
    for name, data in pack_entry_blocks(ordered, lambda key: paths.pop(key).read_bytes(), level=_COMPRESSION_LEVEL):
        has_blocks |= name.startswith(BLOCK_PREFIX)
        yield name, data
    if has_blocks:
        manifest["runtime_requirements"] = sorted(set(manifest.get("runtime_requirements", ())) | {RUNTIME_REQUIREMENT})
    for name in sorted(paths):
        if name != "unified_llvm.json":
            yield name, paths[name].read_bytes()


def _collect(directory, archive):
    manifest = json.loads((directory / "view.json").read_bytes())
    module = directory / "module"
    rows = json.loads((module / "unified_llvm.json").read_bytes())
    roots = {row["key"]: row["value"] for row in rows}
    if not roots or len(roots) != len(rows):
        raise ValueError("Invalid unified LLVM entry table")
    members = {}
    written = set()
    for relative, data in _module_members(module, manifest):
        digest = hashlib.sha256(data).hexdigest()
        physical = "records/" + digest
        members[relative] = physical
        if physical not in written:
            _write_member(archive, physical, data)
            written.add(physical)
    return manifest, {"entries": roots, "members": members}


def _write_units(archive, directory):
    with (directory / "units.bin").open("rb") as stream, (directory / "units.index").open(encoding="ascii") as index:
        for row in index:
            digest, offset, size = row.split()
            stream.seek(int(offset))
            data = stream.read(int(size))
            if hashlib.sha256(data).hexdigest() != digest or not data.startswith(b"BC\xc0\xde"):
                raise ValueError(f"Invalid unified LLVM unit: {digest}")
            _write_member(archive, "code_units/" + digest + ".bc", data)


def _init_options():
    from taichi_forge._lib import core
    from taichi_forge.lang import impl

    runtime = impl.get_runtime()
    if runtime.prog is None:
        return {}
    cfg = impl.current_cfg()
    options = {name: getattr(cfg, name) for name, source in core._get_compile_option_sources(cfg).items()
               if source in ("init_keyword", "direct_assignment")}
    for name in ("fast_math", "debug", "default_fp", "default_ip"):
        value = getattr(cfg, name)
        options[name] = str(value) if name.startswith("default_") else value
    options["short_circuit_operators"] = runtime.short_circuit_operators
    return options


def _restore_leaf_cache(directory, workspace, context):
    if context is None:
        return
    try:
        record = json.loads((directory / "latest.json").read_bytes())
        if not isinstance(record, dict) or record.get("version") != 3 or record.get("context") != context:
            return
        archive = Path(record["output"]) / "code.tcm"
        if not zipfile.is_zipfile(archive):
            return
        bodies = workspace / "bodies"
        bodies.mkdir()
        for field, filename in (("catalog", "roots.tcb"), ("helpers", "helpers.tcb")):
            digest = record.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                continue
            try:
                catalog = (directory / (digest + ".tcb")).read_bytes()
            except OSError:
                continue
            if hashlib.sha256(catalog).hexdigest() == digest:
                (bodies / filename).write_bytes(catalog)
        (bodies / "baseline-path").write_bytes(os.fsencode(archive))
    except (OSError, ValueError, TypeError, KeyError):
        return


def _publish_cache_file(source, target):
    # Most local builds can commit a completed file with a single rename.
    # A cache on another volume still requires a private copy before replace.
    try:
        os.replace(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV and getattr(error, "winerror", None) != 17:
            raise
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temp = Path(stream.name)
            try:
                with source.open("rb") as reader:
                    shutil.copyfileobj(reader, stream)
                stream.close()
                os.replace(temp, target)
            finally:
                stream.close()
                temp.unlink(missing_ok=True)


def _record_leaf_cache(cache_dir, workspace, output):
    directory = Path(cache_dir) / "unified"
    receipt = workspace / "portable-statistics.json"
    if not receipt.is_file():
        return
    try:
        record = json.loads(receipt.read_bytes())
        record["output"] = str(output)
        directory.mkdir(parents=True, exist_ok=True)
        for field, filename in (("catalog", "roots.tcb"), ("helpers", "helpers.tcb")):
            source = workspace / "bodies" / filename
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            _publish_cache_file(source, directory / (digest + ".tcb"))
            record[field] = digest
        declarations = directory.parent / "portable" / "declarations"
        declarations.mkdir(parents=True, exist_ok=True)
        _publish_cache_file(workspace / "portable-declarations.json", declarations / "latest.json")
        receipt.write_bytes(_bytes(record))
        _publish_cache_file(receipt, directory / "latest.json")
    except OSError as error:
        warnings.warn(f"Unified package published; optional producer cache update failed: {error}",
                      RuntimeWarning, stacklevel=2)


def export_library(package, output, arch, caps, source_root, build_cache):
    from taichi_forge._lib import core
    from taichi_forge_runtime.aot._package_manifest import encode_manifest
    from ._package import PackageExportError, _publish_package
    from ._package_sources import package_roots

    if arch is None:
        arch = ("x64", "cuda", "vulkan")
    elif not isinstance(arch, (tuple, list)):
        arch = (arch,)
    names = [getattr(value, "name", value) for value in arch]
    targets = list(dict.fromkeys("x64" if name == "cpu" else name for name in names))
    if not targets or set(targets) - {"x64", "cuda", "vulkan"}:
        raise ValueError("Unified LLVM targets must be x64, cuda or vulkan")
    capability_sets = caps if isinstance(caps, dict) else {name: list(caps or ()) for name in targets}
    roots = package_roots(package, source_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".forge-unified-build-", dir=output.parent) as temporary:
        workspace = Path(temporary)
        request = dict(directory=str(workspace), compiler=core.__file__, options=_init_options(),
                       roots=[(root.__name__, str(Path(root.__file__).resolve())) for root in roots],
                       source_root=str(Path(source_root).resolve()) if source_root else None,
                       cache=str(Path(build_cache).resolve()) if build_cache is not None else None)
        request_path = workspace / "request.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(str(Path(path or ".").resolve()) for path in sys.path)
        # A Program owns the frontend's field trees, kernel factories and target
        # selection. Isolate those lifetimes from the caller and from one another.
        worker = ("import sys; from pathlib import Path; "
                  "from taichi_forge.aot._unified_package import _run_worker; "
                  "_run_worker(Path(sys.argv[1]))")
        staging = Path(tempfile.mkdtemp(prefix=".forge-package-", dir=output.parent))
        complete = False
        try:
            request["staging"] = str(staging)
            request_path.write_bytes(_bytes(request))
            result = subprocess.run([sys.executable, "-c", worker, str(request_path)], env=env)
            if result.returncode:
                error_path = workspace / "export-error.txt"
                if error_path.is_file():
                    raise PackageExportError(error_path.read_text(encoding="utf8"))
                result.check_returncode()
            with zipfile.ZipFile(staging / "code.tcm", "w") as archive:
                base, index = _collect(workspace / "portable", archive)
                del base["frontend_target"]
                _write_units(archive, workspace / "bodies")
                _write_member(archive, "unified_llvm.json", _bytes(dict(version=2, **index)))
            digest = hashlib.sha256((staging / "code.tcm").read_bytes()).hexdigest()
            document = dict(kind="forge_unified_definition_library", version=2, base=encode_manifest(base),
                            targets=targets, target_capabilities=capability_sets, tcm_sha256=digest)
            (staging / "package.json").write_bytes(_bytes(document))
            complete = True
            _publish_package(staging, output)
            if build_cache is not None:
                _record_leaf_cache(build_cache, workspace, output)
            return document
        finally:
            if not complete:
                shutil.rmtree(staging)
