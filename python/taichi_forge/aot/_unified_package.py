"""Produce one LLVM body library from the engine's device-dependent declarations."""

import errno
import hashlib
import json
import os
from contextlib import nullcontext
from copy import copy
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


def _ordered_entries(manifest):
    entries = manifest["entries"]
    return sorted(entries, key=lambda name: (entries[name]["definition"], name))


def _signature_references(module_dir):
    index = module_dir.parent.parent / "bodies" / "signatures.index"
    if not index.is_file():
        return {}
    return {"signatures/" + digest + ".json": digest for digest in index.read_text(encoding="ascii").splitlines()}


def _link_inputs(directory, module, manifest):
    # These are the linker's actual inputs after native emission/restoration.
    # Unit offsets describe scratch storage; the archive uses digest order.
    native = {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in directory.rglob("*") if path.is_file()}
    native.update(_signature_references(directory))
    with (directory.parent.parent / "bodies" / "units.index").open(encoding="ascii") as index:
        units = [line.split()[0] for line in index]
    # Preserve order inside each contract: it determines the ordered list of
    # shared references written by the entry encoder.
    entries = [(name, hashlib.sha256(_bytes(contract)).hexdigest())
               for name, contract in sorted(module._component_entries.items())]
    return hashlib.sha256(_bytes([sorted(native.items()), units, entries, _ordered_entries(manifest)])).hexdigest()


def _reuse_archive(record, inputs, snapshot, destination):
    if not record or not inputs or not snapshot.is_file():
        return False
    receipt = [inputs, record.get("tcm_sha256"), record.get("entry_blocks")]
    if record.get("link") != hashlib.sha256(_bytes(receipt)).hexdigest():
        return False
    snapshot.replace(destination)
    return True


class _Projection:
    def __init__(self, directory, staging, *, cache_links=False, previous=None):
        self.directory = directory
        self.staging = staging
        self.cache_links = cache_links
        self.previous = previous
        self.link_inputs = None
        self.reused = False

    def finish(self, module, manifest, payloads):
        # All targets use the same LLVM emitter. Expose loose compiler outputs
        # to the single library transaction; there is no target archive here.
        directory = self.directory / "module"
        # Component roots can share a physical contract directory with entries.
        # Preserve their ordinary writer order for modules containing both.
        defer_entries = not module._component_roots
        module._save(directory, shared_component_contracts=True, defer_entries=defer_entries)
        absent_inputs = {}
        for symbol, entry in manifest["entries"].items():
            for index, spec in enumerate(entry["captures"], start=len(entry["arguments"])):
                for target in spec.get("absent_devices", ()):
                    absent_inputs.setdefault(target, {}).setdefault(symbol, []).append(index)
        if absent_inputs:
            (self.directory / "module" / "unified_absent_inputs.json").write_bytes(_bytes(absent_inputs, sort_keys=True))
        if defer_entries and self.cache_links:
            self.link_inputs = _link_inputs(directory, module, manifest)
            self.reused = _reuse_archive(self.previous, self.link_inputs, self.directory.parent / "baseline.tcm",
                                         self.staging / "code.tcm")
            if self.reused and self.previous["entry_blocks"]:
                _require_entry_blocks(manifest)
        for relative in sorted(payloads, key=Path):
            data = payloads[relative]
            path = self.staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            manifest["files"][relative] = hashlib.sha256(data).hexdigest()
        documents = module._encode_component_entries(shared=True) if defer_entries and not self.reused else {}
        return manifest, documents


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
    cache = previous = None
    if cache_dir:
        cache = PackageBuildCache(Path(cache_dir) / "portable" / "declarations", files, modules,
                                  packages, (), "unified-bc-library", discovery=discovery, assets=assets)
        previous = _restore_leaf_cache(Path(cache_dir) / "unified", directory.parent, cache)
    declarations = _preflight_package(roots, inputs=inputs, build_cache=cache, portable=True)
    preflight = time.perf_counter()
    del inputs, files, modules
    with PackageCompilation(portable=True) as compilation:
        projection = _Projection(directory, Path(request["staging"]),
                                 cache_links=cache is not None and cache.object_context is not None,
                                 previous=previous)
        manifest, documents = _export_package(directory, declarations, source_root, (), "unified-bc-library", compilation,
                                              cache, assets, projection=projection)
        if cache is not None and cache.object_context is not None:
            counts = compilation.module._aot_builder._streaming_export_statistics()
            counts.update(preflight_seconds=preflight - start, compile_and_seal_seconds=time.perf_counter() - preflight)
            counts["unified_archive_reuses"] = int(projection.reused)
            (directory.parent / "portable-declarations.json").write_bytes(_bytes(cache.declaration_record()))
    # Link the encoded documents directly after retiring the compiler. Passing
    # them through individual scratch files would require a second lookup and
    # read per entry even for a two-root incremental rebuild.
    if not projection.reused:
        snapshot = directory.parent / "baseline.tcm"
        prior = zipfile.ZipFile(snapshot) if snapshot.is_file() else nullcontext(None)
        with prior as source, zipfile.ZipFile(projection.staging / "code.tcm", "w") as archive:
            baseline = _CompressedMembers(source)
            index = _collect(directory / "module", archive, baseline, manifest, documents)
            _write_units(archive, directory.parent / "bodies", baseline)
            _write_member(archive, "unified_llvm.json", _bytes(dict(version=2, **index)))
    if cache is not None and cache.object_context is not None:
        from taichi_forge_runtime.aot._entry_blocks import RUNTIME_REQUIREMENT
        (directory.parent / "portable-statistics.json").write_bytes(_bytes(dict(
            version=3, context=cache.object_context, native_context=cache.native_context,
            produce=counts, link_inputs=projection.link_inputs,
            entry_blocks=RUNTIME_REQUIREMENT in manifest.get("runtime_requirements", ()))))
    (directory / "view.json").write_bytes(_bytes(manifest))


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


def _module_members(module, manifest, documents):
    from taichi_forge_runtime.aot._entry_blocks import (
        BLOCK_PREFIX, ENTRY_PREFIX, pack_entry_blocks,
    )

    names = [ENTRY_PREFIX + quote(name, safe="-_.") + ".json"
             for name in _ordered_entries(manifest)]
    paths = {path.relative_to(module).as_posix(): path for path in module.rglob("*") if path.is_file()}
    def read(name):
        if name in documents:
            return documents.pop(name)
        return paths.pop(name).read_bytes()

    ordered = [name for name in names if name in documents or name in paths]
    has_blocks = False
    for name, data in pack_entry_blocks(ordered, read, level=_COMPRESSION_LEVEL):
        has_blocks |= name.startswith(BLOCK_PREFIX)
        yield name, data
    if has_blocks:
        _require_entry_blocks(manifest)
    references = _signature_references(module)
    for name in sorted(paths.keys() | documents.keys() | references.keys()):
        if name != "unified_llvm.json":
            yield name, None if name in references else read(name)


def _require_entry_blocks(manifest):
    from taichi_forge_runtime.aot._entry_blocks import RUNTIME_REQUIREMENT

    manifest["runtime_requirements"] = sorted(set(manifest.get("runtime_requirements", ())) | {RUNTIME_REQUIREMENT})


class _CompressedMembers:
    """Transfer local ZIP records from a verified Forge archive, in new link order."""

    def __init__(self, baseline):
        self.source = baseline
        self.spans = {}
        if baseline is not None:
            end = baseline.start_dir
            for info in sorted(baseline.infolist(), key=lambda item: item.header_offset, reverse=True):
                self.spans[info.filename] = (info, end - info.header_offset)
                end = info.header_offset

    def copy_to(self, target, name):
        member = self.spans.get(name)
        if member is None:
            return False
        info, remaining = member
        self.source.fp.seek(info.header_offset)
        # zipfile has no public precompressed-member writer. Keep its central
        # directory and ZIP64 bookkeeping; only the local record is identical.
        moved = copy(info)
        moved.header_offset = target.start_dir
        target.fp.seek(target.start_dir)
        while remaining:
            chunk = self.source.fp.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise EOFError(f"Incomplete cached ZIP member {name}")
            target.fp.write(chunk)
            remaining -= len(chunk)
        target.start_dir = target.fp.tell()
        target.filelist.append(moved)
        target.NameToInfo[name] = moved
        return True


def _collect(module, archive, baseline, manifest, documents):
    rows = json.loads((module / "unified_llvm.json").read_bytes())
    roots = {row["key"]: row["value"] for row in rows}
    if not roots or len(roots) != len(rows):
        raise ValueError("Invalid unified LLVM entry table")
    members = {}
    written = set()
    for relative, data in _module_members(module, manifest, documents):
        digest = Path(relative).stem if data is None else hashlib.sha256(data).hexdigest()
        physical = "records/" + digest
        members[relative] = physical
        if physical not in written:
            if not baseline.copy_to(archive, physical):
                if data is None:
                    raise ValueError(f"Missing referenced unified LLVM signature: {digest}")
                _write_member(archive, physical, data)
            written.add(physical)
    return {"entries": roots, "members": members}


def _write_units(archive, directory, baseline):
    with (directory / "units.bin").open("rb") as stream, (directory / "units.index").open(encoding="ascii") as index:
        for row in index:
            digest, *span = row.split()
            member = "code_units/" + digest + ".bc"
            if span == ["baseline"]:
                if not baseline.copy_to(archive, member):
                    raise ValueError(f"Missing referenced unified LLVM unit: {digest}")
                continue
            offset, size = span
            stream.seek(int(offset))
            data = stream.read(int(size))
            if hashlib.sha256(data).hexdigest() != digest or not data.startswith(b"BC\xc0\xde"):
                raise ValueError(f"Invalid unified LLVM unit: {digest}")
            _write_member(archive, member, data)


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


def _pin_baseline(archive, expected, workspace):
    if not isinstance(expected, str):
        return False
    snapshot = workspace / "baseline.tcm"
    try:
        digest = hashlib.sha256()
        with archive.open("rb") as source, snapshot.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest() == expected:
            with zipfile.ZipFile(snapshot) as previous:
                members = [name for name in previous.namelist() if name.startswith(("code_units/", "records/"))]
            (workspace / "bodies" / "baseline-members.index").write_text("\n".join(members), encoding="ascii")
            return True
    except (OSError, zipfile.BadZipFile):
        pass
    (workspace / "bodies" / "baseline-members.index").unlink(missing_ok=True)
    snapshot.unlink(missing_ok=True)
    return False


def _restore_leaf_cache(directory, workspace, cache):
    if cache.object_context is None:
        return
    try:
        record = json.loads((directory / "latest.json").read_bytes())
        if not isinstance(record, dict) or record.get("version") != 3:
            return
        same_producer = record.get("context") == cache.object_context
        if not same_producer and record.get("native_context") != cache.native_context:
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
        if not _pin_baseline(archive, record.get("tcm_sha256"), workspace):
            (bodies / "baseline-path").write_bytes(os.fsencode(archive))
        # A change to the frontend or linker still requires a fresh archive.
        return record if same_producer else None
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


def _record_leaf_cache(cache_dir, workspace, output, archive_digest):
    directory = Path(cache_dir) / "unified"
    receipt = workspace / "portable-statistics.json"
    if not receipt.is_file():
        return
    try:
        record = json.loads(receipt.read_bytes())
        record["output"] = str(output)
        inputs = record.pop("link_inputs", None)
        if inputs is not None:
            record["tcm_sha256"] = archive_digest
            record["link"] = hashlib.sha256(_bytes([inputs, archive_digest, record["entry_blocks"]])).hexdigest()
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
            base = json.loads((workspace / "portable" / "view.json").read_bytes())
            del base["frontend_target"]
            digest = hashlib.sha256((staging / "code.tcm").read_bytes()).hexdigest()
            document = dict(kind="forge_unified_definition_library", version=2, base=encode_manifest(base),
                            targets=targets, target_capabilities=capability_sets, tcm_sha256=digest)
            (staging / "package.json").write_bytes(_bytes(document))
            complete = True
            _publish_package(staging, output)
            if build_cache is not None:
                _record_leaf_cache(build_cache, workspace, output, digest)
            return document
        finally:
            if not complete:
                shutil.rmtree(staging)
