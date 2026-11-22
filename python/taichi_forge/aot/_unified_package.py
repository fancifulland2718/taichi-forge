"""Produce one LLVM body library from the engine's device-dependent declarations."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import warnings
import zipfile


def _bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()


class _LeafCache:
    """Reuse typed, tree-independent bodies even when the application changes."""

    def __init__(self, directory, workspace):
        from taichi_forge.lang._build_cache import value_digest
        from ._package_build_cache import _producer_inputs, _UntrackedInput

        self.directory = Path(directory) / "unified"
        self.toolchain = None
        self.files = {}
        try:
            previous = json.loads((self.directory / "latest.json").read_bytes())
        except (OSError, ValueError):
            previous = {}
        try:
            self.toolchain = value_digest(_producer_inputs(previous.get("toolchain_files", {}), self.files))
            if previous.get("toolchain") != self.toolchain or previous.get("version") != 2:
                return
            digest = previous["catalog"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                return
            catalog = (self.directory / (digest + ".tcb")).read_bytes()
            archive = Path(previous["output"]) / "code.tcm"
            if hashlib.sha256(catalog).hexdigest() != digest or not zipfile.is_zipfile(archive):
                return
            bodies = workspace / "bodies"
            bodies.mkdir()
            (bodies / "roots.tcb").write_bytes(catalog)
            # The worker hashes individual referenced bodies while reading the
            # previous archive. Missing or altered optional bodies recompile.
            (bodies / "baseline-path").write_bytes(os.fsencode(archive))
        except (_UntrackedInput, OSError, ValueError, TypeError, KeyError):
            return

    def record(self, workspace, output):
        if self.toolchain is None:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            catalog = (workspace / "bodies" / "roots.tcb").read_bytes()
            digest = hashlib.sha256(catalog).hexdigest()
            stats = json.loads((workspace / "portable-statistics.json").read_bytes())
            record = dict(version=2, toolchain=self.toolchain, toolchain_files=self.files,
                          catalog=digest, output=str(output), produce=stats)
            declarations = workspace / "portable-declarations.json"
            target_cache = self.directory.parent / "portable" / "declarations"
            target_cache.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target_cache, delete=False) as stream:
                temp = Path(stream.name)
                try:
                    with declarations.open("rb") as reader:
                        shutil.copyfileobj(reader, stream)
                    stream.close()
                    os.replace(temp, target_cache / "latest.json")
                finally:
                    temp.unlink(missing_ok=True)
            for name, payload in ((digest + ".tcb", catalog), ("latest.json", _bytes(record))):
                with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as stream:
                    temp = Path(stream.name)
                    try:
                        stream.write(payload)
                        stream.close()
                        os.replace(temp, self.directory / name)
                    finally:
                        temp.unlink(missing_ok=True)
        except OSError as error:
            warnings.warn(f"Unified package published; optional producer cache update failed: {error}",
                          RuntimeWarning, stacklevel=2)
