"""Named module layouts exist without executable entries or helper kernels."""

import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import taichi_forge as ti
from taichi_forge._lib import utils
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_named_structure_layout_without_kernel(tmp_path):
    record = ti.types.struct(tag=ti.i32, state=ti.types.struct(value=ti.f32, v=ti.math.vec3))
    module = ti.aot.Module()
    module.add_ndarray_layout("record", record)
    module.add_ndarray_layout("alias", record)
    module.add_ndarray_layout("record", record)
    assert len(module._content) == 2
    with pytest.raises((ValueError, RuntimeError), match="Conflicting"):
        module.add_ndarray_layout("record", ti.types.struct(value=ti.i32))
    with pytest.raises(TypeError):
        module.add_ndarray_layout("scalar", ti.f32)
    with pytest.raises(ValueError):
        module.add_ndarray_layout("", record)
    assert not module._exported_kernels
    artifact = tmp_path / "layout.tcm"
    module.archive(str(artifact))
    with ZipFile(artifact) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    assert not any(name.endswith((".bc", ".ptx", ".cubin", ".spv")) for name in contents)
    metadata = json.loads(contents["array_layouts.json"])
    assert len(metadata["types"]) == 1
    names = {item["key"]: item["value"] for item in metadata["layouts"]}
    assert names["record"] == names["alias"]
    metadata["version"] = 100
    with ZipFile(tmp_path / "invalid.tcm", "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in contents.items():
            archive.writestr(name, json.dumps(metadata) if name == "array_layouts.json" else payload)
    original = contents["array_layouts.json"]
    for suffix, invalid in (
        ("trailing", original + b"{}"),
        ("nul", original + b"\0"),
        ("duplicate", b'{"version":1,' + original[1:]),
    ):
        with ZipFile(tmp_path / f"invalid-{suffix}.tcm", "w", compression=ZIP_DEFLATED) as archive:
            for name, payload in contents.items():
                archive.writestr(name, invalid if name == "array_layouts.json" else payload)
    # Re-saving a different generation must remove stale layout metadata.
    module.save(tmp_path / "directory")
    ti.aot.Module().save(tmp_path / "directory")
    assert not (tmp_path / "directory/array_layouts.json").exists()
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join([str(root / "packaging/runtime/src"), *sys.path]),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
    )
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("aot_named_layout_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "Validation Error" not in output and "SYNC-HAZARD" not in output, output
