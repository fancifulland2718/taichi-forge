"""Sparse/list storage is retired before the same module is loaded again."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_sparse_list_close_reload(tmp_path):
    _sparse_module(tmp_path, "bc")


@pytest.mark.parametrize("artifact_format", ["ptx", "cubin"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_sparse_device_artifact_isolation(tmp_path, artifact_format):
    if artifact_format == "cubin" and not (os.environ.get("TI_CUDA_PTXAS_PATH") or shutil.which("ptxas")):
        pytest.skip("External ptxas is required for cubin export")
    _sparse_module(tmp_path, artifact_format)


def _sparse_module(tmp_path, artifact_format):
    # Ensure the producer tree ID differs from the compact artifact ID.
    unrelated = ti.field(ti.i32, shape=1)
    unrelated[0] = 7
    values = ti.field(ti.f32, needs_grad=True)
    blocks = ti.root.pointer(ti.i, 4)
    blocks.bitmasked(ti.i, 2).place(values, values.grad)
    entries = ti.field(ti.f32)
    dynamic = ti.root.dynamic(ti.i, 8, chunk_size=4)
    dynamic.place(entries)

    @ti.kernel
    def activate(scale: ti.f32):
        values[2] = scale
        values[5] = 2 * scale
        values.grad[2] = 0
        values.grad[5] = 0
        ti.append(dynamic, [], scale)
        ti.append(dynamic, [], 3 * scale)

    @ti.kernel
    def gather(out: ti.types.ndarray(ti.f32, 1)):
        for i in values:
            ti.atomic_add(out[0], values[i])
        for i in entries:
            ti.atomic_add(out[1], entries[i])
        out[2] = ti.length(dynamic, [])

    @ti.kernel
    def deactivate():
        for i in range(4):
            ti.deactivate(blocks, i)
        ti.deactivate(dynamic, [])

    @ti.kernel
    def energy(out: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(2):
            loss[0] += 0.5 * out[i] * out[i]

    @ti.kernel
    def sparse_energy(loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in values:
            loss[0] += 0.5 * values[i] * values[i]

    @ti.kernel
    def gather_grad(out: ti.types.ndarray(ti.f32, 1)):
        for i in values:
            out[0] += values.grad[i]

    arch = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    start = time.perf_counter()
    caps = (
        [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
        if artifact_format == "cubin"
        else None
    )
    module = ti.aot.Module(artifact_format=artifact_format, caps=caps)
    for kernel in (activate, gather, deactivate, energy, sparse_energy, gather_grad):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    module.add_kernel(sparse_energy.grad, name="sparse_backward")
    artifact = tmp_path / "sparse.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        metadata = json.loads(archive.read("aot_metadata.json"))
    assert metadata["resource_binding"] == "sparse_v1"
    (tmp_path / "producer.json").write_text(
        json.dumps(
            dict(
                compile_export_ms=(time.perf_counter() - start) * 1000,
                bytes=artifact.stat().st_size,
                format=artifact_format,
            )
        )
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_sparse_retirement_probe.py")),
                arch,
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Failed to retire" not in result.stdout + result.stderr
