"""Sparse/list storage is retired before the same module is loaded again."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_sparse_list_close_reload(tmp_path):
    # Legacy sparse artifacts retain producer tree IDs. Leave ID 0 for the
    # borrowed Program's default root; relocation is a separate contract.
    unrelated = ti.field(ti.i32, shape=1)
    unrelated[0] = 7
    values = ti.field(ti.f32)
    blocks = ti.root.pointer(ti.i, 4)
    blocks.bitmasked(ti.i, 2).place(values)
    entries = ti.field(ti.f32)
    dynamic = ti.root.dynamic(ti.i, 8, chunk_size=4)
    dynamic.place(entries)

    @ti.kernel
    def activate(scale: ti.f32):
        values[2] = scale
        values[5] = 2 * scale
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

    arch = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    start = time.perf_counter()
    module = ti.aot.Module(artifact_format="ptx" if arch == "cuda" else "bc")
    for kernel in (activate, gather, deactivate, energy):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    artifact = tmp_path / "sparse.tcm"
    module.archive(str(artifact))
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(compile_export_ms=(time.perf_counter() - start) * 1000, bytes=artifact.stat().st_size))
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
