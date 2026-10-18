"""AOT resources and new JIT fields must use independent tree IDs."""

import os
import json
from pathlib import Path
import subprocess
import sys
import time

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_borrowed_aot_and_later_jit_trees(tmp_path):
    counter = ti.field(ti.i32, shape=4)

    @ti.kernel
    def step(out: ti.types.ndarray(ti.i32, 1)):
        counter[0] += 1
        out[0] = counter[0]

    arch = "cuda" if ti.cfg.arch == ti.cuda else "vulkan" if ti.cfg.arch == ti.vulkan else "cpu"
    started = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(step)
    artifact = tmp_path / "tree_ids.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(
        json.dumps(
            {
                "compile_export_ms": (time.perf_counter() - started) * 1000,
                "artifact_bytes": artifact.stat().st_size,
            }
        )
    )
    ti.reset()
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("aot_tree_id_isolation_probe.py")), arch, str(artifact)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Failed to retire" not in result.stdout + result.stderr
