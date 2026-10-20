"""Qualify synchronous and actually lowered async BLS in independent AOT owners."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("tile", [16, 2048], ids=["synchronous", "async"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_bls_primal_gradient_and_sparse_owner(tmp_path, tile):
    capability = ti.lang.impl.get_cuda_compute_capability()
    if tile == 2048 and capability < 80:
        pytest.skip("async BLS requires CUDA compute capability 8.0 or newer")
    count = tile * 2
    x, y = ti.field(ti.f32, needs_grad=True), ti.field(ti.f32, needs_grad=True)
    ti.root.pointer(ti.i, 2).dense(ti.i, tile).place(x, y, x.grad, y.grad)

    @ti.kernel
    def initialize(scale: ti.f32):
        for i in range(count):
            x[i] = (i % 23 - 11) * scale
            y[i] = -99
            x.grad[i] = 0
            y.grad[i] = 1

    @ti.kernel
    def compute():
        ti.loop_config(block_dim=min(128, tile))
        ti.block_local(x)
        for i in x:
            y[i] = x[i] * x[i] + 0.5 * x[i]

    @ti.kernel
    def gather(out: ti.types.ndarray(ti.f32, 2)):
        for i in range(count):
            out[i, 0] = y[i]
            out[i, 1] = x.grad[i]

    # Prove this derivative is already supported by the source DSL.
    initialize(1)
    compute()
    compute.grad()
    expected = np.arange(count) % 23 - 11
    np.testing.assert_array_equal(y.to_numpy(), expected * expected + 0.5 * expected)
    np.testing.assert_array_equal(x.grad.to_numpy(), 2 * expected + 0.5)
    program = ti.lang.impl.get_runtime().prog
    before = dict(program._cuda_async_tile_status())
    start = time.perf_counter()
    module = ti.aot.Module(
        caps=[ti.DeviceCapability.cuda_compute_capability(capability)]
    )
    for kernel in (initialize, compute, gather):
        module.add_kernel(kernel)
    module.add_kernel(compute.grad, name="backward")
    module.add_execution_plan(
        "bls",
        ti.aot.ExecutionPlan(
            [
                dict(kind="kernel", entry="initialize", arguments=["scale"]),
                dict(kind="kernel", entry="compute", arguments=[]),
                dict(kind="kernel", entry="backward", arguments=[]),
                dict(kind="kernel", entry="gather", arguments=["out"]),
            ]
        ),
    )
    artifact = tmp_path / "bls.tcm"
    module.archive(str(artifact))
    export_ms = (time.perf_counter() - start) * 1000
    after = dict(program._cuda_async_tile_status())
    lowered = after["lowered_specializations"] - before["lowered_specializations"]
    assert lowered > 0 if tile == 2048 else lowered == 0
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_bls_probe.py")),
                mode,
                str(artifact),
                str(count),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Failed to retire" not in result.stdout + result.stderr
    (tmp_path / "producer.json").write_text(
        json.dumps(
            dict(
                count=count,
                export_ms=export_ms,
                artifact_bytes=artifact.stat().st_size,
                before=before,
                after=after,
                aot_async_specializations=lowered,
            )
        ),
        encoding="utf-8",
    )
