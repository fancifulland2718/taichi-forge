"""Cold/warm mixed sort export preserves native order and downstream AD."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.algorithms import _algorithms as algorithms
from tests import test_utils


@pytest.mark.parametrize("field_keys", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_mixed_sort_export_cold_warm_and_downstream_ad(tmp_path, field_keys, monkeypatch):
    n = 17
    fixed, mixed = ti.field(ti.i32, n), ti.ndarray(ti.i32, n)
    x = ti.field(ti.f32, n, needs_grad=True)
    loss = ti.field(ti.f32, (), needs_grad=True)

    @ti.kernel
    def seed(
        source_keys: ti.types.ndarray(ti.i32, 1),
        source: ti.types.ndarray(ti.f32, 1),
        mixed: ti.types.ndarray(ti.i32, 1),
    ):
        for i in range(n):
            if ti.static(field_keys):
                fixed[i], mixed[i] = source_keys[i], i
            else:
                mixed[i], fixed[i] = source_keys[i], i
            x[i], x.grad[i] = source[i], 0
        loss[None], loss.grad[None] = 0, 1

    @ti.kernel
    def energy(mixed: ti.types.ndarray(ti.i32, 1)):
        for i in range(n):
            j = fixed[i]
            if ti.static(field_keys):
                j = mixed[i]
            loss[None] += (i + 1) * x[j] ** 2

    @ti.kernel
    def report(
        mixed: ti.types.ndarray(ti.i32, 1),
        out_keys: ti.types.ndarray(ti.i32, 1),
        out_ids: ti.types.ndarray(ti.i32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
        out_loss: ti.types.ndarray(ti.f32, 1),
    ):
        for i in range(n):
            if ti.static(field_keys):
                out_keys[i], out_ids[i] = fixed[i], mixed[i]
            else:
                out_keys[i], out_ids[i] = mixed[i], fixed[i]
            dx[i] = x.grad[i]
        out_loss[0] = loss[None]

    dtypes = dict(
        source_keys=ti.i32, source=ti.f32, mixed=ti.i32, out_keys=ti.i32, out_ids=ti.i32, dx=ti.f32, out_loss=ti.f32
    )
    args = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1) for name, dtype in dtypes.items()}
    keys, values = (fixed, mixed) if field_keys else (mixed, fixed)
    sequence = ti.algorithms.primitive_sequence().sort_by_key([keys], values)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(seed, args["source_keys"], args["source"], args["mixed"])
    builder.append_native(sequence)
    builder.dispatch(energy, args["mixed"])
    builder.dispatch(energy.grad, args["mixed"])
    builder.dispatch(report, *(args[name] for name in ("mixed", "out_keys", "out_ids", "dx", "out_loss")))
    graph = builder.compile()
    module = ti.aot.Module()
    fixed.fill(123)
    mixed.fill(456)

    def no_host(*args, **kwargs):
        raise AssertionError("Export or replay changed to host sorting")

    monkeypatch.setattr(algorithms, "_host_stable_sort", no_host)
    cold = module.add_execution_plan("cold", graph, source_bindings={"mixed": mixed}, snapshot_state=True)
    assert [op["kind"] for op in json.loads(cold.to_json())["operations"]] == [
        "kernel",
        "sort",
        "kernel",
        "kernel",
        "kernel",
    ]
    assert sequence._calls[0].workspace._prepared_storage_sort_plan is None
    np.testing.assert_array_equal(fixed.to_numpy(), [123] * n)
    np.testing.assert_array_equal(mixed.to_numpy(), [456] * n)
    arrays = {name: ti.ndarray(dtype, 1 if name == "out_loss" else n) for name, dtype in dtypes.items()}
    arrays["mixed"] = mixed
    source_keys = np.array([(i * 17) % 7 - 3 for i in range(n)], np.int32)
    source = np.arange(n, dtype=np.float32) / 8 - 1
    arrays["source_keys"].from_numpy(source_keys)
    arrays["source"].from_numpy(source)
    graph.run(arrays)
    expected = np.argsort(source_keys, kind="stable")
    gradient = np.zeros(n, np.float32)
    gradient[expected] = 2 * np.arange(1, n + 1) * source[expected]
    np.testing.assert_array_equal(arrays["out_ids"].to_numpy(), expected)
    np.testing.assert_array_equal(arrays["dx"].to_numpy(), gradient)
    plan = sequence._calls[0].workspace._prepared_storage_sort_plan
    # A module has one initial snapshot per Field tree. Restore that state
    # before adding the warmed recipe, while retaining its prepared bindings.
    fixed.fill(123)
    x.fill(0)
    x.grad.fill(0)
    loss.fill(0)
    loss.grad.fill(0)
    module.add_execution_plan("warm", graph, source_bindings={"mixed": mixed}, snapshot_state=True)
    assert sequence._calls[0].workspace._prepared_storage_sort_plan is plan
    assert len(module._exported_kernels) == 4  # Sort adds no helper code.
    artifact = tmp_path / "mixed-sort.tcm"
    module.archive(str(artifact))
    graph.close()
    sequence.clear()
    assert plan.report()["closed"]
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_mixed_sort_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
        assert not any(
            marker in result.stdout + result.stderr for marker in ("VUID-", "Validation Error", "SYNC-HAZARD")
        ), (result.stdout + result.stderr)
