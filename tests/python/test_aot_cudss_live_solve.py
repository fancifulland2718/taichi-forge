"""Ordinary cuDSS solve exports only factors matching the current matrix."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.hardware._cudss_config import configured_plan
from tests import test_utils
from tests.python.test_cudss_config import _problem


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_live_cudss_solve_export_freezes_current_factors_and_gradients(tmp_path):
    matrix, values, dense, vendor = _problem()
    owner = configured_plan(
        matrix,
        {"version": 1, "reordering": "amd", "solve": "general"},
        matrix_type="spd",
        matrix_view="full",
        library_path=vendor,
    )
    owner.compute()
    adapter = owner.provider_identity["library_candidate"]

    @ti.kernel
    def seed(dx: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in dx:
            dx[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            loss[0] += 0.5 * x[i] * x[i]

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("rhs", "x", "dx", "db", "loss", "dloss")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(seed, symbols["dx"], symbols["loss"], symbols["dloss"])
    builder.append_native(owner.recording(rhs="rhs", solution="x"))
    builder.dispatch(energy, symbols["x"], symbols["loss"])
    builder.dispatch(energy.grad, symbols["x"], symbols["loss"])
    # The explicit solve adjoint is valid for this symmetric matrix only.
    builder.append_native(owner.recording(rhs="dx", solution="db"))
    graph = builder.compile()
    module = ti.aot.Module()

    def export(name):
        before = owner._program._cuda_cudss_plan_configuration(owner._handle)
        plan = module.add_execution_plan(name, graph, gradients=dict(x="dx", loss="dloss"))
        after = owner._program._cuda_cudss_plan_configuration(owner._handle)
        assert after["solve_seed_snapshot_count"] - before["solve_seed_snapshot_count"] == 1
        assert after["solve_seed_snapshot_bytes"] - before["solve_seed_snapshot_bytes"] == 100
        document = json.loads(plan.to_json())
        resources = [item for item in document["resources"] if item["kind"] == "native"]
        assert len(resources) == 1
        assert resources[0]["parameters"] == dict(
            rows=4, nonzeros=10, matrix_type="spd", matrix_view="full", phase="solve", reordering="amd", solve="general"
        )
        operations = [item for item in document["operations"] if item["kind"] == "native"]
        assert len(operations) == 2
        assert operations[0]["bindings"]["plan"] == operations[1]["bindings"]["plan"]
        return plan.to_json()

    frozen = export("original")
    values.from_numpy(values.to_numpy() * 2)
    matrix.update_values(values)
    with pytest.raises(RuntimeError, match="stale"):
        export("stale")
    owner.factorize()
    # Display fields cannot change the native reconstruction policy.
    owner.matrix_type, owner.matrix_view = "general", "upper"
    export("updated")
    assert module._execution_plans["original"].to_json() == frozen
    assert len(module._data_payloads) == 4  # Shared row/column data, two values versions.
    assert len(module._exported_kernels) == 3

    rhs, solution = ti.ndarray(ti.f32, 4), ti.ndarray(ti.f32, 4)
    rhs.from_numpy(np.arange(1, 5, dtype=np.float32))
    owner.solve(rhs, solution)
    np.testing.assert_allclose(2 * dense @ solution.to_numpy(), np.arange(1, 5), rtol=2e-5)
    program, handle = owner._program, owner._handle
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        module.add_execution_plan("closed", graph, gradients=dict(x="dx", loss="dloss"))
    with pytest.raises(RuntimeError, match="stale|closed"):
        program._snapshot_cuda_cudss_solve_seed(handle, matrix.matrix)
    graph.close()
    artifact = tmp_path / "live-cudss-solve.tcm"
    module.archive(str(artifact))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_cudss_live_solve_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_live_cudss_solve_snapshot_rejects_unqualified_factor_state():
    matrix, values, _dense, vendor = _problem()
    with ti.hardware.linalg.CudssPlan(matrix, matrix_type="spd", library_path=vendor) as owner:
        snapshot = lambda m=matrix: owner._program._snapshot_cuda_cudss_solve_seed(owner._handle, m.matrix)
        with pytest.raises(RuntimeError, match="factorization"):
            snapshot()
        owner.compute()
        other, _values, _dense, _vendor = _problem()
        with pytest.raises(RuntimeError, match="stale"):
            snapshot(other)
        rhs, solution = ti.ndarray(ti.f32, 4), ti.ndarray(ti.f32, 4)
        rhs.fill(1)
        owner.refactor_solve(values, rhs, solution)
        ti.sync()
        with pytest.raises(RuntimeError, match="explicit"):
            snapshot()
        owner.factorize()
        assert len(snapshot()[-1]) == 40
        # Transactional refactor remains unqualified; do not silently freeze it.
        builder = ti.graph.GraphBuilder()
        builder.append_native(owner.record_refactor_solve())
        graph = builder.compile()
        with pytest.raises(RuntimeError, match="unavailable"):
            ti.aot.Module().add_execution_plan("transactional", graph)
        graph.close()
