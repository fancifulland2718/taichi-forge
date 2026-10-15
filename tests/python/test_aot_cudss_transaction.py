"""Ordinary cuDSS transactions preserve explicit values and completion ownership."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_cudss_config import _problem


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_transaction_export_analysis_only_and_real_gradients(tmp_path):
    matrix, values, _dense, vendor = _problem()
    owners = [ti.hardware.linalg.CudssPlan(matrix, matrix_type="spd", library_path=vendor) for _ in range(2)]
    for owner in owners:
        owner.analyze()
    # These values cannot be factorized. Reconstruction must only analyze them.
    original = values.to_numpy()
    values.fill(0)
    matrix.update_values(values)
    adapter = owners[0].provider_identity["library_candidate"]

    @ti.kernel
    def seed(
        v: ti.types.ndarray(ti.f32, 1),
        live: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
    ):
        for i in v:
            live[i] = v[i]
        for i in dx:
            dx[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            loss[0] += 0.5 * x[i] * x[i]

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("values", "live", "rhs", "x", "dx", "db", "loss", "dloss")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(seed, *(symbols[name] for name in ("values", "live", "dx", "loss", "dloss")))
    builder.append_native(owners[0].record_refactor_solve(values="live", rhs="rhs", solution="x"))
    builder.dispatch(energy, symbols["x"], symbols["loss"])
    builder.dispatch(energy.grad, symbols["x"], symbols["loss"])
    # Independent transaction owner for the symmetric solve adjoint.
    builder.append_native(owners[1].record_refactor_solve(values="live", rhs="dx", solution="db"))
    graph = builder.compile()
    module = ti.aot.Module()

    def export(name):
        plan = module.add_execution_plan(name, graph, gradients=dict(x="dx", loss="dloss"))
        document = json.loads(plan.to_json())
        resources = [item for item in document["resources"] if item["kind"] == "native"]
        assert len(resources) == 2
        assert all(item["operation"] == "transaction_plan_f32" for item in resources)
        assert all(item["parameters"]["reordering"] == "default" for item in resources)
        assert all("phase" not in item["parameters"] for item in resources)
        operations = [item for item in document["operations"] if item["kind"] == "native"]
        assert [item["operation"] for item in operations] == ["refactor_solve_f32"] * 2
        for owner in owners:
            info = owner._program._cuda_cudss_plan_configuration(owner._handle)
            assert info["refactor_seed_snapshot_bytes"] == 100 * info["refactor_seed_snapshot_count"]
        return plan

    export("analysis_only")
    rhs, x = ti.ndarray(ti.f32, 4), ti.ndarray(ti.f32, 4)
    rhs.fill(1)
    values.from_numpy(original)
    owners[0].refactor_solve(values, rhs, x)
    # An in-flight ordinary owner must not be silently cloned or waited on.
    with pytest.raises(RuntimeError, match="in flight"):
        export("inflight")
    ti.sync()
    export("explicit_factors")  # No dependency on the source's explicit factors.
    owners[0]._debug_fail_next_refactor_solve()
    with pytest.raises(RuntimeError, match="Injected"):
        owners[0].refactor_solve(values, rhs, x)
    ti.sync()
    export("after_failure")
    assert len(module._data_payloads) == 3  # Pattern/config seeds shared across owners/exports.
    assert len(module._exported_kernels) == 3
    for owner in owners:
        owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        export("closed")
    graph.close()
    artifact = tmp_path / "cudss-transactions.tcm"
    module.archive(str(artifact))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_cudss_transaction_probe.py")),
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
def test_cudss_transaction_export_validates_analysis_and_resource_policy():
    matrix, _values, _dense, vendor = _problem()
    with ti.hardware.linalg.CudssPlan(matrix, matrix_type="spd", library_path=vendor) as owner:
        snapshot = lambda: owner._program._snapshot_cuda_cudss_solve_seed(owner._handle, matrix.matrix, True)
        with pytest.raises(RuntimeError, match="analysis"):
            snapshot()
        owner.analyze()
        assert snapshot()[4:7] == (-1, 0, 0)
        row, column, alternate_values = ti.ndarray(ti.i32, 5), ti.ndarray(ti.i32, 10), ti.ndarray(ti.f32, 10)
        # Same dimensions and nonzero count do not prove the analyzed pattern.
        row.from_numpy(np.array([0, 2, 5, 8, 10], dtype=np.int32))
        column.from_numpy(np.array([0, 2, 0, 1, 2, 1, 2, 3, 2, 3], dtype=np.int32))
        alternate_values.fill(1)
        alternate = ti.linalg.SparsePattern.csr(4, 4, row, column).matrix(alternate_values)
        with pytest.raises(RuntimeError, match="CSR index changed"):
            owner._program._snapshot_cuda_cudss_solve_seed(owner._handle, alternate.matrix, True)
        owner.factorize()
        # Solves read frozen factors; transactions mutate state using live values.
        # Reusing a cached resource must not silently replace either contract.
        for first in (True, False):
            builder = ti.graph.GraphBuilder()
            recordings = [owner.recording(), owner.record_refactor_solve()]
            for recording in recordings[:: 1 if first else -1]:
                builder.append_native(recording)
            graph = builder.compile()
            with pytest.raises((ValueError, RuntimeError), match="policy changed"):
                ti.aot.Module().add_execution_plan("mixed", graph)
            graph.close()
