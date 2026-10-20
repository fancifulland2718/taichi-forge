"""Capability-first CUDA LDLT selection and reconstruction of the chosen factors."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core
from tests import test_utils
from tests.python.test_cuda_sparse_solver_core import _matrix


def _unexpected(*_args, **_kwargs):
    raise AssertionError("LDLT must not use a Cholesky fallback or its cost comparison")


@pytest.mark.parametrize("failure", ["missing", "plan", "driver"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_ldlt_auto_failure_preserves_unresolved_state(failure, monkeypatch):
    matrix = _matrix([[2, 3, 0], [3, 2, 0], [0, 0, -2]])
    solver = ti.linalg.SparseSolver(solver_type="LDLT")
    monkeypatch.setattr(core, "make_cusparse_solver", _unexpected)
    monkeypatch.setattr(solver, "_evaluate_cudss_admission", _unexpected)
    probes = []

    def probe(*args, **kwargs):
        probes.append((args, kwargs))
        return SimpleNamespace(
            operations=[
                SimpleNamespace(
                    descriptor=SimpleNamespace(operation_id="linalg.solve.cudss"),
                    discovery="missing" if failure == "missing" else "available",
                    unavailable_reason="external_library_not_found",
                )
            ]
        )

    def plan_failure(*_args, **_kwargs):
        raise RuntimeError("test plan allocation failure")

    monkeypatch.setattr(ti.hardware, "probe", probe)
    monkeypatch.setattr(solver, "_make_cudss_plan", plan_failure)
    if failure == "driver":
        monkeypatch.setattr(core, "cuda_driver_api_version", lambda: 11000)
    reason = {
        "missing": "external_library_not_found",
        "plan": "cudss_plan_creation_failed:RuntimeError",
        "driver": "cudss_cuda_driver_too_old",
    }[failure]
    for _ in range(2):
        with pytest.raises(RuntimeError, match="CUDA LDLT requires") as error:
            solver.compute(matrix)
        assert reason in str(error.value)
        assert solver.selected_provider == "unresolved"
        assert solver.solver is None and not solver.info()
        assert solver.provider_status()["admission"] == dict(
            admitted=False, route="unavailable", reason=reason
        )
    assert len(probes) == (0 if failure == "driver" else 2)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cuda_ldlt_auto_export_and_explicit_adjoint(tmp_path, monkeypatch):
    vendor = os.environ.get("TI_CUDSS_TEST_LIBRARY")
    if not vendor:
        pytest.skip("set TI_CUDSS_TEST_LIBRARY to the user-managed cuDSS library")
    dense = np.array([[2, 3, 0], [3, 2, 0], [0, 0, -2]], dtype=np.float32)
    matrix = _matrix(dense)
    solver = ti.linalg.SparseSolver(solver_type="LDLT", library_path=vendor)
    monkeypatch.setattr(core, "make_cusparse_solver", _unexpected)
    monkeypatch.setattr(solver, "_evaluate_cudss_admission", _unexpected)
    real_probe = ti.hardware.probe
    probes = []

    def repaired_probe(*args, **kwargs):
        probes.append((args, kwargs))
        if len(probes) == 1:
            return SimpleNamespace(
                operations=[
                    SimpleNamespace(
                        descriptor=SimpleNamespace(operation_id="linalg.solve.cudss"),
                        discovery="missing",
                        unavailable_reason="external_library_not_found",
                    )
                ]
            )
        return real_probe(*args, **kwargs)

    monkeypatch.setattr(ti.hardware, "probe", repaired_probe)
    with pytest.raises(RuntimeError, match="external_library_not_found"):
        solver.compute(matrix)
    solver.compute(matrix)
    assert solver.selected_provider == "cudss" and solver.info()
    assert solver.provider_status()["fallback_reason"] is None
    assert (
        solver.provider_status()["admission"]["reason"]
        == "required_factorization_capability"
    )
    solver.compute(matrix)
    for entries in ([1, 2, 3], [-1, 3, 0.5]):
        rhs = ti.ndarray(ti.f32, 3)
        rhs.from_numpy(np.array(entries, dtype=np.float32))
        result = solver.solve(rhs).to_numpy()
        np.testing.assert_allclose(
            result, np.linalg.solve(dense, entries), rtol=2e-5, atol=1e-6
        )
        np.testing.assert_allclose(dense @ result, entries, rtol=2e-5, atol=1e-6)
    assert len(probes) == 2  # No discovery on repeated factorization or warm solves.
    adapter = solver.solver.provider_identity["library_candidate"]

    @ti.kernel
    def seed(
        dx: ti.types.ndarray(ti.f32, 1),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
    ):
        for i in dx:
            dx[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in x:
            loss[0] += 0.5 * x[i] * x[i]

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("rhs", "x", "dx", "db", "loss", "dloss")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(seed, symbols["dx"], symbols["loss"], symbols["dloss"])
    builder.append_native(solver.record_solve(rhs="rhs", solution="x"))
    builder.dispatch(energy, symbols["x"], symbols["loss"])
    builder.dispatch(energy.grad, symbols["x"], symbols["loss"])
    builder.append_native(solver.record_solve(rhs="dx", solution="db"))
    graph = builder.compile()
    x = ti.ndarray(ti.f32, 3, needs_grad=True)
    loss = ti.ndarray(ti.f32, 1, needs_grad=True)
    forward_builder = ti.graph.GraphBuilder()
    forward_builder.append_native(solver.record_solve(rhs="rhs", solution="x"))
    forward_graph = forward_builder.compile()
    forward_graph.run(dict(rhs=rhs, x=x))
    # Python JIT Graphs do not bind ndarray gradients. Check JIT differentiation
    # with direct kernels, and the complete graph with explicit AOT bindings below.
    seed(x.grad, loss, loss.grad)
    energy(x, loss)
    energy.grad(x, loss)
    expected = np.linalg.solve(dense, entries)
    np.testing.assert_allclose(x.to_numpy(), expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(x.grad.to_numpy(), expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(
        solver.solve(x.grad).to_numpy(),
        np.linalg.solve(dense, expected),
        rtol=2e-5,
        atol=1e-6,
    )
    module = ti.aot.Module()
    plan = module.add_execution_plan(
        "ldlt", graph, gradients=dict(x="dx", loss="dloss")
    )
    resources = [
        r for r in json.loads(plan.to_json())["resources"] if r["kind"] == "native"
    ]
    assert len(resources) == 1
    assert resources[0]["provider"] == "cudss"
    assert resources[0]["parameters"]["matrix_type"] == "symmetric"
    artifact = tmp_path / "ldlt.tcm"
    module.archive(str(artifact))
    solver.solver.close()
    forward_graph.close()
    graph.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("aot_sparse_ldlt_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
    (tmp_path / "provider.json").write_text(
        json.dumps(dict(adapter=adapter, vendor=vendor))
    )
