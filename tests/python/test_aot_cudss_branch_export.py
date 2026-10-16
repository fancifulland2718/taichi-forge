"""Selected provider branches retain resources and explicit reverse solves."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from taichi_forge.hardware._cudss_capture import CudssCaptureRecording
from tests import test_utils
from tests.python.test_graph_cudss_capture import _owner


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_masked_branch_rejects_provider_before_export(monkeypatch):
    from tests.python.test_graph_cudss_branch import build_branch

    monkeypatch.setenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "1")
    owner, _, _ = _owner()
    graph = build_branch(owner, "if")
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="flat physical kernel body"):
        module.add_execution_plan("invalid", graph)
    assert not module._content and not module._exported_kernels and not module._execution_plans
    assert not module._graph_export_failed
    graph.close()
    del graph
    gc.collect()
    owner.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_branch_export_and_reverse(tmp_path, monkeypatch):
    monkeypatch.delenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", raising=False)
    owner, _, _ = _owner()
    adapter, vendor = owner.provider_identity["library_candidate"], owner.library_path

    @ti.kernel
    def seed(
        x: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
        gradient: ti.types.ndarray(ti.f32, 1),
        checks: ti.types.ndarray(ti.i32, 0),
    ):
        checks[None] = 0
        for i in x:
            x[i], dx[i], loss[i], dloss[i], gradient[i] = -71, 0, -91, 1, 0

    @ti.kernel
    def condition(
        choice: ti.types.ndarray(ti.i32, 1),
        selected: ti.types.ndarray(ti.i32, 0),
        checks: ti.types.ndarray(ti.i32, 0),
    ):
        selected[None] = choice[0]
        checks[None] += 1

    @ti.kernel
    def energy(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in x:
            loss[i] = x[i] * x[i]

    @ti.kernel
    def mark(selected: ti.types.ndarray(ti.i32, 0)):
        selected[None] = -77

    @ti.kernel
    def publish(
        selected: ti.types.ndarray(ti.i32, 0),
        checks: ti.types.ndarray(ti.i32, 0),
        state: ti.types.ndarray(ti.i32, 1),
    ):
        state[0], state[1] = selected[None], checks[None]

    stats = dict(build_ms=0, export_ms=0, consumer={})
    module = ti.aot.Module()
    for kind in ("if", "switch"):
        start = time.perf_counter()
        builder = ti.graph.GraphBuilder()
        symbols = {
            name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
            for name in ("x", "dx", "loss", "dloss", "gradient")
        }
        choice = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "choice", ti.i32, ndim=1)
        state = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.i32, ndim=1)
        selected = builder.private_ndarray("selected", ti.i32, ())
        checks = builder.private_ndarray("checks", ti.i32, ())
        parent, cond, first, second, default = (builder.create_sequential() for _ in range(5))
        parent.dispatch(seed, *[symbols[n] for n in ("x", "dx", "loss", "dloss", "gradient")], checks)
        cond.dispatch(condition, choice, selected, checks)
        for index, body in enumerate((first, second)):
            body.append_native(
                CudssCaptureRecording(
                    owner, phase="refactor_solve", values=f"values{index}", rhs="rhs", solution="x"
                )
            )
            body.dispatch(energy, symbols["x"], symbols["loss"])
            body.dispatch(energy.grad, symbols["x"], symbols["loss"])
            # The fixed SPD matrix makes the transpose solve identical.
            body.append_native(
                CudssCaptureRecording(
                    owner, phase="refactor_solve", values=f"values{index}", rhs="dx", solution="gradient"
                )
            )
            body.dispatch(mark, selected)
        default.dispatch(mark, selected)
        if kind == "if":
            parent.if_then_else(
                cond, first, predicate=selected, else_region=second, lowering_mode="native_required"
            )
        else:
            parent.switch(
                cond,
                [first, second],
                selector=selected,
                default_region=default,
                lowering_mode="native_required",
            )
        parent.dispatch(publish, selected, checks, state)
        builder.append(parent)
        graph = builder.compile()
        stats["build_ms"] += (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        plan = module.add_execution_plan(kind, graph, gradients=dict(x="dx", loss="dloss"))
        stats["export_ms"] += (time.perf_counter() - start) * 1000
        document = json.loads(plan.to_json())
        assert document["version"] == 11
        branch = next(op for op in document["operations"] if op["kind"] == "branch")
        for body in branch["bodies"][:2]:
            assert [op["kind"] for op in body] == ["native", "kernel", "kernel", "native", "kernel"]
        # All four captures use the same reconstructed solver, not four copies.
        assert len([r for r in document["resources"] if r.get("provider") == "cudss"]) == 1
        graph.close()
        del graph, builder, parent, cond, first, second, default, body
    artifact = tmp_path / "cudss-branches.tcm"
    start = time.perf_counter()
    module.archive(str(artifact))
    stats["archive_ms"] = (time.perf_counter() - start) * 1000
    stats["artifact_bytes"] = artifact.stat().st_size
    gc.collect()
    owner.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cudss_branch_probe.py")),
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
        stats["consumer"][mode] = json.loads(result.stdout.splitlines()[-1])
    (tmp_path / "cudss-branch-costs.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
