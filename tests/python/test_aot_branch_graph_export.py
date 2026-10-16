"""Freeze selected branch code and bindings through the public Graph exporter."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from tests import test_utils


def branch_factory():
    @ti.kernel
    def seed(
        y: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
        dy: ti.types.ndarray(ti.f32, 1),
        checks: ti.types.ndarray(ti.i32, 0),
        last: ti.types.ndarray(ti.i32, 0),
    ):
        checks[None] = 0
        last[None] = 0
        for i in y:
            y[i] = -91
            dx[i] = 0
            dy[i] = 1

    @ti.kernel
    def condition(
        choice: ti.types.ndarray(ti.i32, 1),
        selected: ti.types.ndarray(ti.i32, 0),
        checks: ti.types.ndarray(ti.i32, 0),
    ):
        selected[None] = choice[0]
        checks[None] += 1

    @ti.kernel
    def forward(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        gain: ti.f32,
    ):
        for i in x:
            y[i] = gain * x[i] * x[i]

    @ti.kernel
    def mark(selected: ti.types.ndarray(ti.i32, 0)):
        selected[None] = -77

    @ti.kernel
    def publish(
        dx: ti.types.ndarray(ti.f32, 1),
        gradient: ti.types.ndarray(ti.f32, 1),
        checks: ti.types.ndarray(ti.i32, 0),
        selected: ti.types.ndarray(ti.i32, 0),
        state: ti.types.ndarray(ti.i32, 1),
    ):
        for i in gradient:
            gradient[i] = dx[i]
        state[0] = checks[None]
        state[1] = selected[None]

    def build(kind, fallback, mode, *, nested=False):
        builder = ti.graph.GraphBuilder(_map_recipe="map2")
        public = {
            n: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.f32, ndim=1) for n in ("x", "y", "dx", "dy", "gradient")
        }
        public.update({n: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.i32, ndim=1) for n in ("choice", "state")})
        selected = builder.private_ndarray("selected", ti.i32, ())
        checks = builder.private_ndarray("checks", ti.i32, ())
        outer = builder.private_ndarray("outer", ti.i32, ()) if nested else None
        parent = builder.create_sequential()
        parent.dispatch(seed, public["y"], public["dx"], public["dy"], checks, selected, label="step/seed")
        cond = builder.create_sequential()
        cond.dispatch(condition, public["choice"], selected, checks, label="step/condition")
        cases = 1 if kind == "if" else 8
        bodies = []
        for i in range(cases + int(fallback)):
            body = builder.create_sequential()
            gain = body._bind_internal_scalar(f"gain{i}", ti.f32, float(i + 1))
            body.dispatch(forward, public["x"], public["y"], gain, label=f"case/{i}/forward")
            body.dispatch(forward.grad, public["x"], public["y"], gain, label=f"case/{i}/reverse")
            body.dispatch(mark, selected, label=f"case/{i}/mark")
            bodies.append(body)
        region = builder.create_sequential() if nested else parent
        if kind == "if":
            region.if_then_else(
                cond,
                bodies[0],
                predicate=selected,
                else_region=bodies[-1] if fallback else None,
                lowering_mode=mode,
            )
        else:
            region.switch(
                cond,
                bodies[:cases],
                selector=selected,
                default_region=bodies[-1] if fallback else None,
                lowering_mode=mode,
            )
        if nested:
            outer_condition = builder.create_sequential()
            outer_condition.dispatch(condition, public["choice"], outer, checks)
            parent.if_then_else(outer_condition, region, predicate=outer, lowering_mode=mode)
        parent.dispatch(
            publish, public["dx"], public["gradient"], checks, selected, public["state"], label="step/publish"
        )
        builder.append(parent)
        return builder.compile()

    return build


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_selected_branch_graph_and_reverse(tmp_path):
    _selected_branch_graph(tmp_path, host=False)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_host_branch_graph_and_reverse(tmp_path):
    _selected_branch_graph(tmp_path, host=True)


def _selected_branch_graph(tmp_path, host):
    arch = (
        "vulkan"
        if ti.lang.impl.current_cfg().arch == ti.vulkan
        else ("cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu")
    )
    build = branch_factory()
    module = ti.aot.Module()
    cases = [
        ("if", "if", False),
        ("if_else", "if", True),
        ("switch", "switch", False),
        ("switch_default", "switch", True),
    ]
    if arch == "cpu" or host:
        cases.append(("nested", "switch", False))
    stats = dict(build_ms=0, export_ms=0, consumer={})
    for name, kind, fallback in cases:
        start = time.perf_counter()
        mode = "portable" if arch == "cuda" and host else "native_required" if arch == "cuda" else "auto"
        graph = build(kind, fallback, mode, nested=name == "nested")
        stats["build_ms"] += (time.perf_counter() - start) * 1000
        # Export cold. JIT Graph deliberately rejects implicit ndarray AD;
        # persistent execution supplies the explicit gradient bindings below.
        # The consumer checks actual kernel.grad results after producer teardown.
        start = time.perf_counter()
        plan = module.add_execution_plan(name, graph, gradients=dict(x="dx", y="dy"))
        stats["export_ms"] += (time.perf_counter() - start) * 1000
        document = json.loads(plan.to_json())
        assert document["version"] == (12 if host else 11)
        assert set(plan.argument_names) == {"x", "y", "dx", "dy", "gradient", "choice", "state"}
        assert any(op["kind"] == "branch" for op in document["operations"])
        count = len(module._exported_kernels)
        module.add_execution_plan(name + "_again", graph, gradients=dict(x="dx", y="dy"))
        assert len(module._exported_kernels) == count
        graph.close()
    artifact = tmp_path / "branch-graph.tcm"
    start = time.perf_counter()
    module.archive(str(artifact))
    stats["archive_ms"] = (time.perf_counter() - start) * 1000
    stats["artifact_bytes"] = artifact.stat().st_size
    with ZipFile(artifact) as archive:
        # Six shared kernels across every branch, default and repeated plan.
        if arch != "vulkan":
            assert len([n for n in archive.namelist() if n.endswith(".bc")]) == 6
        else:
            assert any(n.endswith(".spv") for n in archive.namelist())
        assert not any(n.endswith(".py") for n in archive.namelist())
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_branch_graph_probe.py")),
                mode,
                arch,
                str(artifact),
                *(["host"] if host else []),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        stats["consumer"][mode] = json.loads(result.stdout.splitlines()[-1])
    (tmp_path / "branch-graph-costs.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_branch_export_keeps_unqualified_routes_closed(monkeypatch):
    build = branch_factory()
    for mode in ("auto", "native_required"):
        if mode == "native_required":
            monkeypatch.setenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "1")
        graph = build("if", True, mode)
        module = ti.aot.Module()
        with pytest.raises(ti.TaichiRuntimeError, match="selected .* branch recipe"):
            module.add_execution_plan("invalid", graph, gradients=dict(x="dx", y="dy"))
        assert not module._content and not module._exported_kernels and not module._execution_plans
        assert not module._graph_export_failed
        graph.close()
