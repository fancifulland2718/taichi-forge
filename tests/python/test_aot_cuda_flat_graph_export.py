"""Selected masked Graph bodies export true adjoints and one code closure."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_flat_masked_graph_exports_gated_adjoint_without_unused_code(tmp_path, monkeypatch):
    monkeypatch.setenv("TI_GRAPH_CUDA_FORCE_MASKED_CONTROL", "1")

    @ti.kernel
    def seed(
        y: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
        dy: ti.types.ndarray(ti.f32, 1),
        count: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
    ):
        count[0], checks[0] = 0, 0
        for i in y:
            y[i], dx[i], dy[i] = -99, 0, 1

    @ti.kernel
    def condition(
        count: ti.types.ndarray(ti.i32, 1),
        predicate: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
        limits: ti.types.ndarray(ti.i32, 1),
    ):
        checks[0] += 1
        predicate[0] = count[0] < limits[0]

    @ti.kernel
    def advance(count: ti.types.ndarray(ti.i32, 1), dy: ti.types.ndarray(ti.f32, 1)):
        count[0] += 1
        for i in dy:
            dy[i] = 1

    @ti.kernel
    def forward(x: ti.types.ndarray(ti.f32, 1, needs_grad=True), y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            y[i] = x[i] * x[i]

    @ti.kernel
    def publish(
        count: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
        terminal: ti.types.ndarray(ti.i32, 1),
    ):
        terminal[0], terminal[1] = count[0], checks[0]

    begin = time.perf_counter()
    builder = ti.graph.GraphBuilder()
    args = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y", "dx", "dy")}
    args.update(
        {
            name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
            for name in ("limits", "predicate", "terminal")
        }
    )
    count = builder.private_ndarray("count", ti.i32, (1,))
    checks = builder.private_ndarray("checks", ti.i32, (1,))
    sequence = builder.create_sequential()
    sequence.dispatch(seed, args["y"], args["dx"], args["dy"], count, checks)
    cond = builder.create_sequential()
    cond.dispatch(condition, count, args["predicate"], checks, args["limits"])
    body = builder.create_sequential()
    body.dispatch(advance, count, args["dy"])
    body.dispatch(forward, args["x"], args["y"])
    body.dispatch(forward.grad, args["x"], args["y"])
    sequence.while_loop(
        cond,
        body,
        predicate=args["predicate"],
        counter=count,
        control_inputs=(count, checks, args["limits"]),
        carried_state=(args["y"], args["dy"]),
        max_iterations=3,
        lowering_mode="native_required",
    )
    sequence.dispatch(publish, count, checks, args["terminal"])
    builder.append(sequence)
    graph = builder.compile()
    costs = dict(build_ms=(time.perf_counter() - begin) * 1000)
    module = ti.aot.Module()
    begin = time.perf_counter()
    for name in ("step", "again"):
        plan = module.add_execution_plan(name, graph, gradients=dict(x="dx", y="dy"))
    costs["export_ms"] = (time.perf_counter() - begin) * 1000
    graph.close()
    document = json.loads(plan.to_json())
    assert document["version"] == 15
    loop = next(op for op in document["operations"] if op["kind"] == "while")
    assert loop["lowering"] == dict(kind="cuda_masked", version=1)
    assert loop["condition"] == [] and len(loop["body"]) == 4
    assert len(module._exported_kernels) == 7  # Three ordinary and four gated.
    assert len(module._kernel_launch_abis) == 4
    assert set(module._exported_kernels) == set(plan.kernel_entries)
    assert all(module._kernel_launch_abis[op["entry"]] == "cuda_graph_gate_v1" for op in loop["body"])
    artifact = tmp_path / "graph.tcm"
    begin = time.perf_counter()
    module.archive(str(artifact))
    costs.update(save_ms=(time.perf_counter() - begin) * 1000, artifact_bytes=artifact.stat().st_size)
    with zipfile.ZipFile(artifact) as archive:
        assert len([name for name in archive.namelist() if name.endswith(".bc")]) == 7
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cuda_flat_masked_probe.py")),
                mode,
                str(artifact),
                "3",
                "--graph-export",
            ],
            env=dict(os.environ, TI_GRAPH_CUDA_FORCE_MASKED_CONTROL="0"),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        costs[mode] = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"load_ms"')))
    (tmp_path / "costs.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")
