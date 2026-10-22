"""Selected portable loop segments retain their explicit reverse dispatches."""

import json
import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_exact_host_while_and_reverse(tmp_path):
    @ti.kernel
    def seed(
        y: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
        dy: ti.types.ndarray(ti.f32, 1),
        state: ti.types.ndarray(ti.i32, 1),
    ):
        state[0] = 0
        state[1] = 0
        for i in y:
            y[i] = -91
            dx[i] = 0
            dy[i] = 1

    @ti.kernel
    def condition(
        limit: ti.types.ndarray(ti.i32, 1), state: ti.types.ndarray(ti.i32, 1), predicate: ti.types.ndarray(ti.i32, 0)
    ):
        predicate[None] = int(state[0] < limit[0])
        state[1] += 1

    @ti.kernel
    def forward(x: ti.types.ndarray(ti.f32, 1, needs_grad=True), y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            y[i] = x[i] * x[i]

    @ti.kernel
    def seed_adjoint(dy: ti.types.ndarray(ti.f32, 1)):
        # Reverse consumes the output adjoint; each iteration differentiates
        # its own forward evaluation and accumulates into dx.
        for i in dy:
            dy[i] = 1

    @ti.kernel
    def advance(state: ti.types.ndarray(ti.i32, 1)):
        state[0] += 1

    arch = "cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu"
    module = ti.aot.Module()
    for budget in (0, 3):
        builder = ti.graph.GraphBuilder()
        args = {n: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.f32, ndim=1) for n in ("x", "y", "dx", "dy")}
        args.update({n: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.i32, ndim=1) for n in ("limit", "state")})
        predicate = builder.private_ndarray("p", ti.i32, ())
        builder.dispatch(seed, args["y"], args["dx"], args["dy"], args["state"])
        cond, body = builder.create_sequential(), builder.create_sequential()
        cond.dispatch(condition, args["limit"], args["state"], predicate)
        body.dispatch(forward, args["x"], args["y"])
        body.dispatch(seed_adjoint, args["dy"])
        body.dispatch(forward.grad, args["x"], args["y"])
        body.dispatch(advance, args["state"])
        builder.while_loop(cond, body, predicate=predicate, max_iterations=budget, lowering_mode="portable")
        graph = builder.compile()
        plan = module.add_execution_plan(f"loop{budget}", graph, gradients=dict(x="dx", y="dy") if budget else None)
        doc = json.loads(plan.to_json())
        if budget and arch == "cuda":
            assert doc["version"] == 26
            assert doc["operations"][-1]["lowering"] == dict(kind="cuda_host", version=1)
        graph.close()
    artifact = tmp_path / "host-while.tcm"
    module.archive(str(artifact))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_host_while_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, result.stdout + result.stderr
