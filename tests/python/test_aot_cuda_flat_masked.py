"""Flat masked plans use precompiled gates and explicit ndarray adjoints."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan, Scalar
from tests import test_utils


def _kernel(entry, *arguments):
    return dict(kind="kernel", entry=entry, arguments=list(arguments))


@pytest.mark.parametrize("budget", (0, 3))
@pytest.mark.parametrize("nonzero, external_condition", ((True, False), (False, True)))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_flat_masked_plan_forward_reverse_and_retirement(budget, nonzero, external_condition, tmp_path):
    @ti.kernel
    def initialize(count: ti.types.ndarray(ti.i32, 1), checks: ti.types.ndarray(ti.i32, 1)):
        count[0], checks[0] = 0, 0

    @ti.kernel
    def condition(
        count: ti.types.ndarray(ti.i32, 1),
        predicate: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
        limits: ti.types.ndarray(ti.i32, 1),
        inverted: ti.i32,
    ):
        checks[0] += 1
        predicate[0] = ti.cast(count[0] < limits[0], ti.i32) ^ inverted

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

    module = ti.aot.Module()
    begin = time.perf_counter()
    for kernel in (initialize, condition, advance, forward, publish):
        module.add_kernel(kernel)
    module.add_kernel(forward.grad, name="reverse")
    costs = dict(compile_ms=(time.perf_counter() - begin) * 1000)
    begin = time.perf_counter()
    for name in ("condition", "advance", "forward", "reverse"):
        compiled = module._exported_kernels[name]
        entry = "gated_" + name
        module._aot_builder._add_with_launch_abi(entry, compiled, "cuda_graph_gate_v1")
        module._exported_kernels[entry] = compiled
        module._kernel_launch_abis[entry] = "cuda_graph_gate_v1"
        module._content.append("kernel:" + entry)
    costs["variant_generation_ms"] = (time.perf_counter() - begin) * 1000
    check = _kernel("condition", "count", "predicate", "checks", "limits", "inverted")
    loop = dict(
        kind="while",
        predicate="predicate",
        max_iterations=budget,
        continue_while_nonzero=nonzero,
        condition=[] if external_condition else [check],
        body=[
            _kernel("gated_advance", "count", "dy"),
            _kernel("gated_forward", "x", "y"),
            _kernel("gated_reverse", "x", "y"),
            dict(check, entry="gated_condition"),
        ],
        lowering=dict(kind="cuda_masked", version=1),
    )
    plan = ExecutionPlan(
        [
            _kernel("initialize", "count", "checks"),
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="fill32", destination="y", value=0xC2C60000),
            *([check] if external_condition else []),
            loop,
            _kernel("publish", "count", "checks", "terminal"),
        ],
        resources=[
            *[dict(kind="ndarray", name=name, dtype="i32", shape=[1]) for name in ("count", "checks")],
            Scalar("i32", int(not nonzero)).resource("inverted"),
        ],
        gradients=dict(x="dx", y="dy"),
    )
    assert json.loads(plan.to_json())["version"] == 15
    for name in ("step", "again"):
        module.add_execution_plan(name, plan)
    assert len(module._exported_kernels) == 10
    artifact = tmp_path / "flat.tcm"
    begin = time.perf_counter()
    module.archive(str(artifact))
    costs.update(save_ms=(time.perf_counter() - begin) * 1000, artifact_bytes=artifact.stat().st_size)
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cuda_flat_masked_probe.py")),
                mode,
                str(artifact),
                str(budget),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        costs[mode] = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"load_ms"')))
    (tmp_path / "costs.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")
