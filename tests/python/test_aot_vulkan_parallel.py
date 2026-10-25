"""Public persistent Vulkan fork/join: real reductions, adjoints and owners."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import AotError, ExecutionPlan
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_source_free_vulkan_parallel_gradients_parameters_and_retirement(tmp_path):
    @ti.kernel
    def square(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            y[i] = x[i] * x[i]

    @ti.kernel
    def reduce_values(x: ti.types.ndarray(ti.f32, 1), result: ti.types.ndarray(ti.f32, 1)):
        total = 0.0
        for i in x:
            total += x[i]
        result[0] = total

    @ti.kernel
    def publish(left: ti.types.ndarray(ti.f32, 1), right: ti.types.ndarray(ti.f32, 1),
                out: ti.types.ndarray(ti.u64, 1), epoch: ti.u64):
        out[0] = epoch
        out[1] = ti.cast(left[0], ti.u64)
        out[2] = ti.cast(right[0], ti.u64)

    module = ti.aot.Module(caps=["spirv_has_int64"])
    module.add_kernel(square)
    module.add_kernel(square.grad, name="backward")
    module.add_kernel(reduce_values)
    module.add_kernel(publish)
    operations = [dict(kind="fill32", destination=f"{name}{lane}", value=value)
                  for lane in range(2) for name, value in (("dx", 0), ("dy", 1073741824))]
    branches = [[dict(kind="kernel", entry=entry, arguments=arguments)
                 for entry, arguments in (
                     ("square", [f"x{lane}", f"y{lane}"]),
                     ("backward", [f"x{lane}", f"y{lane}"]),
                     ("reduce_values", [f"y{lane}", f"sum{lane}"]))]
                for lane in range(2)]
    operations.extend((dict(kind="vulkan_parallel", branches=branches),
                       dict(kind="kernel", entry="publish", arguments=["sum0", "sum1", "out", "epoch"])))
    plan = ExecutionPlan(
        operations,
        resources=[dict(kind="ndarray", name=f"sum{lane}", dtype="f32", shape=[1]) for lane in range(2)],
        constraints=[dict(kind="disjoint", left=f"{a}0", right=f"{b}1")
                     for a in ("x", "y", "dx", "dy") for b in ("x", "y", "dx", "dy")],
    )
    assert json.loads(plan.to_json())["version"] == 27
    module.add_execution_plan("step", plan)
    artifact = tmp_path / "parallel.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_vulkan_parallel_probe.py")), mode, str(artifact)],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90,
        )
        (tmp_path / f"{mode}.log").write_text(result.stdout + result.stderr, encoding="utf8")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "VULKAN_PARALLEL_OK" in result.stdout
        assert not any(s in result.stderr for s in ("VUID-", "SYNC-HAZARD", "cleanup failed"))


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_vulkan_parallel_rejects_wrong_backend(tmp_path):
    @ti.kernel
    def increment(x: ti.types.ndarray(ti.i32, 1)):
        for i in x:
            x[i] += 1

    module = ti.aot.Module()
    module.add_kernel(increment)
    artifact = tmp_path / "kernel.tcm"
    module.archive(str(artifact))
    plan = ExecutionPlan([dict(kind="vulkan_parallel", branches=[
        [dict(kind="kernel", entry="increment", arguments=[name])] for name in ("a", "b")])])
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(str(artifact))
        arrays = {name: runtime.ndarray("i32", 1) for name in ("a", "b")}
        for value in arrays.values():
            value.write(bytes(4))
        with pytest.raises(AotError, match="independent Vulkan"):
            plan.prepare(loaded, arrays)
        assert all(value.read() == bytes(4) for value in arrays.values())
