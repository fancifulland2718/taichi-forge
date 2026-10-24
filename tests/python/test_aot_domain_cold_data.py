"""Owned deployment data keeps its construction and explicit gradient contract."""

import array
from contextlib import ExitStack
import os

import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_private_initial_data_in_independent_instances(tmp_path):
    @ti.kernel
    def square(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(2):
            y[i] = x[i] * x[i]

    backend = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if backend in ("x64", "arm64"):
        backend = "cpu"
    module = ti.aot.Module()
    module.add_kernel(square)
    module.add_kernel(square.grad, name="backward")
    key = module.add_data(array.array("f", [3, -2]).tobytes())
    plan = ti.aot.ExecutionPlan(
        [dict(kind="fill32", destination="private_dx", value=0),
         dict(kind="kernel", entry="square", arguments=["x", "y"]),
         dict(kind="kernel", entry="backward", arguments=["x", "y"]),
         dict(kind="copy", source="private_dx", destination="dx")],
        resources=[dict(kind="ndarray", name="x", dtype="f32", shape=[2], initial_data=key),
                   dict(kind="ndarray", name="private_dx", dtype="f32", shape=[2])],
        gradients=dict(x="private_dx", y="dy"))
    module.add_execution_plan("step", plan)
    artifact = tmp_path / "private-data.tcm"
    module.archive(str(artifact))
    ti.reset()

    def read(value):
        result = array.array("f")
        result.frombytes(value.read())
        return list(result)

    with ExitStack() as scope:
        runtime = scope.enter_context(aot.Runtime(
            backend, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0))
            if backend == "vulkan" else 0))
        loaded = scope.enter_context(runtime.load_module(artifact))
        instances = []
        for seed in ([1, 2], [0.5, -1]):
            bindings = {name: runtime.ndarray("f32", 2) for name in ("y", "dy", "dx")}
            bindings["dy"].write(array.array("f", seed).tobytes())
            domain = None if backend == "cpu" else scope.enter_context(
                runtime.execution_domain(capacity=1, instance_capacity=1))
            prepared = scope.enter_context(loaded.execution_plan("step").prepare(
                loaded, bindings, execution_domain=domain))
            prepared.initialize()
            instances.append((prepared, bindings))
        if backend == "cpu":
            for prepared, _ in instances:
                prepared.run()
            runtime.wait()
        else:
            with ExitStack() as tickets:
                pending = [tickets.enter_context(prepared.submit()) for prepared, _ in instances]
                for ticket in pending:
                    ticket.wait()
        for (_, bindings), expected in zip(instances, ([6, -8], [3, 4])):
            assert read(bindings["y"]) == [9, 4]
            assert read(bindings["dx"]) == expected
        instances[0][0].close()
        prepared, bindings = instances[1]
        bindings["dy"].write(array.array("f", [2, 3]).tobytes())
        if backend == "cpu":
            prepared.run()
            runtime.wait()
        else:
            with prepared.submit() as ticket:
                ticket.wait()
        assert read(bindings["y"]) == [9, 4]
        assert read(bindings["dx"]) == [12, -12]
