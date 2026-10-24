"""Owned deployment data keeps its construction and explicit gradient contract."""

import array
from contextlib import ExitStack
import os
import subprocess

import pytest

import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_private_initialization_reuses_single_submission_slot(tmp_path):
    backend = ti.cfg.arch.name
    module = ti.aot.Module()
    key = module.add_data(array.array("Q", [0x12345678DEADBEEF, 0xFEDCBA9876543210]).tobytes())
    module.add_execution_plan("private", ti.aot.ExecutionPlan(
        [dict(kind="copy", source="seed", destination="output")],
        initialization=[dict(kind="fill32", destination="output", value=0)],
        resources=[dict(kind="ndarray", name="seed", dtype="u64", shape=[2], initial_data=key),
                   dict(kind="ndarray", name="output", dtype="u64", shape=[2])]))
    path = tmp_path / "private.tcm"
    module.archive(str(path))
    ti.reset()
    with ExitStack() as scope:
        runtime = scope.enter_context(aot.Runtime(
            backend, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0))
            if backend == "vulkan" else 0))
        loaded = scope.enter_context(runtime.load_module(path))
        domain = scope.enter_context(runtime.execution_domain(capacity=1, instance_capacity=2))
        calls = [scope.enter_context(loaded.prepare_execution("private", {}, execution_domain=domain))
                 for _ in range(2)]
        for call in calls:
            call.initialize()
        for call in calls * 2:
            with call.submit() as ticket:
                ticket.wait()
        calls[0].close()
        with calls[1].submit() as ticket:
            ticket.wait()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_kernel_preparation_uses_actual_field_dependencies(tmp_path):
    executable = os.environ.get("TI_TEST_DOMAIN_EXECUTABLE")
    if not executable:
        pytest.skip("Requires the native execution-domain gate fixture")
    backend = ti.cfg.arch.name
    counter = ti.field(ti.u32, shape=())

    @ti.kernel
    def rootless(out: ti.types.ndarray(ti.u32, 1)):
        out[0] = 42

    @ti.kernel
    def field(out: ti.types.ndarray(ti.u32, 1)):
        counter[None] += 1
        out[0] = counter[None]

    module = ti.aot.Module()
    module.add_field("counter", counter)
    module.add_kernel(rootless)
    module.add_kernel(field)
    folder = tmp_path / "module"
    module.save(str(folder))
    ti.reset()
    env = os.environ.copy()
    env["TI_TEST_PREPARATION_AOT_DIR"] = str(folder)
    selected = ("DomainPreparation.CudaUsesActualKernelFieldDependencies" if backend == "cuda" else
                "VulkanColdPreparation.RootlessKernelIgnoresDefaultAndPeer:"
                "VulkanColdPreparation.FieldKernelPreservesProducerPrefix")
    result = subprocess.run(
        [executable, "--gtest_filter=" + selected],
        env=env, capture_output=True, text=True, timeout=30)
    print(result.stdout, result.stderr)
    assert result.returncode == 0
    assert f"[  PASSED  ] {1 if backend == 'cuda' else 2} test" in result.stdout
    assert "[  SKIPPED ]" not in result.stdout


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
