"""Explicit CUDA binding families share private state and once-only setup."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_binding_family_forward_reverse_and_retirement(tmp_path):
    @ti.kernel
    def advance(state: ti.types.ndarray(ti.f32, ndim=1)) -> ti.f32:
        for i in range(17):
            state[i] += 1
        return state[0]

    @ti.kernel
    def initialize(seed: ti.types.ndarray(ti.f32, ndim=1),
                   state: ti.types.ndarray(ti.f32, ndim=1)) -> ti.f32:
        for i in range(17):
            state[i] = seed[i]
        return state[0]

    @ti.kernel
    def forward(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        state: ti.types.ndarray(ti.f32, ndim=1),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in range(17):
            y[i] = x[i] * state[i]

    module = ti.aot.Module()
    module.add_kernel(advance)
    module.add_kernel(initialize)
    module.add_kernel(forward)
    module.add_kernel(forward.grad, name="reverse")
    plan = ti.aot.ExecutionPlan(
        [
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="kernel", entry="advance", arguments=["state"]),
            dict(kind="kernel", entry="forward", arguments=["x_view", "state", "y"]),
            dict(kind="kernel", entry="reverse", arguments=["x_view", "state", "y"]),
        ],
        initialization=[dict(kind="kernel", entry="initialize", arguments=["seed", "state"])],
        resources=[
            dict(kind="ndarray", name="state", dtype="f32", shape=[dict(array="seed", axis=0)]),
            dict(kind="ndarray", name="dy", dtype="f32", shape=[17]),
            dict(kind="ndarray_view", name="x_view", array="x", offset=0, shape=[17]),
        ],
        gradients=dict(x_view="dx", y="dy"),
        constraints=[dict(kind="disjoint_writes", reads=["seed", "x"], writes=[dict(array="y")])],
        binding_strategy="cuda_immutable_argument_frames",
    )
    assert json.loads(plan.to_json())["version"] == 10
    module.add_execution_plan("step", plan)
    artifact = tmp_path / "binding-family.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_binding_family_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Unable to release execution resource" not in result.stdout + result.stderr


def test_binding_family_schema():
    from taichi_forge_runtime.aot import ExecutionPlan

    strategy = "cuda_immutable_argument_frames"
    plan = ExecutionPlan([dict(kind="fill32", destination="x", value=0)], binding_strategy=strategy)
    assert ExecutionPlan.from_json(plan.to_json()).binding_strategy == strategy
    bad_version = json.loads(plan.to_json())
    bad_version["version"] = 9
    with pytest.raises(ValueError, match="version 10"):
        ExecutionPlan.from_json(json.dumps(bad_version))
    with pytest.raises(ValueError, match="strategy"):
        ExecutionPlan([], binding_strategy="guess")


@test_utils.test(arch=[ti.cpu, ti.vulkan], offline_cache=False)
def test_binding_family_rejects_other_backends(tmp_path):
    from taichi_forge_runtime.aot import AotError

    module = ti.aot.Module()
    artifact = tmp_path / "empty"
    module.save(artifact)
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        array = runtime.ndarray("i32", (17,))
        plan = ti.aot.ExecutionPlan(
            [dict(kind="fill32", destination="x", value=0)],
            binding_strategy="cuda_immutable_argument_frames",
        )
        with pytest.raises(AotError, match="requires its selected CUDA"):
            loaded.prepare_execution(plan, dict(x=array))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_binding_family_shared_fft_plan(tmp_path):
    directory = os.environ.get("TI_TEST_CUFFT_DIRECTORY")
    dll = os.add_dll_directory(directory) if directory and os.name == "nt" else None
    if not ti.hardware.fft.is_available():
        pytest.skip("external cuFFT required")
    fft = ti.hardware.fft.CufftPlan1D(17)
    builder = ti.graph.GraphBuilder()
    builder.append_native(fft.record(input="x", output="spectrum"))
    builder.append_native(fft.record(input="spectrum", output="y", direction="inverse"))
    graph = builder.compile()
    module = ti.aot.Module()
    original = module.add_execution_plan("original", graph)
    description = json.loads(original.to_json())
    description.update(version=10, binding_strategy="cuda_immutable_argument_frames")
    module.add_execution_plan("step", ti.aot.ExecutionPlan.from_json(json.dumps(description)))
    graph.close()
    fft.close()
    artifact = tmp_path / "binding-fft.tcm"
    module.archive(str(artifact))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("aot_binding_family_probe.py")),
            "standalone",
            str(artifact),
            "fft",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "plan destruction returned status" not in result.stdout + result.stderr
    if dll is not None:
        dll.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_binding_family_fixed_native_inputs(tmp_path):
    import numpy as np

    if not ti.hardware.linalg.cusparse_is_available():
        pytest.skip("external cuSPARSE required")

    def upload(dtype, values):
        array = ti.ndarray(dtype, len(values))
        array.from_numpy(np.asarray(values, np.int32 if dtype == ti.i32 else np.float32))
        return array

    pattern = ti.linalg.SparsePattern.csr(3, 3, upload(ti.i32, [0, 1, 2, 3]), upload(ti.i32, [0, 1, 2]))
    matrix = pattern.matrix(upload(ti.f32, [2, 3, 4]))
    builder = ti.graph.GraphBuilder()
    builder.append_native(ti.hardware.linalg.CusparseSpmvRecording(matrix, input="x", output="y"))
    graph = builder.compile()
    module = ti.aot.Module()
    original = module.add_execution_plan("original", graph, source_bindings=dict(coefficients=matrix))
    document = json.loads(original.to_json())
    document.update(version=10, binding_strategy="cuda_immutable_argument_frames")
    module.add_execution_plan("step", ti.aot.ExecutionPlan.from_json(json.dumps(document)))

    def rename_output(value):
        if isinstance(value, dict):
            return {key: rename_output(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename_output(item) for item in value]
        return "state" if value == "y" else value

    initial = rename_output(document)
    module.add_execution_plan(
        "init_step",
        ti.aot.ExecutionPlan(
            [dict(kind="copy", source="state", destination="out")],
            initialization=initial["operations"],
            resources=initial["resources"]
            + [dict(kind="ndarray", name="state", dtype="f32", shape=[dict(array="coefficients", axis=0)])],
            constraints=initial.get("constraints"),
            binding_strategy="cuda_immutable_argument_frames",
        ),
    )
    graph.close()
    artifact = tmp_path / "binding-sparse.tcm"
    module.archive(str(artifact))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("aot_binding_family_probe.py")),
            "standalone",
            str(artifact),
            "sparse",
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Unable to release execution resource" not in result.stdout + result.stderr
