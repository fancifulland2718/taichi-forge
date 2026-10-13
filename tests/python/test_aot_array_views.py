"""Packed AOT views share storage, dependency lifetimes and gradient offsets."""

import array
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Scalar
from tests import test_utils


def view(name, parent, offset, shape):
    return dict(kind="ndarray_view", name=name, array=parent, offset=offset, shape=shape)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_views_with_actual_gradients_and_parent_retirement(tmp_path):
    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        for i in range(x.shape[0]):
            y[i] = x[i] * x[i]

    @ti.kernel
    def increment(x: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] += 1

    @ti.kernel
    def copy(x: ti.types.ndarray(ti.f32, ndim=1), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(x.shape[0]):
            y[i] = x[i]

    @ti.kernel
    def components(x: ti.types.ndarray(ti.types.vector(2, ti.f32), ndim=1), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(x.shape[0]):
            y[i] = x[i][0] + 2 * x[i][1]

    @ti.kernel
    def rows(x: ti.types.ndarray(ti.f32, ndim=2), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(x.shape[0]):
            value = 0.0
            for j in range(x.shape[1]):
                value += x[i, j]
            y[i] = value

    module = ti.aot.Module()
    for kernel in (square, increment, copy, components, rows):
        module.add_kernel(kernel)
    module.add_kernel(square.grad, name="reverse")
    operations = [
        dict(kind="fill32", destination="dx", value=0),
        dict(kind="fill32", destination="dy", value=0x3F800000),
        dict(kind="kernel", entry="square", arguments=["x", "y"]),
        dict(kind="kernel", entry="reverse", arguments=["x", "y"]),
    ]
    resources = [
        view("x_base", "xp", 1, [8]),
        view("x", "x_base", 2, [5]),
        view("y", "yp", 2, [5]),
        view("dx", "dxp", 5, [5]),
        view("dy", "dyp", 1, [5]),
    ]
    module.add_execution_plan(
        "gradient", ti.aot.ExecutionPlan(operations, resources=resources, gradients={"x": "dx", "y": "dy"})
    )
    module.add_execution_plan(
        "zero_offset",
        ti.aot.ExecutionPlan(
            operations, resources=[view("x", "xp", 0, [5]), *resources[2:]], gradients={"x": "dx", "y": "dy"}
        ),
    )
    payload = module.add_data(array.array("f", range(12)))
    state = ti.aot.ExecutionPlan(
        [
            dict(kind="kernel", entry="increment", arguments=["selected"]),
            dict(kind="copy", source="selected", destination="output"),
        ],
        initialization=[dict(kind="fill32", destination="selected", value=0x40000000)],
        resources=[
            dict(kind="ndarray", name="state", dtype="f32", shape=[12], initial_data=payload),
            view("middle", "state", 1, [9]),
            view("selected", "middle", 2, [5]),
        ],
    )
    assert state.argument_names == ("output",) and state.data_entries == (payload,)
    module.add_execution_plan("state", state)
    module.add_execution_plan(
        "metadata",
        ti.aot.ExecutionPlan(
            [
                dict(kind="fill32", destination="scratch", value=0x40A00000),
                dict(kind="copy", source="scratch", destination="output"),
            ],
            resources=[
                dict(kind="ndarray", name="parent", dtype="f32", shape=[1024]),
                view("shape_only", "parent", 4, [5]),
                dict(kind="ndarray", name="scratch", dtype="f32", shape=[{"array": "shape_only", "axis": 0}]),
            ],
        ),
    )
    module.add_execution_plan(
        "tensor",
        ti.aot.ExecutionPlan(
            [dict(kind="kernel", entry="components", arguments=["selected", "output"])],
            resources=[view("selected", "parent", 2, [5])],
        ),
    )
    artifact = tmp_path / "array-views.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_array_views_probe.py")),
                mode,
                "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Unable to release" not in result.stdout + result.stderr


def test_aot_view_descriptions_reject_cycles_and_type_reinterpretation():
    operation = [dict(kind="fill32", destination="v", value=0)]
    for resource in (
        view("v", "v", 0, [1]),
        dict(view("v", "p", 0, [1]), dtype="i32"),
        view("v", "p", -1, [1]),
        view("v", "p", 1 << 64, [1]),
    ):
        with pytest.raises(ValueError):
            ti.aot.ExecutionPlan(operation, resources=[resource])
    with pytest.raises(ValueError, match="ndarray"):
        ti.aot.ExecutionPlan(operation, resources=[view("v", "p", 0, [1]), Scalar("i32", 3).resource("p")])
