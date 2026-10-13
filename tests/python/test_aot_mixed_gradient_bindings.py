"""A plan-wide gradient can be used by only some dispatches of its primal."""

import pytest
import taichi_forge as ti

from taichi_forge_runtime.aot import AotError
from tests import test_utils
from tests.python.aot_graph_recording_probe import read, write


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_mixed_gradient_bindings_follow_each_kernel_abi(tmp_path):
    @ti.kernel
    def copy(x: ti.types.ndarray(ti.f32, ndim=1), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(17):
            y[i] = x[i]

    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        for i in range(17):
            y[i] = x[i] * x[i]

    module = ti.aot.Module()
    module.add_kernel(copy)
    module.add_kernel(square)
    module.add_kernel(square.grad, name="reverse")
    module.add_execution_plan(
        "step",
        ti.aot.ExecutionPlan(
            [
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="fill32", destination="dy", value=0x3F800000),
                dict(kind="kernel", entry="copy", arguments=["x", "y"]),
                dict(kind="kernel", entry="square", arguments=["x", "y"]),
                dict(kind="kernel", entry="reverse", arguments=["x", "y"]),
                dict(kind="kernel", entry="copy", arguments=["y", "output"]),
            ],
            gradients=dict(x="dx", y="dy"),
        ),
    )
    module.add_execution_plan(
        "unused",
        ti.aot.ExecutionPlan(
            [
                dict(kind="kernel", entry="copy", arguments=["x", "output"]),
            ],
            gradients=dict(x="dx"),
        ),
    )
    artifact = tmp_path / "mixed-gradients.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        arrays = {name: runtime.ndarray("f32", (17,)) for name in ("x", "y", "dx", "dy", "output")}
        prepared = loaded.prepare_execution("step", arrays)
        for shift in (0, 2):
            values = [float(i - 8 + shift) for i in range(17)]
            write(arrays["x"], values)
            prepared()
            assert read(arrays["output"]) == [v * v for v in values]
            assert read(arrays["dx"]) == [2 * v for v in values]
        with pytest.raises(AotError, match="unused"):
            loaded.prepare_execution("unused", {name: arrays[name] for name in ("x", "dx", "output")})
        prepared()
        assert read(arrays["dx"]) == [2 * v for v in values]
