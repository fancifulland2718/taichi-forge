"""Dynamic loop bounds retain their launch-time or device-snapshot semantics."""

import json

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_range_end_jit_graph_and_aot(tmp_path):
    out = ti.ndarray(ti.i32, 37)

    @ti.kernel
    def fill(n: ti.i32, out: ti.types.ndarray(ti.i32, 1)):
        for i in range(n):
            out[i] = i + n

    builder = ti.graph.GraphBuilder()
    child = builder.create_sequential()
    child.dispatch(
        fill,
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "n", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1),
    )
    builder.append(child)
    parent = builder.compile()
    for n in (37, 7, 0, -3):
        expected = np.full(37, -91, np.int32)
        expected[: max(n, 0)] = np.arange(max(n, 0)) + n
        for invoke in (lambda: fill(n, out), lambda: parent.run(dict(n=n, out=out))):
            out.fill(-91)
            invoke()
            np.testing.assert_array_equal(out.to_numpy(), expected)

    module = ti.aot.Module()
    module.add_execution_plan("fill", parent)
    module.save(tmp_path)
    if ti.lang.impl.current_cfg().arch == ti.vulkan:
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert len(metadata["kernels"]) == 1
        assert [
            task["task_type"] for task in metadata["kernels"][0]["tasks_attribs"]
        ] == [1]
    artifact = tmp_path / "ranges.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        result = runtime.ndarray("i32", 37)
        for n in (37, 5, 0, -3):
            result.write(np.full(37, -91, np.int32).tobytes())
            with loaded.prepare_execution("fill", dict(n=n, out=result)) as call:
                call()
            expected = np.full(37, -91, np.int32)
            expected[: max(n, 0)] = np.arange(max(n, 0)) + n
            np.testing.assert_array_equal(
                np.frombuffer(result.read(), np.int32), expected
            )
        result.close()
    parent.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_dynamic_start_shape_end_and_device_snapshot():
    out = ti.ndarray(ti.i32, 37)
    bounds = ti.ndarray(ti.i32, 2)

    @ti.kernel
    def from_shape(start: ti.i32, out: ti.types.ndarray(ti.i32, 1)):
        for i in range(start, out.shape[0]):
            out[i] = i

    @ti.kernel
    def shape_start(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(out.shape[0] - 7, out.shape[0]):
            out[i] = i

    @ti.kernel
    def snapshot(
        n: ti.i32, out: ti.types.ndarray(ti.i32, 1), bounds: ti.types.ndarray(ti.i32, 1)
    ):
        bounds[0] = 3
        bounds[1] = n
        for i in range(bounds[0], bounds[1]):
            out[i] = i
            if i == 3:
                bounds[1] = 3  # The already selected range must not shrink.

    for start in (7, 37, 39):
        out.fill(-91)
        from_shape(start, out)
        expected = np.full(37, -91, np.int32)
        expected[start:] = np.arange(start, 37)
        np.testing.assert_array_equal(out.to_numpy(), expected)
    out.fill(-91)
    shape_start(out)
    expected = np.full(37, -91, np.int32)
    expected[30:] = np.arange(30, 37)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    for end in (37, 9, 3):
        out.fill(-91)
        snapshot(end, out, bounds)
        expected = np.full(37, -91, np.int32)
        expected[3:end] = np.arange(3, end)
        np.testing.assert_array_equal(out.to_numpy(), expected)
        assert bounds[1] == 3


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_range_end_reverse_mode():
    x = ti.field(ti.f32, 37, needs_grad=True)
    y = ti.field(ti.f32, 37, needs_grad=True)

    @ti.kernel
    def square(n: ti.i32):
        for i in range(n):
            y[i] = x[i] * x[i]

    x.from_numpy(np.arange(37, dtype=np.float32))
    for n in (37, 7, 0):
        y.fill(-91)
        x.grad.fill(0)
        y.grad.fill(1)
        square(n)
        square.grad(n)
        forward = np.full(37, -91, np.float32)
        forward[:n] = np.arange(n) ** 2
        reverse = np.zeros(37, np.float32)
        reverse[:n] = 2 * np.arange(n)
        np.testing.assert_array_equal(y.to_numpy(), forward)
        np.testing.assert_array_equal(x.grad.to_numpy(), reverse)
