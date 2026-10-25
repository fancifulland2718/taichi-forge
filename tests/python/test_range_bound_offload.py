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
    def mixed(start: ti.i32, out: ti.types.ndarray(ti.i32, 1), bounds: ti.types.ndarray(ti.i32, 1)):
        for i in range(start, bounds[1]):
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
    bounds[1] = 37
    out.fill(-91)
    mixed(7, out, bounds)
    expected = np.full(37, -91, np.int32)
    expected[7:] = np.arange(7, 37)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    if ti.lang.impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        task = next(task for task in mixed.task_manifest(7, out, bounds) if task.task_type == "range_for")
        assert task.range_bounds.begin.kind == "global_tmp_i32"
        assert task.range_bounds.end.kind == "global_tmp_i32"
        assert task.device_range_dispatch_version == 1


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


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_direct_begin_end_signed_graph_aot_and_clone(tmp_path):
    @ti.kernel
    def fill(start: ti.i32, end: ti.i32, out: ti.types.ndarray(ti.i32, 1)):
        for i in range(start + 2, end + 2):
            out[i + 8] = i * 3

    out = ti.ndarray(ti.i32, 48)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        fill,
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "start", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "end", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1),
    )
    graph = builder.compile()
    for start, end in ((-7, 37), (5, 11), (5, 5), (8, 3)):
        expected = np.full(48, -91, np.int32)
        indices = np.arange(start + 2, end + 2)
        expected[indices + 8] = indices * 3
        for invoke in (lambda: fill(start, end, out), lambda: graph.run(dict(start=start, end=end, out=out))):
            out.fill(-91)
            invoke()
            np.testing.assert_array_equal(out.to_numpy(), expected)
    if ti.lang.impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        assert [task.task_type for task in fill.task_manifest(0, 0, out)] == ["range_for"]
    module = ti.aot.Module()
    module.add_execution_plan("fill", graph)
    artifact = tmp_path / "direct-bounds.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as owner, owner.load_module(artifact) as loaded:
        target = owner.ndarray("i32", (48,))
        try:
            for start, end in ((-7, 37), (8, 3)):
                target.write(np.full(48, -91, np.int32).tobytes())
                with loaded.prepare_execution("fill", dict(start=start, end=end, out=target)) as call:
                    call()
                expected = np.full(48, -91, np.int32)
                indices = np.arange(start + 2, end + 2)
                expected[indices + 8] = indices * 3
                np.testing.assert_array_equal(np.frombuffer(target.read(), np.int32), expected)
        finally:
            target.close()
    # Export/clone must not steal the producer's direct bound expressions.
    out.fill(-91)
    fill(5, 5, out)
    np.testing.assert_array_equal(out.to_numpy(), np.full(48, -91, np.int32))
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, advanced_optimization=True)
def test_direct_start_reverse_mode_and_equal_bound_expression():
    x = ti.field(ti.f32, 37, needs_grad=True)
    y = ti.field(ti.f32, 37, needs_grad=True)

    @ti.kernel
    def square(start: ti.i32, end: ti.i32):
        for i in range(start, end):
            y[i] = x[i] * x[i]

    @ti.kernel
    def empty(n: ti.i32):
        for i in range(n + 1, n + 1):
            y[i] = -100

    x.from_numpy(np.arange(37, dtype=np.float32))
    for start, end in ((3, 37), (10, 10), (7, 3)):
        y.fill(-91)
        x.grad.fill(0)
        y.grad.fill(1)
        square(start, end)
        square.grad(start, end)
        empty(3)
        forward = np.full(37, -91, np.float32)
        reverse = np.zeros(37, np.float32)
        forward[start:end] = np.arange(start, end) ** 2
        reverse[start:end] = 2 * np.arange(start, end)
        np.testing.assert_array_equal(y.to_numpy(), forward)
        np.testing.assert_array_equal(x.grad.to_numpy(), reverse)
