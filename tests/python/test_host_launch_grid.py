"""Host-known ranges share code while selecting launch work per binding."""
import json
from contextlib import ExitStack

import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, debug=True)
def test_debug_affine_end_retains_serial_evaluation():
    @ti.kernel
    def fill(x: ti.types.ndarray(ti.i32, 1), n: ti.i32):
        for i in range(3 * n - 1):
            x[i] = i + 1

    x = ti.ndarray(ti.i32, 8)
    fill(x, 3)
    np.testing.assert_array_equal(x.to_numpy(), np.arange(1, 9))
    assert any(t.task_type == "serial" for t in fill.task_manifest(x, 3))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_immutable_end_and_scalar_body_do_not_need_a_serial_task():
    @ti.kernel
    def fill(x: ti.types.ndarray(ti.f32, 1), n: ti.i32, delta: ti.f32):
        for i in range(3 * n - 1):
            x[i] = i + delta

    @ti.kernel
    def shape(x: ti.types.ndarray(ti.f32, 2), delta: ti.f32):
        for i, j in x:
            x[i, j] = i * x.shape[1] + j + delta

    x = ti.ndarray(ti.f32, 1025)
    for n in (0, 1, 171, 342, 2):
        x.fill(-7)
        fill(x, n, 2.5)
        count = max(0, 3 * n - 1)
        expected = np.full(1025, -7, np.float32)
        expected[:count] = np.arange(count) + 2.5
        np.testing.assert_array_equal(x.to_numpy(), expected)
    y = ti.ndarray(ti.f32, (3, 129))
    shape(y, 2.5)
    np.testing.assert_array_equal(y.to_numpy(), np.arange(387).reshape(3, 129) + 2.5)
    if impl.current_cfg().arch != ti.cpu:
        for kernel, arguments in ((fill, (x, 1, 2.5)), (shape, (y, 2.5))):
            tasks = kernel.task_manifest(*arguments)
            assert [task.task_type for task in tasks] == ["range_for"]
            assert tasks[0].range_bounds.end.kind == "unknown"
            assert tasks[0].actual_geometry_kind == "host_argument_bound"


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_affine_bounds_keep_signed_semantics_and_graph_rebinding():
    @ti.kernel
    def affine(x: ti.types.ndarray(ti.i32, 1), first: ti.i32, last: ti.i32):
        for i in range(2 * first + 1, 3 * last - 1):
            x[i + 5] += i + 17

    @ti.kernel
    def reversed_affine(x: ti.types.ndarray(ti.i32, 1), n: ti.i32):
        for i in range(7 - n):
            x[i] = i + 1

    builder = ti.graph.GraphBuilder()
    symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.i32, ndim=1)
    builder.dispatch(affine, symbol,
                     ti.graph.Arg(ti.graph.ArgKind.SCALAR, "first", ti.i32),
                     ti.graph.Arg(ti.graph.ArgKind.SCALAR, "last", ti.i32))
    graph = builder.compile()
    array = ti.ndarray(ti.i32, 1030)
    for first, last in ((-3, 0), (0, 22), (2, 86), (2, 342), (5, 1), (-3, -2)):
        expected = np.zeros(1030, dtype=np.int32)
        begin, end = 2 * first + 1, 3 * last - 1
        if end > begin:
            expected[begin + 5:end + 5] = np.arange(begin, end) + 17
        array.fill(0)
        affine(array, first, last)
        np.testing.assert_array_equal(array.to_numpy(), expected)
        graph.run(dict(x=array, first=first, last=last))
        np.testing.assert_array_equal(array.to_numpy(), expected * 2)
    for n in (-4, 3, 7, 11):
        array.fill(0)
        reversed_affine(array, n)
        expected = np.zeros(1030, dtype=np.int32)
        expected[:max(7 - n, 0)] = np.arange(1, max(7 - n, 0) + 1)
        np.testing.assert_array_equal(array.to_numpy(), expected)
    if impl.current_cfg().arch != ti.cpu:
        for kernel, arguments in ((affine, (array, 0, 22)),
                                  (reversed_affine, (array, 3))):
            tasks = [t for t in kernel.task_manifest(*arguments) if t.task_type == "range_for"]
            assert len(tasks) == 1
            assert tasks[0].actual_geometry_kind == "host_argument_bound"


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_shape_products_and_graph_rebinding(tmp_path):
    @ti.kernel
    def fill(x: ti.types.ndarray(ti.i32, ndim=2)):
        ti.loop_config(block_dim=128)
        for i, j in x:
            x[i, j] = i * x.shape[1] + j + 7

    builder = ti.graph.GraphBuilder()
    symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.i32, ndim=2)
    builder.dispatch(fill, symbol)
    graph = builder.compile()
    for shape in ((2, 32), (3, 129), (1, 1), (7, 147), (2, 32)):
        array = ti.ndarray(ti.i32, shape)
        expected = np.arange(np.prod(shape), dtype=np.int32).reshape(shape) + 7
        fill(array)
        np.testing.assert_array_equal(array.to_numpy(), expected)
        if impl.current_cfg().arch != ti.cpu:
            tasks = [task for task in fill.task_manifest(array)
                     if task.task_type == "range_for"]
            assert len(tasks) == 1
            assert tasks[0].actual_geometry_kind == "host_argument_bound"
            assert tasks[0].actual_grid_size is None
        array.fill(-1)
        for _ in range(2):
            graph.run({"x": array})
            np.testing.assert_array_equal(array.to_numpy(), expected)

    module = ti.aot.Module()
    module.add_kernel(fill)
    module.save(tmp_path)
    if impl.current_cfg().arch == ti.vulkan:
        metadata = json.loads((tmp_path / "metadata.json").read_text())
        bounds = [task["range_for_attribs"]["host_launch"]
                  for kernel in metadata["kernels"]
                  for task in kernel["tasks_attribs"]
                  if task["task_type"] == 1]
        assert len(bounds) == 1 and bounds[0] is not None
        assert bounds[0]["end"]["shape_product"]
        assert len(bounds[0]["end"]["offsets"]) == 2


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scalar_range_empty_signed_and_graph_parameter_updates():
    @ti.kernel
    def fill(x: ti.types.ndarray(ti.i32, ndim=1), first: ti.i32, last: ti.i32):
        ti.loop_config(block_dim=128)
        for i in range(first, last):
            x[i + 3] += i + 9

    builder = ti.graph.GraphBuilder()
    symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.i32, ndim=1)
    first = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "first", ti.i32)
    last = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "last", ti.i32)
    builder.dispatch(fill, symbol, first, last)
    builder.dispatch(fill, symbol, first, last)
    graph = builder.compile()
    array = ti.ndarray(ti.i32, 1025)
    for stop in (-7, -3, 61, 254, 1022, 62, -3):
        expected = np.zeros(1025, dtype=np.int32)
        if stop > -3:
            expected[:stop + 3] = np.arange(-3, stop, dtype=np.int32) + 9
        array.fill(0)
        fill(array, -3, stop)
        np.testing.assert_array_equal(array.to_numpy(), expected)
        if impl.current_cfg().arch != ti.cpu:
            tasks = [task for task in fill.task_manifest(array, -3, stop)
                     if task.task_type == "range_for"]
            assert len(tasks) == 1
            assert tasks[0].actual_geometry_kind == "host_argument_bound"
        array.fill(0)
        for repeat in (1, 2):
            graph.run({"x": array, "first": -3, "last": stop})
            np.testing.assert_array_equal(array.to_numpy(), expected * 2 * repeat)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_instances_and_gradients_keep_dynamic_shape(tmp_path):
    @ti.kernel
    def square(x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(x.shape[0] - 1):
            y[i] = x[i] * x[i]

    module = ti.aot.Module()
    module.add_kernel(square)
    module.add_kernel(square.grad, name="backward")
    module.add_execution_plan("differentiate", ti.aot.ExecutionPlan(
        [dict(kind="fill32", destination="dx", value=0),
         dict(kind="fill32", destination="dy", value=1065353216),
         dict(kind="kernel", entry="square", arguments=["x", "y"]),
         dict(kind="kernel", entry="backward", arguments=["x", "y"])],
        gradients={"x": "dx", "y": "dy"}))
    artifact = tmp_path / "shape.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        arrays, calls = [], []
        for size in (65, 1025):
            bindings = {name: runtime.ndarray("f32", (size,))
                        for name in ("x", "y", "dx", "dy")}
            arrays.append(bindings)
            bindings["y"].write(np.zeros(size, dtype=np.float32).tobytes())
            calls.append(loaded.prepare_execution("differentiate", bindings))
        for iteration in (1, 2):
            for bindings, call in zip(arrays, calls):
                values = np.arange(bindings["x"].shape[0], dtype=np.float32) + iteration
                bindings["x"].write(values.tobytes())
                call()
                expected = values ** 2
                expected[-1] = 0
                np.testing.assert_array_equal(
                    np.frombuffer(bindings["y"].read(), dtype=np.float32), expected)
                expected = values * 2
                expected[-1] = 0
                np.testing.assert_array_equal(
                    np.frombuffer(bindings["dx"].read(), dtype=np.float32), expected)
        calls[0].close()
        calls[1]()
        calls[1].close()
        for bindings in arrays:
            for array in bindings.values():
                array.close()
        loaded.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_mutable_bound_and_unrelated_parameter(tmp_path):
    from taichi_forge_runtime.aot import Scalar

    @ti.kernel
    def fill(x: ti.types.ndarray(ti.i32, 1), n: ti.i32, value: ti.i32):
        for i in range(3 * n - 1):
            x[i] = value
        for i in x:
            x[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.i32, ndim=1),
                     ti.graph.Arg(ti.graph.ArgKind.SCALAR, "n", ti.i32),
                     ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32))
    graph = builder.compile()
    artifact = ti.aot.Module()
    artifact.add_execution_plan("fill", graph)
    path = tmp_path / "parameters.tcm"
    artifact.archive(str(path))
    with ExitStack() as scope:
        array = ti.ndarray(ti.i32, 1025)
        domain = scope.enter_context(ti.graph.execution_domain(capacity=2))
        jit = scope.enter_context(graph.prepare(dict(x=array, n=1, value=0),
                                  execution_domain=domain, mutable_parameters=("n", "value")))
        runtime = scope.enter_context(ti.aot.borrow_runtime())
        loaded = scope.enter_context(runtime.load_module(path))
        output = runtime.ndarray("i32", (1025,))
        scope.callback(output.close)
        aot_domain = scope.enter_context(runtime.execution_domain(capacity=2))
        aot = scope.enter_context(loaded.prepare_execution(
            "fill", dict(x=output, n=Scalar("i32", 1), value=Scalar("i32", 0)),
            execution_domain=aot_domain, mutable_parameters=("n", "value")))
        for call in (jit, aot):
            call.initialize()
        for stop in (0, 1, 171, 342, 2):
            array.fill(0)
            ti.sync()
            output.write(np.zeros(1025, dtype=np.int32).tobytes())
            for call in (jit, aot):
                ticket = call.submit(values=dict(n=stop, value=7))
                assert ticket is not None
                ticket.wait()
                ticket.close()
            expected = np.ones(1025, dtype=np.int32)
            expected[:max(3 * stop - 1, 0)] = 8
            np.testing.assert_array_equal(array.to_numpy(), expected)
            np.testing.assert_array_equal(np.frombuffer(output.read(), dtype=np.int32), expected)
