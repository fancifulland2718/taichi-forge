"""Host-known ranges share code while selecting launch work per binding."""
import json
from contextlib import ExitStack

import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


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
        for i in x:
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
            calls.append(loaded.prepare_execution("differentiate", bindings))
        for iteration in (1, 2):
            for bindings, call in zip(arrays, calls):
                values = np.arange(bindings["x"].shape[0], dtype=np.float32) + iteration
                bindings["x"].write(values.tobytes())
                call()
                np.testing.assert_array_equal(
                    np.frombuffer(bindings["y"].read(), dtype=np.float32), values ** 2)
                np.testing.assert_array_equal(
                    np.frombuffer(bindings["dx"].read(), dtype=np.float32), values * 2)
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
        for i in range(n):
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
        for stop in (0, 1, 513, 1025, 3):
            array.fill(0)
            ti.sync()
            output.write(np.zeros(1025, dtype=np.int32).tobytes())
            for call in (jit, aot):
                ticket = call.submit(values=dict(n=stop, value=7))
                assert ticket is not None
                ticket.wait()
                ticket.close()
            expected = np.ones(1025, dtype=np.int32)
            expected[:stop] = 8
            np.testing.assert_array_equal(array.to_numpy(), expected)
            np.testing.assert_array_equal(np.frombuffer(output.read(), dtype=np.int32), expected)
