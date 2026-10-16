"""Retained tensor arrays keep public layout and explicit adjoint bindings."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("element_shape", ((3,), (2, 2)))
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_tensor_source_bindings_and_gradient(element_shape, tmp_path):
    dtype = ti.types.vector(3, ti.f32) if len(element_shape) == 1 else ti.types.matrix(2, 2, ti.f32)

    @ti.kernel
    def square(x: ti.types.ndarray(dtype, 1, needs_grad=True), y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in y:
            y[i] = x[i].norm_sqr()

    @ti.kernel
    def seed(dx: ti.types.ndarray(dtype, 1), dy: ti.types.ndarray(ti.f32, 1)):
        for i in dy:
            dy[i] = 1
            if ti.static(len(element_shape) == 1):
                for j in ti.static(range(3)):
                    dx[i][j] = 0
            else:
                for j, k in ti.static(ti.ndrange(2, 2)):
                    dx[i][j, k] = 0

    source = ti.Vector.ndarray(3, ti.f32, 3) if len(element_shape) == 1 else ti.Matrix.ndarray(2, 2, ti.f32, 3)
    symbol = lambda name, t: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, t, ndim=1)
    x, dx = symbol("retained", dtype), symbol("dx", dtype)
    y, dy = symbol("y", ti.f32), symbol("dy", ti.f32)
    start = time.perf_counter()
    builder = ti.graph.GraphBuilder()
    builder._bind_internal_runtime_arg(x, source)
    builder.dispatch(square, x, y)
    builder.dispatch(seed, dx, dy)
    builder.dispatch(square.grad, x, y)
    graph = builder.compile()
    costs = dict(build_ms=(time.perf_counter() - start) * 1000)
    module = ti.aot.Module()
    start = time.perf_counter()
    plan = module.add_execution_plan(
        "step", graph, source_bindings={"x": source}, gradients=dict(retained="dx", y="dy")
    )
    costs["export_ms"] = (time.perf_counter() - start) * 1000
    assert plan.argument_names == ("dx", "dy", "x", "y")
    layout = next(
        c for c in json.loads(plan.to_json())["constraints"] if c["kind"] == "array_layout" and c["array"] == "x"
    )
    assert layout["element_shape"] == list(element_shape) and layout["exact_shape"] == [3]
    assert not module._data_payloads  # The consumer owns data; export does not snapshot it.
    entries = len(module._exported_kernels)
    module.add_execution_plan("again", graph, source_bindings={"x": source}, gradients=dict(retained="dx", y="dy"))
    assert len(module._exported_kernels) == entries
    graph.close()
    del graph, builder, source
    gc.collect()
    artifact = tmp_path / "tensor.tcm"
    start = time.perf_counter()
    module.archive(str(artifact))
    costs.update(
        archive_ms=(time.perf_counter() - start) * 1000, artifact_bytes=artifact.stat().st_size, entries=entries
    )
    arch = {ti.cpu: "cpu", ti.cuda: "cuda", ti.vulkan: "vulkan"}[ti.cfg.arch]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_tensor_source_probe.py")),
                mode,
                arch,
                str(artifact),
                *map(str, element_shape),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        costs[mode] = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"load_ms"')))
    (tmp_path / "costs.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_tensor_source_rejects_kernel_shape_before_publication():
    dtype = ti.types.matrix(3, 1, ti.f32)

    @ti.kernel
    def copy(x: ti.types.ndarray(dtype, 1), y: ti.types.ndarray(dtype, 1)):
        for i in y:
            y[i] = x[i]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(copy, *(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1) for name in ("x", "y")))
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="compiled kernel argument"):
        module.add_execution_plan("bad", graph, source_bindings={"x": ti.Vector.ndarray(3, ti.f32, 3)})
    assert not module._content and not module._exported_kernels and not module._execution_plans
    assert not module._graph_export_failed
    graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_retained_tensor_views_use_external_parent_storage(tmp_path):
    values, output = (ti.Vector.ndarray(2, ti.f32, 9) for _ in range(2))
    flags, count = ti.ndarray(ti.i32, 9), ti.ndarray(ti.i32, 1)
    view = lambda owner, start, stop: ti.experimental.ndarray_view(owner, slices=slice(start, stop))
    prepared = ti.algorithms.prepare_compact(view(values, 2, 7), view(flags, 2, 7), view(output, 1, 6), count)
    builder = ti.graph.GraphBuilder()
    builder.append_native(prepared.record())
    graph = builder.compile()
    module = ti.aot.Module()
    plan = module.add_execution_plan(
        "step", graph, source_bindings=dict(values=values, flags=flags, output=output, count=count)
    )
    assert len([r for r in json.loads(plan.to_json())["resources"] if r["kind"] == "ndarray_view"]) == 3
    assert not module._data_payloads and not module._exported_kernels
    graph.close()
    prepared.close()
    artifact = tmp_path / "tensor-views.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        arrays = {name: runtime.ndarray("f32", 9, element_shape=(2,)) for name in ("values", "output")}
        arrays.update(flags=runtime.ndarray("i32", 9), count=runtime.ndarray("i32", 1))
        source = np.arange(18, dtype=np.float32).reshape(9, 2)
        arrays["values"].write(source.tobytes())
        arrays["output"].write(np.full((9, 2), -99, np.float32).tobytes())
        arrays["flags"].write(np.array([0, 0, 1, 0, -1, 2, 0, 0, 0], np.int32).tobytes())
        with loaded.prepare_execution("step", arrays) as call:
            call()
            expected = np.full((9, 2), -99, np.float32)
            expected[1:4] = source[[2, 4, 5]]
            np.testing.assert_array_equal(np.frombuffer(arrays["output"].read(), np.float32).reshape(9, 2), expected)
            np.testing.assert_array_equal(np.frombuffer(arrays["count"].read(), np.int32), [3])
