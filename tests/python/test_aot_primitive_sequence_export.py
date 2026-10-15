"""Preserve native sorting between producer and gather/adjoint kernels."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_primitive_sequence_sort_graph_export(tmp_path):
    n = 17
    keys, order = ti.ndarray(ti.f32, n), ti.ndarray(ti.i32, n)

    @ti.kernel
    def initialize(
        source_keys: ti.types.ndarray(ti.f32, 1), keys: ti.types.ndarray(ti.f32, 1), order: ti.types.ndarray(ti.i32, 1)
    ):
        for i in keys:
            keys[i], order[i] = source_keys[i], i

    @ti.kernel
    def gather(
        source: ti.types.ndarray(ti.f32, 1),
        order: ti.types.ndarray(ti.i32, 1),
        output: ti.types.ndarray(ti.f32, 1),
        gradient: ti.types.ndarray(ti.f32, 1),
    ):
        for i in order:
            j = order[i]
            output[i] = (i + 1) * source[j] * source[j]
            gradient[j] = 2 * (i + 1) * source[j]

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("source_keys", "keys", "source", "output", "gradient")
    }
    symbols["order"] = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "order", ti.i32, ndim=1)
    sequence = ti.algorithms.primitive_sequence().sort(keys, order)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, symbols["source_keys"], symbols["keys"], symbols["order"])
    builder.append_native(sequence)
    builder.dispatch(gather, symbols["source"], symbols["order"], symbols["output"], symbols["gradient"])
    graph = builder.compile()
    frames = {name: ti.ndarray(ti.f32, n) for name in ("source_keys", "source", "output", "gradient")}
    frames.update(keys=keys, order=order)
    source_keys = np.array([math.nan, 0.0, -0.0, 3, -7, 3, math.inf, -math.inf, math.nan] + list(range(8)), np.float32)
    source = np.arange(n, dtype=np.float32) / 4 - 1
    frames["source_keys"].from_numpy(source_keys)
    frames["source"].from_numpy(source)
    graph.run(frames)
    expected_order = sorted(
        range(n), key=lambda i: (math.isnan(source_keys[i]), 0 if math.isnan(source_keys[i]) else source_keys[i])
    )
    expected = np.array([(i + 1) * source[j] ** 2 for i, j in enumerate(expected_order)])
    gradient = np.zeros(n)
    for i, j in enumerate(expected_order):
        gradient[j] = 2 * (i + 1) * source[j]
    np.testing.assert_array_equal(order.to_numpy(), expected_order)
    np.testing.assert_allclose(frames["output"].to_numpy(), expected)
    np.testing.assert_allclose(frames["gradient"].to_numpy(), gradient)
    module = ti.aot.Module()
    before = keys.to_numpy().tobytes(), order.to_numpy().tobytes()
    plan = module.add_execution_plan("step", graph, source_bindings=dict(keys=keys, order=order))
    assert before == (keys.to_numpy().tobytes(), order.to_numpy().tobytes()), "Export mutated live storage"
    assert [op["kind"] for op in json.loads(plan.to_json())["operations"]] == ["kernel", "sort", "kernel"]
    module.add_execution_plan("again", graph, source_bindings=dict(keys=keys, order=order))
    assert len(module._exported_kernels) == 2
    standalone_builder = ti.graph.GraphBuilder()
    standalone_builder.append_native(sequence)
    standalone_graph = standalone_builder.compile()
    module.add_execution_plan("sort", standalone_graph, source_bindings=dict(keys=keys, order=order))
    standalone_graph.close()
    inverse_builder = ti.graph.GraphBuilder()
    inverse_builder.append_native(ti.algorithms.primitive_sequence().sort(keys, order).sort(order, keys))
    inverse_graph = inverse_builder.compile()
    module.add_execution_plan("roundtrip", inverse_graph, source_bindings=dict(keys=keys, order=order))
    inverse_graph.close()
    graph.close()
    sequence.clear()
    artifact = tmp_path / "sequence.tcm"
    module.archive(str(artifact))
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_primitive_sequence_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_primitive_sequence_export_preflight_and_binding_identity():
    keys, order = ti.ndarray(ti.f32, 3), ti.ndarray(ti.i32, 3)
    sequence = ti.algorithms.primitive_sequence().sort(keys, order)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    for bindings in (
        None,
        dict(keys=keys),
        dict(keys=keys, alias=keys, order=order),
        dict(keys=keys, order=order, unused=ti.ndarray(ti.i32, 1)),
    ):
        with pytest.raises((ValueError, ti.TaichiRuntimeError)):
            module.add_execution_plan("bad", graph, source_bindings=bindings)
        assert not module._content and not module._exported_kernels and not module._execution_plans
    sequence._calls[0].kwargs["descending"] = True
    with pytest.raises(ti.TaichiRuntimeError, match="strategy"):
        module.add_execution_plan("bad", graph, source_bindings=dict(keys=keys, order=order))
    sequence._calls[0].kwargs["descending"] = False
    sequence.scan(order)
    with pytest.raises(ti.TaichiRuntimeError, match="'scan'"):
        module.add_execution_plan("bad", graph, source_bindings=dict(keys=keys, order=order))
    assert not module._content and not module._exported_kernels
    sequence._calls.pop()
    module.add_execution_plan("good", graph, source_bindings=dict(keys=keys, order=order))
    graph.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_source_bindings_externalize_fixed_storage_and_reject_inherited_export(tmp_path):
    from taichi_forge.algorithms._algorithms import _PrimitiveSequenceGraphExecutable
    from taichi_forge.graph._native import NativeGraphNode

    @ti.kernel
    def copy(source: ti.types.ndarray(ti.f32, 1), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = source[i] + 2

    source = ti.ndarray(ti.f32, 3)
    symbol = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder._bind_internal_runtime_arg(symbol("internal"), source)
    builder.dispatch(copy, symbol("internal"), symbol("output"))
    graph = builder.compile()
    module = ti.aot.Module()
    plan = module.add_execution_plan("copy", graph, source_bindings={"input": source})
    assert plan.argument_names == ("input", "output")
    artifact = tmp_path / "externalized.tcm"
    graph.close()
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        arrays = {name: runtime.ndarray("f32", 3) for name in ("input", "output")}
        arrays["input"].write(np.array([3, 1, 2], np.float32).tobytes())
        with loaded.prepare_execution("copy", arrays) as call:
            call()
            np.testing.assert_array_equal(np.frombuffer(arrays["output"].read(), np.float32), [5, 3, 4])

    class ExtraBehavior(_PrimitiveSequenceGraphExecutable):
        def run(self):
            super().run()
            source.fill(42)

    class ExtraNode(NativeGraphNode):
        def compile(self):
            return ExtraBehavior(ti.algorithms.primitive_sequence().sort(source))

    builder = ti.graph.GraphBuilder()
    builder.append_native(ExtraNode())
    graph = builder.compile()
    target = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="reconstruction description"):
        target.add_execution_plan("bad", graph, source_bindings={"input": source})
    assert not target._content
    graph.close()
