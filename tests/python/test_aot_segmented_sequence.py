"""Segmented sequence export preserves selected math, topology and owner scopes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


def _graph(sequence):
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_segmented_sequence_runtime_only_and_borrowed(tmp_path, monkeypatch):
    arch = ti.lang.impl.current_cfg().arch
    module = ti.aot.Module(caps=["spirv_has_int64", "spirv_has_float64"] if arch == ti.vulkan else [])
    cases = []
    configs = [(dtype, method, 19, [0, 0, 1, 5, 5, 13, 13])
               for dtype in (ti.i32, ti.u64, ti.f32, ti.f64) for method in ("auto", "serial")]
    configs.extend((ti.i32, method, 7, [0, 0, 0, 0]) for method in ("auto", "serial"))
    if arch == ti.cpu:
        configs.append((ti.f32, "grouped", 262151, [0, 262147, 262147]))
    for dtype, method, capacity, offsets in configs:
        if arch == ti.vulkan and method != "serial" and not ti.lang.impl.get_runtime().prog.vulkan_grouped_reduce_atomic_value_type_available(
            ti.algorithms._algorithms._grouped_reduce_value_type(dtype)
        ):
            continue
        scalar = np.dtype(str(dtype).replace("i", "int").replace("u", "uint").replace("f", "float")).type
        layout = ti.algorithms.SegmentedLayout.from_offsets(np.asarray(offsets, np.int32), capacity=capacity)
        values, output = ti.ndarray(dtype, capacity), ti.ndarray(dtype, len(offsets) - 1)
        sequence = ti.algorithms.primitive_sequence().segmented_reduce(values, layout, output, method=method)
        samples = []
        for iteration in (1, 2, 0):
            source = (np.arange(capacity) % 7).astype(scalar) * scalar(iteration)
            if np.issubdtype(scalar, np.integer) and iteration:
                source[1:3] = np.iinfo(scalar).max
            elif iteration == 2:
                source[:3] = [-0.0, np.nan, np.inf]
            if capacity > 100:
                source.fill(0)
                for part in range(4):
                    begin, end = capacity * part // 4, min(capacity * (part + 1) // 4, offsets[-1])
                    source[begin:begin + 2] = [1e20, 1]
                    source[end - 2:end] = [-1e20, iteration + 1]
                expected = np.asarray([4 * (iteration + 1), 0], scalar)
            else:
                expected = np.zeros(len(offsets) - 1, scalar)
                with np.errstate(over="ignore", invalid="ignore"):
                    for segment, (begin, end) in enumerate(zip(offsets, offsets[1:])):
                        acc = scalar(0)
                        for value in source[begin:end]:
                            acc = scalar(acc + value)
                        expected[segment] = acc
            # Padding is outside every segment, including nonfinite values.
            source[offsets[-1]:] = np.nan if np.issubdtype(scalar, np.floating) else 97
            values.from_numpy(source)
            if method != "serial" or dtype != ti.f32:
                sequence.run()
                np.testing.assert_equal(output.to_numpy(), expected)
            samples.append(dict(source=source.tobytes().hex(), expected=expected.tobytes().hex()))
        output.fill(73)
        graph = _graph(sequence)
        name = f"{dtype}_{method}_{capacity}"
        with monkeypatch.context() as cold:
            cold.setattr(type(layout._offsets), "to_numpy", lambda *a, **k: pytest.fail("device read during export"))
            plan = module.add_execution_plan(name, graph, source_bindings=dict(values=values, output=output))
        np.testing.assert_array_equal(output.to_numpy(), np.full(len(offsets) - 1, 73, scalar))
        doc = json.loads(plan.to_json())
        data = [r for r in doc["resources"] if "initial_data" in r]
        assert len(data) == 1
        assert data[0]["shape"] == [len(offsets) if method == "serial" else capacity]
        assert len(doc["operations"]) == 1
        operation = doc["operations"][0]
        assert operation["kind"] == ("kernel" if method == "serial" else "native")
        if method != "serial":
            assert operation["parameters"]["size"] == capacity
            if arch == ti.cpu:
                assert operation["parameters"]["cpu_max_threads"] == 4
        cases.append(dict(name=name, dtype=str(dtype), size=capacity, segments=len(offsets) - 1,
                          initial=np.full(len(offsets) - 1, 73, scalar).tobytes().hex(), samples=samples))
        graph.close()
        sequence.clear()
    artifact = tmp_path / "segmented-sequences.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, *(["-S"] if mode == "standalone" else []),
             str(Path(__file__).with_name("aot_segmented_sequence_probe.py")), mode, backend, str(artifact)],
            env=os.environ.copy(), capture_output=True, text=True, timeout=60,
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "Validation Error" not in text and "SYNC-HAZARD" not in text, text


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_segmented_sequence_shared_workspace_and_cold_export(tmp_path):
    workspace = ti.algorithms.SegmentedWorkspace(max_items=17, max_segments=3)
    values = ti.ndarray(ti.i32, 17)
    left, right = ti.ndarray(ti.i32, 3), ti.ndarray(ti.i32, 3)
    layouts = [ti.algorithms.SegmentedLayout.from_offsets(np.asarray(o, np.int32), capacity=17)
               for o in ([0, 2, 2, 7], [0, 1, 3, 5])]
    sequence = ti.algorithms.primitive_sequence()
    for layout, output in zip(layouts, (left, right)):
        sequence.segmented_reduce(values, layout, output, workspace=workspace)
    module = ti.aot.Module()
    for name, warm in (("cold", False), ("warm", True)):
        if warm:
            values.fill(1)
            sequence.run()  # The workspace's last plan belongs to right, not left.
        left.fill(73)
        right.fill(74)
        graph = _graph(sequence)
        plan = module.add_execution_plan(name, graph, source_bindings=dict(values=values, left=left, right=right))
        assert len(json.loads(plan.to_json())["operations"]) == 2
        np.testing.assert_array_equal(left.to_numpy(), [73] * 3)
        np.testing.assert_array_equal(right.to_numpy(), [74] * 3)
        graph.close()
    artifact = tmp_path / "shared.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        args = dict(values=runtime.ndarray("i32", 17), left=runtime.ndarray("i32", 3), right=runtime.ndarray("i32", 3))
        for name in ("cold", "warm"):
            with loaded.prepare_execution(name, args) as call:
                for value in (3, 0):
                    args["values"].write(np.full(17, value, np.int32))
                    call()
                    np.testing.assert_array_equal(np.frombuffer(args["left"].read(), np.int32), np.asarray([2, 0, 5]) * value)
                    np.testing.assert_array_equal(np.frombuffer(args["right"].read(), np.int32), np.asarray([1, 2, 2]) * value)
        for value in args.values():
            value.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("device_layout", [False, True])
def test_segmented_serial_field_snapshot_and_explicit_adjoint(tmp_path, device_layout):
    values = ti.field(ti.f32, shape=9, offset=5)
    reduced = ti.field(ti.f32, shape=4, offset=11)
    values.from_numpy(np.arange(9, dtype=np.float32))
    offsets = [0, 0, 2, 2, 7]
    if device_layout:
        counts = ti.ndarray(ti.i32, 4)
        counts.from_numpy(np.diff(np.asarray(offsets, np.int32)))
        layout = ti.algorithms.SegmentedLayout._from_device_counts(
            counts, num_items=7, max_segment_length=5, capacity=9
        )
    else:
        layout = ti.algorithms.SegmentedLayout.from_offsets(np.asarray(offsets, np.int32), capacity=9)
    sequence = ti.algorithms.primitive_sequence().segmented_reduce(values, layout, reduced)

    @ti.kernel
    def publish(out: ti.types.ndarray(ti.f32, 1), gradient: ti.types.ndarray(ti.f32, 1)):
        for j in range(4):
            out[j] = reduced[j + 11] * reduced[j + 11]
        for i in range(9):
            gradient[i] = 0
            if i < 2:
                gradient[i] = 2 * reduced[12]
            elif i < 7:
                gradient[i] = 2 * reduced[14]

    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(publish, *(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, n, ti.f32, ndim=1) for n in ("out", "gradient")))
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state"):
        module.add_execution_plan("no_snapshot", graph)
    plan = module.add_execution_plan("fields", graph, snapshot_state=True)
    assert len([r for r in json.loads(plan.to_json())["resources"] if "initial_data" in r]) == 1
    artifact = tmp_path / "fields.tcm"
    module.archive(str(artifact))
    graph.close()
    values.fill(99)
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        args = dict(out=runtime.ndarray("f32", 4), gradient=runtime.ndarray("f32", 9))
        with loaded.prepare_execution("fields", args) as call:
            for _ in range(2):
                call()
                np.testing.assert_equal(np.frombuffer(args["out"].read(), np.float32), [0, 1, 0, 400])
                np.testing.assert_equal(np.frombuffer(args["gradient"].read(), np.float32), [2, 2, 40, 40, 40, 40, 40, 0, 0])
        for value in args.values():
            value.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_segmented_sequence_rejects_unknown_topology_and_changed_cached_strategy():
    values, output = ti.ndarray(ti.i32, 9), ti.ndarray(ti.i32, 2)
    layout = ti.algorithms.SegmentedLayout.from_offsets(np.asarray([0, 4, 7], np.int32), capacity=9)
    sequence = ti.algorithms.primitive_sequence().segmented_reduce(values, layout, output)
    values.fill(1)
    sequence.run()
    graph = _graph(sequence)
    workspace = sequence._calls[0].workspace._grouped_reduce_workspace
    workspace._native_grouped_reduce_plan.method_name = "unknown_strategy"
    with pytest.raises(ti.TaichiRuntimeError, match="selected cached strategy"):
        ti.aot.Module().add_execution_plan("bad_strategy", graph, source_bindings=dict(values=values, output=output))
    graph.close()
    counts = ti.ndarray(ti.i32, 2)
    counts.from_numpy(np.asarray([4, 3], np.int32))
    layout = ti.algorithms.SegmentedLayout._from_device_counts(counts, num_items=7, max_segment_length=4, capacity=9)
    layout._encoding = "unknown_generation"
    graph = _graph(ti.algorithms.primitive_sequence().segmented_reduce(values, layout, output))
    with pytest.raises(ti.TaichiRuntimeError, match="published immutable topology"):
        ti.aot.Module().add_execution_plan("device_layout", graph, source_bindings=dict(values=values, output=output))
    graph.close()
