"""Sequence scan closures preserve math, padding, aliasing and private state."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


def reference(source, offsets, inclusive, method, initial):
    scalar = source.dtype.type
    result = initial.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        if method == "global_scan":
            result[:] = np.cumsum(source, dtype=source.dtype)
        for begin, end in zip(offsets, offsets[1:]):
            acc = scalar(0)
            for index in range(begin, end):
                if inclusive:
                    acc = source[index] if index == begin else scalar(acc + source[index])
                    result[index] = acc
                else:
                    result[index] = acc
                    acc = scalar(acc + source[index])
    return result


def graph_for(sequence):
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=2)
def test_sequence_scan_runtime_only_and_borrowed(tmp_path):
    arch = ti.lang.impl.current_cfg().arch
    module = ti.aot.Module(caps=["spirv_has_int64", "spirv_has_float64"] if arch == ti.vulkan else [])
    cases = []
    configs = [
        (ti.f32, "auto", True, False, False, False),
        (ti.f64, "serial", False, True, True, False),
        (ti.i32, "global_scan", True, False, True, False),
        (ti.u64, "global_scan", False, True, False, False),
        (ti.u32, "serial", False, False, False, False),
        (ti.i64, "global_scan", True, False, False, False),
        (ti.i32, "serial", True, False, True, True),
        (ti.i32, "global_scan", False, False, True, True),
    ]
    for dtype, method, inclusive, inplace, device, empty in configs:
        capacity, offsets = (7, [0, 0, 0, 0]) if empty else (277, [0, 0, 1, 3, 3, 257, 269, 269])
        scalar = np.dtype(str(dtype).replace("i", "int").replace("u", "uint").replace("f", "float")).type
        if device:
            counts = ti.ndarray(ti.i32, len(offsets) - 1)
            counts.from_numpy(np.diff(np.asarray(offsets, np.int32)))
            layout = ti.algorithms.SegmentedLayout._from_device_counts(
                counts, num_items=offsets[-1], max_segment_length=max(np.diff(offsets)), capacity=capacity
            )
            counts.fill(99)
        else:
            layout = ti.algorithms.SegmentedLayout.from_offsets(np.asarray(offsets, np.int32), capacity=capacity)
        values = ti.ndarray(dtype, capacity)
        output = values if inplace else ti.ndarray(dtype, capacity)
        sequence = ti.algorithms.primitive_sequence().segmented_scan(
            values, layout, output, method=method, inclusive=inclusive
        )
        sources = dict(values=values) if inplace else dict(values=values, output=output)
        graph = graph_for(sequence)
        name = f"{dtype}_{method}_{empty}"
        values.fill(73)
        output.fill(73)
        plan = module.add_execution_plan(name, graph, source_bindings=sources)
        np.testing.assert_array_equal(output.to_numpy(), np.full(capacity, 73, scalar))
        document = json.loads(plan.to_json())
        data = [r for r in document["resources"] if "initial_data" in r]
        assert len(data) == 1 and data[0]["shape"] == [len(offsets)]
        assert len(document["operations"]) == (3 + int(not inplace) if method == "global_scan" else 1)
        scratch = [r for r in document["resources"] if r["kind"] == "ndarray" and "initial_data" not in r]
        assert len(scratch) == int(method == "global_scan")
        samples = []
        for iteration in (1, 2, 0):
            source = ((np.arange(capacity) % 7) * iteration).astype(scalar)
            if np.issubdtype(scalar, np.integer) and iteration:
                source[1:3] = np.iinfo(scalar).max
            elif iteration == 2:
                source[:3] = [-0.0, np.nan, np.inf]
            values.from_numpy(source)
            initial = source if inplace else np.full(capacity, 73, scalar)
            if not inplace:
                output.from_numpy(initial)
            expected = reference(source, offsets, inclusive, method, initial)
            sequence.run()
            np.testing.assert_equal(output.to_numpy(), expected)
            assert output.to_numpy().tobytes()[:np.dtype(scalar).itemsize] == expected.tobytes()[:np.dtype(scalar).itemsize]
            samples.append(dict(source=source.tobytes().hex(), initial=initial.tobytes().hex(),
                                expected=expected.tobytes().hex()))
        # Warm export must retain this request, without replaying it.
        before = output.to_numpy()
        warm = module.add_execution_plan(name + "_warm", graph, source_bindings=sources)
        assert [op["kind"] for op in json.loads(warm.to_json())["operations"]] == [
            op["kind"] for op in document["operations"]
        ]
        np.testing.assert_equal(output.to_numpy(), before)
        cases.append(dict(name=name, dtype=str(dtype), size=capacity, inplace=inplace, samples=samples))
        graph.close()
        sequence.clear()
    artifact = tmp_path / "sequence-scans.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, *(["-S"] if mode == "standalone" else []),
             str(Path(__file__).with_name("aot_segmented_sequence_scan_probe.py")), mode, backend, str(artifact)],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90,
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "Validation Error" not in text and "SYNC-HAZARD" not in text, text


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_sequence_scan_field_snapshot_and_downstream_ad(tmp_path, monkeypatch):
    from taichi_forge.algorithms._segmented_definition import SegmentedDefinition

    layout = ti.algorithms.SegmentedLayout.from_offsets(np.asarray([0, 0, 1, 3, 3, 7], np.int32), capacity=9)
    floating = ti.field(ti.f32, shape=9, offset=5)
    serial = ti.field(ti.f32, shape=9, offset=11)
    integer = ti.field(ti.i32, shape=9, offset=17)
    global_scan = ti.field(ti.i32, shape=9, offset=21)
    floating.from_numpy(np.arange(9, dtype=np.float32))
    integer.from_numpy(np.arange(9, dtype=np.int32))
    serial.fill(73)
    sequence = ti.algorithms.primitive_sequence()
    sequence.segmented_scan(floating, layout, serial, inclusive=False)
    sequence.segmented_scan(integer, layout, global_scan, method="global_scan")
    sequence.run()

    def no_redefine(*args, **kwargs):
        raise AssertionError("Scan replay/export must reuse the ordinary call's definition")

    weights = ti.field(ti.f32, shape=9, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    weights.fill(2)

    @ti.kernel
    def objective():
        for i in range(9):
            loss[None] += serial[i + 11] * weights[i]

    @ti.kernel
    def publish(out: ti.types.ndarray(ti.f32, 1), integers: ti.types.ndarray(ti.i32, 1)):
        for i in range(9):
            out[i] = weights.grad[i]
            integers[i] = global_scan[i + 21]

    # Only the downstream DSL objective is differentiated; segmented scan has
    # no automatic adjoint and is not silently differentiated by this export.
    with ti.ad.Tape(loss):
        objective()
    weights.grad.fill(0)
    loss.grad[None] = 1
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(objective.grad)
    builder.dispatch(publish, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1),
                     ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "integers", ti.i32, ndim=1))
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state"):
        module.add_execution_plan("bad", graph)
    with monkeypatch.context() as patch:
        patch.setattr(SegmentedDefinition, "__init__", no_redefine)
        sequence.run()
        module.add_execution_plan("fields", graph, snapshot_state=True)
    artifact = tmp_path / "fields.tcm"
    module.archive(str(artifact))
    graph.close()
    floating.fill(99)
    integer.fill(99)
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        args = dict(out=runtime.ndarray("f32", 9), integers=runtime.ndarray("i32", 9))
        with loaded.prepare_execution("fields", args) as call:
            call()
            np.testing.assert_equal(np.frombuffer(args["out"].read(), np.float32), [0, 0, 1, 0, 3, 7, 12, 73, 73])
            np.testing.assert_equal(np.frombuffer(args["integers"].read(), np.int32), [0, 1, 3, 3, 7, 12, 18, 28, 36])
        for value in args.values():
            value.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_sequence_scan_shared_workspace_auto_and_cache_identity(tmp_path):
    size = 65539
    values, left, right = (ti.ndarray(ti.i32, size) for _ in range(3))
    large = ti.algorithms.SegmentedLayout.from_offsets(np.asarray([0, size - 2], np.int32), capacity=size)
    small = ti.algorithms.SegmentedLayout.from_offsets(np.asarray([0, 2], np.int32), capacity=size)
    workspace = ti.algorithms.SegmentedWorkspace()
    sequence = ti.algorithms.primitive_sequence()
    sequence.segmented_scan(values, large, left, workspace=workspace)
    sequence.segmented_scan(values, small, right, workspace=workspace)
    values.fill(1)
    right.fill(73)
    sequence.run()
    assert workspace._last_scan_method == "serial"
    graph = graph_for(sequence)
    module = ti.aot.Module()
    plan = module.add_execution_plan("shared", graph, source_bindings=dict(values=values, left=left, right=right))
    assert [op.get("operation", op["kind"]) for op in json.loads(plan.to_json())["operations"]] == [
        "transform", "scan", "kernel", "kernel", "kernel"
    ]
    artifact = tmp_path / "shared.tcm"
    module.archive(str(artifact))
    executor = workspace._scan_executors[size]
    executor._native_scan_plan.method_name = "unknown_strategy"
    with pytest.raises(ti.TaichiRuntimeError, match="selected cached strategy"):
        module.add_execution_plan("bad", graph, source_bindings=dict(values=values, left=left, right=right))
    graph.close()
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        args = {name: runtime.ndarray("i32", size) for name in ("values", "left", "right")}
        args["values"].write(np.ones(size, np.int32))
        args["right"].write(np.full(size, 73, np.int32))
        with loaded.prepare_execution("shared", args) as call:
            call()
            np.testing.assert_equal(np.frombuffer(args["left"].read(), np.int32), np.arange(1, size + 1))
            expected = np.full(size, 73, np.int32)
            expected[:2] = [1, 2]
            np.testing.assert_equal(np.frombuffer(args["right"].read(), np.int32), expected)
        for value in args.values():
            value.close()
