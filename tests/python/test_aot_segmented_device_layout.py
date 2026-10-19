"""Capture only selected immutable topology once, after structural preflight."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_device_layout_sequence_snapshot_runtime_only_and_borrowed(tmp_path, monkeypatch):
    module = ti.aot.Module()
    cases = []
    configurations = (
        (ti.i32, 277, [0, 0, 3, 3, 257, 269, 269]),
        (ti.f32, 19, [0, 0, 1, 5, 5, 13, 13]),
        (ti.i32, 7, [0, 0, 0, 0]),
    )
    for dtype, capacity, offsets in configurations:
        scalar = np.int32 if dtype == ti.i32 else np.float32
        counts = ti.ndarray(ti.i32, len(offsets) - 1)
        counts.from_numpy(np.diff(np.asarray(offsets, np.int32)))
        layout = ti.algorithms.SegmentedLayout._from_device_counts(
            counts, num_items=offsets[-1], max_segment_length=max(np.diff(offsets)), capacity=capacity
        )
        counts.fill(-99)  # The published layout no longer borrows construction counts.
        values = ti.ndarray(dtype, capacity)
        output = ti.ndarray(dtype, len(offsets) - 1)
        for method in ("grouped", "serial"):
            if ti.cfg.arch == ti.vulkan and method == "grouped" and not ti.lang.impl.get_runtime().prog.vulkan_grouped_reduce_atomic_value_type_available(
                ti.algorithms._algorithms._grouped_reduce_value_type(dtype)
            ):
                continue
            sequence = ti.algorithms.primitive_sequence()
            for _ in range(2):
                sequence.segmented_reduce(values, layout, output, method=method)
            builder = ti.graph.GraphBuilder()
            builder.append_native(sequence)
            graph = builder.compile()
            output.fill(73)
            program = ti.lang.impl.get_runtime().prog
            original = type(program).copy_ndarrays_to_host
            copies = []

            def capture(program, sources, destinations, **options):
                assert options == {"snapshot": True}
                selected = layout._segment_ids if method == "grouped" else layout._offsets
                assert sources == [selected.arr]
                copies.append(sum(destination.nbytes for destination in destinations))
                return original(program, sources, destinations, **options)

            name = f"{dtype}_{capacity}_{method}"
            with monkeypatch.context() as cold:
                cold.setattr(type(program), "copy_ndarrays_to_host", capture)
                plan = module.add_execution_plan(name, graph, source_bindings=dict(values=values, output=output))
            assert copies == [4 * (capacity if method == "grouped" else len(offsets))]
            document = json.loads(plan.to_json())
            assert len(document["operations"]) == 2
            frozen = [r for r in document["resources"] if "initial_data" in r]
            assert len(frozen) == 1
            np.testing.assert_array_equal(output.to_numpy(), np.full(len(offsets) - 1, 73, scalar))
            samples = []
            for iteration in (1, 2, 0):
                source = (np.arange(capacity) % 7 * iteration).astype(scalar)
                expected = np.asarray([sum(source[a:b]) for a, b in zip(offsets, offsets[1:])], scalar)
                source[offsets[-1]:] = 91
                samples.append(dict(source=source.tobytes().hex(), expected=expected.tobytes().hex()))
            cases.append(dict(name=name, dtype=str(dtype), size=capacity, segments=len(offsets) - 1,
                              initial=np.full(len(offsets) - 1, 73, scalar).tobytes().hex(), samples=samples))
            graph.close()
            sequence.clear()
        layout._offsets.fill(-77)
        layout._segment_ids.fill(-77)  # Export owns the captured generation.
    artifact = tmp_path / "device-layout.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(cases))
    arch = "cpu" if ti.cfg.arch == ti.cpu else str(ti.cfg.arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, *(["-S"] if mode == "standalone" else []),
             str(Path(__file__).with_name("aot_segmented_sequence_probe.py")), mode, arch, str(artifact)],
            env=os.environ.copy(), capture_output=True, text=True, timeout=60,
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "Validation Error" not in text and "SYNC-HAZARD" not in text, text


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_device_layout_does_not_snapshot_rejected_export(monkeypatch):
    counts = ti.ndarray(ti.i32, 2)
    counts.fill(2)
    layout = ti.algorithms.SegmentedLayout._from_device_counts(counts, num_items=4, max_segment_length=2)
    values, output, extra = ti.ndarray(ti.i32, 4), ti.ndarray(ti.i32, 2), ti.ndarray(ti.i32, 2)
    sequence = ti.algorithms.primitive_sequence()
    sequence.segmented_reduce(values, layout, output)
    sequence.transform(output, extra, method="kernel")
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    program = ti.lang.impl.get_runtime().prog

    def forbidden(*args, **kwargs):
        pytest.fail("Rejected Graph must not copy device topology")

    monkeypatch.setattr(type(program), "copy_ndarrays_to_host", forbidden)
    with pytest.raises(ti.TaichiRuntimeError, match="available native strategy"):
        ti.aot.Module().add_execution_plan(
            "rejected", graph, source_bindings=dict(values=values, output=output, extra=extra)
        )
    graph.close()
