import json
import tempfile
from pathlib import Path

import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_bounded_runtime_trace_records_program_boundaries_and_exports_json():
    @ti.kernel
    def increment(values: ti.types.ndarray()):
        for i in values:
            values[i] += 1

    values = ti.ndarray(ti.i32, shape=4)
    increment(values)
    ti.sync()

    prog = ti.lang.impl.get_runtime().prog
    disabled = prog._runtime_trace_snapshot()
    assert disabled["enabled"] is False
    assert disabled["event_capacity"] == 0
    assert disabled["allocated_bytes"] == 0
    counters_before = prog._runtime_statistics_snapshot()["trace"]

    started = prog._runtime_trace_start(max_threads=2, events_per_thread=32)
    assert started["enabled"] is True
    assert started["event_capacity"] == 64
    assert 0 < started["allocated_bytes"] <= 64 * 32 + 2 * 32

    values.from_numpy(np.arange(4, dtype=np.int32))
    increment(values)
    result = values.to_numpy()
    ti.sync()
    np.testing.assert_array_equal(result, np.arange(4, dtype=np.int32) + 1)

    stopped = prog._runtime_trace_stop()
    assert stopped["enabled"] is False
    assert stopped["recorded_events"] >= 4
    assert stopped["dropped_events"] == 0
    counters_after = prog._runtime_statistics_snapshot()["trace"]
    assert (
        counters_after["recorded_events"] - counters_before["recorded_events"]
        == stopped["recorded_events"]
    )
    assert (
        counters_after["dropped_events"] - counters_before["dropped_events"]
        == stopped["dropped_events"]
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "runtime-trace.json"
        assert prog._runtime_trace_export(str(path)) is True
        exported = json.loads(path.read_text(encoding="utf-8"))

    metadata = exported["taichiRuntimeTrace"]
    assert metadata["schemaVersion"] == 1
    assert metadata["programDomain"] == stopped["program_domain"]
    assert metadata["session"] == stopped["session"]
    assert metadata["recordedEvents"] == stopped["recorded_events"]
    assert metadata["droppedEvents"] == 0
    events = exported["traceEvents"]
    assert len(events) == stopped["recorded_events"]
    names = {event["name"] for event in events}
    assert "runtime.kernel.submit" in names
    assert "runtime.transfer.h2d" in names
    assert "runtime.transfer.d2h" in names
    assert "runtime.synchronize" in names
    for event in events:
        assert event["ph"] == "X"
        assert event["ts"] >= 0
        assert event["dur"] >= 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_bounded_runtime_trace_overflow_drops_without_growing():
    @ti.kernel
    def step(value: ti.types.ndarray()):
        value[0] += 1

    value = ti.ndarray(ti.i32, shape=1)
    step(value)
    ti.sync()

    prog = ti.lang.impl.get_runtime().prog
    counters_before = prog._runtime_statistics_snapshot()["trace"]
    started = prog._runtime_trace_start(max_threads=1, events_per_thread=2)
    for _ in range(5):
        step(value)
    stopped = prog._runtime_trace_stop()
    ti.sync()

    assert stopped["event_capacity"] == started["event_capacity"] == 2
    assert stopped["allocated_bytes"] == started["allocated_bytes"]
    assert 0 < stopped["allocated_bytes"] <= 2 * 32 + 32
    assert stopped["recorded_events"] == 2
    assert stopped["dropped_events"] == 3
    counters_after = prog._runtime_statistics_snapshot()["trace"]
    assert (
        counters_after["recorded_events"] - counters_before["recorded_events"]
        == 2
    )
    assert counters_after["dropped_events"] - counters_before["dropped_events"] == 3

    restarted = prog._runtime_trace_start(max_threads=1, events_per_thread=1)
    assert restarted["session"] == stopped["session"] + 1
    assert restarted["recorded_events"] == 0
    assert restarted["dropped_events"] == 0
    prog._runtime_trace_stop()
