"""Static CPU partitions eliminate empty work without specializing call values."""

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=4, offline_cache=False)
def test_legacy_adaptive_flag_does_not_mutate_global_configuration(monkeypatch):
    from taichi_forge.lang import misc

    messages = []
    monkeypatch.setattr(misc._logging, "warn", messages.append)

    @ti.kernel
    def run() -> ti.i32:
        value = 0
        ti.loop_config(block_dim_adaptive=False, parallelize=2)
        for i in range(17):
            value += i
        return value

    assert run() == 136
    assert len(messages) == 1 and "has no effect" in messages[0]
    assert ti.lang.impl.current_cfg().cpu_block_dim_adaptive
    assert run() == 136
    assert len(messages) == 1  # Never warn from the warm execution path.


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=16, offline_cache=False)
def test_static_chunks_preserve_parallelize_and_dynamic_calls():
    output = ti.ndarray(ti.i32, 1025)

    @ti.kernel
    def fixed(out: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(parallelize=4)
        for i in range(-17, 1008):
            out[i + 17] = i * 3

    @ti.kernel
    def dynamic(out: ti.types.ndarray(ti.i32, 1), end: ti.i32):
        ti.loop_config(parallelize=4)
        for i in range(-17, end):
            out[i + 17] = i * 3

    fixed(output)
    ti.profiler.set_cpu_scheduler_telemetry(True, reset=True)
    try:
        fixed(output)
        stats = ti.profiler.query_cpu_scheduler_telemetry()
        assert stats["tasks_requested"] == 3
        assert stats["tasks_completed"] == 3
        assert stats["max_requested_threads"] == 4
    finally:
        ti.profiler.set_cpu_scheduler_telemetry(False)
    expected = np.arange(-17, 1008, dtype=np.int32) * 3
    np.testing.assert_array_equal(output.to_numpy(), expected)
    for end in (-17, -16, 512, 1008):
        output.fill(-91)
        dynamic(output, end)
        reference = np.full(1025, -91, dtype=np.int32)
        reference[:end + 17] = expected[:end + 17]
        np.testing.assert_array_equal(output.to_numpy(), reference)


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=16, offline_cache=False)
def test_static_reduction_and_serial_side_effects(tmp_path):
    @ti.kernel
    def sum_range(out: ti.types.ndarray(ti.i32, 1)):
        out[0] = 9
        for i in range(-17, 1008):
            ti.atomic_add(out[0], i)
        ti.loop_config(serialize=True)
        for i in range(3):
            out[0] = out[0] * 2 + i

    expected = 9 + sum(range(-17, 1008))
    for i in range(3):
        expected = expected * 2 + i
    output = ti.ndarray(ti.i32, 1)
    sum_range(output)
    np.testing.assert_array_equal(output.to_numpy(), [expected])
    builder = ti.graph.GraphBuilder()
    builder.dispatch(sum_range, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1))
    graph = builder.compile()
    graph.run(dict(out=output))
    np.testing.assert_array_equal(output.to_numpy(), [expected])
    module = ti.aot.Module()
    module.add_execution_plan("sum", graph)
    archive = tmp_path / "sum.tcm"
    module.archive(str(archive))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(archive) as loaded:
        target = runtime.ndarray("i32", (1,))
        try:
            with loaded.prepare_execution("sum", dict(out=target)) as call:
                call()
                call()
            np.testing.assert_array_equal(np.frombuffer(target.read(), np.int32), [expected])
        finally:
            target.close()
    graph.close()
