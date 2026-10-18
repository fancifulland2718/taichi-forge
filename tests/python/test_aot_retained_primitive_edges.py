"""Cold export boundaries and retained native storage layouts."""

import json

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_retained_unique_partial_empty_and_untouched_tails(tmp_path):
    keys, output, count = ti.ndarray(ti.i32, 7), ti.ndarray(ti.i32, 9), ti.ndarray(ti.i32, 2)
    module = ti.aot.Module()
    plans = {}
    for size in (0, 2, 5, 7):
        sequence = ti.algorithms.primitive_sequence().unique(keys, output, count, size=size)
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()
        sources = dict(count=count) if size == 0 else dict(keys=keys, output=output, count=count)
        plans[size] = module.add_execution_plan(str(size), graph, source_bindings=sources)
        if size == 0:
            assert plans[size].argument_names == ("count",)
            assert not module._exported_kernels
        graph.close()
        sequence.clear()
    assert len(module._exported_kernels) == 1
    artifact = tmp_path / "partial.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        arrays = {name: runtime.ndarray("i32", size) for name, size in (("keys", 7), ("output", 9), ("count", 2))}
        arrays["keys"].write(np.array([3, 3, 1, 1, 5, 5, 5], np.int32).tobytes())
        for size, expected in ((7, [3, 1, 5]), (0, []), (2, [3]), (5, [3, 1, 5])):
            arrays["output"].write(np.full(9, -17, np.int32).tobytes())
            arrays["count"].write(np.array([99, 71], np.int32).tobytes())
            bindings = {name: arrays[name] for name in plans[size].argument_names}
            with loaded.prepare_execution(str(size), bindings) as call:
                call()
                assert call.info["resource_bytes"] == (0 if size == 0 else 7 * 4)
                np.testing.assert_array_equal(np.frombuffer(arrays["count"].read(), np.int32), [len(expected), 71])
                np.testing.assert_array_equal(
                    np.frombuffer(arrays["output"].read(), np.int32), expected + [-17] * (9 - len(expected))
                )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_retained_field_export_failure_does_not_publish_views():
    keys, output = ti.field(ti.i32, shape=5), ti.field(ti.i32, shape=5)
    count = ti.field(ti.i32, shape=())
    sequence = ti.algorithms.primitive_sequence().unique(keys, output, count).count_if(output)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="'check'"):
        module.add_execution_plan("bad", graph, snapshot_state=True)
    assert not module._fields and not module._array_fields and not module._exported_kernels
    assert not module._execution_plans and not module._content
    sequence._calls.pop()
    module.add_execution_plan("valid", graph, snapshot_state=True)
    assert len(module._array_fields) == 3 and len(module._exported_kernels) == 1
    graph.close()
    sequence.clear()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_fixed_compact_unaligned_multi_column_views(tmp_path):
    module = ti.aot.Module()
    names = ("keys", "payload", "flags", "key_output", "payload_output", "count")
    module.add_execution_plan("step", ti.aot.ExecutionPlan(
        [dict(kind="compact", flags="flags_view", count="count_view", columns=[
            dict(values="keys_view", output="key_output_view"),
            dict(values="payload_view", output="payload_output_view"),
        ])],
        resources=[dict(kind="ndarray_view", name=name + "_view", array=name,
                        offset=1, shape=[1 if name == "count" else 7]) for name in names],
    ))
    artifact = tmp_path / "compact-offsets.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        arrays = {name: runtime.ndarray(dtype, 3 if name == "count" else 9)
                  for name, dtype in zip(names, ("u64", "f32", "i32", "u64", "f32", "i32"))}
        keys = np.arange(9, dtype=np.uint64) + 2**40
        payload = np.arange(9, dtype=np.float32) / 4
        expected_keys, expected_payload = np.full(9, 91, np.uint64), np.full(9, 92, np.float32)
        arrays["keys"].write(keys.tobytes())
        arrays["payload"].write(payload.tobytes())
        arrays["key_output"].write(expected_keys.tobytes())
        arrays["payload_output"].write(expected_payload.tobytes())
        arrays["count"].write(np.array([73, -1, 74], np.int32).tobytes())
        with loaded.prepare_execution("step", arrays) as call:
            for flags in ([1] * 9, [0, -1, 0, 2, 0, 0, 3, 0, 0], [0] * 9):
                arrays["flags"].write(np.array(flags, np.int32).tobytes())
                selected = [i for i in range(1, 8) if flags[i]]
                expected_keys[1:1 + len(selected)] = keys[selected]
                expected_payload[1:1 + len(selected)] = payload[selected]
                call()
                assert arrays["keys"].read() == keys.tobytes()
                assert arrays["payload"].read() == payload.tobytes()
                assert arrays["key_output"].read() == expected_keys.tobytes()
                assert arrays["payload_output"].read() == expected_payload.tobytes()
                assert arrays["count"].read() == np.array([73, len(selected), 74], np.int32).tobytes()
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_compact_exports_offset_record_views(tmp_path):
    values, output = (ti.Vector.field(2, ti.f32, shape=9) for _ in range(2))
    flags, count = ti.field(ti.i32, shape=9), ti.field(ti.i32, shape=())
    source = np.arange(18, dtype=np.float32).reshape(9, 2)
    values.from_numpy(source)
    output.from_numpy(np.full((9, 2), -99, np.float32))
    flags.from_numpy(np.array([0, 0, 1, 0, -1, 2, 0, 0, 0], np.int32))
    count[None] = -1
    view = lambda owner, start, stop: ti.experimental.ndarray_view(owner, slices=slice(start, stop))
    plan = ti.algorithms.prepare_compact(view(values, 2, 7), view(flags, 2, 7), view(output, 1, 6), count)

    @ti.kernel
    def read(result: ti.types.ndarray(ti.f32, 2), counts: ti.types.ndarray(ti.i32, 1)):
        for i in range(9):
            for j in ti.static(range(2)):
                result[i, j] = output[i][j]
        counts[0] = count[None]

    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record())
    builder.dispatch(
        read,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=2),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counts", ti.i32, ndim=1),
    )
    graph = builder.compile()
    module = ti.aot.Module()
    exported = module.add_execution_plan("step", graph, snapshot_state=True)
    assert len([r for r in json.loads(exported.to_json())["resources"] if r["kind"] == "ndarray_view"]) == 3
    artifact = tmp_path / "offset-records.tcm"
    module.archive(str(artifact))
    graph.close()
    plan.close()
    flags.fill(0)  # Consumer state must be detached from its producer's snapshot.
    expected = np.full((9, 2), -99, np.float32)
    expected[1:4] = source[[2, 4, 5]]
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        arrays = dict(result=runtime.ndarray("f32", (9, 2)), counts=runtime.ndarray("i32", 1))
        with loaded.prepare_execution("step", arrays) as call:
            call()
            assert call.info["resource_bytes"] == 0
            np.testing.assert_array_equal(np.frombuffer(arrays["result"].read(), np.float32).reshape(9, 2), expected)
            np.testing.assert_array_equal(np.frombuffer(arrays["counts"].read(), np.int32), [3])
