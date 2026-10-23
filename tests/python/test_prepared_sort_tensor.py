"""Prepared sort keeps whole packed tensor records and its cold binding contract."""

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("persistent", (False, True))
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_tensor_sort_offsets_recording_and_retirement(persistent):
    if persistent and ti.cfg.arch == ti.cpu:
        pytest.skip("Persistent recording is a GPU policy")
    n = 19
    keys = ti.ndarray(ti.u64, n)
    identities = []
    for dtype, shape in ((ti.f32, (3,)), (ti.f32, (1, 3)), (ti.i64, (2, 2))):
        values = (
            ti.Vector.ndarray(shape[0], dtype, n)
            if len(shape) == 1
            else ti.Matrix.ndarray(*shape, dtype, n)
        )
        key_view = ti.experimental.ndarray_view(keys, slices=(slice(1, 18),))
        value_view = ti.experimental.ndarray_view(values, slices=(slice(1, 18),))
        source = ((np.arange(n, dtype=np.uint64) * 7) % 5) + np.uint64(1 << 60)
        payload = np.arange(
            n * np.prod(shape), dtype=np.int64 if dtype == ti.i64 else np.float32
        ).reshape((n, *shape))
        if dtype == ti.f32:
            payload.view(np.uint32).reshape(-1)[:5] = [
                0x80000000,
                0,
                0x7FC00017,
                0xFFC00023,
                0x7F800000,
            ]
        else:
            payload += np.int64(1 << 60)
        keys.from_numpy(source)
        values.from_numpy(payload)
        ti.sync()
        program = ti.lang.impl.get_runtime().prog
        before = program._runtime_statistics_snapshot()
        plan = ti.algorithms.prepare_sort(key_view, value_view)
        builder = ti.graph.GraphBuilder()
        builder.append_native(plan.record(persistent=persistent))
        graph = builder.compile()
        after = program._runtime_statistics_snapshot()
        assert before["submission"] == after["submission"]
        # Explicit private-workspace preparation acquires the backend lock;
        # that instrumentation counter is not a GPU wait or synchronization.
        synchronization = lambda snapshot: {
            key: value for key, value in snapshot["synchronization"].items()
            if not persistent or key != "backend_lock_samples"
        }
        assert synchronization(before) == synchronization(after)
        assert values.to_numpy().tobytes() == payload.tobytes()
        assert plan.report()["workspace_bytes_last_observed"] is None
        identities.append(plan.report()["physical_plan_id"])
        assert plan.report()["value_element_shape"] == list(shape)
        for execute in (plan.run, lambda: graph.run({}), plan.run):
            keys.from_numpy(source)
            values.from_numpy(payload)
            execute()
            order = np.argsort(source[1:18], kind="stable")
            expected_keys, expected_payload = source.copy(), payload.copy()
            expected_keys[1:18] = source[1:18][order]
            expected_payload[1:18] = payload[1:18][order]
            assert keys.to_numpy().tobytes() == expected_keys.tobytes()
            assert values.to_numpy().tobytes() == expected_payload.tobytes()
            source = source[::-1].copy()
        keys.from_numpy(source)
        values.from_numpy(payload)
        ticket = graph.submit({})
        graph.close()
        plan.close()
        ticket.wait()
        with pytest.raises(RuntimeError, match="closed"):
            plan.run()
        expected_payload = payload.copy()
        expected_payload[1:18] = payload[1:18][np.argsort(source[1:18], kind="stable")]
        assert values.to_numpy().tobytes() == expected_payload.tobytes()
    assert len(set(identities)) == len(identities)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_tensor_sort_dense_field_offset_and_retirement():
    builder = ti.FieldsBuilder()
    sentinel = ti.field(ti.i32)
    values = ti.Vector.field(3, ti.f32)
    builder.dense(ti.i, 1).place(sentinel)
    builder.dense(ti.i, 17).place(values)
    tree = builder.finalize()
    keys = ti.ndarray(ti.i32, 17)
    sentinel[0] = 1234567
    source = np.arange(17, dtype=np.int32)[::-1].copy()
    payload = np.arange(51, dtype=np.float32).reshape(17, 3)
    keys.from_numpy(source)
    values.from_numpy(payload)
    from taichi_forge.lang._storage_view import describe_storage

    description = describe_storage(values)
    assert description.descriptor.byte_offset > 0
    assert description.descriptor.byte_offset % 12 != 0
    plan = ti.algorithms.prepare_sort(keys, values)
    plan.run()
    np.testing.assert_array_equal(values.to_numpy(), payload[::-1])
    assert sentinel[0] == 1234567
    tree.destroy()
    with pytest.raises(
        RuntimeError, match="retired|destroyed|generation|stale|inactive"
    ):
        plan.run()
    plan.close()
