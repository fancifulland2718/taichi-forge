import threading

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _expected_runs(keys):
    if keys.size == 0:
        return (
            keys.copy(),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int64),
        )
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(keys[1:] != keys[:-1]).astype(np.int64) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.array([keys.size], dtype=np.int64)))
    return keys[starts], (ends - starts).astype(np.int32), starts


def _fill_outputs(unique_keys, run_lengths, count, sentinel=-97):
    unique_keys.fill(sentinel)
    run_lengths.fill(sentinel)
    count.fill(sentinel)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_rle_unique_ndarray_auto_parity_and_workspace_reuse():
    n = 257
    keys_np = np.repeat(
        np.array([7, 7, -3, 4, 4, 4, 9, 2, 2, 11], dtype=np.int32),
        [1, 3, 2, 1, 5, 4, 2, 6, 3, 1],
    )
    keys_np = np.resize(keys_np, n).astype(np.int32)
    payload_np = (np.arange(n, dtype=np.int32) * 13 - 9).astype(np.int32)
    expected_keys, expected_lengths, starts = _expected_runs(keys_np)

    keys = ti.ndarray(ti.i32, shape=n)
    payload = ti.ndarray(ti.i32, shape=n)
    unique_keys = ti.ndarray(ti.i32, shape=n)
    unique_payload = ti.ndarray(ti.i32, shape=n)
    run_lengths = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)
    keys.from_numpy(keys_np)
    payload.from_numpy(payload_np)
    _fill_outputs(unique_keys, run_lengths, count)
    unique_payload.fill(-97)

    workspace = ti.algorithms.RunLengthWorkspace(max_items=n)
    assert (
        ti.algorithms.experimental_run_length_encode(
            keys,
            unique_keys,
            run_lengths,
            count,
            workspace=workspace,
        )
        is workspace
    )
    run_count = int(count.to_numpy()[0])
    assert run_count == expected_keys.size
    assert np.array_equal(unique_keys.to_numpy()[:run_count], expected_keys)
    assert np.array_equal(run_lengths.to_numpy()[:run_count], expected_lengths)
    assert workspace.workspace_bytes_peak >= n * 12

    unique_keys.fill(-97)
    count.fill(-97)
    ti.algorithms.experimental_unique(
        keys,
        unique_keys,
        count,
        workspace=workspace,
    )
    assert int(count.to_numpy()[0]) == expected_keys.size
    assert np.array_equal(
        unique_keys.to_numpy()[: expected_keys.size], expected_keys
    )

    unique_keys.fill(-97)
    unique_payload.fill(-97)
    count.fill(-97)
    ti.algorithms.experimental_unique_by_key(
        keys,
        payload,
        unique_keys,
        unique_payload,
        count,
        workspace=workspace,
    )
    assert int(count.to_numpy()[0]) == expected_keys.size
    assert np.array_equal(
        unique_keys.to_numpy()[: expected_keys.size], expected_keys
    )
    assert np.array_equal(
        unique_payload.to_numpy()[: expected_keys.size], payload_np[starts]
    )

    active_size = 73
    prefix_keys, prefix_lengths, _ = _expected_runs(keys_np[:active_size])
    ti.algorithms.experimental_run_length_encode(
        keys,
        unique_keys,
        run_lengths,
        count,
        size=active_size,
        workspace=workspace,
    )
    prefix_count = int(count.to_numpy()[0])
    assert prefix_count == prefix_keys.size
    assert np.array_equal(unique_keys.to_numpy()[:prefix_count], prefix_keys)
    assert np.array_equal(run_lengths.to_numpy()[:prefix_count], prefix_lengths)

    workspace.clear()
    assert workspace.workspace_bytes_current == 0
    assert workspace.workspace_bytes_peak == 0


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_rle_unique_dense_field_auto_parity():
    n = 193
    keys_np = ((np.arange(n, dtype=np.int32) // 5) % 11 - 4).astype(np.int32)
    expected_keys, expected_lengths, starts = _expected_runs(keys_np)
    payload_np = (np.arange(n, dtype=np.int32) * 3 + 1).astype(np.int32)

    keys = ti.field(ti.i32, shape=n)
    payload = ti.field(ti.i32, shape=n)
    unique_keys = ti.field(ti.i32, shape=n)
    unique_payload = ti.field(ti.i32, shape=n)
    run_lengths = ti.field(ti.i32, shape=n)
    count = ti.field(ti.i32, shape=())
    keys.from_numpy(keys_np)
    payload.from_numpy(payload_np)
    unique_keys.fill(-51)
    unique_payload.fill(-51)
    run_lengths.fill(-51)
    count[None] = -51

    workspace = ti.algorithms.RunLengthWorkspace(max_items=n)
    ti.algorithms.experimental_run_length_encode(
        keys,
        unique_keys,
        run_lengths,
        count,
        workspace=workspace,
    )
    run_count = int(count[None])
    assert run_count == expected_keys.size
    assert np.array_equal(unique_keys.to_numpy()[:run_count], expected_keys)
    assert np.array_equal(run_lengths.to_numpy()[:run_count], expected_lengths)

    ti.algorithms.experimental_unique_by_key(
        keys,
        payload,
        unique_keys,
        unique_payload,
        count,
        workspace=workspace,
    )
    assert int(count[None]) == expected_keys.size
    assert np.array_equal(
        unique_payload.to_numpy()[: expected_keys.size], payload_np[starts]
    )


@test_utils.test(arch=ti.cpu)
@pytest.mark.parametrize(
    "ti_dtype,np_dtype",
    [
        (ti.i32, np.int32),
        (ti.u32, np.uint32),
        (ti.i64, np.int64),
        (ti.u64, np.uint64),
    ],
)
def test_rle_integer_key_dtypes(ti_dtype, np_dtype):
    keys_np = np.array([0, 0, 1, 2, 2, 2, 9, 9], dtype=np_dtype)
    expected_keys, expected_lengths, _ = _expected_runs(keys_np)
    n = keys_np.size
    keys = ti.ndarray(ti_dtype, shape=n)
    unique_keys = ti.ndarray(ti_dtype, shape=n)
    run_lengths = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)
    keys.from_numpy(keys_np)

    ti.algorithms.experimental_run_length_encode(
        keys, unique_keys, run_lengths, count
    )
    run_count = int(count.to_numpy()[0])
    assert np.array_equal(unique_keys.to_numpy()[:run_count], expected_keys)
    assert np.array_equal(run_lengths.to_numpy()[:run_count], expected_lengths)


@test_utils.test(arch=ti.cpu)
def test_rle_empty_and_single_ndarray():
    empty = ti.ndarray(ti.i32, shape=1)
    empty_output = ti.ndarray(ti.i32, shape=1)
    empty_lengths = ti.ndarray(ti.i32, shape=1)
    count = ti.ndarray(ti.i32, shape=1)
    count.fill(-1)
    workspace = ti.algorithms.RunLengthWorkspace(max_items=1)

    ti.algorithms.experimental_run_length_encode(
        empty,
        empty_output,
        empty_lengths,
        count,
        size=0,
        workspace=workspace,
    )
    assert count.to_numpy()[0] == 0
    ti.algorithms.experimental_unique(
        empty, empty_output, count, size=0, workspace=workspace
    )
    assert count.to_numpy()[0] == 0

    one = ti.ndarray(ti.i32, shape=1)
    one_output = ti.ndarray(ti.i32, shape=1)
    one_lengths = ti.ndarray(ti.i32, shape=1)
    one.from_numpy(np.array([42], dtype=np.int32))
    ti.algorithms.experimental_run_length_encode(
        one, one_output, one_lengths, count
    )
    assert count.to_numpy()[0] == 1
    assert one_output.to_numpy()[0] == 42
    assert one_lengths.to_numpy()[0] == 1


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_unique_by_key_struct_payload_preserves_first_run_value():
    n = 96
    payload_type = ti.types.struct(tag=ti.i32, pair=ti.types.vector(2, ti.i32))
    keys_np = (np.arange(n, dtype=np.int32) // 4).astype(np.int32)
    _, _, starts = _expected_runs(keys_np)
    keys = ti.ndarray(ti.i32, shape=n)
    keys.from_numpy(keys_np)
    values = ti.ndarray(payload_type, shape=n)
    output = ti.ndarray(payload_type, shape=n)
    values_np = np.zeros(n, dtype=values.numpy_dtype)
    values_np["tag"] = np.arange(n, dtype=np.int32) * 7 + 3
    values_np["pair"] = np.arange(n * 2, dtype=np.int32).reshape(n, 2)
    values.from_numpy(values_np)
    unique_keys = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)

    ti.algorithms.experimental_unique_by_key(
        keys, values, unique_keys, output, count
    )
    run_count = int(count.to_numpy()[0])
    result = output.to_numpy()[:run_count]
    assert run_count == starts.size
    assert np.array_equal(result["tag"], values_np["tag"][starts])
    assert np.array_equal(result["pair"], values_np["pair"][starts])


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_unique_by_key_i32_matrix_field_first_payload():
    n = 64
    keys_np = (np.arange(n, dtype=np.int32) // 4).astype(np.int32)
    _, _, starts = _expected_runs(keys_np)
    keys = ti.field(ti.i32, shape=n)
    values = ti.Matrix.field(2, 2, ti.i32, shape=n)
    unique_keys = ti.field(ti.i32, shape=n)
    output = ti.Matrix.field(2, 2, ti.i32, shape=n)
    count = ti.field(ti.i32, shape=())
    values_np = np.arange(n * 4, dtype=np.int32).reshape(n, 2, 2)
    keys.from_numpy(keys_np)
    values.from_numpy(values_np)

    ti.algorithms.experimental_unique_by_key(
        keys, values, unique_keys, output, count
    )
    run_count = int(count[None])
    assert run_count == starts.size
    assert np.array_equal(output.to_numpy()[:run_count], values_np[starts])


@test_utils.test(arch=ti.cpu)
def test_unique_by_key_matrix_field_contract_rejects_before_write():
    n = 8
    keys = ti.field(ti.i32, shape=n)
    values = ti.Matrix.field(2, 2, ti.f32, shape=n)
    unique_keys = ti.field(ti.i32, shape=n)
    output = ti.Matrix.field(2, 2, ti.f32, shape=n)
    count = ti.field(ti.i32, shape=())
    unique_keys.fill(71)
    output.fill(72)
    count[None] = 73

    with pytest.raises(TypeError, match="MatrixField payloads.*ti.i32"):
        ti.algorithms.experimental_unique_by_key(
            keys, values, unique_keys, output, count
        )
    assert np.all(unique_keys.to_numpy() == 71)
    assert np.all(output.to_numpy() == 72)
    assert count[None] == 73


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_rle_primitive_sequence_graph_native_replay():
    n = 128
    keys = ti.ndarray(ti.i32, shape=n)
    unique_keys = ti.ndarray(ti.i32, shape=n)
    run_lengths = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)
    sequence = ti.algorithms.primitive_sequence().run_length_encode(
        keys, unique_keys, run_lengths, count
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence, prewarm=False)
    graph = builder.compile()
    assert graph._spec.native_count == 1

    for divisor in (3, 7):
        keys_np = (np.arange(n, dtype=np.int32) // divisor).astype(np.int32)
        expected_keys, expected_lengths, _ = _expected_runs(keys_np)
        keys.from_numpy(keys_np)
        graph.run({})
        run_count = int(count.to_numpy()[0])
        assert np.array_equal(unique_keys.to_numpy()[:run_count], expected_keys)
        assert np.array_equal(
            run_lengths.to_numpy()[:run_count], expected_lengths
        )
    assert sequence.call_count == 1
    assert sequence.workspace_bytes_peak >= n * 12


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_rle_independent_workspaces_submit_from_two_threads():
    n = 4096
    cases = []
    for worker_id in range(2):
        keys_np = (
            (np.arange(n, dtype=np.int32) // (worker_id + 3)) % 97
        ).astype(np.int32)
        expected_keys, expected_lengths, _ = _expected_runs(keys_np)
        keys = ti.ndarray(ti.i32, shape=n)
        unique_keys = ti.ndarray(ti.i32, shape=n)
        run_lengths = ti.ndarray(ti.i32, shape=n)
        count = ti.ndarray(ti.i32, shape=1)
        keys.from_numpy(keys_np)
        workspace = ti.algorithms.RunLengthWorkspace(max_items=n)
        ti.algorithms.experimental_run_length_encode(
            keys,
            unique_keys,
            run_lengths,
            count,
            workspace=workspace,
        )
        cases.append(
            (
                keys,
                unique_keys,
                run_lengths,
                count,
                workspace,
                expected_keys,
                expected_lengths,
            )
        )
    ti.sync()

    errors = []

    def submit(case):
        try:
            keys, unique_keys, run_lengths, count, workspace, _, _ = case
            for _ in range(10):
                ti.algorithms.experimental_run_length_encode(
                    keys,
                    unique_keys,
                    run_lengths,
                    count,
                    workspace=workspace,
                )
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    threads = [threading.Thread(target=submit, args=(case,)) for case in cases]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    ti.sync()
    for (
        _,
        unique_keys,
        run_lengths,
        count,
        _,
        expected_keys,
        expected_lengths,
    ) in cases:
        run_count = int(count.to_numpy()[0])
        assert np.array_equal(unique_keys.to_numpy()[:run_count], expected_keys)
        assert np.array_equal(
            run_lengths.to_numpy()[:run_count], expected_lengths
        )


@test_utils.test(arch=ti.cpu)
def test_rle_validation_rejects_before_output_write():
    n = 8
    keys = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    lengths = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)
    keys.from_numpy(np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32))
    output.fill(91)
    lengths.fill(92)
    count.fill(93)

    with pytest.raises(ValueError, match="does not support input/output aliasing"):
        ti.algorithms.experimental_run_length_encode(
            keys, keys, lengths, count
        )
    assert np.all(output.to_numpy() == 91)
    assert np.all(lengths.to_numpy() == 92)
    assert count.to_numpy()[0] == 93

    with pytest.raises(ValueError, match="mode='consecutive'"):
        ti.algorithms.experimental_unique(
            keys, output, count, mode="global"
        )
    assert np.all(output.to_numpy() == 91)
    assert count.to_numpy()[0] == 93

    short = ti.ndarray(ti.i32, shape=n - 1)
    with pytest.raises(ValueError, match="capacity"):
        ti.algorithms.experimental_unique(keys, short, count)
    assert count.to_numpy()[0] == 93

    with pytest.raises(ValueError, match="0 <= size <= input capacity"):
        ti.algorithms.experimental_unique(
            keys, output, count, size=n + 1
        )
    assert np.all(output.to_numpy() == 91)
    assert count.to_numpy()[0] == 93


@test_utils.test(arch=ti.cpu)
def test_rle_discrete_ad_rejects_before_output_write():
    n = 8
    keys = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    count = ti.ndarray(ti.i32, shape=1)
    keys.fill(1)
    output.fill(77)
    count.fill(78)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    with pytest.raises(RuntimeError, match="not differentiable"):
        with ti.ad.Tape(loss):
            ti.algorithms.experimental_unique(keys, output, count)
    assert np.all(output.to_numpy() == 77)
    assert count.to_numpy()[0] == 78

    param = ti.field(ti.f32, shape=1, needs_dual=True)
    fwd_loss = ti.field(ti.f32, shape=1, needs_dual=True)
    with pytest.raises(RuntimeError, match="not differentiable.*FwdMode"):
        with ti.ad.FwdMode(loss=fwd_loss, param=param):
            ti.algorithms.experimental_unique(keys, output, count)
    assert np.all(output.to_numpy() == 77)
    assert count.to_numpy()[0] == 78
