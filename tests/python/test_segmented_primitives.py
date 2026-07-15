import threading

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _offsets():
    return np.array([0, 3, 3, 9, 14, 14, 23], dtype=np.int32)


def _segmented_reduce_expected(values, offsets):
    result = np.zeros(offsets.size - 1, dtype=values.dtype)
    for segment in range(result.size):
        result[segment] = np.sum(
            values[offsets[segment] : offsets[segment + 1]],
            dtype=values.dtype,
        )
    return result


def _segmented_scan_expected(values, offsets, inclusive):
    result = np.zeros_like(values)
    for segment in range(offsets.size - 1):
        begin = offsets[segment]
        end = offsets[segment + 1]
        if begin == end:
            continue
        scanned = np.cumsum(values[begin:end], dtype=values.dtype)
        if inclusive:
            result[begin:end] = scanned
        else:
            result[begin] = 0
            result[begin + 1 : end] = scanned[:-1]
    return result


def _assert_equal(actual, expected):
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        np.testing.assert_array_equal(actual, expected)


@test_utils.test(arch=ti.cpu)
def test_segmented_layout_validates_and_normalizes_both_encodings():
    offsets = _offsets()
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets, capacity=32)
    assert layout.encoding == "offsets"
    assert layout.num_items == 23
    assert layout.capacity == 32
    assert layout.num_segments == 6
    assert layout.max_segment_length == 9
    assert layout.topology_bytes == (32 + 7) * 4

    ids = np.array([0, 0, 2, 2, 4, 4, 4, 5], dtype=np.int64)
    from_ids = ti.algorithms.SegmentedLayout.from_segment_ids(
        ids, 6, size=7, capacity=12
    )
    assert from_ids.encoding == "segment_ids"
    assert from_ids.num_items == 7
    assert from_ids.capacity == 12
    assert from_ids.num_segments == 6
    assert from_ids.max_segment_length == 3

    values_np = np.arange(12, dtype=np.int32) + 1
    values = ti.ndarray(ti.i32, shape=12)
    output = ti.ndarray(ti.i32, shape=6)
    values.from_numpy(values_np)
    ti.algorithms.experimental_segmented_reduce(values, from_ids, output)
    expected = np.array(
        [
            values_np[:2].sum(),
            0,
            values_np[2:4].sum(),
            0,
            values_np[4:7].sum(),
            0,
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(output.to_numpy(), expected)


@test_utils.test(arch=ti.cpu)
def test_segmented_layout_rejects_invalid_topology_before_device_use():
    with pytest.raises(TypeError, match="integer"):
        ti.algorithms.SegmentedLayout.from_offsets(
            np.array([0.0, 1.0], dtype=np.float32)
        )
    with pytest.raises(ValueError, match="start at zero"):
        ti.algorithms.SegmentedLayout.from_offsets(
            np.array([1, 2], dtype=np.int32)
        )
    with pytest.raises(ValueError, match="nondecreasing"):
        ti.algorithms.SegmentedLayout.from_offsets(
            np.array([0, 4, 3], dtype=np.int32)
        )
    with pytest.raises(ValueError, match="fixed layout capacity"):
        ti.algorithms.SegmentedLayout.from_offsets(
            np.array([0, 4], dtype=np.int32), capacity=3
        )
    with pytest.raises(ValueError, match=r"\[0, num_segments\)"):
        ti.algorithms.SegmentedLayout.from_segment_ids(
            np.array([0, 3], dtype=np.int32), 3
        )
    with pytest.raises(ValueError, match="nondecreasing"):
        ti.algorithms.SegmentedLayout.from_segment_ids(
            np.array([0, 2, 1], dtype=np.int32), 3
        )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_segmented_ndarray_reduce_and_scan_parity():
    capacity = 32
    offsets = _offsets()
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        offsets, capacity=capacity
    )
    workspace = ti.algorithms.SegmentedWorkspace(
        max_items=capacity, max_segments=layout.num_segments
    )

    for dtype, np_dtype in ((ti.i32, np.int32), (ti.f32, np.float32)):
        values_np = (
            (np.arange(capacity, dtype=np.float64) % 9) - 4
        ).astype(np_dtype)
        values = ti.ndarray(dtype, shape=capacity)
        reduced = ti.ndarray(dtype, shape=layout.num_segments)
        scanned = ti.ndarray(dtype, shape=capacity)
        values.from_numpy(values_np)

        ti.algorithms.experimental_segmented_reduce(
            values, layout, reduced, workspace=workspace
        )
        _assert_equal(
            reduced.to_numpy(),
            _segmented_reduce_expected(values_np, offsets),
        )

        for inclusive in (True, False):
            scanned.fill(97)
            ti.algorithms.experimental_segmented_scan(
                values,
                layout,
                scanned,
                inclusive=inclusive,
                workspace=workspace,
            )
            expected = _segmented_scan_expected(
                values_np, offsets, inclusive
            )
            _assert_equal(
                scanned.to_numpy()[: layout.num_items],
                expected[: layout.num_items],
            )

    assert workspace.last_scan_method == "serial"
    assert workspace.workspace_bytes_current == 0
    assert workspace.workspace_bytes_peak == 0
    workspace.clear()
    assert workspace.workspace_bytes_current == 0
    assert workspace.workspace_bytes_peak == 0
    assert workspace.last_scan_method is None


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_segmented_integer_scan_in_place():
    capacity = 32
    offsets = _offsets()
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        offsets, capacity=capacity
    )
    values_np = (np.arange(capacity, dtype=np.int32) % 7) - 3
    values = ti.ndarray(ti.i32, shape=capacity)

    for inclusive in (True, False):
        values.from_numpy(values_np)
        ti.algorithms.experimental_segmented_scan(
            values, layout, values, inclusive=inclusive
        )
        expected = _segmented_scan_expected(values_np, offsets, inclusive)
        np.testing.assert_array_equal(
            values.to_numpy()[: layout.num_items],
            expected[: layout.num_items],
        )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_segmented_integer_scan_auto_dispatch_is_coarse_and_observable():
    short_layout = ti.algorithms.SegmentedLayout.from_offsets(
        np.arange(0, 33, 4, dtype=np.int32), capacity=32
    )
    short_values = ti.ndarray(ti.i32, shape=32)
    short_output = ti.ndarray(ti.i32, shape=32)
    short_workspace = ti.algorithms.SegmentedWorkspace()
    short_values.fill(1)
    ti.algorithms.experimental_segmented_scan(
        short_values,
        short_layout,
        short_output,
        workspace=short_workspace,
    )
    assert short_workspace.last_scan_method == "serial"

    capacity = 1 << 16
    segment_length = 1 << 12
    long_layout = ti.algorithms.SegmentedLayout.from_offsets(
        np.arange(0, capacity + 1, segment_length, dtype=np.int32),
        capacity=capacity,
    )
    long_values = ti.ndarray(ti.i32, shape=capacity)
    long_output = ti.ndarray(ti.i32, shape=capacity)
    long_workspace = ti.algorithms.SegmentedWorkspace()
    long_values.fill(1)
    ti.algorithms.experimental_segmented_scan(
        long_values,
        long_layout,
        long_output,
        workspace=long_workspace,
    )
    expected_method = (
        "global_scan"
        if ti.lang.impl.current_cfg().arch == ti.cuda
        else "serial"
    )
    assert long_workspace.last_scan_method == expected_method
    result = long_output.to_numpy()
    np.testing.assert_array_equal(
        result.reshape(-1, segment_length)[:, -1],
        np.full(long_layout.num_segments, segment_length, dtype=np.int32),
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_segmented_dense_field_offsets_and_float_stability():
    capacity = 32
    offsets = _offsets()
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        offsets, capacity=capacity
    )
    values_np = ((np.arange(capacity, dtype=np.float32) % 11) - 5) * 0.25
    values = ti.field(ti.f32, shape=capacity, offset=-7)
    reduced = ti.field(ti.f32, shape=layout.num_segments, offset=3)
    scanned = ti.field(ti.f32, shape=capacity, offset=11)
    values.from_numpy(values_np)

    ti.algorithms.experimental_segmented_reduce(
        values, layout, reduced, method="serial"
    )
    np.testing.assert_allclose(
        reduced.to_numpy(),
        _segmented_reduce_expected(values_np, offsets),
        rtol=1e-6,
        atol=1e-6,
    )

    ti.algorithms.experimental_segmented_scan(values, layout, scanned)
    expected = _segmented_scan_expected(values_np, offsets, True)
    np.testing.assert_allclose(
        scanned.to_numpy()[: layout.num_items],
        expected[: layout.num_items],
        rtol=1e-6,
        atol=1e-6,
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_segmented_serial_ndarray_uses_true_zero_identity_with_inf():
    offsets = np.array([0, 0, 2, 4], dtype=np.int32)
    layout = ti.algorithms.SegmentedLayout.from_offsets(offsets, capacity=4)
    values_np = np.array([np.inf, 1.0, 3.0, 4.0], dtype=np.float32)
    values = ti.ndarray(ti.f32, shape=4)
    reduced = ti.ndarray(ti.f32, shape=3)
    scanned = ti.ndarray(ti.f32, shape=4)
    values.from_numpy(values_np)

    ti.algorithms.experimental_segmented_reduce(
        values, layout, reduced, method="serial"
    )
    ti.algorithms.experimental_segmented_scan(
        values, layout, scanned, method="serial"
    )

    np.testing.assert_array_equal(
        reduced.to_numpy(),
        np.array([0.0, np.inf, 7.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        scanned.to_numpy(),
        np.array([np.inf, np.inf, 3.0, 7.0], dtype=np.float32),
    )


@test_utils.test(arch=ti.cpu)
def test_segmented_validation_rejects_before_output_write():
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        np.array([0, 4, 8], dtype=np.int32), capacity=8
    )
    values = ti.ndarray(ti.f32, shape=8)
    reduced = ti.ndarray(ti.f32, shape=2)
    scanned = ti.ndarray(ti.f32, shape=8)
    values.fill(1)
    reduced.fill(71)
    scanned.fill(72)

    alias_layout = ti.algorithms.SegmentedLayout.from_offsets(
        np.arange(9, dtype=np.int32), capacity=8
    )
    with pytest.raises(ValueError, match="must not alias"):
        ti.algorithms.experimental_segmented_reduce(
            values, alias_layout, values
        )
    assert np.all(values.to_numpy() == 1)
    assert np.all(reduced.to_numpy() == 71)

    with pytest.raises(ValueError, match="global_scan is integer-only"):
        ti.algorithms.experimental_segmented_scan(
            values, layout, scanned, method="global_scan"
        )
    assert np.all(scanned.to_numpy() == 72)

    short = ti.ndarray(ti.f32, shape=7)
    with pytest.raises(ValueError, match="layout capacity"):
        ti.algorithms.experimental_segmented_scan(
            short, layout, scanned
        )
    assert np.all(scanned.to_numpy() == 72)

    with pytest.raises(NotImplementedError, match="method 'missing'"):
        ti.algorithms.experimental_segmented_reduce(
            values, layout, reduced, method="missing"
        )
    assert np.all(reduced.to_numpy() == 71)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_segmented_primitive_sequence_graph_replay():
    capacity = 64
    offsets = np.array([0, 4, 4, 19, 37, 64], dtype=np.int32)
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        offsets, capacity=capacity
    )
    values = ti.ndarray(ti.i32, shape=capacity)
    reduced = ti.ndarray(ti.i32, shape=layout.num_segments)
    scanned = ti.ndarray(ti.i32, shape=capacity)
    sequence = (
        ti.algorithms.primitive_sequence()
        .segmented_reduce(values, layout, reduced)
        .segmented_scan(values, layout, scanned, inclusive=False)
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence, prewarm=False)
    graph = builder.compile()
    assert graph._spec.native_count == 1

    for scale in (1, 3):
        values_np = ((np.arange(capacity, dtype=np.int32) % 7) - 3) * scale
        values.from_numpy(values_np)
        graph.run({})
        np.testing.assert_array_equal(
            reduced.to_numpy(),
            _segmented_reduce_expected(values_np, offsets),
        )
        np.testing.assert_array_equal(
            scanned.to_numpy(),
            _segmented_scan_expected(values_np, offsets, False),
        )
    assert sequence.workspace_bytes_peak == 0


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_segmented_independent_workspaces_submit_from_two_threads():
    capacity = 2048
    offsets = np.arange(0, capacity + 1, 8, dtype=np.int32)
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        offsets, capacity=capacity
    )
    cases = []
    for worker_id in range(2):
        values_np = (
            (np.arange(capacity, dtype=np.int32) + worker_id) % 11
        ) - 5
        values = ti.ndarray(ti.i32, shape=capacity)
        reduced = ti.ndarray(ti.i32, shape=layout.num_segments)
        scanned = ti.ndarray(ti.i32, shape=capacity)
        values.from_numpy(values_np)
        workspace = ti.algorithms.SegmentedWorkspace(
            max_items=capacity, max_segments=layout.num_segments
        )
        ti.algorithms.experimental_segmented_reduce(
            values, layout, reduced, workspace=workspace
        )
        ti.algorithms.experimental_segmented_scan(
            values, layout, scanned, workspace=workspace
        )
        cases.append((values, reduced, scanned, workspace, values_np))
    ti.sync()

    errors = []

    def submit(case):
        values, reduced, scanned, workspace, _values_np = case
        try:
            for _ in range(5):
                ti.algorithms.experimental_segmented_reduce(
                    values, layout, reduced, workspace=workspace
                )
                ti.algorithms.experimental_segmented_scan(
                    values, layout, scanned, workspace=workspace
                )
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    threads = [
        threading.Thread(target=submit, args=(case,)) for case in cases
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    ti.sync()
    for _values, reduced, scanned, _workspace, values_np in cases:
        np.testing.assert_array_equal(
            reduced.to_numpy(),
            _segmented_reduce_expected(values_np, offsets),
        )
        np.testing.assert_array_equal(
            scanned.to_numpy(),
            _segmented_scan_expected(values_np, offsets, True),
        )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
    offline_cache=False,
)
def test_segmented_reduce_grouped_reverse_ad_and_padded_tail():
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        np.array([0, 2, 3, 6], dtype=np.int32), capacity=8
    )
    values = ti.ndarray(ti.f32, shape=8, needs_grad=True)
    output = ti.ndarray(ti.f32, shape=3, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    values.from_numpy(np.ones(8, dtype=np.float32))

    @ti.kernel
    def weighted_sum(arr: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        loss[None] += arr[0] * 2.0
        loss[None] += arr[1] * 3.0
        loss[None] += arr[2] * 5.0

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_segmented_reduce(
            values, layout, output
        )
        weighted_sum(output)

    np.testing.assert_allclose(
        values.grad.to_numpy(),
        np.array([2, 2, 3, 5, 5, 5, 0, 0], dtype=np.float32),
    )


@test_utils.test(arch=ti.cpu)
def test_segmented_scan_and_serial_reduce_reject_ad_before_write():
    layout = ti.algorithms.SegmentedLayout.from_offsets(
        np.array([0, 4, 8], dtype=np.int32), capacity=8
    )
    values = ti.ndarray(ti.f32, shape=8, needs_grad=True)
    reduced = ti.ndarray(ti.f32, shape=2, needs_grad=True)
    scanned = ti.ndarray(ti.f32, shape=8, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    values.fill(1)
    reduced.fill(71)
    scanned.fill(72)

    with pytest.raises(RuntimeError, match="segmented_scan.*not differentiable"):
        with ti.ad.Tape(loss):
            ti.algorithms.experimental_segmented_scan(
                values, layout, scanned
            )
    assert np.all(scanned.to_numpy() == 72)

    with pytest.raises(RuntimeError, match="reverse AD requires"):
        with ti.ad.Tape(loss):
            ti.algorithms.experimental_segmented_reduce(
                values, layout, reduced, method="serial"
            )
    assert np.all(reduced.to_numpy() == 71)
