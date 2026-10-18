import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize(
    "strategy,wide",
    [("single", False), ("direct", True), ("private_shared", True), ("private", False)],
)
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_histogram_field_views_keep_graph_parameters_and_neighbors(
    strategy, wide, monkeypatch
):
    monkeypatch.setenv(
        "TI_VULKAN_HISTOGRAM_SINGLE_SHARED_MAX_N",
        "4096" if strategy == "single" else "0",
    )
    monkeypatch.setenv(
        "TI_VULKAN_HISTOGRAM_PRIVATE_MIN_N",
        "0" if strategy.startswith("private") else "65536",
    )
    monkeypatch.setenv("TI_VULKAN_HISTOGRAM_PRIVATE_MAX_BINS", "0")
    if (
        wide
        and not ti.lang.impl.get_runtime().prog.vulkan_histogram_value_type_available(
            2, 4
        )
    ):
        pytest.skip("Vulkan i64 histogram capability unavailable")
    value_type, bin_type = (ti.u32, ti.i64) if wide else (ti.i32, ti.i32)
    value_numpy, bin_numpy = (np.uint32, np.int64) if wide else (np.int32, np.int32)
    marker = ti.field(ti.i32, 5)
    dimensions = (
        [(4099, 4097), (17, 4099)] if strategy == "private" else [(17, 7), (9, 11)]
    )
    arrays = [
        (ti.field(value_type, n), ti.field(bin_type, count)) for n, count in dimensions
    ]
    tail = ti.field(ti.i32, 3)
    marker.fill(91)
    tail.fill(83)
    sequences, graphs = [], []
    try:
        for values, bins in arrays:
            values.fill(0)
            sequence = ti.algorithms.primitive_sequence().histogram(
                values, bins, method="vulkan_native"
            )
            builder = ti.graph.GraphBuilder()
            builder.append_native(sequence)
            graph = builder.compile()
            sequences.append(sequence)
            graphs.append(graph)
            graph.run({})
            ti.sync()
        # The second materialization changes counts/bindings and reuses a
        # potentially larger partial allocation. Each Graph keeps its own view.
        for shift in (0, 3):
            for graph, (values, bins), (n, count) in zip(graphs, arrays, dimensions):
                data = (
                    (np.arange(n, dtype=np.int32) + shift) % (count + 1) - 1
                ).astype(value_numpy)
                values.from_numpy(data)
                bins.fill(37)
                graph.run({})
                valid = (
                    data[data < count] if wide else data[(data >= 0) & (data < count)]
                )
                np.testing.assert_array_equal(
                    bins.to_numpy(),
                    np.bincount(valid.astype(np.int64), minlength=count).astype(
                        bin_numpy
                    ),
                )
                np.testing.assert_array_equal(values.to_numpy(), data)
                np.testing.assert_array_equal(
                    marker.to_numpy(), np.full(5, 91, np.int32)
                )
                np.testing.assert_array_equal(tail.to_numpy(), np.full(3, 83, np.int32))
    finally:
        for graph in graphs:
            graph.close()
        for sequence in sequences:
            sequence.clear()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_histogram_empty_input_clears_only_output():
    values, bins = ti.field(ti.i32, 3), ti.field(ti.i32, 7)
    values.from_numpy(np.array([0, 2, 4], np.int32))
    view = ti.algorithms._algorithms._primitive_view
    prog = ti.lang.impl.get_runtime().prog
    for n in (0, 2, 0):
        bins.fill(37)
        prog.vulkan_histogram_dense_field(
            view(values).snode, view(bins).snode, 0, 0, n, 5
        )
        expected = np.array([0, 0, 0, 0, 0, 37, 37], np.int32)
        if n:
            expected[0] = expected[2] = 1
        np.testing.assert_array_equal(bins.to_numpy(), expected)
        np.testing.assert_array_equal(values.to_numpy(), [0, 2, 4])
