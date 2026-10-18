import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_strided_indexed_copy_graphs_keep_immutable_parameters():
    payload = ti.types.struct(
        tag=ti.i32, vec=ti.types.vector(3, ti.f32), mat=ti.types.matrix(2, 2, ti.i32)
    )
    source, gathered, scattered = (ti.ndarray(payload, n) for n in (17, 9, 23))
    indices = ti.ndarray(ti.i32, 9)
    data = np.zeros(17, source.numpy_dtype)
    data["tag"] = 71
    data["vec"] = np.arange(51, dtype=np.float32).reshape(17, 3) - 13
    data["mat"] = np.arange(68, dtype=np.int32).reshape(17, 2, 2) * 3 - 7
    source.from_numpy(data)
    gather_initial = np.zeros(9, gathered.numpy_dtype)
    gather_initial["tag"] = 91
    gather_initial["mat"] = data["mat"][:9]
    scatter_initial = np.zeros(23, scattered.numpy_dtype)
    scatter_initial["tag"] = 113
    scatter_initial["vec"] = -17
    scatter_initial["mat"] = -19
    gathered.from_numpy(gather_initial)
    workspace = ti.algorithms.IndexedCopyWorkspace(max_items=9)
    sequences = [
        ti.algorithms.primitive_sequence().gather(
            source.field("vec"),
            indices,
            gathered.field("vec"),
            method="vulkan_native",
            workspace=workspace,
        ),
        ti.algorithms.primitive_sequence().scatter(
            gathered.field("mat"),
            indices,
            scattered.field("mat"),
            method="vulkan_native",
            workspace=workspace,
        ),
    ]
    graphs = []
    for sequence in sequences:
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graphs.append(builder.compile())
    try:
        for index_data in (
            np.array([16, 2, -1, 8, 20, 3, 6, 0, 99], np.int32),
            np.array([0, 1, 4, 9, 10, 12, 13, 15, 22], np.int32),
        ):
            indices.from_numpy(index_data)
            scattered.from_numpy(scatter_initial)
            # Different payload widths and bounds share one device cache. A
            # later materialization must not overwrite an earlier graph's data.
            graphs[0].run({})
            graphs[1].run({})
            graphs[0].run({})
            ti.sync()
            expected_gather = gather_initial.copy()
            expected_scatter = scatter_initial.copy()
            for i, index in enumerate(index_data):
                expected_gather["vec"][i] = data["vec"][index] if 0 <= index < 17 else 0
                if 0 <= index < 23:
                    expected_scatter["mat"][index] = gather_initial["mat"][i]
            np.testing.assert_array_equal(gathered.to_numpy(), expected_gather)
            np.testing.assert_array_equal(scattered.to_numpy(), expected_scatter)
            np.testing.assert_array_equal(source.to_numpy(), data)
        assert workspace.workspace_bytes_peak == 0
    finally:
        for graph in reversed(graphs):
            graph.close()
        for sequence in sequences:
            sequence.clear()
