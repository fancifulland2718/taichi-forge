import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_graph_builder_private_ndarray_is_instance_owned_and_not_public():
    size = 33

    @ti.kernel
    def stage(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            scratch[i] = source[i] * 3

    @ti.kernel
    def consume(
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in output:
            output[i] = scratch[i] + 2

    source_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "private_source", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "private_output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    scratch_arg = builder.private_ndarray("private_scratch", ti.i32, size)
    builder.dispatch(stage, source_arg, scratch_arg)
    builder.dispatch(consume, scratch_arg, output_arg)
    graph = builder.compile(workspace_lanes=2)

    source = ti.ndarray(ti.i32, shape=size)
    output = ti.ndarray(ti.i32, shape=size)
    source.from_numpy(np.arange(size, dtype=np.int32))
    output.fill(0)
    graph.run({"private_source": source, "private_output": output})

    np.testing.assert_array_equal(
        output.to_numpy(), np.arange(size, dtype=np.int32) * 3 + 2
    )
    assert "private_scratch" not in graph._spec.runtime_arg_names
    assert graph._spec.internal_storage_bytes == size * 4
    assert len(graph._instance._internal_storages) == 1
    assert graph._instance._exclusive_internal_storage
    memory = graph.execution_stats().memory
    assert memory.persistent_internal_storage_bytes == size * 4
    assert memory.internal_storage_exclusive
    assert memory.workspace_lane_capacity == 2


def test_graph_owned_ndarray_is_a_provider_neutral_declaration():
    requirement = ti.graph.GraphOwnedNdarray(
        ti.f32, (7, 3), exclusive_submission=True
    )

    assert requirement.dtype == ti.f32
    assert requirement.shape == (7, 3)
    assert requirement.storage_bytes == 7 * 3 * 4
    assert requirement.exclusive_submission


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_appended_sequence_propagates_private_storage_ownership():
    size = 9

    @ti.kernel
    def initialize(scratch: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in scratch:
            scratch[i] = i + 4

    @ti.kernel
    def copy_out(
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in output:
            output[i] = scratch[i]

    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "sequence_output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    sequence = builder.create_sequential()
    scratch_arg = sequence.private_ndarray(
        "sequence_scratch", ti.i32, size
    )
    sequence.dispatch(initialize, scratch_arg)
    sequence.dispatch(copy_out, scratch_arg, output_arg)
    builder.append(sequence)
    graph = builder.compile()
    output = ti.ndarray(ti.i32, shape=size)
    output.fill(0)

    graph.run({"sequence_output": output})

    np.testing.assert_array_equal(
        output.to_numpy(), np.arange(size, dtype=np.int32) + 4
    )
    assert "sequence_scratch" not in graph._spec.runtime_arg_names
    assert graph._spec.internal_storage_bytes == size * 4
