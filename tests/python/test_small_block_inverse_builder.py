import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _array(dtype, values):
    values = np.asarray(values)
    result = ti.ndarray(dtype, shape=values.size)
    result.from_numpy(values.reshape(-1))
    return result


@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_small_block_inverse_builder_matches_numpy_and_graph(block_size):
    block_count = 3
    blocks = np.empty((block_count, block_size, block_size), np.float32)
    for index in range(block_count):
        seed = np.arange(block_size * block_size, dtype=np.float32).reshape(block_size, block_size)
        blocks[index] = np.eye(block_size, dtype=np.float32) * (2.0 + index) + (seed + seed.T) * np.float32(0.01)
    source = _array(ti.f32, blocks)
    builder = ti.linalg.SmallBlockInverseBuilder(block_size, block_count, pivot_tolerance=1.0e-7)
    direct = builder.build(source)
    assert np.all(direct.status.to_numpy() == 0)
    expected = np.linalg.inv(blocks.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(
        direct.inverse_blocks.to_numpy().reshape(blocks.shape),
        expected,
        rtol=2e-5,
        atol=2e-5,
    )

    blocks_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "blocks", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "inverse", ti.f32, ndim=1)
    status_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "status", ti.i32, ndim=1)
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.append_native(builder.graph_action(blocks_arg, output_arg, status_arg))
    graph = graph_builder.compile()
    output = ti.ndarray(ti.f32, shape=blocks.size)
    status = ti.ndarray(ti.i32, shape=block_count)
    bindings = graph.bind(
        {"blocks": source, "inverse": output, "status": status}
    )
    assert bindings.fast_path_qualified, bindings.statistics()
    graph.run(bindings)
    graph.run(bindings)
    assert graph.binding_statistics()["version_fast_replays"] == 2
    revision = bindings.revision
    with pytest.raises(RuntimeError, match="must be proven disjoint"):
        bindings.update(inverse=source)
    assert bindings.revision == revision
    graph.run(bindings)
    assert graph.binding_statistics()["version_fast_replays"] == 3
    assert np.all(status.to_numpy() == 0)
    np.testing.assert_allclose(output.to_numpy().reshape(blocks.shape), expected, rtol=2e-5, atol=2e-5)
    assert graph._debug_info["nodes"][0]["dispatch_count"] == 1


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_small_block_inverse_graph_action_snapshots_builder_configuration():
    builder = ti.linalg.SmallBlockInverseBuilder(
        1, 2, regularization=0.0, pivot_tolerance=1.0e-6
    )
    blocks_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "blocks", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "inverse", ti.f32, ndim=1)
    status_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "status", ti.i32, ndim=1)
    node = builder.graph_action(blocks_arg, output_arg, status_arg)
    executable = node.compile()
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.append_native(node)
    graph = graph_builder.compile()

    source = _array(ti.f32, [2.0, 4.0])
    output = ti.ndarray(ti.f32, shape=2)
    status = ti.ndarray(ti.i32, shape=2)
    runtime_args = {"blocks": source, "inverse": output, "status": status}

    replacement = ti.linalg.SmallBlockInverseBuilder(2, 1)
    builder.block_size = replacement.block_size
    builder.block_count = replacement.block_count
    builder.coefficient_count = replacement.coefficient_count
    builder.regularization = 0.5
    builder.pivot_tolerance = 0.25
    builder._kernel = replacement._kernel

    def reject_mutable_builder_fallback(*_args, **_kwargs):
        pytest.fail("compiled Graph action consulted its mutable builder")

    builder.build = reject_mutable_builder_fallback

    assert executable.debug_info["block_size"] == 1
    assert executable.debug_info["block_count"] == 2
    executable.validate_graph_bindings(runtime_args)
    with pytest.raises(RuntimeError, match="exactly 2 scalar"):
        executable.validate_graph_bindings(
            {
                "blocks": ti.ndarray(ti.f32, shape=4),
                "inverse": ti.ndarray(ti.f32, shape=4),
                "status": ti.ndarray(ti.i32, shape=1),
            }
        )

    graph.run(runtime_args)
    np.testing.assert_allclose(output.to_numpy(), [0.5, 0.25])
    np.testing.assert_array_equal(status.to_numpy(), [0, 0])

    fallback_output = ti.ndarray(ti.f32, shape=2)
    fallback_status = ti.ndarray(ti.i32, shape=2)
    executable.run(
        {
            "blocks": source,
            "inverse": fallback_output,
            "status": fallback_status,
        }
    )
    np.testing.assert_allclose(fallback_output.to_numpy(), [0.5, 0.25])
    np.testing.assert_array_equal(fallback_status.to_numpy(), [0, 0])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_small_block_inverse_builder_status_and_regularization():
    singular = np.asarray(
        [[[1.0, 1.0], [1.0, 1.0]], [[np.nan, 0.0], [0.0, 1.0]]],
        np.float32,
    )
    source = _array(ti.f32, singular)
    failed = ti.linalg.SmallBlockInverseBuilder(2, 2, pivot_tolerance=1.0e-6).build(source)
    np.testing.assert_array_equal(failed.status.to_numpy(), [2, 1])
    np.testing.assert_array_equal(failed.inverse_blocks.to_numpy(), np.zeros(singular.size, np.float32))

    regularized_source = _array(ti.f32, singular[:1])
    regularized = ti.linalg.SmallBlockInverseBuilder(2, 1, regularization=0.25, pivot_tolerance=1.0e-6).build(
        regularized_source
    )
    np.testing.assert_array_equal(regularized.status.to_numpy(), [0])
    expected = np.linalg.inv(singular[0].astype(np.float64) + np.eye(2) * 0.25)
    np.testing.assert_allclose(
        regularized.inverse_blocks.to_numpy().reshape(2, 2),
        expected,
        rtol=2e-5,
        atol=2e-5,
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_small_block_inverse_builder_uses_scale_relative_pivot_tolerance():
    matrices = np.asarray(
        [
            [[2.0e-12, 0.25e-12], [0.25e-12, 1.0e-12]],
            [[2.0e12, 0.25e12], [0.25e12, 1.0e12]],
            [[1.0, 0.0], [0.0, 1.0e-10]],
        ],
        np.float32,
    )
    result = ti.linalg.SmallBlockInverseBuilder(
        2, 3, pivot_tolerance=1.0e-6
    ).build(_array(ti.f32, matrices))
    np.testing.assert_array_equal(result.status.to_numpy(), [0, 0, 2])
    expected = np.linalg.inv(matrices[:2].astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(
        result.inverse_blocks.to_numpy().reshape(3, 2, 2)[:2],
        expected,
        rtol=3e-5,
        atol=1e-20,
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_small_block_inverse_builder_rejects_unrepresentable_f32_controls():
    with pytest.raises(RuntimeError, match="representable as f32"):
        ti.linalg.SmallBlockInverseBuilder(2, 1, regularization=1.0e100)
    with pytest.raises(RuntimeError, match="representable as f32|positive"):
        ti.linalg.SmallBlockInverseBuilder(2, 1, pivot_tolerance=1.0e-100)
    with pytest.raises(RuntimeError, match="representable as f32"):
        ti.linalg.SmallBlockInverseBuilder(2, 1, regularization=1.0e-100)
