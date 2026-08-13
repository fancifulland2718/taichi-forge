import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _stable_group_sum(keys, values, groups):
    result = np.zeros(groups, dtype=values.dtype)
    for key, value in zip(keys, values):
        if 0 <= key < groups:
            result[key] = np.asarray(result[key] + value, dtype=values.dtype)
    return result


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_deterministic_scatter_reduce_preserves_source_order_and_graph_replay():
    keys = np.asarray([2, 0, 2, 1, 0, 2, -1, 3, 9, 1], dtype=np.int32)
    values_host = np.asarray(
        [1.0e20, 1.5, 1.0, 7.0, 2.5, -1.0e20, 99.0, -3.0, 88.0, 4.0],
        dtype=np.float32,
    )
    groups = 4
    values = ti.ndarray(ti.f32, shape=keys.size)
    output = ti.ndarray(ti.f32, shape=groups)
    values.from_numpy(values_host)

    plan = ti.algorithms.DeterministicScatterReducePlan(keys, groups)
    binding = plan.bind(values, output).prewarm()
    expected = _stable_group_sum(keys, values_host, groups)
    for _ in range(5):
        output.fill(np.nan)
        binding.run()
        np.testing.assert_array_equal(output.to_numpy(), expected)

    builder = ti.graph.GraphBuilder()
    builder.append_native(binding.graph_action())
    graph = builder.compile()
    output.fill(np.nan)
    graph.run({})
    np.testing.assert_array_equal(output.to_numpy(), expected)
    physical = graph.physical_plan()
    assert physical["logical_submission_count"] == 1
    assert physical["logical_node_count"] == 1
    assert physical["physical_dispatch_count"] == 2
    assert physical["backend_recording_complete"]
    assert not physical["fragmented_native_plan"]

    report = binding.report()
    assert report.source_count == keys.size
    assert report.valid_count == 8
    assert report.ignored_count == 2
    assert report.group_count == groups
    assert report.reduction_order == "stable_source_ordinal_within_group"
    assert report.floating_point_deterministic
    assert report.ordered_value_bytes == report.valid_count * 4


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_deterministic_scatter_reduce_supports_root_dense_fields():
    keys = np.asarray([1, 0, 1, 2, 0, 1], dtype=np.int32)
    values_host = np.asarray([3, 4, 5, 6, 7, 8], dtype=np.int32)
    values = ti.field(ti.i32, shape=keys.size)
    output = ti.field(ti.i32, shape=3)
    values.from_numpy(values_host)
    binding = ti.algorithms.DeterministicScatterReducePlan(keys, 3).bind(values, output)
    binding.run()
    np.testing.assert_array_equal(output.to_numpy(), _stable_group_sum(keys, values_host, 3))
    output.fill(0)
    builder = ti.graph.GraphBuilder()
    builder.append_native(binding.graph_action())
    graph = builder.compile()
    graph.run({})
    np.testing.assert_array_equal(output.to_numpy(), _stable_group_sum(keys, values_host, 3))
    physical = graph.physical_plan()
    assert physical["physical_dispatch_count"] == 2
    assert physical["backend_recording_complete"]
    assert not physical["fragmented_native_plan"]


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_deterministic_scatter_reduce_handles_empty_valid_topology():
    keys = np.asarray([-1, 4, 9], dtype=np.int32)
    values = ti.ndarray(ti.f32, shape=keys.size)
    output = ti.ndarray(ti.f32, shape=4)
    values.from_numpy(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    binding = ti.algorithms.DeterministicScatterReducePlan(keys, 4).bind(values, output)
    output.fill(np.nan)
    binding.run()
    np.testing.assert_array_equal(output.to_numpy(), np.zeros(4, dtype=np.float32))
    report = binding.report()
    assert report.valid_count == 0
    assert report.ignored_count == keys.size


def test_deterministic_scatter_reduce_validates_topology_and_binding():
    ti.init(arch=ti.cpu)
    plan = None
    try:
        with pytest.raises(ValueError, match="num_groups"):
            ti.algorithms.DeterministicScatterReducePlan([0], 0)
        plan = ti.algorithms.DeterministicScatterReducePlan([0, 1, 0], 2)
        values = ti.ndarray(ti.f32, shape=2)
        output = ti.ndarray(ti.f32, shape=2)
        with pytest.raises(ValueError, match="values length"):
            plan.bind(values, output)
    finally:
        ti.reset()
    with pytest.raises(ti.TaichiRuntimeError, match="stale"):
        plan.report()
