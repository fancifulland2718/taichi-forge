"""Compiler metadata is shared across families, occurrence bindings are not."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.graph import _graph
from taichi_forge.graph._recipes.task_facts import GraphKernelTaskFacts
from taichi_forge.lang import impl
from tests import test_utils


def test_task_facts_retry_failed_query(monkeypatch):
    calls = []
    tasks = (SimpleNamespace(backend="cuda", task_type="serial"),)

    def query(kernel):
        calls.append(kernel)
        if len(calls) == 1:
            raise RuntimeError("temporary compiler query failure")
        return tasks

    monkeypatch.setattr(_graph, "_kernel_task_manifests", query)
    facts = GraphKernelTaskFacts(object())
    with pytest.raises(RuntimeError, match="temporary"):
        facts.snapshot
    result = facts.snapshot
    assert facts.snapshot is result
    assert result.manifests is tasks
    assert not result.fusible_ranges
    assert not result.bounded_sparse_parent
    assert not result.device_range
    assert len(calls) == 2


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_dispatch_families_share_facts_preserve_bindings(monkeypatch):
    @ti.kernel
    def write(out: ti.types.ndarray(ti.i32, 1), limits: ti.types.ndarray(ti.i32, 1), value: ti.i32):
        for i in range(limits[0]):
            out[i] = value + i

    calls = []
    query = _graph._kernel_task_manifests

    def observed(kernel):
        calls.append(kernel)
        return query(kernel)

    monkeypatch.setattr(_graph, "_kernel_task_manifests", observed)
    builder = ti.graph.GraphBuilder()
    count = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "count", ti.i32, ndim=1)
    for occurrence in range(8):
        name = "left" if occurrence % 2 == 0 else "right"
        builder.dispatch(
            write,
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1),
            count,
            ti.graph.Arg(ti.graph.ArgKind.SCALAR, name + "_value", ti.i32),
        )
    gpu = impl.current_cfg().arch in (ti.cuda, ti.vulkan)
    assert len(calls) == int(gpu)
    if gpu:
        sources = builder._graph_range_dispatch_sources
        assert len(sources) == 8
        assert len({id(source.baseline_manifests) for source in sources}) == 1
        assert not builder._graph_offload_fusion_sources
        assert not builder._graph_sparse_traversal_sources
        for source in builder._graph_memory_sources:
            source._prepare_definition()
            assert source.baseline_manifests is sources[0].baseline_manifests
        assert len(calls) == 1

    graph = builder.compile()
    left, right = ti.ndarray(ti.i32, 9), ti.ndarray(ti.i32, 9)
    limits = ti.ndarray(ti.i32, 1)
    limits.from_numpy(np.array([9], np.int32))
    for offset in (10, 30):
        graph.run(dict(left=left, right=right, count=limits, left_value=offset, right_value=offset + 100))
        np.testing.assert_array_equal(left.to_numpy(), np.arange(9) + offset)
        np.testing.assert_array_equal(right.to_numpy(), np.arange(9) + offset + 100)
    assert len(calls) == int(gpu)
