"""Fusion preserves every observable store and follows sequence mutations."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("dtype", [ti.i32, ti.u32, ti.f32])
@pytest.mark.parametrize("inplace", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_indexed_transform_sequence_preserves_all_outputs(dtype, inplace):
    n = 5
    src = ti.ndarray(dtype, n)
    transformed = src if inplace else ti.ndarray(dtype, n)
    indices = ti.ndarray(ti.i32, 6)
    gathered = ti.ndarray(dtype, 6)
    dst = ti.ndarray(dtype, 8)
    np_dtype = {ti.i32: np.int32, ti.u32: np.uint32, ti.f32: np.float32}[dtype]
    method = {ti.cpu: "cpu_native", ti.cuda: "cuda_device", ti.vulkan: "vulkan_native"}[ti.cfg.arch]
    sequence = ti.algorithms.primitive_sequence()
    sequence.transform(src, transformed, scale=3, bias=5, method=method)
    sequence.gather(transformed, indices, gathered, method=method)
    sequence.scatter(gathered, indices, dst, method=method)
    for iteration in range(2):
        values = np.arange(n, dtype=np_dtype) + 13 + iteration
        index_values = np.asarray([4, 0, 5, -1, 2, 9] if iteration == 0 else [1, 4, 6, -2, 1, 8], np.int32)
        if ti.cfg.arch == ti.cpu and iteration == 1:
            # CPU plain scatter requires unique valid destination indices.
            index_values[4] = 2
        src.from_numpy(values)
        if not inplace:
            transformed.fill(77)
        indices.from_numpy(index_values)
        gathered.fill(88)
        dst.fill(99)
        if iteration == 0:
            sequence.prewarm()
        else:
            graph.run({})
        full = (values * 3 + 5).astype(np_dtype)
        selected = np.asarray([full[i] if 0 <= i < n else 0 for i in index_values], np_dtype)
        expected_dst = np.full(8, 99, np_dtype)
        for i, value in zip(index_values, selected):
            if 0 <= i < 8:
                expected_dst[i] = value
        np.testing.assert_array_equal(transformed.to_numpy(), full)
        np.testing.assert_array_equal(gathered.to_numpy(), selected)
        np.testing.assert_array_equal(dst.to_numpy(), expected_dst)
        if iteration == 0:
            builder = ti.graph.GraphBuilder()
            builder.append_native(sequence)
            graph = builder.compile()
            graph._prewarm()
    assert sequence.fused_plan_count == int(ti.cfg.arch == ti.vulkan and not inplace)
    graph.close()
    sequence.clear()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_appending_call_invalidates_selected_fusion():
    src, tmp, gathered, dst, extra = [ti.ndarray(ti.i32, 7) for _ in range(5)]
    indices = ti.ndarray(ti.i32, 7)
    src.from_numpy(np.arange(7, dtype=np.int32))
    indices.from_numpy(np.arange(6, -1, -1, dtype=np.int32))
    sequence = ti.algorithms.primitive_sequence()
    sequence.transform(src, tmp, scale=3, bias=5, method="vulkan_native")
    sequence.gather(tmp, indices, gathered, method="vulkan_native")
    sequence.scatter(gathered, indices, dst, method="vulkan_native")
    sequence.prewarm()
    assert sequence.fused_plan_count == 1
    sequence.transform(dst, extra, scale=2, bias=1, method="vulkan_native")
    extra.fill(-1)
    sequence.run()
    np.testing.assert_array_equal(extra.to_numpy(), (np.arange(7) * 3 + 5) * 2 + 1)
    assert sequence.fused_plan_count == 0
    sequence.clear()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_fusion_updates_transform_tail_with_short_indices():
    n = 513
    src, tmp, dst = [ti.ndarray(ti.i32, n) for _ in range(3)]
    indices, gathered = ti.ndarray(ti.i32, 3), ti.ndarray(ti.i32, 3)
    src.from_numpy(np.arange(n, dtype=np.int32))
    indices.from_numpy(np.asarray([-1, 0, n - 1], np.int32))
    dst.fill(-17)
    sequence = ti.algorithms.primitive_sequence()
    sequence.transform(src, tmp, scale=3, bias=5)
    sequence.gather(tmp, indices, gathered)
    sequence.scatter(gathered, indices, dst)
    sequence.run()
    assert sequence.fused_plan_count == 1
    np.testing.assert_array_equal(tmp.to_numpy(), np.arange(n) * 3 + 5)
    np.testing.assert_array_equal(gathered.to_numpy(), [0, 5, (n - 1) * 3 + 5])
    reference = np.full(n, -17)
    reference[[0, n - 1]] = [5, (n - 1) * 3 + 5]
    np.testing.assert_array_equal(dst.to_numpy(), reference)
    sequence.clear()


