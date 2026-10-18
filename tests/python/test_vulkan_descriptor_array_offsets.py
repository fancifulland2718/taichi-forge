"""Descriptor-relative offsets share one contract in ordinary and Graph calls."""

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_descriptor_views_rebind_without_overwriting_neighbors():
    source, output = ti.ndarray(ti.f32, 12), ti.ndarray(ti.f32, 12)
    values = np.arange(12, dtype=np.float32) - 3
    source.from_numpy(values)
    xs = [ti.experimental.ndarray_view(source, slices=slice(start, start + 3)) for start in (1, 5)]
    ys = [ti.experimental.ndarray_view(output, slices=slice(start, start + 3)) for start in (2, 7)]

    @ti.kernel
    def transform(x: ti.types.ndarray(ti.f32, 1), scale: ti.f32, y: ti.types.ndarray(ti.f32, 1), bias: ti.i32):
        for i in range(x.shape[0]):
            y[i] = scale * x[i] + bias

    def check(index, scale, bias):
        expected = np.full(12, -91, np.float32)
        start, end = (1, 2) if index == 0 else (5, 7)
        expected[end : end + 3] = scale * values[start : start + 3] + bias
        np.testing.assert_array_equal(output.to_numpy(), expected)
        np.testing.assert_array_equal(source.to_numpy(), values)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        transform,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.i32),
    )
    graph = builder.compile()
    binding = graph.bind(dict(x=xs[0], y=ys[0], scale=2.0, bias=7))
    for index, scale, bias in ((0, 2.0, 7), (1, -3.0, 11), (0, 4.0, -5)):
        output.fill(-91)
        transform(xs[index], scale, ys[index], bias)
        check(index, scale, bias)
        output.fill(-91)
        binding.update(x=xs[index], y=ys[index], scale=scale, bias=bias)
        graph.run(binding)
        check(index, scale, bias)
    graph.close()
