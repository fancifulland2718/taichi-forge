"""Kernel metadata snapshots must not freeze array contents or later calls."""

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aliased_payload_and_changed_argument_metadata():
    @ti.kernel
    def update(a: ti.types.ndarray(ti.i32, 2), b: ti.types.ndarray(ti.i32, 2), repeats: ti.i32):
        for i, j in a:
            for _ in range(repeats):
                a[i, j] += 1
                b[i, j] += a[i, j]

    for shape, repeats in (((3, 5), 1), ((7, 2), 3), ((3, 5), 2), ((2, 4), 0)):
        data = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
        array = ti.ndarray(ti.i32, shape=shape)
        array.from_numpy(data)
        # These arguments intentionally alias: only ABI metadata is immutable.
        update(array, array, repeats)
        expected = data * (2**repeats) + 2 * (2**repeats - 1)
        np.testing.assert_array_equal(array.to_numpy(), expected)
    assert len(update._primal.compiled_kernels) == 1


@pytest.mark.parametrize("rank", [0, 2])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_affine_graph_refreshes_strides_and_preserves_aliased_matrix_elements(rank):
    matrix = ti.types.matrix(2, 3, ti.i32)

    @ti.kernel
    def update(a: ti.types.ndarray(matrix, rank), b: ti.types.ndarray(matrix, rank)):
        for index in ti.grouped(a):
            for row, column in ti.static(ti.ndrange(2, 3)):
                a[index][row, column] += 1
                b[index][row, column] += a[index][row, column]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(update, *[ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, matrix, ndim=rank) for name in ("a", "b")])
    graph = builder.compile()
    try:
        for shape in (((5, 7), (8, 6)) if rank else ((), ())):
            array = ti.Matrix.ndarray(2, 3, ti.i32, shape=shape)
            expected = np.arange(int(np.prod(shape)) * 6, dtype=np.int32).reshape(shape + (2, 3))
            array.from_numpy(expected)
            # The same compiled affine-capable entry first sees canonical
            # descriptors, then strided views, then canonical descriptors again.
            selected = (slice(1, None, 2), slice(None, None, 3)) if rank else ()
            view = ti.experimental.ndarray_view(array, slices=selected) if rank else array
            for binding, subset in ((array, (...,)), (view, selected), (array, (...,))):
                graph.run(dict(a=binding, b=binding))
                expected[subset] = 2 * expected[subset] + 2
                np.testing.assert_array_equal(array.to_numpy(), expected)
        assert len(update._primal.compiled_kernels) == 1
    finally:
        graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_affine_stride_selection_uses_real_function_context():
    @ti.real_func
    def read(values: ti.types.ndarray(ti.i32, 2), i: ti.i32, j: ti.i32) -> ti.i32:
        return values[i, j]

    @ti.kernel
    def copy(values: ti.types.ndarray(ti.i32, 2), out: ti.types.ndarray(ti.i32, 2)):
        for i, j in out:
            out[i, j] = read(values, i, j)

    for shape in ((7, 8), (9, 11)):
        source = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
        array = ti.ndarray(ti.i32, shape=shape)
        array.from_numpy(source)
        view = ti.experimental.ndarray_view(array, slices=(slice(1, None, 2), slice(2, None, 3)))
        out = ti.ndarray(ti.i32, shape=view.shape)
        copy(view, out)
        np.testing.assert_array_equal(out.to_numpy(), source[1::2, 2::3])
