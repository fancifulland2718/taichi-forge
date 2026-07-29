import platform
import gc
import threading
import time
import weakref

import numpy as np
import pytest
from taichi_forge.lang.exception import TaichiCompilationError, TaichiRuntimeError

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from taichi_forge.lang import impl
from taichi_forge.graph._graph import (
    _GraphTemporaryArena,
    _new_runtime_graph_builder,
    gen_cpp_kernel,
)
from taichi_forge.graph._ir import (
    DispatchNode,
    GraphAccess,
    ResourceEffect,
    IfRegion,
    RuntimeBinding,
    SequentialRegion,
    SwitchRegion,
    TemporaryRequirement,
    WhileRegion,
    analyze_elementwise_fusion,
    analyze_graph_ir,
    graph_ir_to_dict,
    plan_temporary_memory,
)
from taichi_forge.graph._native import (
    DispatchGraphAction,
    NativeGraphExecutable,
    NativeGraphNode,
)
from tests import test_utils

supported_floating_types = (
    [ti.f32] if platform.system() == "Darwin" else [ti.f32, ti.f64]
)

supported_archs_cgraph = [ti.vulkan, ti.opengl]


def test_ndarray_arg_tensor_descriptor_roundtrip():
    vector_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        "vector_arg",
        ti.types.vector(3, ti.f32),
        ndim=2,
    )
    assert vector_arg.dtype() == ti.f32
    assert tuple(vector_arg.element_shape) == (3,)
    assert tuple(vector_arg.element_dtype().shape()) == (3,)
    assert vector_arg.element_dtype().element_type() == ti.f32

    matrix_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        "matrix_arg",
        ti.types.matrix(2, 4, ti.i32),
        ndim=1,
    )
    assert matrix_arg.dtype() == ti.i32
    assert tuple(matrix_arg.element_shape) == (2, 4)
    assert tuple(matrix_arg.element_dtype().shape()) == (2, 4)
    assert matrix_arg.element_dtype().element_type() == ti.i32


def test_matrix_arg_uses_canonical_tensor_shape():
    vector_arg = ti.graph.Arg(
        ti.graph.ArgKind.MATRIX,
        "vector_arg",
        ti.types.vector(3, ti.f32),
    )
    assert tuple(vector_arg.element_shape) == (3,)
    assert tuple(vector_arg.element_dtype().shape()) == (3,)

    matrix_arg = ti.graph.Arg(
        ti.graph.ArgKind.MATRIX,
        "matrix_arg",
        ti.types.matrix(2, 4, ti.i32),
    )
    assert tuple(matrix_arg.element_shape) == (2, 4)
    assert tuple(matrix_arg.element_dtype().shape()) == (2, 4)


def test_rank_three_ndarray_tensor_descriptor_rejected_early():
    tensor_type = ti_core.get_type_factory_instance().get_tensor_type([2, 3, 4], ti.f32)
    with pytest.raises(ValueError, match="rank 1.*rank 2"):
        ti.types.ndarray(dtype=tensor_type, ndim=1)
    with pytest.raises(TaichiRuntimeError, match="rank 1.*rank 2"):
        ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY,
            "tensor_arg",
            tensor_type,
            ndim=1,
        )
    with pytest.raises(RuntimeError, match="rank-1 vector.*rank-2 matrix"):
        ti_core.Arg(
            ti.graph.ArgKind.NDARRAY,
            "tensor_arg",
            tensor_type,
            1,
            [],
        )


def test_legacy_matrix_symbolic_arg_adapter_is_bounded():
    from taichi_forge.graph._graph import flatten_args

    entries = [
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, f"entry_{i}", ti.f32) for i in range(4)
    ]
    assert flatten_args(([entries[:2], entries[2:]],)) == entries
    with pytest.raises(
        TaichiRuntimeError,
        match="must be a nested list",
    ):
        flatten_args((entries,))


def test_graph_struct_ndarray_descriptor_rejected_explicitly():
    element = ti.types.struct(value=ti.f32, index=ti.i32)
    with pytest.raises(
        TaichiRuntimeError,
        match="Graph StructNdarray arguments are not supported",
    ):
        ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY,
            "struct_arg",
            element,
            ndim=1,
        )


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_tensor_descriptor_runtime_binding():
    vector_type = ti.types.vector(3, ti.f32)
    matrix_type = ti.types.matrix(2, 2, ti.i32)

    @ti.kernel
    def fill(
        vectors: ti.types.ndarray(dtype=vector_type, ndim=1),
        matrices: ti.types.ndarray(dtype=matrix_type, ndim=1),
    ):
        vectors[0] = ti.Vector([1.0, 2.0, 3.0])
        matrices[0] = ti.Matrix([[1, 2], [3, 4]])

    vector_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "vectors", vector_type, ndim=1)
    matrix_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "matrices", matrix_type, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, vector_arg, matrix_arg)
    graph = builder.compile()

    vectors = ti.Vector.ndarray(3, dtype=ti.f32, shape=1)
    matrices = ti.Matrix.ndarray(2, 2, dtype=ti.i32, shape=1)
    graph.run({"vectors": vectors, "matrices": matrices})

    np.testing.assert_array_equal(
        vectors.to_numpy(), np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        matrices.to_numpy(), np.array([[[1, 2], [3, 4]]], dtype=np.int32)
    )


@test_utils.test(arch=ti.cpu)
def test_matrix_runtime_shape_is_not_flattened_by_element_count():
    matrix_type = ti.types.matrix(2, 3, ti.i32)

    @ti.kernel
    def consume(value: matrix_type, out: ti.types.ndarray(ti.i32, ndim=1)):
        out[0] = value[0, 0]

    value_arg = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "value", matrix_type)
    out_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(consume, value_arg, out_arg)
    graph = builder.compile()
    out = ti.ndarray(ti.i32, shape=1)

    with pytest.raises(RuntimeError, match="Matrix argument value with shape"):
        graph.run(
            {
                "value": ti.Matrix([[1, 2], [3, 4], [5, 6]], ti.i32),
                "out": out,
            }
        )


@test_utils.test(arch=ti.cpu)
def test_graph_matrix_injection_cache_uses_structural_key():
    from taichi_forge.aot.utils import produce_injected_args_for_graph

    vector_type = ti.types.vector(3, ti.i32)

    @ti.kernel
    def consume(value: vector_type, factor: ti.template()):
        pass

    first_arg = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "value", vector_type)
    second_arg = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "other_name", vector_type)
    first = produce_injected_args_for_graph(
        consume._primal, (first_arg,), template_args={"factor": 1}
    )
    cache = consume._primal._graph_template_injection_cache
    second = produce_injected_args_for_graph(
        consume._primal, (second_arg,), template_args={"factor": 2}
    )

    assert cache[1] == consume._primal._graph_template_injection_cache[1]
    assert first[0] is second[0]
    assert first[1] == 1
    assert second[1] == 2


@test_utils.test(arch=ti.cpu)
def test_graph_rwtexture_descriptor_mismatch_rejected_before_compile():
    from taichi_forge.aot.utils import produce_injected_args_for_graph

    @ti.kernel
    def write(
        tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
    ):
        pass

    bad_format = ti.graph.Arg(
        ti.graph.ArgKind.RWTEXTURE,
        "tex",
        ndim=2,
        fmt=ti.Format.rgba8,
    )
    with pytest.raises(
        TaichiCompilationError,
        match="RWTexture format mismatch",
    ):
        produce_injected_args_for_graph(write._primal, (bad_format,))

    bad_ndim = ti.graph.Arg(
        ti.graph.ArgKind.RWTEXTURE,
        "tex",
        ndim=3,
        fmt=ti.Format.r32f,
    )
    with pytest.raises(
        TaichiCompilationError,
        match="RWTexture descriptor mismatch",
    ):
        produce_injected_args_for_graph(write._primal, (bad_ndim,))


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_int():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(n):
            pos[i] = 1

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.i32, ndim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=(n,))
    g.run({"pos": a})
    assert (a.to_numpy() == np.ones(4)).all()


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_1dim_scalar():
    @ti.kernel
    def ti_test_debug(arr: ti.types.ndarray(ndim=1)):
        arr[0] = 0

    debug_arr = ti.ndarray(ti.i32, shape=5)
    sym_debug_arr = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "debug_arr", ti.types.vector(1, ti.f32), ndim=1
    )

    g_builder = ti.graph.GraphBuilder()
    g_builder.dispatch(ti_test_debug, sym_debug_arr)


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_0dim():
    @ti.kernel
    def test(pos: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        pos[None] = 1

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.i32, ndim=0)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=())
    g.run({"pos": a})
    assert a.to_numpy() == 1


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_float():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, ndim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.f32, shape=(n,))
    g.run({"pos": a})
    assert (a.to_numpy() == (np.ones(4) * 2.5)).all()


@test_utils.test(arch=supported_archs_cgraph)
def test_arg_mismatched_ndim():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, ndim=2)
    g_init = ti.graph.GraphBuilder()
    with pytest.raises(
        TaichiCompilationError, match="doesn't match kernel's annotated ndim"
    ):
        g_init.dispatch(test, sym_pos)


@test_utils.test(arch=supported_archs_cgraph)
def test_arg_mismatched_ndim_ndarray():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, 1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.f32, shape=(n, n))
    with pytest.raises(RuntimeError, match="Dispatch node is compiled for"):
        g.run({"pos": a})


@test_utils.test(arch=supported_archs_cgraph)
def test_repeated_arg_name():
    n = 4

    @ti.kernel
    def test1(pos: ti.types.ndarray(ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    @ti.kernel
    def test2(v: ti.f32):
        for i in range(n):
            print(v)

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, ndim=1)
    sym_pos1 = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "pos", ti.f32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(test1, sym_pos)

    with pytest.raises(RuntimeError):
        builder.dispatch(test2, sym_pos1)


@test_utils.test(arch=supported_archs_cgraph)
def test_arg_mismatched_scalar_dtype():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(ndim=1), val: ti.f32):
        for i in range(n):
            pos[i] = val

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, 1)
    sym_val = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "val", ti.i32)
    g_init = ti.graph.GraphBuilder()
    with pytest.raises(
        TaichiCompilationError, match="doesn't match kernel's annotated dtype"
    ):
        g_init.dispatch(test, sym_pos, sym_val)


@test_utils.test(arch=supported_archs_cgraph)
def test_arg_mismatched_ndarray_dtype():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.i32, 1)
    g_init = ti.graph.GraphBuilder()
    with pytest.raises(
        TaichiCompilationError, match="doesn't match kernel's annotated dtype"
    ):
        g_init.dispatch(test, sym_pos)


@test_utils.test(arch=supported_archs_cgraph)
def test_ndarray_dtype_mismatch_runtime():
    n = 4

    @ti.kernel
    def test(pos: ti.types.ndarray(ndim=1)):
        for i in range(n):
            pos[i] = 2.5

    sym_pos = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pos", ti.f32, ndim=1)
    g_init = ti.graph.GraphBuilder()
    g_init.dispatch(test, sym_pos)
    g = g_init.compile()

    a = ti.ndarray(ti.i32, shape=(n,))
    with pytest.raises(RuntimeError, match="but got an ndarray with dtype="):
        g.run({"pos": a})


def build_graph_vector(N, dtype):
    @ti.kernel
    def vector_sum(
        mat: ti.types.vector(N, dtype), res: ti.types.ndarray(dtype=dtype, ndim=1)
    ):
        res[0] = mat.sum() + mat[2]

    sym_A = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "mat", ti.types.vector(N, dtype))
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "res", dtype, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(vector_sum, sym_A, sym_res)
    graph = builder.compile()
    return graph


def build_graph_matrix(N, dtype):
    @ti.kernel
    def matrix_sum(
        mat: ti.types.matrix(N, 2, dtype), res: ti.types.ndarray(dtype=dtype, ndim=1)
    ):
        res[0] = mat.sum()

    sym_A = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "mat", ti.types.matrix(N, 2, dtype))
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "res", dtype, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(matrix_sum, sym_A, sym_res)
    graph = builder.compile()
    return graph


@pytest.mark.sm70
@pytest.mark.parametrize(
    "dt", [ti.u8, ti.u16, ti.u32, ti.u64, ti.i8, ti.i16, ti.i32, ti.i64]
)
@test_utils.test(arch=supported_archs_cgraph)
def test_matrix_int(dt):
    if ti.lang.impl.current_cfg().arch == ti.opengl and dt not in [ti.u32, ti.i32]:
        return
    n = 4
    A = ti.Matrix([4, 5] * n, dt)
    res = ti.ndarray(dt, shape=(1,))
    graph = build_graph_matrix(n, dtype=dt)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == 36


@pytest.mark.parametrize("dt", supported_floating_types)
@test_utils.test(arch=supported_archs_cgraph)
def test_matrix_float(dt):
    if ti.lang.impl.current_cfg().arch == ti.opengl and dt not in [ti.f32]:
        return
    n = 4
    A = ti.Matrix([4.2, 5.7] * n, dt)
    res = ti.ndarray(dt, shape=(1,))
    graph = build_graph_matrix(n, dtype=dt)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(39.6, rel=1e-5)


@pytest.mark.sm70
@test_utils.test(arch=[ti.vulkan])
def test_matrix_float16():
    n = 4
    A = ti.Matrix([4.0, 5.0] * n, ti.f16)
    res = ti.ndarray(ti.f16, shape=(1,))
    graph = build_graph_matrix(n, dtype=ti.f16)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(36.0, rel=1e-5)


@pytest.mark.sm70
@pytest.mark.parametrize(
    "dt", [ti.u8, ti.u16, ti.u32, ti.u64, ti.i8, ti.i16, ti.i32, ti.i64]
)
@test_utils.test(arch=supported_archs_cgraph)
def test_vector_int(dt):
    if ti.lang.impl.current_cfg().arch == ti.opengl and dt not in [ti.u32, ti.i32]:
        return
    n = 12
    A = ti.Vector([1, 3, 13, 4, 5, 6, 7, 2, 3, 4, 1, 25], dt)
    res = ti.ndarray(dt, shape=(1,))
    graph = build_graph_vector(n, dtype=dt)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == 87


@pytest.mark.parametrize("dt", supported_floating_types)
@test_utils.test(arch=supported_archs_cgraph)
def test_vector_float(dt):
    if ti.lang.impl.current_cfg().arch == ti.opengl and dt not in [ti.f32]:
        return
    n = 8
    A = ti.Vector([1.4, 3.7, 13.2, 4.5, 5.6, 6.1, 7.2, 2.6], dt)
    res = ti.ndarray(dt, shape=(1,))
    graph = build_graph_vector(n, dtype=dt)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(57.5, rel=1e-5)


@pytest.mark.sm70
@test_utils.test(arch=[ti.vulkan])
def test_vector_float16():
    n = 4
    A = ti.Vector([1.4, 3.7, 13.2, 4.5], ti.f16)
    res = ti.ndarray(ti.f16, shape=(1,))
    graph = build_graph_vector(n, dtype=ti.f16)
    graph.run({"mat": A, "res": res})
    assert res.to_numpy()[0] == test_utils.approx(36.0, rel=1e-2)


@pytest.mark.parametrize("dt", supported_floating_types)
@test_utils.test(arch=supported_archs_cgraph)
def test_arg_float(dt):
    @ti.kernel
    def foo(a: dt, b: ti.types.ndarray(dtype=dt, ndim=1)):
        b[0] = a

    k = ti.ndarray(dt, shape=(1,))

    sym_A = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "mat", dt)
    sym_B = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "b", dt, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(foo, sym_A, sym_B)
    graph = builder.compile()
    graph.run({"mat": 3.12, "b": k})
    assert k.to_numpy()[0] == test_utils.approx(3.12, rel=1e-5)


@test_utils.test(arch=supported_archs_cgraph)
def test_mixed_runtime_arg_cache_updates_dynamic_values():
    n = 8

    @ti.kernel
    def scale(
        src: ti.types.ndarray(dtype=ti.i32, ndim=1),
        factor: ti.i32,
        bias: ti.i32,
        dst: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in src:
            dst[i] = src[i] * factor + bias

    sym_src = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "src", ti.i32, ndim=1)
    sym_factor = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "factor", ti.i32)
    sym_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.i32)
    sym_dst = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(scale, sym_src, sym_factor, sym_bias, sym_dst)
    graph = builder.compile()

    src_np = np.arange(n, dtype=np.int32)
    src = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(src_np)
    dst = ti.ndarray(ti.i32, shape=n)

    graph.run({"src": src, "factor": 2, "bias": 3, "dst": dst})
    assert np.array_equal(dst.to_numpy(), src_np * 2 + 3)

    graph.run({"src": src, "factor": 5, "bias": -1, "dst": dst})
    assert np.array_equal(dst.to_numpy(), src_np * 5 - 1)

    other_np = (src_np + 10).astype(np.int32)
    other = ti.ndarray(ti.i32, shape=n)
    other.from_numpy(other_np)
    graph.run({"src": other, "factor": 4, "bias": 7, "dst": dst})
    assert np.array_equal(dst.to_numpy(), other_np * 4 + 7)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_graph_ndarray_registry_lifetime_after_runtime_arg_gc():
    sink = ti.field(ti.i32, shape=())

    @ti.kernel
    def consume(arr: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        sink[None] += arr[0]

    sym_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "arr", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(consume, sym_arr)
    graph = builder.compile()
    prog = impl.get_runtime().prog
    # Graph compilation uses temporary concrete exemplars for specialization;
    # drain those wrappers before taking the lifecycle baseline.
    gc.collect()
    ti.sync()
    baseline = prog._debug_ndarray_resource_stats()

    arr = ti.ndarray(ti.i32, shape=1)
    arr.fill(11)
    graph.run({"arr": arr})
    launched = prog._debug_ndarray_resource_stats()
    if impl.current_cfg().arch == ti.cpu:
        assert launched["inflight"] == baseline["inflight"]
    else:
        assert launched["inflight"] == baseline["inflight"] + 1

    del arr
    gc.collect()
    ti.sync()
    completed = prog._debug_ndarray_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]
    assert completed["created_total"] == baseline["created_total"] + 1
    assert completed["retired_total"] == baseline["retired_total"] + 1
    assert completed["released_total"] == baseline["released_total"] + 1
    assert sink[None] == 11


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_graph_submit_returns_one_public_completion_ticket():
    result = ti.ndarray(ti.i32, shape=1)

    @ti.kernel
    def first(dst: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        dst[0] = 20

    @ti.kernel
    def second(dst: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        dst[0] += 22

    dst_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(first, dst_arg)
    builder.dispatch(second, dst_arg)
    graph = builder.compile()
    prog = impl.get_runtime().prog

    next_before_run = prog._debug_runtime_completion_stats()["next_sequence"]
    assert graph.run({"dst": result}) is None
    next_after_run = prog._debug_runtime_completion_stats()["next_sequence"]
    assert next_after_run == next_before_run

    ticket = graph.submit({"dst": result})
    assert isinstance(ticket, ti.graph.SubmissionTicket)
    next_after_submit = prog._debug_runtime_completion_stats()["next_sequence"]
    assert ticket.sequence == next_before_run
    assert next_after_submit == next_before_run + 1
    assert ticket.backend == ti_core.arch_name(impl.current_cfg().arch)
    if impl.current_cfg().arch == ti.cpu:
        assert not ticket._has_backend_work
    else:
        # A short GPU graph may complete during the nonblocking collection at
        # the end of completion recording. Both pending and already-completed
        # tokens preserve the same sequence/ordering contract.
        assert ticket._has_backend_work or ticket.done()
    assert ticket.wait() is None
    assert ticket.done()
    assert ticket.done()
    assert ticket.wait() is None
    assert result.to_numpy()[0] == 42

    pacer = ti.graph.SubmissionPacer(2, max_in_flight_per_lane=1, max_queued=4)
    paced = graph.submit({"dst": result}, pacer=pacer, lane="simulation")
    assert paced.wait() is None
    pacing = pacer.statistics()
    assert pacing["admission_calls"] == 1
    assert pacing["grants"] == 1
    assert pacing["completed"] == 1
    assert pacing["in_flight"] == 0
    assert pacing["lanes"]["simulation"]["completed"] == 1

    with pytest.raises(TaichiRuntimeError, match="require a SubmissionPacer"):
        graph.submit({"dst": result}, lane="unpaced")


@test_utils.test(arch=ti.cpu)
def test_runtime_submission_owner_registry_retains_until_ready():
    runtime = impl.get_runtime()

    class FakeCompletion:
        program_domain = -id(runtime)
        sequence = id(runtime)

        def __init__(self):
            self.ready = False

        def done(self):
            return self.ready

    class Owner:
        pass

    completion = FakeCompletion()
    owner = Owner()
    owner_ref = weakref.ref(owner)
    runtime.retain_runtime_submission_owner(completion, owner)
    del owner
    gc.collect()
    assert owner_ref() is not None

    completion.ready = True
    runtime.collect_ready_runtime_submission_owners()
    gc.collect()
    assert owner_ref() is None


@test_utils.test(arch=ti.cpu)
def test_graph_rejects_runtime_arg_key_mismatch():
    @ti.kernel
    def fill(value: ti.i32, out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in out:
            out[i] = value

    sym_value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    sym_out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, sym_value, sym_out)
    graph = builder.compile()
    out = ti.ndarray(ti.i32, shape=4)

    with pytest.raises(
        TaichiRuntimeError, match="Missing graph runtime arguments: value"
    ):
        graph.run({"out": out})
    with pytest.raises(
        TaichiRuntimeError, match="Unexpected graph runtime arguments: typo"
    ):
        graph.run({"value": 3, "out": out, "typo": 1})
    with pytest.raises(TaichiRuntimeError, match=r"Graph\.run\(\) expects a dict"):
        graph.run([out])
    with pytest.raises(
        TaichiRuntimeError, match="Missing graph runtime arguments: value"
    ):
        graph.submit({"out": out})
    with pytest.raises(
        TaichiRuntimeError,
        match=r"Graph\.submit\(\) expects a dict",
    ):
        graph.submit([out])


@test_utils.test(arch=ti.cpu)
def test_same_graph_two_thread_invocations_do_not_interleave():
    counter = ti.field(ti.i32, shape=())

    @ti.kernel
    def increment():
        counter[None] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment)
    graph = builder.compile()
    original_run_impl = graph._run_impl
    state_lock = threading.Lock()
    start = threading.Barrier(2)
    active = 0
    max_active = 0
    errors = []

    def observed_run(args):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.001)
            original_run_impl(args)
        finally:
            with state_lock:
                active -= 1

    graph._run_impl = observed_run

    def worker():
        try:
            start.wait()
            for _ in range(8):
                graph.run({})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ti.sync()
    assert not errors
    assert max_active == 1
    assert counter[None] == 16


@test_utils.test(arch=ti.cpu)
def test_independent_graph_submit_balances_runtime_ad_exclusion_count():
    counter = ti.field(ti.i32, shape=())

    @ti.kernel
    def increment():
        counter[None] += 1

    first_builder = ti.graph.GraphBuilder()
    first_builder.dispatch(increment)
    first = first_builder.compile()
    second_builder = ti.graph.GraphBuilder()
    second_builder.dispatch(increment)
    second = second_builder.compile()
    start = threading.Barrier(2)
    errors = []

    def worker(graph):
        try:
            start.wait()
            for _ in range(32):
                graph.submit({}).wait()
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(first,)),
        threading.Thread(target=worker, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert ti.lang.impl.get_runtime()._active_graph_submissions == 0
    assert counter[None] == 64


@test_utils.test(arch=ti.cpu)
def test_graph_recovers_runtime_args_from_legacy_low_level_dispatch():
    """Strict validation remains compatible with precompiled-kernel adapters."""

    @ti.kernel
    def fill(value: ti.i32, out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in out:
            out[i] = value

    from taichi_forge.graph._graph import flatten_args, gen_cpp_kernel

    sym_value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    sym_out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    symbolic_args = flatten_args((sym_value, sym_out))
    kernel_cpp = gen_cpp_kernel(fill, (sym_value, sym_out))

    # Compatibility path used by adapters that instantiate template arguments
    # before dispatch and therefore cannot call GraphBuilder.dispatch().
    builder = ti.graph.GraphBuilder()
    builder._aot_graph_plan.dispatch(kernel_cpp, symbolic_args)
    builder._ensure_runtime_graph_builder().dispatch(kernel_cpp, symbolic_args)
    builder._dispatch_count += 1
    graph = builder.compile()

    out = ti.ndarray(ti.i32, shape=4)
    graph.run({"value": 7, "out": out})
    assert np.array_equal(out.to_numpy(), np.full(4, 7, dtype=np.int32))

    with pytest.raises(
        TaichiRuntimeError, match="Unexpected graph runtime arguments: typo"
    ):
        graph.run({"value": 7, "out": out, "typo": 1})


@test_utils.test(arch=ti.cpu)
def test_graph_dispatch_rejects_non_kernel_callables():
    @ti.func
    def helper():
        return 1

    builder = ti.graph.GraphBuilder()
    with pytest.raises(
        TaichiCompilationError,
        match="decorated Taichi kernel or an explicit kernel.grad",
    ):
        builder.dispatch(lambda: None)

    sequential = builder.create_sequential()
    with pytest.raises(
        TaichiCompilationError,
        match="Python callables and ti.func objects",
    ):
        sequential.dispatch(helper)


@pytest.mark.parametrize("mutation", ["builder", "sequential"])
@test_utils.test(arch=ti.cpu)
def test_compiled_graph_freezes_aot_plan(mutation):
    @ti.kernel
    def increment(value: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        value[None] += 1

    sym_value = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "value", ti.i32, ndim=0)
    builder = ti.graph.GraphBuilder()
    sequential = None
    if mutation == "builder":
        builder.dispatch(increment, sym_value)
    else:
        sequential = builder.create_sequential()
        sequential.dispatch(increment, sym_value)
        builder.append(sequential)

    graph = builder.compile()
    if mutation == "builder":
        builder.dispatch(increment, sym_value)
    else:
        sequential.dispatch(increment, sym_value)

    value = ti.ndarray(ti.i32, shape=())
    value.fill(0)
    graph._compiled_graph.jit_run(
        ti.lang.impl.current_cfg(),
        {"value": value.arr},
    )
    ti.sync()
    assert value.to_numpy()[()] == 1


@test_utils.test(arch=ti.cpu)
def test_reused_sequential_append_freezes_each_version():
    @ti.kernel
    def increment(value: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        value[None] += 1

    sym_value = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "value", ti.i32, ndim=0)
    builder = ti.graph.GraphBuilder()
    sequential = builder.create_sequential()
    sequential.dispatch(increment, sym_value)
    builder.append(sequential)
    sequential.dispatch(increment, sym_value)
    builder.append(sequential)
    graph = builder.compile()

    runtime_value = ti.ndarray(ti.i32, shape=())
    runtime_value.fill(0)
    graph.run({"value": runtime_value})
    ti.sync()
    assert runtime_value.to_numpy()[()] == 3

    aot_value = ti.ndarray(ti.i32, shape=())
    aot_value.fill(0)
    graph._compiled_graph.jit_run(
        ti.lang.impl.current_cfg(),
        {"value": aot_value.arr},
    )
    ti.sync()
    assert aot_value.to_numpy()[()] == 3


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_aot_jit_graph_pins_ndarray_runtime_arg_until_sync():
    sink = ti.field(ti.i32, shape=())

    @ti.kernel
    def consume(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            ti.atomic_add(sink[None], values[i])

    sym_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(consume, sym_values)
    graph = builder.compile()

    values = ti.ndarray(ti.i32, shape=16)
    values.fill(3)
    native_view = values.arr
    graph._compiled_graph.jit_run(
        ti.lang.impl.current_cfg(),
        {"values": native_view},
    )

    # The high-level wrapper requests retirement immediately. GPU execution is
    # allowed to finish later, so the low-level JIT/AOT path must keep its own
    # generation-qualified in-flight lease until Program synchronization.
    del values
    gc.collect()
    ti.sync()
    assert sink[None] == 48


def test_elementwise_fusion_analysis_requires_explicit_safe_metadata():
    safe_effects = (
        ResourceEffect("source", GraphAccess.READ),
        ResourceEffect("destination", GraphAccess.WRITE),
    )
    root = SequentialRegion(
        (
            DispatchNode(
                "map_a",
                effects=safe_effects,
                iteration_domain="range:n",
                opaque=False,
                elementwise=True,
            ),
            DispatchNode(
                "map_b",
                effects=safe_effects,
                iteration_domain="range:n",
                opaque=False,
                elementwise=True,
            ),
            DispatchNode(
                "atomic_map",
                effects=(ResourceEffect("sum", GraphAccess.ATOMIC),),
                iteration_domain="range:n",
                opaque=False,
                elementwise=True,
            ),
            DispatchNode(
                "other_domain",
                effects=safe_effects,
                iteration_domain="range:m",
                opaque=False,
                elementwise=True,
            ),
            DispatchNode(
                "random_map",
                effects=safe_effects,
                iteration_domain="range:m",
                opaque=False,
                elementwise=True,
                side_effects=("random",),
            ),
        )
    )
    plan = analyze_elementwise_fusion(root).to_dict()
    assert plan == {
        "candidate_groups": 1,
        "candidate_dispatches": 2,
        "eligible_dispatches": 3,
        "blocked_dispatches": 2,
        "blockers": {"atomic_effect": 1, "side_effect": 1},
        "applied_groups": 0,
        "lowering_available": False,
        "decision": "cross_kernel_ir_composer_unavailable",
    }

    opaque = analyze_elementwise_fusion(
        SequentialRegion((DispatchNode("ordinary"),))
    ).to_dict()
    assert opaque["candidate_groups"] == 0
    assert opaque["blockers"] == {"opaque_dispatch": 1}
    assert opaque["decision"] == "no_safe_candidates"


def test_structured_control_ir_validates_and_serializes_fixed_schema():
    condition = SequentialRegion((DispatchNode("condition"),), name="condition")
    body = SequentialRegion((DispatchNode("body"),), name="body")
    while_region = WhileRegion(
        predicate="continue_flag",
        max_iterations=32,
        condition=condition,
        body=body,
        control_inputs=("residual", "tolerance", "active_count"),
        carried_state=("solution", "residual"),
        counter="iteration",
        status="terminal_status",
        chunk_size=4,
        masked_execution=True,
    )
    if_region = IfRegion(
        predicate="restart_flag",
        condition=condition,
        then_region=body,
        else_region=body,
        control_inputs=("status",),
    )
    switch_region = SwitchRegion(
        selector="phase",
        condition=condition,
        branches=(body, body),
        default_region=body,
        control_inputs=("status",),
    )
    root = SequentialRegion((while_region, if_region, switch_region))
    analysis = analyze_graph_ir(root)
    assert analysis.while_regions == 1
    assert analysis.if_regions == 1
    assert analysis.switch_regions == 1
    serialized = graph_ir_to_dict(root)
    serialized_while = serialized["children"][0]
    assert serialized_while["control_inputs"] == (
        "residual",
        "tolerance",
        "active_count",
    )
    assert serialized_while["status"] == "terminal_status"
    assert serialized_while["carried_state"] == (
        "solution",
        "residual",
    )
    assert serialized["children"][1]["has_else"]
    assert serialized["children"][2]["branch_count"] == 2

    with pytest.raises(ValueError, match="duplicate resources"):
        WhileRegion(
            predicate="continue_flag",
            max_iterations=1,
            condition=condition,
            body=body,
            control_inputs=("status", "status"),
        )
    with pytest.raises(ValueError, match="distinct control resource"):
        WhileRegion(
            predicate="continue_flag",
            max_iterations=1,
            condition=condition,
            body=body,
            status="continue_flag",
        )
    with pytest.raises(ValueError, match="cannot be a control input"):
        IfRegion(
            predicate="predicate",
            condition=condition,
            then_region=body,
            control_inputs=("predicate",),
        )


def test_temporary_memory_plan_reuses_only_nonoverlapping_intervals():
    sequential = SequentialRegion(
        (
            DispatchNode(
                "first",
                temporaries=(TemporaryRequirement("a", 64, 16),),
            ),
            DispatchNode(
                "second",
                temporaries=(TemporaryRequirement("b", 32, 8),),
            ),
        )
    )
    plan = plan_temporary_memory(sequential).to_dict()
    assert plan == {
        "declared_bytes": 96,
        "logical_bytes": 96,
        "planned_peak_bytes": 64,
        "reused_bytes": 32,
        "alignment_padding_bytes": 0,
        "slot_count": 1,
        "conflicting_requirements": 0,
        "opaque_bytes": 0,
        "materialized": False,
    }

    overlapping = SequentialRegion(
        (
            DispatchNode(
                "begin_a",
                temporaries=(TemporaryRequirement("a", 64, 16),),
            ),
            DispatchNode(
                "use_b",
                temporaries=(TemporaryRequirement("b", 32, 8),),
            ),
            DispatchNode(
                "end_a",
                temporaries=(TemporaryRequirement("a", 64, 16),),
            ),
        )
    )
    overlap_plan = plan_temporary_memory(overlapping).to_dict()
    assert overlap_plan["planned_peak_bytes"] == 96
    assert overlap_plan["reused_bytes"] == 0
    assert overlap_plan["slot_count"] == 2

    conflicting = SequentialRegion(
        (
            DispatchNode(
                "small",
                temporaries=(TemporaryRequirement("same", 64, 16),),
            ),
            DispatchNode(
                "large",
                temporaries=(TemporaryRequirement("same", 128, 16),),
            ),
        )
    )
    conflict_plan = plan_temporary_memory(conflicting).to_dict()
    assert conflict_plan["conflicting_requirements"] == 1
    assert conflict_plan["opaque_bytes"] == 128
    assert conflict_plan["planned_peak_bytes"] == 0


class _RecordedDispatchExecutable(NativeGraphExecutable):
    def __init__(self, kernel, args, tracker, lease, fixed_bindings=None):
        self._action = DispatchGraphAction(
            ((kernel, args),), fixed_bindings=fixed_bindings
        )
        self._tracker = tracker
        self._lease = lease
        fixed_names = frozenset((fixed_bindings or {}).keys())
        runtime_names = tuple(
            dict.fromkeys(arg.name for arg in args if arg.name not in fixed_names)
        )
        self._runtime_arg_schema = tuple(
            RuntimeBinding(name, "ndarray") for name in runtime_names
        )
        self._resource_effects = tuple(
            ResourceEffect(name, GraphAccess.READ_WRITE) for name in runtime_names
        )

    def run(self, runtime_args):
        self._tracker["fallback_runs"] += 1
        raise AssertionError("recordable native node was not lowered")

    @property
    def runtime_arg_schema(self):
        return self._runtime_arg_schema

    @property
    def resource_effects(self):
        return self._resource_effects

    @property
    def lifetime_leases(self):
        return (self._lease,)

    @property
    def recordable_action(self):
        return self._action

    @property
    def debug_info(self):
        return {"kind": "recorded_dispatch"}


class _RecordedDispatchNode(NativeGraphNode):
    def __init__(self, kernel, args, tracker, lease, fixed_bindings=None):
        self._kernel = kernel
        self._args = tuple(args)
        self._tracker = tracker
        self._lease = lease
        self._fixed_bindings = fixed_bindings

    def compile(self):
        return _RecordedDispatchExecutable(
            gen_cpp_kernel(self._kernel, self._args),
            self._args,
            self._tracker,
            self._lease,
            self._fixed_bindings,
        )


class _ValidatingLease:
    def __init__(self):
        self.valid = True
        self.validations = 0

    def validate_graph_lifetime(self):
        self.validations += 1
        if not self.valid:
            raise TaichiRuntimeError(
                "recordable provider generation changed; rebuild the Graph"
            )


class _OpaqueExecutable(NativeGraphExecutable):
    def __init__(self, tracker):
        self._tracker = tracker

    def run(self):
        self._tracker["runs"] += 1

    @property
    def debug_info(self):
        return {"kind": "opaque_test"}


class _OpaqueNode(NativeGraphNode):
    def __init__(self, tracker):
        self._tracker = tracker

    def compile(self):
        return _OpaqueExecutable(self._tracker)


class _TemporaryExecutable(_OpaqueExecutable):
    def __init__(self, tracker, temporary):
        super().__init__(tracker)
        self._temporary = temporary

    def run_with_graph_temporaries(self, temporaries, runtime_args=None):
        binding = temporaries[self._temporary.name]
        self._tracker.setdefault("buffers", []).append(
            (
                id(binding.storage),
                binding.offset,
                binding.bytes,
                binding.alignment,
                binding.slot,
            )
        )
        return super().run()

    @property
    def temporary_requirements(self):
        return (self._temporary,)


class _TemporaryNode(NativeGraphNode):
    def __init__(self, tracker, temporary):
        self._tracker = tracker
        self._temporary = temporary

    def compile(self):
        return _TemporaryExecutable(self._tracker, self._temporary)


class _ArenaDispatchAction(DispatchGraphAction):
    def __init__(self, dispatches, temporary_symbol, requirement, tracker):
        super().__init__(dispatches, conditional_body_safe=True)
        self._temporary_symbol = temporary_symbol
        self._requirement = requirement
        self._tracker = tracker

    @property
    def temporary_bindings(self):
        return {self._temporary_symbol: self._requirement.name}

    def bind_graph_temporaries(self, temporaries):
        binding = temporaries[self._requirement.name]
        if binding.offset != 0 or binding.bytes != self._requirement.bytes:
            return None
        self._tracker["temporary_binds"] += 1
        return {self._temporary_symbol: binding.storage}


class _ArenaDispatchExecutable(NativeGraphExecutable):
    def __init__(
        self,
        dispatches,
        source_name,
        output_name,
        temporary_symbol,
        requirement,
        tracker,
    ):
        self._action = _ArenaDispatchAction(
            dispatches, temporary_symbol, requirement, tracker
        )
        self._source_name = source_name
        self._output_name = output_name
        self._requirement = requirement
        self._tracker = tracker

    def run(self, runtime_args):
        self._tracker["fallback_runs"] += 1
        raise AssertionError("temporary recordable action was not lowered")

    @property
    def runtime_arg_schema(self):
        return (
            RuntimeBinding(self._source_name, "ndarray"),
            RuntimeBinding(self._output_name, "ndarray"),
        )

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self._source_name, GraphAccess.READ),
            ResourceEffect(self._output_name, GraphAccess.WRITE),
        )

    @property
    def temporary_requirements(self):
        return (self._requirement,)

    @property
    def recordable_action(self):
        return self._action


class _ArenaDispatchNode(NativeGraphNode):
    def __init__(self, dispatches, source, output, scratch, requirement, tracker):
        self._dispatches = dispatches
        self._source = source
        self._output = output
        self._scratch = scratch
        self._requirement = requirement
        self._tracker = tracker

    def compile(self):
        return _ArenaDispatchExecutable(
            self._dispatches,
            self._source.name,
            self._output.name,
            self._scratch.name,
            self._requirement,
            self._tracker,
        )


def _temporary_recordable_node(size, tracker, source, output):
    @ti.kernel
    def stage(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(size):
            scratch[index] = source[index] * 2

    @ti.kernel
    def finish(
        scratch: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in range(size):
            output[index] = scratch[index] + 1

    scratch = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "__provider_scratch", ti.i32, ndim=1
    )
    requirement = TemporaryRequirement("provider_scratch", size * 4, 16)
    dispatches = (
        (gen_cpp_kernel(stage, (source, scratch)), (source, scratch)),
        (gen_cpp_kernel(finish, (scratch, output)), (scratch, output)),
    )
    return _ArenaDispatchNode(
        dispatches, source, output, scratch, requirement, tracker
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_graph_materializes_and_reuses_temporary_arena():
    tracker = {"runs": 0}
    builder = ti.graph.GraphBuilder()
    builder.append_native(
        _TemporaryNode(tracker, TemporaryRequirement("first", 64, 16))
    )
    builder.append_native(
        _TemporaryNode(tracker, TemporaryRequirement("second", 32, 8))
    )
    graph = builder.compile()
    graph.run({})
    assert tracker["runs"] == 2
    first_storage, first_offset, first_bytes, first_alignment, first_slot = tracker[
        "buffers"
    ][0]
    second_storage, second_offset, second_bytes, second_alignment, second_slot = (
        tracker["buffers"][1]
    )
    assert first_storage == second_storage
    assert first_offset == second_offset == 0
    assert first_slot == second_slot == 0
    assert (first_bytes, first_alignment) == (64, 16)
    assert (second_bytes, second_alignment) == (32, 8)

    plan = graph._ir_debug_info["temporary_memory_plan"]
    assert plan["planned_peak_bytes"] == 64
    assert plan["reused_bytes"] == 32
    assert not plan["materialized"]
    memory = graph.execution_stats().memory
    assert memory.transient_temporary_bytes == 64
    assert memory.planned_temporary_bytes == 64
    assert memory.temporary_reuse_bytes == 32
    assert memory.opaque_temporary_bytes == 0
    assert memory.temporary_plan_materialized
    assert memory.persistent_temporary_bytes == 64
    assert memory.temporary_arena_capacity == 4
    assert memory.temporary_arena_slots == 1
    assert memory.temporary_arena_allocations == 1
    assert memory.temporary_arena_reuses == 0
    assert memory.temporary_arena_waits == 0
    assert memory.opaque_driver_bytes is None

    graph.run({})
    assert tracker["runs"] == 4
    assert tracker["buffers"][2:] == tracker["buffers"][:2]
    memory = graph.execution_stats().memory
    assert memory.temporary_arena_allocations == 1
    assert memory.temporary_arena_reuses == 1


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_recordable_provider_binds_graph_temporary_arena_without_public_scratch():
    size = 256
    tracker = {"temporary_binds": 0, "fallback_runs": 0}
    source_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(
        _temporary_recordable_node(
            size, tracker, source_arg, output_arg
        )
    )
    graph = builder.compile()

    source = ti.ndarray(ti.i32, shape=size)
    output = ti.ndarray(ti.i32, shape=size)
    values = np.arange(size, dtype=np.int32)
    source.from_numpy(values)
    graph.run({"source": source, "output": output})
    np.testing.assert_array_equal(output.to_numpy(), values * 2 + 1)
    assert tracker["fallback_runs"] == 0
    assert tracker["temporary_binds"] >= 1
    initial_binds = tracker["temporary_binds"]
    graph.run({"source": source, "output": output})
    assert tracker["temporary_binds"] == initial_binds
    with pytest.raises(TaichiRuntimeError, match="Unexpected graph runtime"):
        graph.run(
            {
                "source": source,
                "output": output,
                "__provider_scratch": ti.ndarray(ti.i32, shape=size),
            }
        )
    memory = graph.execution_stats().memory
    assert memory.planned_temporary_bytes == size * 4
    assert memory.persistent_temporary_bytes == size * 4

    second_source = ti.ndarray(ti.i32, shape=size)
    second_output = ti.ndarray(ti.i32, shape=size)
    second_values = values + 17
    second_source.from_numpy(second_values)
    first_ticket = graph.submit({"source": source, "output": output})
    second_ticket = graph.submit(
        {"source": second_source, "output": second_output}
    )
    first_ticket.wait()
    second_ticket.wait()
    np.testing.assert_array_equal(output.to_numpy(), values * 2 + 1)
    np.testing.assert_array_equal(
        second_output.to_numpy(), second_values * 2 + 1
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_recordable_provider_temporary_embeds_in_structured_while():
    size = 64
    tracker = {"temporary_binds": 0, "fallback_runs": 0}

    @ti.kernel
    def evaluate(
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(counter[None] < target)

    @ti.kernel
    def advance(
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            for index in range(size):
                source[index] = output[index]
            counter[None] += 1

    source_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    predicate_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0
    )
    counter_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=0
    )
    target_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(evaluate, predicate_arg, counter_arg, target_arg)
    body = builder.create_sequential()
    body.append_native(
        _temporary_recordable_node(
            size, tracker, source_arg, output_arg
        )
    )
    body.dispatch(
        advance,
        output_arg,
        source_arg,
        predicate_arg,
        counter_arg,
    )
    arch = ti.lang.impl.current_cfg().arch
    builder.while_loop(
        condition,
        body,
        predicate=predicate_arg,
        control_inputs=(counter_arg, target_arg),
        carried_state=(source_arg, output_arg),
        counter=counter_arg,
        max_iterations=8,
        lowering_mode=("native_required" if arch == ti.cuda else "auto"),
        name="temporary_provider_iteration",
    )
    builder.observe(counter_arg, name="terminal")
    graph = builder.compile()

    source = ti.ndarray(ti.i32, shape=size)
    output = ti.ndarray(ti.i32, shape=size)
    predicate = ti.ndarray(ti.i32, shape=())
    counter = ti.ndarray(ti.i32, shape=())
    values = np.arange(size, dtype=np.int32)
    source.from_numpy(values)
    output.fill(0)
    predicate.fill(0)
    counter.fill(0)
    args = {
        "source": source,
        "output": output,
        "predicate": predicate,
        "counter": counter,
        "target": 3,
    }
    if arch == ti.cuda:
        assert graph.submit(args).observations() == {
            "terminal": {"counter": 3}
        }
    else:
        graph.run(args)
        assert graph.latest_observations() == {"terminal": {"counter": 3}}
        assert graph.control_flow_stats()[0].logical_iterations == 3
    np.testing.assert_array_equal(source.to_numpy(), values * 8 + 7)
    assert tracker["fallback_runs"] == 0
    assert tracker["temporary_binds"] >= 1
    assert graph.execution_stats().memory.planned_temporary_bytes == size * 4


class _PendingGraphCompletion:
    has_backend_work = True

    def __init__(self):
        self.is_done = False
        self.waits = 0

    def done(self):
        return self.is_done

    def wait(self):
        self.waits += 1
        self.is_done = True


@test_utils.test(arch=ti.cpu)
def test_graph_temporary_arena_bounds_async_inflight_slots():
    plan = plan_temporary_memory(
        SequentialRegion(
            (
                DispatchNode(
                    "scratch",
                    temporaries=(TemporaryRequirement("scratch", 64, 16),),
                ),
            )
        )
    )
    arena = _GraphTemporaryArena(plan, capacity=2)
    first = arena.acquire()
    first_completion = _PendingGraphCompletion()
    first.attach(first_completion)
    second = arena.acquire()
    second_completion = _PendingGraphCompletion()
    second.attach(second_completion)

    third = arena.acquire()
    assert first_completion.waits == 1
    assert second_completion.waits == 0
    third.cancel()
    assert arena.stats == {
        "materialized": True,
        "capacity": 2,
        "slots": 2,
        "reserved_bytes": 128,
        "allocations": 2,
        "reuses": 1,
        "waits": 1,
    }


def _build_observation_graph():
    @ti.kernel
    def advance(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        residual: ti.types.ndarray(dtype=ti.f32, ndim=0),
    ):
        state[None] += 1
        residual[None] *= 0.5

    arg_state = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.i32, ndim=0)
    arg_residual = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "residual", ti.f32, ndim=0)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance, arg_state, arg_residual)
    builder.observe(arg_state, arg_residual, name="tail")
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_graph_observation_defers_readback_and_reports_memory(monkeypatch):
    monkeypatch.setenv("TI_GRAPH_OBSERVATION_SLOTS", "2")
    graph = _build_observation_graph()
    state = ti.ndarray(ti.i32, shape=())
    residual = ti.ndarray(ti.f32, shape=())
    state.fill(4)
    residual.fill(8.0)

    ticket = graph.submit({"state": state, "residual": residual})
    report = graph.execution_stats()
    memory = report.memory
    assert report.observation_node_count == 1
    assert report.segments[-1].kind == "observation"
    assert report.segments[-1].last_path == "asynchronous_snapshot"
    assert memory.observation_arena_capacity == 2
    assert memory.observation_arena_slots == 1
    assert memory.observation_arena_allocations == 1
    assert memory.observation_arena_reuses == 0
    assert memory.observation_materializations == 0
    assert memory.observation_host_readback_bytes == 0
    assert memory.persistent_observation_bytes == 8

    observed = ticket.observations()
    assert observed == {"tail": {"state": 5, "residual": 4.0}}
    assert ticket.observations() == observed
    memory = graph.execution_stats().memory
    assert memory.observation_materializations == 1
    assert memory.observation_host_readback_bytes == 8
    assert memory.persistent_observation_bytes >= 8

    state.fill(10)
    residual.fill(4.0)
    graph.run({"state": state, "residual": residual})
    assert graph.latest_observations() == {"tail": {"state": 11, "residual": 2.0}}
    memory = graph.execution_stats().memory
    assert memory.observation_arena_allocations == 1
    assert memory.observation_arena_reuses == 1
    assert memory.observation_materializations == 2
    assert memory.observation_host_readback_bytes == 16

    ir = graph._ir_debug_info
    assert ir["analysis"]["observation_nodes"] == 1
    observation = ir["root"]["children"][-1]
    assert observation["kind"] == "observation"
    assert not observation["synchronization"]
    assert not observation["opaque"]
    with pytest.raises(TaichiRuntimeError, match="serialized as AOT"):
        graph._compiled_graph


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_graph_observation_ring_preserves_unconsumed_snapshots(monkeypatch):
    monkeypatch.setenv("TI_GRAPH_OBSERVATION_SLOTS", "2")
    graph = _build_observation_graph()
    states = []
    residuals = []
    tickets = []
    for state_value in (0, 10, 20):
        state = ti.ndarray(ti.i32, shape=())
        residual = ti.ndarray(ti.f32, shape=())
        state.fill(state_value)
        residual.fill(8.0)
        states.append(state)
        residuals.append(residual)
        tickets.append(graph.submit({"state": state, "residual": residual}))

    memory = graph.execution_stats().memory
    assert memory.observation_arena_slots == 2
    assert memory.observation_arena_allocations == 2
    assert memory.observation_arena_reuses == 1
    assert memory.observation_materializations == 1
    assert memory.observation_host_readback_bytes == 8

    expected_states = (1, 11, 21)
    for ticket, expected_state in zip(tickets, expected_states):
        assert ticket.observations() == {
            "tail": {"state": expected_state, "residual": 4.0}
        }
    memory = graph.execution_stats().memory
    assert memory.observation_materializations == 3
    assert memory.observation_host_readback_bytes == 24


@test_utils.test(arch=ti.cpu)
def test_graph_observation_rejects_unsupported_values_and_runtime_shape():
    scalar = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scalar", ti.i32)
    vector = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "vector", ti.i32, ndim=1)
    value = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "value", ti.i32, ndim=0)

    with pytest.raises(TaichiRuntimeError, match="symbolic ndarray"):
        ti.graph.GraphBuilder().observe(scalar)
    with pytest.raises(TaichiRuntimeError, match="scalar ndarrays"):
        ti.graph.GraphBuilder().observe(vector)
    with pytest.raises(TaichiRuntimeError, match="at least one value"):
        ti.graph.GraphBuilder().observe()
    with pytest.raises(TaichiRuntimeError, match="unique argument names"):
        ti.graph.GraphBuilder().observe(value, value)

    builder = ti.graph.GraphBuilder()
    builder.observe(value, name="tail")
    with pytest.raises(TaichiRuntimeError, match="already defined"):
        builder.observe(value, name="tail")
    graph = builder.compile()
    wrong_shape = ti.ndarray(ti.i32, shape=1)
    with pytest.raises(TaichiRuntimeError, match="scalar ndarray"):
        graph.run({"value": wrong_shape})


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_compiler_metadata_enables_safe_elementwise_graph_candidates(monkeypatch):
    monkeypatch.setenv("TI_GRAPH_TWO_MAP_COMPOSER", "1")

    @ti.kernel
    def first_map(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            temporary[i] = source[i] * 2

    @ti.kernel
    def second_map(
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = temporary[i] + 3

    sym_source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    sym_temporary = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "temporary", ti.i32, ndim=1)
    sym_output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(first_map, sym_source, sym_temporary)
    builder.dispatch(second_map, sym_output, sym_temporary, sym_source)
    graph = builder.compile()

    ir = graph._ir_debug_info
    plan = ir["fusion_plan"]
    assert plan["candidate_groups"] == 1
    assert plan["candidate_dispatches"] == 2
    assert plan["applied_groups"] == 1
    assert plan["lowering_available"]
    assert plan["decision"] == "applied"
    assert plan["eligible_dispatches"] == 2
    assert plan["blocked_dispatches"] == 0
    assert graph._debug_info["nodes"] == [
        {
            "kind": "cgraph",
            "dispatch_count": 2,
            "physical_dispatch_count": 1,
            "composed_two_map_groups": 1,
        }
    ]
    assert graph._compiled_graph._composer_stats["physical_dispatches"] == 2
    dispatches = ir["pre_optimization_root"]["children"][0]["children"]
    assert len(dispatches) == 2
    assert all(not dispatch["opaque"] for dispatch in dispatches)
    assert all(dispatch["elementwise"] for dispatch in dispatches)
    assert (
        dispatches[0]["iteration_domain"]
        == dispatches[1]["iteration_domain"]
        == "external_tensor:source:axis:0"
    )

    n = 257
    source_np = np.arange(n, dtype=np.int32)
    source = ti.ndarray(ti.i32, shape=n)
    temporary = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=n)
    source.from_numpy(source_np)
    graph.run(
        {
            "source": source,
            "temporary": temporary,
            "output": output,
        }
    )
    ti.sync()
    np.testing.assert_array_equal(output.to_numpy(), source_np * 2 + 3)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_two_map_composer_auto_gate_preserves_vulkan_latency(monkeypatch):
    class ProbeBuilder:
        def __init__(self):
            self.enabled = False

        def _enable_two_map_composer(self):
            self.enabled = True

    monkeypatch.delenv("TI_GRAPH_TWO_MAP_COMPOSER", raising=False)
    monkeypatch.setattr(ti_core, "GraphBuilder", ProbeBuilder)
    builder = _new_runtime_graph_builder()
    expected = impl.current_cfg().arch != ti_core.Arch.vulkan
    assert builder.enabled == expected


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_compiler_metadata_fails_closed_for_atomic_and_stencil_access():
    @ti.kernel
    def atomic_reduce(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        total: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        for i in source:
            ti.atomic_add(total[None], source[i])

    @ti.kernel
    def stencil(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = source[(i + 1) % source.shape[0]]

    sym_source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    sym_total = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "total", ti.i32, ndim=0)
    sym_output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(atomic_reduce, sym_source, sym_total)
    builder.dispatch(stencil, sym_source, sym_output)
    graph = builder.compile()

    ir = graph._ir_debug_info
    plan = ir["fusion_plan"]
    assert plan["candidate_groups"] == 0
    assert plan["eligible_dispatches"] == 0
    assert plan["blocked_dispatches"] == 2
    assert plan["blockers"] == {"opaque_dispatch": 2}
    assert plan["applied_groups"] == 0
    assert plan["lowering_available"]
    assert plan["decision"] == "no_safe_candidates"
    assert graph._debug_info["nodes"] == [{"kind": "cgraph", "dispatch_count": 2}]
    dispatches = ir["pre_optimization_root"]["children"][0]["children"]
    assert all(dispatch["opaque"] for dispatch in dispatches)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_mixed_recordable_native_node_lowers_to_one_backend_region():
    @ti.kernel
    def add_one(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    @ti.kernel
    def add_fixed(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        offset: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        for i in values:
            values[i] += offset[None]

    @ti.kernel
    def add_three(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 3

    sym_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    sym_offset = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY,
        "__recordable_test_offset",
        ti.i32,
        ndim=0,
    )
    fixed_offset = ti.ndarray(ti.i32, shape=())
    fixed_offset.fill(2)
    tracker = {"fallback_runs": 0}
    lease = object()
    builder = ti.graph.GraphBuilder()
    builder.dispatch(add_one, sym_values)
    builder.append_native(
        _RecordedDispatchNode(
            add_fixed,
            (sym_values, sym_offset),
            tracker,
            lease,
            fixed_bindings={sym_offset.name: fixed_offset},
        )
    )
    builder.dispatch(add_three, sym_values)
    graph = builder.compile()

    assert graph._instance_debug_info == {"kind": "mixed_backend_region"}
    debug = graph._debug_info
    assert debug["node_count"] == 1
    assert debug["dispatch_count"] == 3
    assert debug["native_count"] == 1
    assert debug["nodes"] == [
        {
            "kind": "mixed_cgraph_native",
            "dispatch_count": 3,
            "lowered_native_count": 1,
        }
    ]
    backend = (
        "cpu"
        if ti.lang.impl.current_cfg().arch == ti.cpu
        else ti_core.arch_name(ti.lang.impl.current_cfg().arch)
    )
    assert debug["optimization"] == {
        "backend": backend,
        "input_segments": 3,
        "output_segments": 1,
        "mixed_backend_regions": 1,
        "lowered_native_nodes": 1,
        "opaque_native_nodes": 0,
    }
    ir = graph._ir_debug_info
    assert not ir["analysis_only"]
    assert ir["optimization"]["mixed_backend_regions"] == 1
    assert len(ir["pre_optimization_root"]["children"]) == 3
    assert len(ir["root"]["children"]) == 1
    assert ir["root"]["children"][0]["name"] == "mixed_backend_region"

    values = ti.ndarray(ti.i32, shape=128)
    values.fill(0)
    graph.execution_stats()
    graph.run({"values": values})
    graph.run({"values": values})
    ti.sync()
    np.testing.assert_array_equal(values.to_numpy(), np.full(128, 12, dtype=np.int32))
    assert tracker["fallback_runs"] == 0
    report = graph.execution_stats()
    assert report.node_count == 1
    assert report.cgraph_segment_count == 1
    assert report.native_node_count == 1
    assert report.dispatch_count == 3
    if ti.lang.impl.current_cfg().arch in (ti.cuda, ti.vulkan):
        assert report.backend_graph_segments == 1
    with pytest.raises(TaichiRuntimeError, match="cannot be serialized"):
        _ = graph._compiled_graph


@test_utils.test(arch=ti.cpu)
def test_recordable_provider_runs_inside_structured_while_and_fails_stale():
    @ti.kernel
    def evaluate_condition(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(state[None] < target)

    @ti.kernel
    def provider_step(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            state[None] += 1
            counter[None] += 1

    arg_state = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.i32, ndim=0)
    arg_predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0)
    arg_counter = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=0)
    arg_target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
    tracker = {"fallback_runs": 0}
    lease = _ValidatingLease()
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(evaluate_condition, arg_state, arg_predicate, arg_target)
    body = builder.create_sequential()
    body.append_native(
        _RecordedDispatchNode(
            provider_step,
            (arg_state, arg_predicate, arg_counter),
            tracker,
            lease,
        )
    )
    builder.while_loop(
        condition,
        body,
        predicate=arg_predicate,
        control_inputs=(arg_state, arg_target),
        carried_state=(arg_state,),
        counter=arg_counter,
        max_iterations=8,
        name="provider_iteration",
    )
    graph = builder.compile()
    state = ti.ndarray(ti.i32, shape=())
    predicate = ti.ndarray(ti.i32, shape=())
    counter = ti.ndarray(ti.i32, shape=())
    state.fill(0)
    predicate.fill(0)
    counter.fill(0)
    args = {
        "state": state,
        "predicate": predicate,
        "counter": counter,
        "target": 4,
    }
    graph.run(args)
    assert state.to_numpy()[()] == 4
    assert counter.to_numpy()[()] == 4
    assert tracker["fallback_runs"] == 0
    assert lease.validations == 1
    assert graph._debug_info["native_count"] == 1

    lease.valid = False
    with pytest.raises(TaichiRuntimeError, match="generation changed"):
        graph.run(args)
    assert state.to_numpy()[()] == 4


@test_utils.test(arch=ti.cpu)
def test_mixed_opaque_native_node_remains_an_explicit_segment():
    @ti.kernel
    def add_one(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    sym_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    tracker = {"runs": 0}
    builder = ti.graph.GraphBuilder()
    builder.dispatch(add_one, sym_values)
    builder.append_native(_OpaqueNode(tracker))
    builder.dispatch(add_one, sym_values)
    graph = builder.compile()

    assert graph._instance_debug_info == {"kind": "dispatch_loop"}
    assert graph._debug_info["optimization"] == {
        "backend": "cpu",
        "input_segments": 3,
        "output_segments": 3,
        "mixed_backend_regions": 0,
        "lowered_native_nodes": 0,
        "opaque_native_nodes": 1,
    }
    values = ti.ndarray(ti.i32, shape=16)
    values.fill(0)
    graph.run({"values": values})
    assert tracker["runs"] == 1
    np.testing.assert_array_equal(values.to_numpy(), np.full(16, 2, dtype=np.int32))


@test_utils.test(arch=ti.cpu)
def test_graph_instance_keeps_single_cgraph_fast_path():
    @ti.kernel
    def fill(value: ti.i32, out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in out:
            out[i] = value

    sym_value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    sym_out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, sym_value, sym_out)
    graph = builder.compile()

    assert graph._instance_debug_info == {"kind": "single_cgraph"}

    out = ti.ndarray(ti.i32, shape=4)
    graph._prewarm()
    graph.run({"value": 42, "out": out})
    assert np.array_equal(out.to_numpy(), np.full(4, 42, dtype=np.int32))


def _build_repeated_inc_graph():
    @ti.kernel
    def inc(arr: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        arr[None] += 1

    sym_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "arr", ti.i32, ndim=0)
    builder = ti.graph.GraphBuilder()
    step = builder.create_sequential()
    step.dispatch(inc, sym_arr)
    for _ in range(4):
        builder.append(step)
    return builder.compile()


def _run_repeated_inc_graph(graph):
    arr = ti.ndarray(ti.i32, shape=())
    arr.fill(0)
    graph.run({"arr": arr})
    assert arr.to_numpy()[()] == 4


@test_utils.test(arch=ti.cpu)
def test_graph_instance_does_not_form_a_self_cycle():
    graph = _build_repeated_inc_graph()
    instance_ref = weakref.ref(graph._instance)

    del graph

    # Runtime registration is weak. The instance and its backend JIT cache
    # must therefore be released immediately, without waiting for cyclic GC
    # that could run only after the Program/Device has been finalized.
    assert instance_ref() is None


@test_utils.test(arch=ti.cpu)
def test_repeated_sequential_keeps_cpu_expanded_runtime_and_aot_graph():
    graph = _build_repeated_inc_graph()
    debug = graph._debug_info
    assert debug["dispatch_count"] == 4
    assert debug["repeat_count"] == 0
    assert debug["nodes"] == [{"kind": "cgraph", "dispatch_count": 4}]
    assert graph._instance_debug_info == {"kind": "single_cgraph"}
    assert graph._compiled_graph is not None
    ir = graph._ir_debug_info
    assert ir["metadata_version"] == 1
    assert ir["analysis_only"]
    assert ir["analysis"]["node_count"] == 6
    assert ir["analysis"]["dispatch_nodes"] == 4
    assert ir["analysis"]["sequential_regions"] == 2
    assert ir["analysis"]["opaque_nodes"] == 4
    assert ir["analysis"]["runtime_bindings"] == 4
    assert ir["root"]["kind"] == "sequential_region"
    assert ir["root"]["children"][0]["kind"] == "sequential_region"
    assert len(ir["root"]["children"][0]["children"]) == 4
    _run_repeated_inc_graph(graph)


@test_utils.test(arch=ti.cuda)
def test_repeated_sequential_keeps_cuda_expanded_runtime_by_default():
    graph = _build_repeated_inc_graph()
    debug = graph._debug_info
    assert debug["dispatch_count"] == 4
    assert debug["repeat_count"] == 0
    assert debug["nodes"] == [{"kind": "cgraph", "dispatch_count": 4}]
    assert graph._instance_debug_info == {"kind": "single_cgraph"}
    assert graph._compiled_graph is not None
    _run_repeated_inc_graph(graph)


def _build_structured_while_graph(
    observe=False,
    *,
    max_iterations,
    chunk_size=4,
    lowering_mode="auto",
    masked_execution=True,
):
    @ti.kernel
    def evaluate_condition(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        enabled: ti.types.ndarray(dtype=ti.i32, ndim=0),
        breakdown: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        predicate[None] = int(
            enabled[None] != 0 and breakdown[None] == 0 and state[None] < target
        )

    @ti.kernel
    def step(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            state[None] += 1
            counter[None] += 1

    arg_state = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.i32, ndim=0)
    arg_predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0)
    arg_counter = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counter", ti.i32, ndim=0)
    arg_enabled = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "enabled", ti.i32, ndim=0)
    arg_breakdown = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "breakdown", ti.i32, ndim=0)
    arg_target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(
        evaluate_condition,
        arg_state,
        arg_predicate,
        arg_enabled,
        arg_breakdown,
        arg_target,
    )
    body = builder.create_sequential()
    body.dispatch(step, arg_state, arg_predicate, arg_counter)
    builder.while_loop(
        condition,
        body,
        predicate=arg_predicate,
        control_inputs=(
            arg_state,
            arg_enabled,
            arg_breakdown,
            arg_target,
        ),
        carried_state=(arg_state,),
        counter=arg_counter,
        max_iterations=max_iterations,
        chunk_size=chunk_size,
        masked_execution=masked_execution,
        lowering_mode=lowering_mode,
        name="adaptive_step",
    )
    if observe:
        builder.observe(
            arg_state,
            arg_predicate,
            arg_counter,
            name="terminal",
        )
    return builder.compile()


def _structured_while_args(*, target, active=True, breakdown=False):
    arrays = {
        name: ti.ndarray(ti.i32, shape=())
        for name in (
            "state",
            "predicate",
            "counter",
            "enabled",
            "breakdown",
        )
    }
    arrays["state"].fill(0)
    arrays["predicate"].fill(0)
    arrays["counter"].fill(0)
    arrays["enabled"].fill(int(active))
    arrays["breakdown"].fill(int(breakdown))
    return {**arrays, "target": target}


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_structured_graph_while_reports_exact_stop_and_backend_overshoot():
    graph = _build_structured_while_graph(max_iterations=20)
    args = _structured_while_args(target=5)
    graph.run(args)

    report = graph.control_flow_stats()[0]
    assert report.logical_iterations == 5
    assert report.final_counter == 5
    assert args["state"].to_numpy()[()] == 5
    assert report.counter_values[-1] == 5
    assert report.observation_batches == len(report.observation_boundaries)
    assert report.observation_scalar_count == 2 * report.observation_batches
    assert report.device_to_host_bytes == 8 * report.observation_batches
    assert report.staging_fallback_batches == 0
    if ti.lang.impl.current_cfg().arch == ti.vulkan:
        assert report.persistent_staging_bytes >= 8
        assert report.staging_allocations == 1
        assert report.staging_reuses == report.observation_batches - 1
        assert report.packed_observation_batches == report.observation_batches
        assert report.direct_observation_batches == 0
        assert report.packed_observation_bytes == report.device_to_host_bytes
    else:
        assert report.persistent_staging_bytes == 0
        assert report.staging_allocations == 0
        assert report.staging_reuses == 0
        assert report.packed_observation_batches == 0
        assert report.direct_observation_batches == report.observation_batches
        assert report.packed_observation_bytes == 0
    memory = graph.execution_stats().memory
    assert memory.persistent_observation_bytes == report.persistent_staging_bytes
    assert memory.persistent_bytes == (
        memory.persistent_argument_bytes + memory.persistent_observation_bytes
    )
    assert memory.transient_temporary_bytes == 0
    assert memory.opaque_driver_bytes is None
    assert args["predicate"].to_numpy()[()] == 0
    assert report.observation_boundaries[0] == 0
    assert report.observation_boundaries[-1] == report.executed_iterations
    assert len(report.predicate_values) == len(report.observation_boundaries)

    if ti.lang.impl.current_cfg().arch == ti.cpu:
        assert report.lowering == "cpu_host_loop"
        assert report.executed_iterations == 5
        assert report.overshoot_iterations == 0
        assert report.chunk_sizes == (1, 1, 1, 1, 1)
    elif report.lowering == "cuda_conditional_graph":
        assert ti.lang.impl.current_cfg().arch == ti.cuda
        assert report.executed_iterations == 5
        assert report.overshoot_iterations == 0
        assert report.chunk_sizes == (5,)
        assert report.observation_boundaries == (0, 5)
        assert report.native_upgrade_reason == "selected"
    else:
        assert report.lowering == "portable_chunk_replay"
        assert report.executed_iterations == 8
        assert report.overshoot_iterations == 3
        assert report.chunk_sizes == (4, 4)
        assert report.observation_boundaries == (0, 4, 8)

    ir = graph._ir_debug_info
    assert ir["analysis"]["while_regions"] == 1
    assert ir["root"]["children"][0]["lowering_mode"] == "auto"
    with pytest.raises(
        TaichiRuntimeError, match="supports structured control only"
    ):
        graph.submit(args)


def _build_structured_status_graph(*, max_iterations=8):
    @ti.kernel
    def evaluate_condition(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        user_stop: ti.types.ndarray(dtype=ti.i32, ndim=0),
        breakdown: ti.types.ndarray(dtype=ti.i32, ndim=0),
        target: ti.i32,
    ):
        if status[None] == 0:
            if breakdown[None] != 0:
                status[None] = 2
            elif user_stop[None] != 0:
                status[None] = 3
            elif state[None] >= target:
                status[None] = 1
        predicate[None] = int(status[None] == 0)

    @ti.kernel
    def step(
        state: ti.types.ndarray(dtype=ti.i32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if predicate[None] != 0:
            state[None] += 1
            counter[None] += 1

    scalar = lambda name: ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=0
    )
    arg_state = scalar("state")
    arg_predicate = scalar("predicate")
    arg_counter = scalar("counter")
    arg_status = scalar("status")
    arg_user_stop = scalar("user_stop")
    arg_breakdown = scalar("breakdown")
    arg_target = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "target", ti.i32)
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(
        evaluate_condition,
        arg_state,
        arg_predicate,
        arg_status,
        arg_user_stop,
        arg_breakdown,
        arg_target,
    )
    body = builder.create_sequential()
    body.dispatch(step, arg_state, arg_predicate, arg_counter)
    builder.while_loop(
        condition,
        body,
        predicate=arg_predicate,
        status=arg_status,
        control_inputs=(arg_state, arg_user_stop, arg_breakdown, arg_target),
        carried_state=(arg_state,),
        counter=arg_counter,
        max_iterations=max_iterations,
        chunk_size=4,
        masked_execution=True,
        name="status_iteration",
    )
    return builder.compile()


def _structured_status_args(*, user_stop=False, breakdown=False):
    args = {
        name: ti.ndarray(ti.i32, shape=())
        for name in (
            "state",
            "predicate",
            "counter",
            "status",
            "user_stop",
            "breakdown",
        )
    }
    for value in args.values():
        value.fill(0)
    args["user_stop"].fill(int(user_stop))
    args["breakdown"].fill(int(breakdown))
    return args


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_structured_graph_while_reports_user_defined_terminal_status():
    graph = _build_structured_status_graph()

    converged = _structured_status_args()
    graph.run({**converged, "target": 3})
    report = graph.control_flow_stats()[0]
    assert report.status_resource == "status"
    assert report.initial_status == 0
    assert report.final_status == 1
    assert report.status_values[0] == 0
    assert report.status_values[-1] == 1
    assert report.logical_iterations == 3
    assert report.observation_scalar_count == 3 * report.observation_batches
    assert report.device_to_host_bytes == 12 * report.observation_batches

    stopped = _structured_status_args(user_stop=True)
    graph.run({**stopped, "target": 3})
    report = graph.control_flow_stats()[0]
    assert report.logical_iterations == 0
    assert report.initial_status == 3
    assert report.final_status == 3

    failed = _structured_status_args(breakdown=True)
    graph.run({**failed, "target": 3})
    report = graph.control_flow_stats()[0]
    assert report.logical_iterations == 0
    assert report.initial_status == 2
    assert report.final_status == 2


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_structured_graph_while_transfer_planner_has_strict_fallback(monkeypatch):
    monkeypatch.setenv("TI_GRAPH_OBSERVATION_TRANSFER_PLANNER", "0")
    graph = _build_structured_while_graph(max_iterations=20)
    args = _structured_while_args(target=5)
    graph.run(args)
    report = graph.control_flow_stats()[0]
    assert report.logical_iterations == 5
    assert report.packed_observation_batches == 0
    assert report.direct_observation_batches == report.observation_batches
    assert report.staging_allocations == 0
    assert report.staging_reuses == 0
    assert report.staging_fallback_batches == 0
    assert args["state"].to_numpy()[()] == 5


@test_utils.test(arch=ti.vulkan)
def test_structured_graph_while_reuses_or_disables_persistent_observation_staging(
    monkeypatch,
):
    graph = _build_structured_while_graph(max_iterations=20)
    first = _structured_while_args(target=5)
    graph.run(first)
    first_report = graph.control_flow_stats()[0]
    assert first_report.staging_allocations == 1
    assert first_report.packed_observation_batches == first_report.observation_batches

    second = _structured_while_args(target=5)
    graph.run(second)
    second_report = graph.control_flow_stats()[0]
    assert second_report.staging_allocations == 0
    assert second_report.staging_reuses == second_report.observation_batches
    assert second_report.packed_observation_batches == second_report.observation_batches
    assert (
        second_report.persistent_staging_bytes == first_report.persistent_staging_bytes
    )

    monkeypatch.setenv("TI_GRAPH_PERSISTENT_OBSERVATION_STAGING", "0")
    disabled = _structured_while_args(target=5)
    graph.run(disabled)
    disabled_report = graph.control_flow_stats()[0]
    assert disabled_report.packed_observation_batches == 0
    assert disabled_report.staging_reuses == 0
    assert (
        disabled_report.direct_observation_batches
        == disabled_report.observation_batches
    )
    assert disabled_report.logical_iterations == 5
    assert disabled["state"].to_numpy()[()] == 5


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_structured_graph_while_honors_initial_stop_and_iteration_cap():
    graph = _build_structured_while_graph(max_iterations=6)

    inactive = _structured_while_args(target=100, active=False)
    graph.run(inactive)
    stopped = graph.control_flow_stats()[0]
    assert stopped.logical_iterations == 0
    assert stopped.executed_iterations == 0
    assert stopped.observation_boundaries == (0,)
    assert stopped.counter_values == (0,)
    assert stopped.observation_batches == 1
    assert stopped.observation_scalar_count == 2
    assert stopped.device_to_host_bytes == 8

    capped = _structured_while_args(target=100)
    graph.run(capped)
    report = graph.control_flow_stats()[0]
    assert report.logical_iterations == 6
    assert report.executed_iterations == 6
    assert report.overshoot_iterations == 0
    if ti.lang.impl.current_cfg().arch == ti.cpu:
        assert report.chunk_sizes == (1, 1, 1, 1, 1, 1)
    elif report.lowering == "cuda_conditional_graph":
        assert report.chunk_sizes == (6,)
        assert report.observation_boundaries == (0, 6)
    else:
        assert report.chunk_sizes == (4, 2)
    assert capped["state"].to_numpy()[()] == 6
    assert capped["predicate"].to_numpy()[()] == 1


@test_utils.test(arch=ti.cuda)
def test_structured_graph_while_forced_portable_keeps_chunk_trace():
    graph = _build_structured_while_graph(max_iterations=20, lowering_mode="portable")
    args = _structured_while_args(target=5)
    graph.run(args)

    report = graph.control_flow_stats()[0]
    assert report.lowering == "portable_chunk_replay"
    assert report.logical_iterations == 5
    assert report.executed_iterations == 8
    assert report.overshoot_iterations == 3
    assert report.observation_boundaries == (0, 4, 8)
    assert report.native_upgrade_reason == "forced_portable"


@test_utils.test(arch=ti.cuda)
def test_structured_graph_while_native_rebind_and_replay_diagnostics():
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    if not (
        capabilities["driver_version_eligible"]
        and capabilities["conditional_graph_symbols_loaded"]
        and capabilities["general_device_setter_lowering_compiled"]
    ):
        pytest.skip("general CUDA conditional Graph is unavailable")

    graph = _build_structured_while_graph(
        max_iterations=20, lowering_mode="native_required"
    )
    graph._graph_stats
    first = _structured_while_args(target=5)
    second = _structured_while_args(target=7)

    graph.run(first)
    first_report = graph.control_flow_stats()[0]
    assert first_report.lowering == "cuda_conditional_graph"
    assert first_report.logical_iterations == 5
    assert first_report.executed_iterations == 5
    assert first_report.observation_boundaries == (0, 5)

    graph.run(second)
    second_report = graph.control_flow_stats()[0]
    assert second_report.logical_iterations == 7
    assert second_report.executed_iterations == 7
    second["state"].fill(0)
    second["predicate"].fill(1)
    second["counter"].fill(0)
    graph.run(second)

    native_stats = graph._graph_stats[0]
    assert native_stats["captures"] == 1
    assert native_stats["patched_replays"] >= 1
    assert native_stats["exact_replays"] >= 1
    assert native_stats["last_path"] == "cuda_exact_replay"


@test_utils.test(arch=ti.cuda)
def test_structured_graph_native_while_submit_defers_terminal_observation():
    capabilities = dict(ti_core.cuda_conditional_graph_capabilities())
    if not (
        capabilities["driver_version_eligible"]
        and capabilities["conditional_graph_symbols_loaded"]
        and capabilities["general_device_setter_lowering_compiled"]
    ):
        pytest.skip("general CUDA conditional Graph is unavailable")

    graph = _build_structured_while_graph(
        observe=True,
        max_iterations=20,
        lowering_mode="native_required",
    )
    inactive = _structured_while_args(target=7, active=False)
    inactive_ticket = graph.submit(inactive)
    assert inactive_ticket.observations() == {
        "terminal": {"state": 0, "predicate": 0, "counter": 0}
    }

    args = _structured_while_args(target=7)
    ticket = graph.submit(args)

    with pytest.raises(TaichiRuntimeError, match="unavailable after asynchronous"):
        graph.control_flow_stats()
    assert ticket.observations() == {
        "terminal": {"state": 7, "predicate": 0, "counter": 7}
    }
    assert args["state"].to_numpy()[()] == 7

    args["state"].fill(0)
    args["predicate"].fill(0)
    args["counter"].fill(0)
    args["target"] = 3
    graph.run(args)
    report = graph.control_flow_stats()[0]
    assert report.lowering == "cuda_conditional_graph"
    assert report.logical_iterations == 3

    unmasked = _build_structured_while_graph(
        observe=True,
        max_iterations=20,
        lowering_mode="native_required",
        masked_execution=False,
    )
    unmasked_inactive = _structured_while_args(target=3, active=False)
    assert unmasked.submit(unmasked_inactive).observations() == {
        "terminal": {"state": 0, "predicate": 0, "counter": 0}
    }
    unmasked_active = _structured_while_args(target=3)
    assert unmasked.submit(unmasked_active).observations() == {
        "terminal": {"state": 3, "predicate": 0, "counter": 3}
    }


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_structured_if_uses_multiple_control_inputs():
    @ti.kernel
    def evaluate_condition(
        left: ti.i32,
        right: ti.i32,
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        predicate[None] = int(left < right)

    @ti.kernel
    def write_then(out: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        out[None] = 7

    @ti.kernel
    def write_else(out: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        out[None] = -3

    arg_left = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "left", ti.i32)
    arg_right = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "right", ti.i32)
    arg_predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0)
    arg_out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=0)
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(evaluate_condition, arg_left, arg_right, arg_predicate)
    then_region = builder.create_sequential()
    then_region.dispatch(write_then, arg_out)
    else_region = builder.create_sequential()
    else_region.dispatch(write_else, arg_out)
    builder.if_then_else(
        condition,
        then_region,
        predicate=arg_predicate,
        control_inputs=(arg_left, arg_right),
        else_region=else_region,
        name="ordered_branch",
    )
    graph = builder.compile()

    predicate = ti.ndarray(ti.i32, shape=())
    out = ti.ndarray(ti.i32, shape=())
    graph.run({"left": 2, "right": 5, "predicate": predicate, "out": out})
    assert out.to_numpy()[()] == 7
    report = graph.control_flow_stats()[0]
    assert report.kind == "if"
    assert report.selected_branch == "then"
    assert report.control_inputs == ("left", "right")

    graph.run({"left": 8, "right": 5, "predicate": predicate, "out": out})
    assert out.to_numpy()[()] == -3
    assert graph.control_flow_stats()[0].selected_branch == "else"
    ir = graph._ir_debug_info
    assert ir["analysis"]["if_regions"] == 1
    assert ir["root"]["children"][0]["has_else"]


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_structured_switch_selects_case_and_default():
    @ti.kernel
    def evaluate_selector(
        choice: ti.i32,
        selector: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        selector[None] = choice

    @ti.kernel
    def write_case_zero(out: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        out[None] = 10

    @ti.kernel
    def write_case_one(out: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        out[None] = 20

    @ti.kernel
    def write_default(out: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        out[None] = 99

    arg_choice = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "choice", ti.i32)
    arg_selector = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "selector", ti.i32, ndim=0)
    arg_out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=0)
    builder = ti.graph.GraphBuilder()
    condition = builder.create_sequential()
    condition.dispatch(evaluate_selector, arg_choice, arg_selector)
    case_zero = builder.create_sequential()
    case_zero.dispatch(write_case_zero, arg_out)
    case_one = builder.create_sequential()
    case_one.dispatch(write_case_one, arg_out)
    default = builder.create_sequential()
    default.dispatch(write_default, arg_out)
    builder.switch(
        condition,
        (case_zero, case_one),
        selector=arg_selector,
        control_inputs=(arg_choice,),
        default_region=default,
        name="choice_switch",
    )
    graph = builder.compile()

    selector = ti.ndarray(ti.i32, shape=())
    out = ti.ndarray(ti.i32, shape=())
    for choice, expected, selected in (
        (0, 10, "case_0"),
        (1, 20, "case_1"),
        (5, 99, "default"),
    ):
        graph.run({"choice": choice, "selector": selector, "out": out})
        assert out.to_numpy()[()] == expected
        report = graph.control_flow_stats()[0]
        assert report.kind == "switch"
        assert report.selected_branch == selected
    ir = graph._ir_debug_info
    assert ir["analysis"]["switch_regions"] == 1
    assert ir["root"]["children"][0]["branch_count"] == 2


@test_utils.test(arch=ti.cuda)
def test_cuda_cgraph_cache_survives_reset_then_delete():
    graph = _build_repeated_inc_graph()
    arr = ti.ndarray(ti.i32, shape=())
    arr.fill(0)
    graph.run({"arr": arr})
    assert arr.to_numpy()[()] == 4

    ti.reset()
    assert graph._spec is None
    report = graph.execution_stats()
    assert report.lifecycle_state == "runtime_invalid"
    assert report.execution_path == "runtime_invalid"
    assert report.fallback_reason == "runtime_invalid"
    del graph
    del arr
    gc.collect()


@test_utils.test(arch=ti.cuda)
def test_cuda_cgraph_recaptures_for_distinct_ndarray_arguments():
    graph = _build_repeated_inc_graph()
    first = ti.ndarray(ti.i32, shape=())
    second = ti.ndarray(ti.i32, shape=())
    first.fill(0)
    second.fill(10)

    # Different ndarray identities force a CUDA graph signature change. The
    # executable must retire capture-owned packets on its own capture stream,
    # then replay with the new argument state without touching another run.
    graph.run({"arr": first})
    graph.run({"arr": second})
    graph.run({"arr": first})

    assert first.to_numpy()[()] == 8
    assert second.to_numpy()[()] == 14


@test_utils.test(arch=ti.cuda)
def test_cuda_cgraph_internal_stats_report_capture_and_replay():
    graph = _build_repeated_inc_graph()
    arr = ti.ndarray(ti.i32, shape=())
    arr.fill(0)

    # Detailed CUDA counters are opt-in so the default replay hot path does not
    # update diagnostic state. Failure recovery owns a separate backoff state.
    initial = graph._graph_stats
    assert len(initial) == 1
    assert initial[0]["backend"] == "none"
    assert initial[0]["attempts"] == 0

    graph.run({"arr": arr})
    first = graph._graph_stats
    assert len(first) == 1
    assert first[0]["backend"] == "cuda"
    assert first[0]["attempts"] == 1
    assert first[0]["capture_attempts"] == 1
    assert first[0]["captures"] == 1
    assert first[0]["last_path"] == "cuda_capture"
    assert first[0]["known_persistent_argument_bytes"] > 0

    graph.run({"arr": arr})
    second = graph._graph_stats[0]
    assert second["attempts"] == 2
    assert second["exact_replays"] == 1
    assert second["ordinary_fallbacks"] == 0
    assert second["last_path"] == "cuda_exact_replay"
    assert second["last_fallback_reason"] == "none"
    assert second["consecutive_transient_failures"] == 0
    assert arr.to_numpy()[()] == 8


@test_utils.test(arch=ti.cuda)
def test_cuda_cgraph_patches_scalar_matrix_and_ndarray_arguments():
    n = 64

    @ti.kernel
    def transform(
        scale: ti.i32,
        offset: ti.types.vector(2, ti.i32),
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in output:
            output[i] = source[i] * scale + offset[0] + offset[1]

    sym_scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.i32)
    sym_offset = ti.graph.Arg(
        ti.graph.ArgKind.MATRIX, "offset", ti.types.vector(2, ti.i32)
    )
    sym_source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    sym_output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(transform, sym_scale, sym_offset, sym_source, sym_output)
    graph = builder.compile()

    source_a_np = np.arange(n, dtype=np.int32)
    source_b_np = np.arange(n, dtype=np.int32)[::-1].copy()
    source_a = ti.ndarray(ti.i32, shape=n)
    source_b = ti.ndarray(ti.i32, shape=n)
    output_a = ti.ndarray(ti.i32, shape=n)
    output_b = ti.ndarray(ti.i32, shape=n)
    source_a.from_numpy(source_a_np)
    source_b.from_numpy(source_b_np)

    graph.run(
        {
            "scale": 2,
            "offset": ti.Vector([1, 3], dt=ti.i32),
            "source": source_a,
            "output": output_a,
        }
    )
    graph.run(
        {
            "scale": -3,
            "offset": ti.Vector([7, -2], dt=ti.i32),
            "source": source_b,
            "output": output_b,
        }
    )
    graph.run(
        {
            "scale": 4,
            "offset": ti.Vector([-5, 2], dt=ti.i32),
            "source": source_a,
            "output": output_a,
        }
    )

    np.testing.assert_array_equal(output_a.to_numpy(), source_a_np * 4 - 3)
    np.testing.assert_array_equal(output_b.to_numpy(), source_b_np * -3 + 5)


@test_utils.test(arch=ti.cuda)
def test_cuda_cgraph_recaptures_when_ndarray_structure_changes():
    @ti.kernel
    def fill(
        value: ti.i32,
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in output:
            output[i] = value + i

    sym_value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    sym_output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, sym_value, sym_output)
    graph = builder.compile()
    small = ti.ndarray(ti.i32, shape=17)
    large = ti.ndarray(ti.i32, shape=37)

    graph.run({"value": 3, "output": small})
    graph.run({"value": 5, "output": large})
    graph.run({"value": 7, "output": small})

    np.testing.assert_array_equal(small.to_numpy(), np.arange(17, dtype=np.int32) + 7)
    np.testing.assert_array_equal(large.to_numpy(), np.arange(37, dtype=np.int32) + 5)


@test_utils.test(arch=ti.cuda)
def test_cuda_cgraph_runtime_state_clear_is_idempotent_and_reusable():
    graph = _build_repeated_inc_graph()
    arr = ti.ndarray(ti.i32, shape=())
    arr.fill(0)
    graph.run({"arr": arr})
    assert arr.to_numpy()[()] == 4

    cache = graph._instance._backend_executable._jit_cache
    cache.clear_runtime_state()
    cache.clear_runtime_state()

    # A cleared cache remains usable and performs one fresh capture rather
    # than retaining any packet, stream, executable, or allocation lease from
    # the retired state.
    graph.run({"arr": arr})
    assert arr.to_numpy()[()] == 8


@test_utils.test(arch=ti.cuda)
def test_cuda_cgraph_signature_tracks_allocation_generation_reuse():
    graph = _build_repeated_inc_graph()
    generations_by_slot = {}
    reused_slot = False

    for iteration in range(32):
        arr = ti.ndarray(ti.i32, shape=())
        arr.fill(iteration)
        allocation_id = arr.arr.device_allocation().alloc_id
        slot = allocation_id & 0xFFFFFFFF
        previous_id = generations_by_slot.get(slot)
        if previous_id is not None:
            assert previous_id != allocation_id
            reused_slot = True
        generations_by_slot[slot] = allocation_id

        graph.run({"arr": arr})
        assert arr.to_numpy()[()] == iteration + 4
        del arr
        gc.collect()

    # The graph lease pins its current allocation, while recapture releases
    # the previous one. Registry slots should therefore be reused with a new
    # generation rather than growing one slot per invocation.
    assert reused_slot


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_cgraph_run_after_reset_is_rejected():
    graph = _build_repeated_inc_graph()
    arr = ti.ndarray(ti.i32, shape=())
    arr.fill(0)
    graph.run({"arr": arr})
    assert arr.to_numpy()[()] == 4

    arch = ti.lang.impl.current_cfg().arch
    ti.reset()
    assert graph._spec is None
    ti.init(arch=arch, enable_fallback=False)
    assert graph._instance is None
    assert graph._instances == {}
    arr_after_reset = ti.ndarray(ti.i32, shape=())
    arr_after_reset.fill(0)
    with pytest.raises(TaichiRuntimeError, match="compiled before ti.reset"):
        graph.run({"arr": arr_after_reset})


@test_utils.test(arch=ti.vulkan)
def test_vulkan_cgraph_replay_identity_survives_cache_churn():
    graph_count = 64

    @ti.kernel
    def add_bias(values: ti.types.ndarray(dtype=ti.i32, ndim=1), bias: ti.i32):
        for i in values:
            values[i] += bias

    sym_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    sym_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.i32)
    values = ti.ndarray(ti.i32, shape=256)
    values.fill(0)

    for iteration in range(graph_count):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(add_bias, sym_values, sym_bias)
        builder.dispatch(add_bias, sym_values, sym_bias)
        graph = builder.compile()
        runtime_args = {"values": values, "bias": iteration + 1}
        # The first eight launches populate the bounded slot ring; launch nine
        # replays slot zero.
        for _ in range(9):
            graph.run(runtime_args)
        del graph
        del builder
        gc.collect()

    ti.sync()
    expected = 18 * sum(range(1, graph_count + 1))
    np.testing.assert_array_equal(
        values.to_numpy(), np.full(256, expected, dtype=np.int32)
    )


@test_utils.test(arch=ti.vulkan)
def test_vulkan_cgraph_clear_retires_in_flight_slots_and_reregisters():
    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    sym_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment, sym_values)
    builder.dispatch(increment, sym_values)
    graph = builder.compile()
    values = ti.ndarray(ti.i32, shape=1 << 14)
    values.fill(0)

    for _ in range(9):
        graph.run({"values": values})
    cache = graph._instance._backend_executable._jit_cache
    cache.clear_runtime_state()
    cache.clear_runtime_state()

    # Reuse the same cache identity while its previous slot generation may
    # still be in flight. The new registration must create independent active
    # state while the runtime-owned retirement queue pins the old resources.
    for _ in range(9):
        graph.run({"values": values})

    ti.sync()
    np.testing.assert_array_equal(
        values.to_numpy(), np.full(1 << 14, 36, dtype=np.int32)
    )


@test_utils.test(arch=ti.vulkan)
def test_vulkan_cgraph_replay_slot_saturation_telemetry_is_monotonic():
    @ti.kernel
    def increment(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    sym_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment, sym_values)
    builder.dispatch(increment, sym_values)
    graph = builder.compile()
    values = ti.ndarray(ti.i32, shape=1 << 14)
    values.fill(0)

    fallback_before = ti_core.query_int64(
        "vulkan_graph_replay_slot_saturation_fallbacks"
    )
    # Detailed per-Graph counters are opt-in; global saturation telemetry
    # remains always-on because it is a bounded safety signal.
    assert graph.execution_stats().execution_path == "not_run"
    launch_count = 12
    for _ in range(launch_count):
        graph.run({"values": values})

    stats = graph._graph_stats
    assert len(stats) == 1
    assert stats[0]["backend"] == "vulkan"
    assert stats[0]["attempts"] == launch_count
    assert (
        stats[0]["records"] + stats[0]["replays"] + stats[0]["ordinary_fallbacks"]
        == launch_count
    )
    assert stats[0]["records"] > 0
    assert stats[0]["known_persistent_argument_bytes"] > 0

    fallback_after = ti_core.query_int64(
        "vulkan_graph_replay_slot_saturation_fallbacks"
    )
    assert fallback_after >= fallback_before

    ti.sync()
    np.testing.assert_array_equal(
        values.to_numpy(), np.full(1 << 14, 2 * launch_count, dtype=np.int32)
    )


@test_utils.test(arch=ti.vulkan)
def test_vulkan_cgraph_patches_same_structure_ndarray_bindings():
    @ti.kernel
    def add_bias(values: ti.types.ndarray(dtype=ti.i32, ndim=1), bias: ti.i32):
        for i in values:
            values[i] += bias

    sym_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    sym_bias = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bias", ti.i32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(add_bias, sym_values, sym_bias)
    builder.dispatch(add_bias, sym_values, sym_bias)
    graph = builder.compile()
    first = ti.ndarray(ti.i32, shape=256)
    second = ti.ndarray(ti.i32, shape=256)
    first.fill(1)
    second.fill(10)

    graph.execution_stats()
    graph.run({"values": first, "bias": 2})
    ti.sync()
    first_stats = graph._graph_stats[0]
    assert first_stats["records"] == 1
    assert first_stats["last_path"] == "vulkan_record"
    first_persistent_bytes = first_stats["known_persistent_argument_bytes"]
    assert first_persistent_bytes > 0

    graph.run({"values": second, "bias": 3})
    ti.sync()
    patched_stats = graph._graph_stats[0]
    patch_supported = patched_stats["last_path"] == "vulkan_patched_replay"
    assert patched_stats["known_persistent_argument_bytes"] == first_persistent_bytes
    if patch_supported:
        assert patched_stats["records"] == 1
        assert patched_stats["patched_replays"] == 1
        assert patched_stats["replays"] == 1
    else:
        assert patched_stats["records"] == 2
        assert patched_stats["patched_replays"] == 0
        assert patched_stats["replays"] == 0
        assert patched_stats["last_path"] == "vulkan_record"

    graph.run({"values": second, "bias": 4})
    ti.sync()
    replay_stats = graph._graph_stats[0]
    assert replay_stats["records"] == (1 if patch_supported else 2)
    assert replay_stats["patched_replays"] == (1 if patch_supported else 0)
    assert replay_stats["replays"] == (2 if patch_supported else 1)
    assert replay_stats["last_path"] == "vulkan_replay"
    np.testing.assert_array_equal(first.to_numpy(), np.full(256, 5, dtype=np.int32))
    np.testing.assert_array_equal(second.to_numpy(), np.full(256, 24, dtype=np.int32))


@test_utils.test(arch=ti.vulkan)
def test_vulkan_cgraph_structural_shape_change_records_again():
    @ti.kernel
    def add_one(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    sym_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(add_one, sym_values)
    builder.dispatch(add_one, sym_values)
    graph = builder.compile()
    small = ti.ndarray(ti.i32, shape=64)
    large = ti.ndarray(ti.i32, shape=128)
    replacement = ti.ndarray(ti.i32, shape=64)
    small.fill(0)
    large.fill(10)
    replacement.fill(20)

    graph.execution_stats()
    graph.run({"values": small})
    ti.sync()
    assert graph._graph_stats[0]["records"] == 1

    graph.run({"values": large})
    ti.sync()
    recapture_stats = graph._graph_stats[0]
    assert recapture_stats["records"] == 2
    assert recapture_stats["last_path"] == "vulkan_record"
    persistent_bytes = recapture_stats["known_persistent_argument_bytes"]

    graph.run({"values": replacement})
    ti.sync()
    replacement_stats = graph._graph_stats[0]
    patch_supported = replacement_stats["last_path"] == "vulkan_patched_replay"
    assert replacement_stats["records"] == (2 if patch_supported else 3)
    assert replacement_stats["known_persistent_argument_bytes"] == persistent_bytes
    np.testing.assert_array_equal(small.to_numpy(), np.full(64, 2, dtype=np.int32))
    np.testing.assert_array_equal(large.to_numpy(), np.full(128, 12, dtype=np.int32))
    np.testing.assert_array_equal(
        replacement.to_numpy(), np.full(64, 22, dtype=np.int32)
    )


@test_utils.test(arch=ti.vulkan)
def test_vulkan_cgraph_alias_topology_change_records_again(monkeypatch):
    # This test owns Vulkan multi-dispatch recording, not kernel composition.
    monkeypatch.setenv("TI_GRAPH_TWO_MAP_COMPOSER", "0")

    @ti.kernel
    def copy_increment(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        destination: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            destination[i] = source[i] + 1

    sym_source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    sym_destination = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "destination", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(copy_increment, sym_source, sym_destination)
    builder.dispatch(copy_increment, sym_source, sym_destination)
    graph = builder.compile()
    aliased = ti.ndarray(ti.i32, shape=64)
    source = ti.ndarray(ti.i32, shape=64)
    destination = ti.ndarray(ti.i32, shape=64)
    replacement_source = ti.ndarray(ti.i32, shape=64)
    replacement_destination = ti.ndarray(ti.i32, shape=64)
    aliased.fill(3)
    source.fill(7)
    destination.fill(0)
    replacement_source.fill(11)
    replacement_destination.fill(0)

    graph.execution_stats()
    graph.run({"source": aliased, "destination": aliased})
    ti.sync()
    assert graph._graph_stats[0]["records"] == 1

    graph.run({"source": source, "destination": destination})
    ti.sync()
    distinct_stats = graph._graph_stats[0]
    assert distinct_stats["records"] == 2
    assert distinct_stats["last_path"] == "vulkan_record"
    persistent_bytes = distinct_stats["known_persistent_argument_bytes"]

    graph.run(
        {
            "source": replacement_source,
            "destination": replacement_destination,
        }
    )
    ti.sync()
    replacement_stats = graph._graph_stats[0]
    patch_supported = replacement_stats["last_path"] == "vulkan_patched_replay"
    assert replacement_stats["records"] == (2 if patch_supported else 3)
    assert replacement_stats["known_persistent_argument_bytes"] == persistent_bytes
    np.testing.assert_array_equal(aliased.to_numpy(), np.full(64, 5, dtype=np.int32))
    np.testing.assert_array_equal(
        destination.to_numpy(), np.full(64, 8, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        replacement_destination.to_numpy(), np.full(64, 12, dtype=np.int32)
    )


@test_utils.test(arch=ti.vulkan)
def test_vulkan_cgraph_structural_patch_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TI_VULKAN_GRAPH_STRUCTURAL_PATCH", "0")

    @ti.kernel
    def add_one(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    sym_values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(add_one, sym_values)
    builder.dispatch(add_one, sym_values)
    graph = builder.compile()
    first = ti.ndarray(ti.i32, shape=64)
    second = ti.ndarray(ti.i32, shape=64)
    first.fill(0)
    second.fill(10)

    graph.execution_stats()
    graph.run({"values": first})
    ti.sync()
    graph.run({"values": second})
    ti.sync()
    stats = graph._graph_stats[0]
    assert stats["records"] == 2
    assert stats["patched_replays"] == 0
    assert stats["replays"] == 0
    assert stats["last_path"] == "vulkan_record"
    np.testing.assert_array_equal(first.to_numpy(), np.full(64, 2, dtype=np.int32))
    np.testing.assert_array_equal(second.to_numpy(), np.full(64, 12, dtype=np.int32))


@pytest.mark.parametrize("hazard_planner", [False, True])
@test_utils.test(arch=ti.vulkan)
def test_vulkan_cgraph_hazard_planner_preserves_dependency_chains(
    monkeypatch, hazard_planner
):
    monkeypatch.setenv(
        "TI_VULKAN_GRAPH_HAZARD_PLANNER",
        "1" if hazard_planner else "0",
    )
    monkeypatch.setenv("TI_GRAPH_TWO_MAP_COMPOSER", "0")

    @ti.kernel
    def read_a_to_b(
        a: ti.types.ndarray(dtype=ti.i32, ndim=1),
        b: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in a:
            b[i] = a[i] + 1

    @ti.kernel
    def read_a_to_c(
        a: ti.types.ndarray(dtype=ti.i32, ndim=1),
        c: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in a:
            c[i] = a[i] * 2

    @ti.kernel
    def fill(values: ti.types.ndarray(dtype=ti.i32, ndim=1), value: ti.i32):
        for i in values:
            values[i] = value

    @ti.kernel
    def copy_twice(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        destination: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            destination[i] = source[i] * 2

    @ti.kernel
    def combine(
        a: ti.types.ndarray(dtype=ti.i32, ndim=1),
        c: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in a:
            output[i] = a[i] + c[i]

    arg_a = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "a", ti.i32, ndim=1)
    arg_b = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "b", ti.i32, ndim=1)
    arg_c = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "c", ti.i32, ndim=1)
    arg_output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(read_a_to_b, arg_a, arg_b)
    builder.dispatch(read_a_to_c, arg_a, arg_c)
    builder.dispatch(fill, arg_a, value)
    builder.dispatch(fill, arg_b, value)
    builder.dispatch(copy_twice, arg_b, arg_c)
    builder.dispatch(combine, arg_a, arg_c, arg_output)
    graph = builder.compile()

    arrays = {name: ti.ndarray(ti.i32, shape=256) for name in ("a", "b", "c", "output")}
    arrays["a"].fill(3)
    graph.execution_stats()
    graph.run({**arrays, "value": 7})

    np.testing.assert_array_equal(
        arrays["output"].to_numpy(), np.full(256, 21, dtype=np.int32)
    )
    stats = graph._graph_stats[0]
    assert stats["effect_reads"] > 0
    assert stats["effect_writes"] > 0
    assert stats["dependency_barriers"] > 0
    assert stats["exit_barriers"] > 0
    if hazard_planner:
        assert stats["barrier_deferrals"] > 0
        assert stats["rar_elisions"] > 0
    else:
        assert stats["barrier_deferrals"] == 0
        assert stats["rar_elisions"] == 0


@pytest.mark.parametrize("dt", [ti.i32, ti.i64, ti.u32, ti.u64])
@test_utils.test(arch=supported_archs_cgraph, exclude=[(ti.vulkan, "Darwin")])
def test_arg_int(dt):
    @ti.kernel
    def foo(a: dt, b: ti.types.ndarray(dtype=dt, ndim=1)):
        b[0] = a

    k = ti.ndarray(dt, shape=(1,))

    sym_A = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "mat", dt)
    sym_B = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "b", dt, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(foo, sym_A, sym_B)
    graph = builder.compile()
    graph.run({"mat": 1234, "b": k})
    assert k.to_numpy()[0] == 1234


@pytest.mark.parametrize("dt", [ti.i16, ti.u16, ti.u8, ti.i8])
@test_utils.test(arch=ti.vulkan)
def test_arg_short(dt):
    @ti.kernel
    def foo(a: dt, b: ti.types.ndarray(dtype=dt, ndim=1)):
        b[0] = a

    k = ti.ndarray(dt, shape=(1,))

    sym_A = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "mat", dt)
    sym_B = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "b", dt, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(foo, sym_A, sym_B)
    graph = builder.compile()
    graph.run({"mat": 123, "b": k})
    assert k.to_numpy()[0] == 123


@test_utils.test(arch=ti.vulkan)
def test_texture():
    res = (256, 256)

    @ti.kernel
    def make_texture(
        tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
    ):
        for i, j in ti.ndrange(128, 128):
            tex.store(ti.Vector([i, j]), ti.Vector([0.1, 0.0, 0.0, 0.0]))

    @ti.kernel
    def paint(
        t: ti.f32,
        pixels: ti.types.ndarray(ndim=2),
        tex: ti.types.texture(num_dimensions=2),
    ):
        for i, j in pixels:
            uv = ti.Vector([i / res[0], j / res[1]])
            warp_uv = (
                uv + ti.Vector([ti.cos(t + uv.x * 5.0), ti.sin(t + uv.y * 5.0)]) * 0.1
            )
            c = ti.math.vec4(0.0)
            if uv.x > 0.5:
                c = tex.sample_lod(warp_uv, 0.0)
            else:
                c = tex.fetch(ti.cast(warp_uv * 128, ti.i32), 0)
            pixels[i, j] = [c.r, c.r, c.r, 1.0]

    _t = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "t", ti.f32)
    _pixels_arr = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "pixels_arr", ti.math.vec4, ndim=2
    )

    _rw_tex = ti.graph.Arg(
        ti.graph.ArgKind.RWTEXTURE,
        "rw_tex",
        ndim=2,
        fmt=ti.Format.r32f,
    )
    _tex = ti.graph.Arg(
        ti.graph.ArgKind.TEXTURE,
        "tex",
        ndim=2,
    )

    g_builder = ti.graph.GraphBuilder()
    g_builder.dispatch(make_texture, _rw_tex)
    g_builder.dispatch(paint, _t, _pixels_arr, _tex)
    g = g_builder.compile()

    pixels_arr = ti.Vector.ndarray(4, dtype=float, shape=res)
    texture = ti.Texture(ti.Format.r32f, (128, 128))
    t = 1

    g.run({"rw_tex": texture, "t": t, "pixels_arr": pixels_arr, "tex": texture})
    pixels = pixels_arr.to_numpy()
    for i in range(res[0]):
        for j in range(res[1]):
            assert test_utils.allclose(pixels[i, j], [0.1, 0.1, 0.1, 1.0])


@test_utils.test(arch=ti.vulkan)
def test_graph_texture_registry_lifetime_after_runtime_arg_gc():
    @ti.kernel
    def write(
        tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0),
    ):
        tex.store(ti.Vector([0, 0]), ti.Vector([9.0, 0.0, 0.0, 0.0]))

    sym_tex = ti.graph.Arg(
        ti.graph.ArgKind.RWTEXTURE,
        "tex",
        ndim=2,
        fmt=ti.Format.r32f,
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(write, sym_tex)
    graph = builder.compile()
    prog = impl.get_runtime().prog
    gc.collect()
    ti.sync()
    baseline = prog._debug_texture_resource_stats()

    texture = ti.Texture(ti.Format.r32f, (1, 1))
    graph.run({"tex": texture})
    launched = prog._debug_texture_resource_stats()
    assert launched["inflight"] == baseline["inflight"] + 1

    texture_ref = weakref.ref(texture)
    del texture
    gc.collect()
    assert texture_ref() is None
    retired = prog._debug_texture_resource_stats()
    assert retired["views"] == baseline["views"]
    assert retired["retiring"] == baseline["retiring"] + 1

    ti.sync()
    completed = prog._debug_texture_resource_stats()
    for key in ("live", "retiring", "leases", "views", "inflight"):
        assert completed[key] == baseline[key]
    assert completed["created_total"] == baseline["created_total"] + 1
    assert completed["retired_total"] == baseline["retired_total"] + 1
    assert completed["released_total"] == baseline["released_total"] + 1


@test_utils.test(arch=supported_archs_cgraph)
def test_ti_func_with_template_args():
    MyStruct = ti.types.struct(
        id=ti.i32,
        val=ti.f32,
        center=ti.types.vector(3, ti.f32),
        color=ti.types.vector(4, ti.i32),
    )

    arr = ti.ndarray(ti.i32, shape=())

    @ti.func
    def test_func(x: ti.template()):
        x.id = 0
        x.val = 1.0
        x.center = ti.Vector([0.0, 0.0, 0.0])
        x.color = ti.Vector([1, 1, 0, 0])

    @ti.kernel
    def test_kernel(arr: ti.types.ndarray()):
        x = MyStruct()
        test_func(x)
        arr[None] = x.color[1]

    sym_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "arr", ti.i32, ndim=0)
    g_builder = ti.graph.GraphBuilder()
    g_builder.dispatch(test_kernel, sym_arr)
    g = g_builder.compile()
    g.run({"arr": arr})
    assert arr.to_numpy() == 1


@test_utils.test(arch=[ti.vulkan])
def test_texture_struct_for():
    res = (128, 128)
    tex = ti.Texture(ti.Format.r32f, res)
    arr = ti.ndarray(ti.f32, res)

    @ti.kernel
    def write(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
        for i, j in tex:
            tex.store(ti.Vector([i, j]), ti.Vector([1.0, 0.0, 0.0, 0.0]))

    @ti.kernel
    def read(tex: ti.types.texture(num_dimensions=2), arr: ti.types.ndarray()):
        for i, j in arr:
            arr[i, j] = tex.fetch(ti.Vector([i, j]), 0).x

    sym_tex = ti.graph.Arg(
        ti.graph.ArgKind.RWTEXTURE,
        "tex",
        fmt=ti.Format.r32f,
        ndim=2,
    )
    sym_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "arr", ti.f32, ndim=2)

    gb = ti.graph.GraphBuilder()
    gb.dispatch(write, sym_tex)
    gb.dispatch(read, sym_tex, sym_arr)
    graph = gb.compile()

    graph.run({"tex": tex, "arr": arr})
    assert arr.to_numpy().sum() == 128 * 128
