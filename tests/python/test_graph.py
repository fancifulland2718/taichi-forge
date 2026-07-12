import platform
import gc
import weakref

import numpy as np
import pytest
from taichi_forge.lang.exception import TaichiCompilationError, TaichiRuntimeError

import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from tests import test_utils

supported_floating_types = [ti.f32] if platform.system() == "Darwin" else [ti.f32, ti.f64]

supported_archs_cgraph = [ti.vulkan, ti.opengl]


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
    sym_debug_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "debug_arr", ti.types.vector(1, ti.f32), ndim=1)

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
    with pytest.raises(TaichiCompilationError, match="doesn't match kernel's annotated ndim"):
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
    with pytest.raises(TaichiCompilationError, match="doesn't match kernel's annotated dtype"):
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
    with pytest.raises(TaichiCompilationError, match="doesn't match kernel's annotated dtype"):
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
    def vector_sum(mat: ti.types.vector(N, dtype), res: ti.types.ndarray(dtype=dtype, ndim=1)):
        res[0] = mat.sum() + mat[2]

    sym_A = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "mat", ti.types.vector(N, dtype))
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "res", dtype, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(vector_sum, sym_A, sym_res)
    graph = builder.compile()
    return graph


def build_graph_matrix(N, dtype):
    @ti.kernel
    def matrix_sum(mat: ti.types.matrix(N, 2, dtype), res: ti.types.ndarray(dtype=dtype, ndim=1)):
        res[0] = mat.sum()

    sym_A = ti.graph.Arg(ti.graph.ArgKind.MATRIX, "mat", ti.types.matrix(N, 2, dtype))
    sym_res = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "res", dtype, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(matrix_sum, sym_A, sym_res)
    graph = builder.compile()
    return graph


@pytest.mark.sm70
@pytest.mark.parametrize("dt", [ti.u8, ti.u16, ti.u32, ti.u64, ti.i8, ti.i16, ti.i32, ti.i64])
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
@pytest.mark.parametrize("dt", [ti.u8, ti.u16, ti.u32, ti.u64, ti.i8, ti.i16, ti.i32, ti.i64])
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


@pytest.mark.parametrize("mutation", ["builder", "sequential"])
@test_utils.test(arch=ti.cpu)
def test_compiled_graph_freezes_aot_plan(mutation):
    @ti.kernel
    def increment(value: ti.types.ndarray(dtype=ti.i32, ndim=0)):
        value[None] += 1

    sym_value = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "value", ti.i32, ndim=0
    )
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

    sym_value = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "value", ti.i32, ndim=0
    )
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


@test_utils.test(arch=ti.cuda)
def test_cuda_cgraph_cache_survives_reset_then_delete():
    graph = _build_repeated_inc_graph()
    arr = ti.ndarray(ti.i32, shape=())
    arr.fill(0)
    graph.run({"arr": arr})
    assert arr.to_numpy()[()] == 4

    ti.reset()
    assert graph._spec is None
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
    sym_source = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1
    )
    sym_output = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        transform, sym_scale, sym_offset, sym_source, sym_output
    )
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
    sym_output = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, sym_value, sym_output)
    graph = builder.compile()
    small = ti.ndarray(ti.i32, shape=17)
    large = ti.ndarray(ti.i32, shape=37)

    graph.run({"value": 3, "output": small})
    graph.run({"value": 5, "output": large})
    graph.run({"value": 7, "output": small})

    np.testing.assert_array_equal(
        small.to_numpy(), np.arange(17, dtype=np.int32) + 7
    )
    np.testing.assert_array_equal(
        large.to_numpy(), np.arange(37, dtype=np.int32) + 5
    )


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
    def add_bias(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1), bias: ti.i32
    ):
        for i in values:
            values[i] += bias

    sym_values = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
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

    sym_values = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
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

    sym_values = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(increment, sym_values)
    builder.dispatch(increment, sym_values)
    graph = builder.compile()
    values = ti.ndarray(ti.i32, shape=1 << 14)
    values.fill(0)

    fallback_before = ti_core.query_int64(
        "vulkan_graph_replay_slot_saturation_fallbacks"
    )
    launch_count = 12
    for _ in range(launch_count):
        graph.run({"values": values})

    stats = graph._graph_stats
    assert len(stats) == 1
    assert stats[0]["backend"] == "vulkan"
    assert stats[0]["attempts"] == launch_count
    assert (
        stats[0]["records"]
        + stats[0]["replays"]
        + stats[0]["ordinary_fallbacks"]
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
    def make_texture(tex: ti.types.rw_texture(num_dimensions=2, fmt=ti.Format.r32f, lod=0)):
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
            warp_uv = uv + ti.Vector([ti.cos(t + uv.x * 5.0), ti.sin(t + uv.y * 5.0)]) * 0.1
            c = ti.math.vec4(0.0)
            if uv.x > 0.5:
                c = tex.sample_lod(warp_uv, 0.0)
            else:
                c = tex.fetch(ti.cast(warp_uv * 128, ti.i32), 0)
            pixels[i, j] = [c.r, c.r, c.r, 1.0]

    _t = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "t", ti.f32)
    _pixels_arr = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pixels_arr", ti.math.vec4, ndim=2)

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
