import gc
import threading
import weakref

import numpy as np
import pytest
import taichi_forge as ti
import taichi_forge.algorithms._algorithms as alg_impl
from taichi_forge.aot.utils import produce_injected_args_from_template
from taichi_forge.graph._graph import flatten_args
from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError
from tests import test_utils


def _assert_plan_reuses_rebuilt_wrappers(name, objects_a, objects_b, semantic_key):
    backend = "cpu_native"
    prog = impl.get_runtime().prog
    plan = alg_impl._NativePrimitivePlan(
        backend=backend,
        method_name=f"test_{name}",
        objects=objects_a,
        semantic_key=semantic_key,
        call_args=(),
        prog=prog,
        value_type=0,
        n=32,
    )
    assert plan.object_keys == tuple(
        alg_impl._primitive_plan_object_key(obj) for obj in objects_a
    )
    assert plan.matches_request(backend, objects_b, semantic_key)
    assert plan.cache_key() == alg_impl._native_plan_cache_key(
        backend, objects_b, semantic_key
    )


def _assert_group_reuses_rebuilt_wrappers(name, objects_a, objects_b, semantic_key):
    backend = "cpu_native"
    prog = impl.get_runtime().prog
    group = alg_impl._NativePrimitivePlanGroup(
        backend,
        objects_a,
        semantic_key,
        plans=(),
        prog=prog,
    )
    assert group.object_keys == tuple(
        alg_impl._primitive_plan_object_key(obj) for obj in objects_a
    )
    assert group.matches_request(backend, objects_b, semantic_key)
    assert group.cache_key() == alg_impl._native_plan_cache_key(
        backend, objects_b, semantic_key
    )


def _native_sort_method_for_current_arch():
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cuda:
        if not prog.cuda_device_radix_sort_available():
            pytest.skip("CUDA Driver stable sort is unavailable in this build/runtime.")
        return "cuda_device"
    if arch == ti.vulkan:
        if not prog.vulkan_radix_sort_available():
            pytest.skip("Vulkan native sort is unavailable in this build/runtime.")
        return "vulkan_native_radix_u32"
    if not prog.cpu_stable_sort_available():
        pytest.skip("CPU native sort is unavailable in this build/runtime.")
    return "cpu_native"


def _require_native_scan_for_current_arch():
    arch = impl.current_cfg().arch
    prog = impl.get_runtime().prog
    if arch == ti.cuda:
        if not prog.cuda_device_scan_available():
            pytest.skip("CUDA Driver scan is unavailable in this build/runtime.")
        return
    if arch == ti.vulkan:
        if not prog.vulkan_scan_available():
            pytest.skip("Vulkan native scan is unavailable in this build/runtime.")
        return
    if not prog.cpu_scan_available():
        pytest.skip("CPU native scan is unavailable in this build/runtime.")


@test_utils.test(arch=ti.cpu)
def test_native_plan_descriptor_reuses_rebuilt_wrappers_for_algorithm_semantics():
    n = 32
    num_groups = 8
    payload = ti.types.struct(
        val=ti.i32,
        vec=ti.types.vector(2, ti.i32),
    )
    src = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(payload, shape=n)
    keys = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    order = ti.ndarray(ti.i32, shape=n)
    offsets = ti.ndarray(ti.i32, shape=num_groups + 1)
    reduce_out = ti.ndarray(ti.i32, shape=1)
    grouped_out = ti.ndarray(ti.i32, shape=num_groups)

    src_field = ti.Vector.field(2, ti.i32, shape=n)
    dst_field = ti.Vector.field(2, ti.i32, shape=n)
    flags_field = ti.field(ti.i32, shape=n)
    count_field = ti.field(ti.i32, shape=())

    _assert_plan_reuses_rebuilt_wrappers(
        "scan",
        (src.field("val"),),
        (src.field("val"),),
        (n,),
    )
    _assert_plan_reuses_rebuilt_wrappers(
        "reduce",
        (src.field("val"), reduce_out),
        (src.field("val"), reduce_out),
        ("sum",),
    )
    _assert_plan_reuses_rebuilt_wrappers(
        "transform",
        (src.field("val"), dst.field("val")),
        (src.field("val"), dst.field("val")),
        (3, -2),
    )
    _assert_plan_reuses_rebuilt_wrappers(
        "gather",
        (src.field("val"), indices, dst.field("val")),
        (src.field("val"), indices, dst.field("val")),
        (False,),
    )
    _assert_plan_reuses_rebuilt_wrappers(
        "scatter",
        (src.field("val"), indices, dst.field("val")),
        (src.field("val"), indices, dst.field("val")),
        (True,),
    )

    scatter_workspace = alg_impl.ScatterAddWorkspace(max_items=n)
    objects_a, semantic_key = scatter_workspace._native_scatter_add_request_signature(
        src.field("val"), indices, dst.field("val"), 0
    )
    objects_b, semantic_key_b = scatter_workspace._native_scatter_add_request_signature(
        src.field("val"), indices, dst.field("val"), 0
    )
    assert semantic_key == semantic_key_b
    _assert_plan_reuses_rebuilt_wrappers(
        "scatter_add",
        objects_a,
        objects_b,
        semantic_key,
    )

    bucket_workspace = alg_impl.BucketBuilderWorkspace(max_items=n, max_bins=num_groups)
    objects_a, semantic_key = bucket_workspace._native_bucket_builder_request_signature(
        keys, src.field("val"), offsets, dst.field("val"), 0, n, num_groups
    )
    objects_b, semantic_key_b = bucket_workspace._native_bucket_builder_request_signature(
        keys, src.field("val"), offsets, dst.field("val"), 0, n, num_groups
    )
    assert semantic_key == semantic_key_b
    _assert_plan_reuses_rebuilt_wrappers(
        "bucket",
        objects_a,
        objects_b,
        semantic_key,
    )

    grouped_workspace = alg_impl.GroupedReduceWorkspace(
        max_items=n,
        max_groups=num_groups,
    )
    objects_a, semantic_key = grouped_workspace._native_grouped_reduce_request_signature(
        keys, src.field("val"), grouped_out, 0, 0, n, num_groups
    )
    objects_b, semantic_key_b = grouped_workspace._native_grouped_reduce_request_signature(
        keys, src.field("val"), grouped_out, 0, 0, n, num_groups
    )
    assert semantic_key == semantic_key_b
    _assert_plan_reuses_rebuilt_wrappers(
        "grouped_reduce",
        objects_a,
        objects_b,
        semantic_key,
    )

    tensor_a = src.field("vec")
    tensor_b = src.field("vec")
    semantic_key = alg_impl._component_group_semantic_key(
        "inplace_order_apply",
        "cpu_native",
        n,
        str(tensor_a.scalar_dtype),
        tensor_a.element_shape,
    )
    _assert_group_reuses_rebuilt_wrappers(
        "sort_order_apply",
        (tensor_a, order, dst.field("vec")),
        (tensor_b, order, dst.field("vec")),
        semantic_key,
    )
    _assert_group_reuses_rebuilt_wrappers(
        "compact_order_apply",
        (src.field("vec"), order, dst.field("vec")),
        (src.field("vec"), order, dst.field("vec")),
        semantic_key,
    )

    _assert_plan_reuses_rebuilt_wrappers(
        "dense_field_component_transform",
        (src_field.get_scalar_field(0), dst_field.get_scalar_field(0)),
        (src_field.get_scalar_field(0), dst_field.get_scalar_field(0)),
        (1, 0),
    )

    compact_workspace = alg_impl.CompactWorkspace(max_items=n)
    compact_key_a = compact_workspace._cpu_field_scan_plan_key(
        src_field.get_scalar_field(0),
        flags_field,
        dst_field.get_scalar_field(0),
        count_field,
        n,
    )
    compact_key_b = compact_workspace._cpu_field_scan_plan_key(
        src_field.get_scalar_field(0),
        flags_field,
        dst_field.get_scalar_field(0),
        count_field,
        n,
    )
    assert compact_key_a == compact_key_b

@test_utils.test(arch=ti.cpu)
def test_execution_plan_records_stable_stage_signature():
    n = 16
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    prog = impl.get_runtime().prog
    backend = "cpu_native"

    scan_plan = alg_impl._NativePrimitivePlan(
        backend=backend,
        method_name="test_scan_stage",
        objects=(src,),
        semantic_key=("scan", n),
        call_args=(),
        prog=prog,
        value_type=0,
        n=n,
    )
    gather_plan = alg_impl._NativePrimitivePlan(
        backend=backend,
        method_name="test_gather_stage",
        objects=(src, indices, dst),
        semantic_key=("gather", n),
        call_args=(),
        prog=prog,
        value_type=0,
        n=n,
    )
    plan_groups = {}
    group = alg_impl._record_native_plan_group(
        plan_groups,
        backend,
        (src, indices, dst),
        ("sequence", n),
        (scan_plan, gather_plan),
        prog=prog,
    )

    expected_signature = (
        (backend, "test_scan_stage", ("scan", n), 0, n),
        (backend, "test_gather_stage", ("gather", n), 0, n),
    )
    assert group.stage_signature == expected_signature
    assert group.stage_signature == alg_impl._primitive_stage_signature(group.plans)
    assert group.stage_calls == tuple(
        (plan.method_descriptor, plan.method_name, plan.call_args)
        for plan in group.plans
    )
    assert group.execution_key() == (group.cache_key(), expected_signature)
    assert scan_plan.execution_key() == scan_plan.cache_key()
    assert plan_groups[group.cache_key()] is group

@test_utils.test(arch=ti.cpu)
def test_native_plan_cached_match_reuses_rebuilt_wrapper_keys():
    n = 16
    payload = ti.types.struct(vec=ti.types.vector(2, ti.i32))
    src = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(payload, shape=n)
    order = ti.ndarray(ti.i32, shape=n)
    prog = impl.get_runtime().prog
    backend = "cpu_native"
    semantic_key = ("order_apply", n)
    objects_a = (src.field("vec"), order, dst.field("vec"))
    src._member_view_cache.clear()
    dst._member_view_cache.clear()
    objects_b = (src.field("vec"), order, dst.field("vec"))
    group = alg_impl._NativePrimitivePlanGroup(
        backend,
        objects_a,
        semantic_key,
        plans=(
            alg_impl._NativePrimitivePlan(
                backend=backend,
                method_name="test_stage_0",
                objects=(src.field("vec", component=0), order, dst.field("vec", component=0)),
                semantic_key=("stage", 0, n),
                call_args=(),
                prog=prog,
                value_type=0,
                n=n,
            ),
        ),
        prog=prog,
    )

    exact_match, object_keys = alg_impl._native_plan_matches_request_cached(
        group, backend, objects_a, semantic_key
    )
    assert exact_match
    assert object_keys is None

    rebuilt_match, object_keys = alg_impl._native_plan_matches_request_cached(
        group, backend, objects_b, semantic_key
    )
    assert rebuilt_match
    assert object_keys == alg_impl._primitive_plan_object_keys(objects_b)

    cached_match, object_keys_again = alg_impl._native_plan_matches_request_cached(
        group, backend, objects_b, semantic_key, object_keys
    )
    assert cached_match
    assert object_keys_again is object_keys


@test_utils.test(arch=ti.cpu)
def test_native_plan_group_cache_lookup_reuses_rebuilt_wrapper_keys():
    n = 16
    payload = ti.types.struct(vec=ti.types.vector(2, ti.i32))
    src = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(payload, shape=n)
    order = ti.ndarray(ti.i32, shape=n)
    prog = impl.get_runtime().prog
    backend = "cpu_native"
    semantic_key = ("cached_group", n)
    objects_a = (src.field("vec"), order, dst.field("vec"))
    objects_b = (src.field("vec"), order, dst.field("vec"))
    stage = alg_impl._NativePrimitivePlan(
        backend=backend,
        method_name="cpu_reduce_available",
        objects=(src.field("vec", component=0),),
        semantic_key=("stage", n),
        call_args=(),
        prog=prog,
        value_type=0,
        n=n,
    )
    group = alg_impl._NativePrimitivePlanGroup(
        backend,
        objects_a,
        semantic_key,
        plans=(stage,),
        prog=prog,
    )
    plan_groups = {group.cache_key(): group}
    observed = []

    matched = alg_impl._try_native_plan_group_from_cache(
        None,
        plan_groups,
        backend,
        objects_b,
        semantic_key,
        lambda matched_group, temp_bytes: observed.append(
            (matched_group, temp_bytes)
        ),
    )

    assert matched
    assert observed == [(group, 1)]

@test_utils.test(arch=ti.cpu)
def test_native_plan_group_cache_lookup_reuses_rebuilt_wrapper_keys():
    n = 16
    payload = ti.types.struct(vec=ti.types.vector(2, ti.i32))
    src = ti.ndarray(payload, shape=n)
    dst = ti.ndarray(payload, shape=n)
    order = ti.ndarray(ti.i32, shape=n)
    prog = impl.get_runtime().prog
    backend = "cpu_native"
    semantic_key = ("cached_group", n)
    objects_a = (src.field("vec"), order, dst.field("vec"))
    objects_b = (src.field("vec"), order, dst.field("vec"))
    stage = alg_impl._NativePrimitivePlan(
        backend=backend,
        method_name="cpu_reduce_available",
        objects=(src.field("vec", component=0),),
        semantic_key=("stage", n),
        call_args=(),
        prog=prog,
        value_type=0,
        n=n,
    )
    group = alg_impl._NativePrimitivePlanGroup(
        backend,
        objects_a,
        semantic_key,
        plans=(stage,),
        prog=prog,
    )
    plan_groups = {group.cache_key(): group}
    observed = []

    matched = alg_impl._try_native_plan_group_from_cache(
        None,
        plan_groups,
        backend,
        objects_b,
        semantic_key,
        lambda matched_group, temp_bytes: observed.append(
            (matched_group, temp_bytes)
        ),
    )

    assert matched
    assert observed == [(group, 1)]

@test_utils.test(arch=ti.cpu)
def test_prog_method_descriptor_cache_invokes_without_bound_lookup():
    prog = impl.get_runtime().prog
    method_name = "cpu_reduce_available"

    descriptor = alg_impl._prog_method_descriptor(prog, method_name)
    assert descriptor is not None
    assert alg_impl._prog_method_descriptor(prog, method_name) is descriptor

    found, result = alg_impl._invoke_prog_method(prog, method_name)
    assert found
    assert result == prog.cpu_reduce_available()

    found, result = alg_impl._invoke_prog_method(
        prog, "__missing_native_method_for_test__"
    )
    assert not found
    assert result is None

    available_plan = alg_impl._NativePrimitivePlan(
        backend="cpu_native",
        method_name=method_name,
        objects=(),
        semantic_key=("available",),
        call_args=(),
        prog=prog,
        value_type=0,
        n=0,
    )
    assert available_plan.method_descriptor is descriptor
    assert available_plan.invoke(prog) == 1

    missing_plan = alg_impl._NativePrimitivePlan(
        backend="cpu_native",
        method_name="__missing_native_method_for_test__",
        objects=(),
        semantic_key=("missing",),
        call_args=(),
        prog=prog,
        value_type=0,
        n=0,
    )
    assert missing_plan.method_descriptor is None
    assert missing_plan.invoke(prog) is None


@test_utils.test(arch=ti.cpu)
def test_default_workspace_reuses_workspace_none_transform_cache_entry():
    n = 32
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(np.arange(n, dtype=np.int32))
    alg_impl.clear_default_workspaces()
    try:
        alg_impl.experimental_transform(src, dst, scale=3, bias=5, workspace=None)
        workspaces = [
            workspace
            for cache in alg_impl._default_workspace_caches.values()
            for workspace in cache.values()
        ]
        assert len(workspaces) == 1
        cached_workspace = workspaces[0]
        alg_impl.experimental_transform(src, dst, scale=3, bias=5, workspace=None)
        workspaces = [
            workspace
            for cache in alg_impl._default_workspace_caches.values()
            for workspace in cache.values()
        ]
    finally:
        alg_impl.clear_default_workspaces()
    assert len(workspaces) == 1
    assert workspaces[0] is cached_workspace
    assert np.array_equal(dst.to_numpy(), np.arange(n, dtype=np.int32) * 3 + 5)


@test_utils.test(arch=ti.cpu)
def test_default_workspace_cache_clears_on_reset():
    n = 8
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    alg_impl.clear_default_workspaces()
    workspace = alg_impl._get_default_workspace(
        "transform",
        (src, dst),
        ("transform", "auto", n, 1, 0),
        lambda: alg_impl.TransformWorkspace(max_items=n),
    )
    assert workspace is alg_impl._get_default_workspace(
        "transform",
        (src, dst),
        ("transform", "auto", n, 1, 0),
        lambda: alg_impl.TransformWorkspace(max_items=n),
    )
    assert alg_impl._default_workspace_caches
    ti.reset()
    assert not alg_impl._default_workspace_caches


@test_utils.test(arch=ti.cpu)
def test_default_workspace_cache_is_owned_per_python_thread():
    n = 16
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    barrier = threading.Barrier(2)
    workspaces = []
    failures = []
    result_lock = threading.Lock()
    alg_impl.clear_default_workspaces()

    def acquire_workspace():
        try:
            barrier.wait(timeout=10)
            workspace = alg_impl._get_default_workspace(
                "transform",
                (src, dst),
                ("transform", "auto", n, 2, 1),
                lambda: alg_impl.TransformWorkspace(max_items=n),
            )
            with result_lock:
                workspaces.append(workspace)
        except BaseException as exc:
            with result_lock:
                failures.append(exc)

    threads = [threading.Thread(target=acquire_workspace) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        assert all(not thread.is_alive() for thread in threads)
        assert failures == []
        assert len(workspaces) == 2
        assert workspaces[0] is not workspaces[1]
        statistics = alg_impl.get_primitive_workspace_statistics()
        assert statistics["default_cache"]["context_count"] == 2
        assert statistics["default_cache"]["entry_count"] == 2
        assert statistics["default_cache"]["ownership"] == "per_python_thread"
    finally:
        alg_impl.clear_default_workspaces()

@test_utils.test(arch=ti.cpu)
def test_primitive_sequence_prewarm_replays_native_plans():
    n = 32
    src_np = np.arange(n, dtype=np.int32)
    idx_np = np.arange(n - 1, -1, -1, dtype=np.int32)
    src = ti.ndarray(ti.i32, shape=n)
    tmp = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(src_np)
    indices.from_numpy(idx_np)

    seq = alg_impl.primitive_sequence()
    assert seq.transform(src, tmp, scale=2, bias=1).gather(tmp, indices, dst) is seq
    assert seq.call_count == 2
    assert seq.direct_plan_count == 0

    seq.prewarm()
    assert seq.direct_plan_count == 2
    assert np.array_equal(dst.to_numpy(), (src_np * 2 + 1)[idx_np])

    src_np = src_np + 10
    src.from_numpy(src_np)
    seq.run()
    assert np.array_equal(dst.to_numpy(), (src_np * 2 + 1)[idx_np])
    assert len(seq.workspaces) == 2
    assert seq.workspace_bytes_peak >= 0


@test_utils.test(arch=ti.cpu)
def test_graph_private_native_sequence_replays_native_plans():
    n = 32
    src_np = np.arange(n, dtype=np.int32)
    idx_np = np.arange(n - 1, -1, -1, dtype=np.int32)
    src = ti.ndarray(ti.i32, shape=n)
    tmp = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(src_np)
    indices.from_numpy(idx_np)

    seq = alg_impl.primitive_sequence()
    seq.transform(src, tmp, scale=2, bias=1).gather(tmp, indices, dst)
    builder = ti.graph.GraphBuilder()
    assert builder._append_native(seq) is builder
    graph = builder.compile()

    graph.run({})
    assert seq.direct_plan_count == 2
    assert np.array_equal(dst.to_numpy(), (src_np * 2 + 1)[idx_np])

    src_np = src_np + 10
    src.from_numpy(src_np)
    graph.run({})
    assert np.array_equal(dst.to_numpy(), (src_np * 2 + 1)[idx_np])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_graph_private_native_sequence_sorts_with_native_backend():
    method = _native_sort_method_for_current_arch()
    n = 32
    keys_np = ((np.arange(n, dtype=np.int32) * 17) % 41 - 20).astype(np.int32)
    values_np = np.arange(n, dtype=np.int32) * 3 + 1
    keys = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(ti.i32, shape=n)

    seq = alg_impl.primitive_sequence()
    seq.sort(keys, values, method=method)
    builder = ti.graph.GraphBuilder()
    assert builder._append_native(seq) is builder
    graph = builder.compile()

    for offset in (0, 7):
        current_keys = (keys_np + offset).astype(np.int32)
        keys.from_numpy(current_keys)
        values.from_numpy(values_np)
        graph.run({})
        order = np.argsort(current_keys, kind="stable")
        assert np.array_equal(keys.to_numpy(), current_keys[order])
        assert np.array_equal(values.to_numpy(), values_np[order])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_graph_private_native_sequence_sort_by_key_with_native_backend():
    method = _native_sort_method_for_current_arch()
    n = 32
    primary_np = (np.arange(n, dtype=np.int32) % 5).astype(np.int32)
    secondary_np = ((np.arange(n, dtype=np.int32) * 7) % 11).astype(np.int32)
    values_np = np.arange(n, dtype=np.int32) * 13 + 3
    primary = ti.ndarray(ti.i32, shape=n)
    secondary = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(ti.i32, shape=n)

    seq = alg_impl.primitive_sequence()
    seq.sort_by_key([primary, secondary], values, method=method)
    builder = ti.graph.GraphBuilder()
    assert builder._append_native(seq) is builder
    graph = builder.compile()

    for offset in (0, 2):
        current_primary = (primary_np + offset).astype(np.int32)
        current_secondary = (secondary_np - offset).astype(np.int32)
        primary.from_numpy(current_primary)
        secondary.from_numpy(current_secondary)
        values.from_numpy(values_np)
        graph.run({})
        order = np.lexsort((np.arange(n), current_secondary, current_primary))
        assert np.array_equal(primary.to_numpy(), current_primary[order])
        assert np.array_equal(secondary.to_numpy(), current_secondary[order])
        assert np.array_equal(values.to_numpy(), values_np[order])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], exclude=[(ti.vulkan, "Darwin")])
def test_graph_private_native_sequence_scan_uses_prefix_sum_executor_plan():
    _require_native_scan_for_current_arch()

    n = 64
    values_np = (np.arange(n, dtype=np.int32) % 13 - 4).astype(np.int32)
    values = ti.ndarray(ti.i32, shape=n)

    seq = alg_impl.primitive_sequence()
    seq.scan(values)
    builder = ti.graph.GraphBuilder()
    assert builder._append_native(seq) is builder
    graph = builder.compile()

    for offset in (0, 5):
        current = (values_np + offset).astype(np.int32)
        values.from_numpy(current)
        graph.run({})
        expected = np.cumsum(current, dtype=np.int32).astype(np.int32)
        assert np.array_equal(values.to_numpy(), expected)
        assert seq.direct_plan_count == 1


@pytest.mark.parametrize("adapter", ["public", "legacy"])
@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_graph_native_sequence_joins_disjoint_cgraph_segments(adapter):
    n = 32
    src_np = np.arange(n, dtype=np.int32)
    idx_np = np.arange(n - 1, -1, -1, dtype=np.int32)
    src = ti.ndarray(ti.i32, shape=n)
    tmp0 = ti.ndarray(ti.i32, shape=n)
    tmp1 = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    gathered = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(src_np)
    indices.from_numpy(idx_np)

    @ti.kernel
    def prepare(
        src_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
        tmp_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in src_arr:
            tmp_arr[i] = src_arr[i] + 10

    @ti.data_oriented
    class Finalizer:
        @ti.kernel
        def apply(
            self,
            gathered_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
            dst_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
        ):
            for i in gathered_arr:
                dst_arr[i] = gathered_arr[i] - 3

    src_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "src", ti.i32, ndim=1)
    tmp0_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "tmp0", ti.i32, ndim=1)
    gathered_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "gathered", ti.i32, ndim=1
    )
    dst_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dst", ti.i32, ndim=1)

    seq = alg_impl.primitive_sequence()
    seq.transform(tmp0, tmp1, scale=2, bias=1).gather(tmp1, indices, gathered)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(prepare, src_arg, tmp0_arg)
    assert builder._append_native(seq) is builder
    finalizer = Finalizer()
    if adapter == "public":
        builder.dispatch(
            finalizer.apply,
            gathered_arg,
            dst_arg,
            template_args={"self": finalizer},
        )
    else:
        kernel = finalizer.apply._primal
        injected = produce_injected_args_from_template(
            kernel,
            {
                "self": finalizer,
                "gathered_arr": gathered,
                "dst_arr": dst,
            },
        )
        key = kernel.ensure_compiled(*injected)
        kernel_cpp = kernel.compiled_kernels[key]
        symbolic_args = flatten_args((gathered_arg, dst_arg))
        builder._aot_graph_plan.dispatch(kernel_cpp, symbolic_args)
        builder._ensure_runtime_graph_builder().dispatch(
            kernel_cpp, symbolic_args
        )
        builder._dispatch_count += 1
    graph = builder.compile()

    assert [
        node.runtime_arg_names for node in graph._spec.nodes
    ] == [
        frozenset({"src", "tmp0"}),
        frozenset(),
        frozenset({"gathered", "dst"}),
    ]

    assert graph.run(
        {"src": src, "tmp0": tmp0, "gathered": gathered, "dst": dst}
    ) is None
    expected = ((src_np + 10) * 2 + 1)[idx_np] - 3
    assert seq.direct_plan_count == 2
    assert np.array_equal(dst.to_numpy(), expected)

    src_np = src_np + 5
    src.from_numpy(src_np)
    prog = impl.get_runtime().prog
    next_sequence = prog._debug_runtime_completion_stats()["next_sequence"]
    ticket = graph.submit(
        {"src": src, "tmp0": tmp0, "gathered": gathered, "dst": dst}
    )
    assert ticket.sequence == next_sequence
    assert (
        prog._debug_runtime_completion_stats()["next_sequence"]
        == next_sequence + 1
    )
    ticket.wait()
    expected = ((src_np + 10) * 2 + 1)[idx_np] - 3
    assert np.array_equal(dst.to_numpy(), expected)

    for stats in graph._graph_stats:
        assert stats["last_fallback_reason"] != "unsupported_arguments"

    with pytest.raises(
        TaichiRuntimeError,
        match="Missing graph runtime arguments: dst",
    ):
        graph.run(
            {"src": src, "tmp0": tmp0, "gathered": gathered}
        )
    with pytest.raises(
        TaichiRuntimeError,
        match="Unexpected graph runtime arguments: extra",
    ):
        graph.run(
            {
                "src": src,
                "tmp0": tmp0,
                "gathered": gathered,
                "dst": dst,
                "extra": 1,
            }
        )


@test_utils.test(
    arch=[ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_graph_submit_retains_native_owner_until_completion():
    n = 1 << 18
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    src.fill(7)

    sequence = alg_impl.primitive_sequence()
    sequence.transform(src, dst, scale=3, bias=1)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()

    ticket = graph.submit({})
    pending = ticket._has_backend_work
    if not pending:
        # Extremely fast CUDA devices may retire even this replay during the
        # completion recorder's final nonblocking query. In that case there is
        # no in-flight Python workspace left to retain.
        assert ticket.done()
    graph_ref = weakref.ref(graph)
    del graph
    del builder
    del sequence
    gc.collect()
    if pending:
        assert graph_ref() is not None

    ticket.wait()
    gc.collect()
    assert graph_ref() is None
    assert np.all(dst.to_numpy() == 22)


@test_utils.test(
    arch=[ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_graph_submit_retains_native_owner_after_ticket_is_dropped():
    src = ti.ndarray(ti.i32, shape=1 << 18)
    dst = ti.ndarray(ti.i32, shape=1 << 18)
    src.fill(4)
    sequence = alg_impl.primitive_sequence()
    sequence.transform(src, dst, scale=5, bias=2)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()

    ticket = graph.submit({})
    pending = ticket._has_backend_work
    graph_ref = weakref.ref(graph)
    del ticket
    del graph
    del builder
    del sequence
    gc.collect()
    if pending:
        assert graph_ref() is not None

    # Program completion tracking, rather than the ticket wrapper, owns the
    # outstanding work. A later synchronization retires the Python owner too.
    ti.sync()
    gc.collect()
    assert graph_ref() is None
    assert np.all(dst.to_numpy() == 22)


@test_utils.test(
    arch=[ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_graph_submit_owner_is_retired_before_reset():
    src = ti.ndarray(ti.i32, shape=1 << 18)
    dst = ti.ndarray(ti.i32, shape=1 << 18)
    src.fill(5)
    sequence = alg_impl.primitive_sequence()
    sequence.transform(src, dst, scale=2, bias=3)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()

    ticket = graph.submit({})
    pending = ticket._has_backend_work
    graph_ref = weakref.ref(graph)
    del graph
    del builder
    del sequence
    gc.collect()
    if pending:
        assert graph_ref() is not None

    # Runtime.clear() synchronizes pending native owners before invalidating
    # weak Graph wrappers and before Program.finalize() tears down workspaces.
    ti.reset()
    gc.collect()
    assert graph_ref() is None
    ticket.wait()


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    exclude=[(ti.vulkan, "Darwin")],
)
def test_graph_field_only_segment_native_sort_and_runtime_segment():
    method = _native_sort_method_for_current_arch()
    n = 32
    state = ti.field(ti.i32, shape=())
    keys = ti.ndarray(ti.i32, shape=n)
    values = ti.ndarray(ti.i32, shape=n)
    output = ti.ndarray(ti.i32, shape=1)
    base_keys = ((np.arange(n, dtype=np.int32) * 17) % 41 - 20).astype(
        np.int32
    )
    base_values = np.arange(n, dtype=np.int32) * 5 + 3

    @ti.kernel
    def advance_field():
        state[None] += 1

    @ti.kernel
    def finalize(
        scale: ti.i32,
        sorted_keys: ti.types.ndarray(dtype=ti.i32, ndim=1),
        sorted_values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        dst: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        dst[0] = (
            state[None] * scale + sorted_keys[0] + sorted_values[0]
        )

    scale_arg = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "scale", ti.i32
    )
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1
    )
    keys_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "keys", ti.i32, ndim=1
    )
    values_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    sequence = alg_impl.primitive_sequence()
    sequence.sort(keys, values, method=method)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(advance_field)
    builder.dispatch(advance_field)
    builder.append_native(sequence)
    builder.dispatch(
        finalize, scale_arg, keys_arg, values_arg, output_arg
    )
    graph = builder.compile()

    assert [
        node.runtime_arg_names for node in graph._spec.nodes
    ] == [
        frozenset(),
        frozenset(),
        frozenset({"scale", "keys", "values", "output"}),
    ]
    initial_report = graph.execution_stats()
    assert initial_report.node_count == 3
    assert initial_report.cgraph_segment_count == 2
    assert initial_report.native_node_count == 1
    assert initial_report.dispatch_count == 3
    assert initial_report.execution_path == "not_run"
    assert [
        segment.runtime_arg_count for segment in initial_report.segments
    ] == [0, 0, 4]
    assert [
        segment.dispatch_count for segment in initial_report.segments
    ] == [2, 0, 1]

    order = np.argsort(base_keys, kind="stable")
    for initial_state, scale in ((0, 3), (7, -2)):
        state[None] = initial_state
        keys.from_numpy(base_keys)
        values.from_numpy(base_values)
        graph.run(
            {
                "scale": scale,
                "keys": keys,
                "values": values,
                "output": output,
            }
        )
        expected = (
            (initial_state + 2) * scale
            + base_keys[order[0]]
            + base_values[order[0]]
        )
        assert output.to_numpy()[0] == expected
        assert np.array_equal(keys.to_numpy(), base_keys[order])
        assert np.array_equal(values.to_numpy(), base_values[order])

    execution_report = graph.execution_stats()
    cgraph_segments = [
        segment
        for segment in execution_report.segments
        if segment.kind == "cgraph"
    ]
    for segment in cgraph_segments:
        assert segment.fallback_reason != "unsupported_arguments"
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        field_segment = cgraph_segments[0]
        assert field_segment.zero_arg_eligible
        assert field_segment.counters.captures == 1
        assert field_segment.counters.exact_replays == 1
        assert field_segment.persistent_argument_bytes == 0


@test_utils.test(arch=ti.cuda)
def test_graph_private_native_sequence_uses_cuda_native_replay_backend():
    n = 32
    src_np = np.arange(n, dtype=np.int32)
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(src_np)

    seq = alg_impl.primitive_sequence()
    seq.transform(src, dst, scale=4, bias=-3)
    builder = ti.graph.GraphBuilder()
    assert builder._append_native(seq) is builder
    graph = builder.compile()

    info = graph._instance_debug_info
    assert info["kind"] == "cuda_native_replay"
    report = graph.execution_stats()
    assert report.execution_path == "native_replay"
    assert report.node_count == 1
    assert report.cgraph_segment_count == 0
    assert report.native_node_count == 1
    assert report.segments[0].backend == "cuda"

    graph._prewarm()
    assert seq.direct_plan_count == 1
    assert np.array_equal(dst.to_numpy(), src_np * 4 - 3)

    src_np = src_np + 7
    src.from_numpy(src_np)
    graph.run({})
    assert np.array_equal(dst.to_numpy(), src_np * 4 - 3)


@test_utils.test(arch=ti.cpu)
def test_graph_private_native_sequence_uses_cpu_native_replay_backend():
    n = 32
    src_np = np.arange(n, dtype=np.int32)
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(src_np)

    seq = alg_impl.primitive_sequence()
    seq.transform(src, dst, scale=4, bias=-3)
    builder = ti.graph.GraphBuilder()
    assert builder._append_native(seq) is builder
    graph = builder.compile()

    info = graph._instance_debug_info
    assert info["kind"] == "cpu_native_replay"
    report = graph.execution_stats()
    assert report.execution_path == "native_replay"
    assert report.node_count == 1
    assert report.cgraph_segment_count == 0
    assert report.native_node_count == 1
    assert report.segments[0].backend == "cpu"

    graph._prewarm()
    assert seq.direct_plan_count == 1
    assert np.array_equal(dst.to_numpy(), src_np * 4 - 3)

    src_np = src_np + 7
    src.from_numpy(src_np)
    graph.run({})
    assert np.array_equal(dst.to_numpy(), src_np * 4 - 3)


@test_utils.test(arch=ti.cpu)
def test_graph_private_native_sequence_rejects_unregistered_native_node():
    class RunOnly:
        def run(self):
            pass

    with pytest.raises(ti.TaichiRuntimeError, match="DSL-defined native graph"):
        ti.graph.GraphBuilder()._append_native(RunOnly())


@test_utils.test(arch=ti.vulkan)
def test_primitive_sequence_vulkan_fuses_indexed_transform_chain():
    prog = ti.lang.impl.get_runtime().prog
    if not hasattr(prog, "vulkan_transform_indexed_affine_ndarray"):
        pytest.skip("Vulkan indexed transform fusion is unavailable.")
    n = 64
    src_np = np.arange(n, dtype=np.int32) - 7
    idx_np = np.arange(n - 1, -1, -1, dtype=np.int32)
    src = ti.ndarray(ti.i32, shape=n)
    tmp = ti.ndarray(ti.i32, shape=n)
    indices = ti.ndarray(ti.i32, shape=n)
    gathered = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(src_np)
    indices.from_numpy(idx_np)
    dst.fill(0)

    seq = alg_impl.primitive_sequence()
    seq.transform(src, tmp, scale=3, bias=5, method="vulkan_native")
    seq.gather(tmp, indices, gathered, method="vulkan_native")
    seq.scatter(gathered, indices, dst, method="vulkan_native")
    seq.prewarm()
    assert seq.fused_plan_method == "vulkan_transform_indexed_affine_ndarray"
    assert np.array_equal(dst.to_numpy(), src_np * 3 + 5)

    src_np = src_np + 11
    src.from_numpy(src_np)
    dst.fill(0)
    seq.run()
    assert np.array_equal(dst.to_numpy(), src_np * 3 + 5)

@test_utils.test(arch=ti.vulkan)
def test_primitive_sequence_vulkan_fused_indexed_transform_value_types():
    prog = ti.lang.impl.get_runtime().prog
    if not hasattr(prog, "vulkan_transform_indexed_affine_ndarray"):
        pytest.skip("Vulkan indexed transform fusion is unavailable.")
    n = 32
    idx_np = np.arange(n - 1, -1, -1, dtype=np.int32)
    cases = (
        (ti.u32, np.uint32, 3, 5),
        (ti.f32, np.float32, 1.5, -2.25),
    )
    for dtype, np_dtype, scale, bias in cases:
        src_np = (np.arange(n, dtype=np_dtype) + np_dtype(2)).astype(np_dtype)
        src = ti.ndarray(dtype, shape=n)
        tmp = ti.ndarray(dtype, shape=n)
        indices = ti.ndarray(ti.i32, shape=n)
        gathered = ti.ndarray(dtype, shape=n)
        dst = ti.ndarray(dtype, shape=n)
        src.from_numpy(src_np)
        indices.from_numpy(idx_np)
        dst.fill(0)
        seq = alg_impl.primitive_sequence()
        seq.transform(src, tmp, scale=scale, bias=bias, method="vulkan_native")
        seq.gather(tmp, indices, gathered, method="vulkan_native")
        seq.scatter(gathered, indices, dst, method="vulkan_native")
        seq.prewarm()
        assert seq.fused_plan_method == "vulkan_transform_indexed_affine_ndarray"
        expected = (src_np * np_dtype(scale) + np_dtype(bias)).astype(np_dtype)
        if dtype == ti.f32:
            assert np.allclose(dst.to_numpy(), expected)
        else:
            assert np.array_equal(dst.to_numpy(), expected)

@test_utils.test(arch=ti.cpu)
def test_primitive_sequence_clear_keeps_calls_and_rebuilds_plans():
    n = 16
    src_np = np.arange(n, dtype=np.int32)
    src = ti.ndarray(ti.i32, shape=n)
    dst = ti.ndarray(ti.i32, shape=n)
    src.from_numpy(src_np)

    seq = alg_impl.PrimitiveSequence().transform(src, dst, scale=4, bias=-3)
    seq.prewarm()
    assert seq.direct_plan_count == 1

    seq.clear()
    assert seq.call_count == 1
    assert seq.direct_plan_count == 0

    seq.run()
    assert seq.direct_plan_count == 1
    assert np.array_equal(dst.to_numpy(), src_np * 4 - 3)
