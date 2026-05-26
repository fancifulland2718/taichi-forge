import numpy as np
import pytest
import taichi_forge as ti
import taichi_forge.algorithms._algorithms as alg_impl
from taichi_forge.lang import impl
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
