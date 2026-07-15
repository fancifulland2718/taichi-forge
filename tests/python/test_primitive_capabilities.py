import json
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

import taichi_forge as ti
import taichi_forge.algorithms._algorithms as alg_impl
import taichi_forge.algorithms._autodiff as autodiff_impl
from tests import test_utils


_FAMILIES = (
    "sort",
    "scan",
    "compact",
    "run_length_encode",
    "unique",
    "unique_by_key",
    "segmented_reduce",
    "segmented_scan",
    "reduce",
    "histogram",
    "transform",
    "gather",
    "scatter",
    "scatter_add",
    "bucket_builder",
    "grouped_reduce",
    "check",
    "metric",
)


def _method_names(capability):
    return frozenset(method.name for method in capability.methods)


def test_static_primitive_capability_catalog_is_complete_and_immutable():
    ti.reset()
    capabilities = ti.algorithms.primitive_capabilities()

    assert tuple(capability.name for capability in capabilities) == _FAMILIES
    assert all(
        capability.schema_version
        == ti.algorithms.PRIMITIVE_CAPABILITY_SCHEMA_VERSION
        == 2
        for capability in capabilities
    )
    assert all(capability.dtypes for capability in capabilities)
    assert all(capability.ranks == (1,) for capability in capabilities)
    assert all(capability.layouts for capability in capabilities)
    assert all(capability.storages for capability in capabilities)
    assert all(capability.operands for capability in capabilities)
    assert all(capability.methods for capability in capabilities)
    assert ti.algorithms.PRIMITIVE_DEPENDENCY_CLASSES == (
        "none",
        "selected_provider",
        "cuda_driver",
        "cuda_driver_or_toolkit_runtime",
        "cuda_toolkit_runtime",
    )
    methods_by_family = {
        capability.name: {method.name: method for method in capability.methods}
        for capability in capabilities
    }
    assert (
        methods_by_family["scan"]["cuda_cub"].dependency_class
        == "cuda_toolkit_runtime"
    )
    assert (
        methods_by_family["transform"]["cuda_device"].dependency_class
        == "cuda_driver"
    )
    for family in ("gather", "scatter"):
        assert (
            methods_by_family[family]["cuda_device"].dependency_class
            == "cuda_driver"
        )
    for family in ("check", "metric"):
        assert (
            methods_by_family[family]["cuda_device"].dependency_class
            == "cuda_driver"
        )
        assert (
            methods_by_family[family]["cuda_cub"].dependency_class
            == "cuda_toolkit_runtime"
        )
    assert (
        methods_by_family["bucket_builder"]["cuda_device"].dependency_class
        == "cuda_toolkit_runtime"
    )
    assert (
        methods_by_family["reduce"]["cpu_native"].dependency_class == "none"
    )
    assert methods_by_family["reduce"]["auto"].dependency_class == (
        "selected_provider"
    )
    assert all(capability.ad.primal == "supported" for capability in capabilities)
    assert all(
        capability.aot == "unsupported_for_native_nodes"
        for capability in capabilities
    )

    reduce_capability = ti.algorithms.primitive_capability(
        "ti.algorithms.experimental_reduce()"
    )
    assert reduce_capability.name == "reduce"
    assert reduce_capability.ad.differentiable_ops == ("sum",)
    assert tuple(operand.name for operand in reduce_capability.operands) == (
        "values",
        "output",
    )
    assert reduce_capability.operands[0].ranks == (1,)
    assert reduce_capability.operands[1].ranks == (0, 1)
    histogram = ti.algorithms.primitive_capability("histogram")
    assert histogram.operands[0].dtypes == ("i32", "u32")
    assert histogram.operands[1].dtypes == ("i32", "i64")
    scatter = ti.algorithms.primitive_capability("scatter")
    assert "unique" in scatter.operands[1].constraints
    assert ti.algorithms.primitive_capability("scan").ad.forward_ad == "unsupported"
    assert (
        ti.algorithms.primitive_capability("grouped_reduce").ad.forward_ad
        == "unsupported"
    )
    assert (
        ti.algorithms.primitive_capability("sort_by_key")
        is ti.algorithms.primitive_capability("sort")
    )
    rle = ti.algorithms.primitive_capability("experimental_run_length_encode")
    assert rle.stability == "stable_consecutive_run_order"
    assert "host_read_synchronizes" in rle.operands[-1].constraints
    assert (
        ti.algorithms.primitive_capability("experimental_unique").ad.forward_ad
        == "not_differentiable"
    )
    unique_by_key = ti.algorithms.primitive_capability("unique_by_key")
    assert (
        "matrix_field_payload_i32_only"
        in unique_by_key.operands[1].constraints
    )
    segmented_reduce = ti.algorithms.primitive_capability(
        "experimental_segmented_reduce"
    )
    assert segmented_reduce.graph_replay == "primitive_sequence_native_node"
    assert (
        "host_validated_at_construction"
        in segmented_reduce.operands[1].constraints
    )
    assert segmented_reduce.ad.forward_ad == "unsupported"
    assert segmented_reduce.ad.reverse_ad == "grouped_ndarray_only"
    assert segmented_reduce.ad.fallback_method is None
    assert (
        ti.algorithms.primitive_capability(
            "experimental_segmented_scan"
        ).ad.reverse_ad
        == "not_differentiable"
    )
    sort_methods = {
        method.name: method
        for method in ti.algorithms.primitive_capability("sort").methods
    }
    assert sort_methods["radix_u32"].backends == ("cuda", "vulkan")
    assert sort_methods["vulkan_graph_radix_u32"].provider_probes == ()
    assert sort_methods["vulkan_graph_radix_u32"].implementation == "composite"
    assert sort_methods["vulkan_radix_u32"].provider_probes == ()
    assert sort_methods["vulkan_radix_u32"].implementation == "composite"
    serialized = json.loads(json.dumps(asdict(reduce_capability)))
    assert serialized["schema_version"] == 2
    assert serialized["operands"][0]["name"] == "values"
    with pytest.raises(FrozenInstanceError):
        reduce_capability.name = "changed"
    with pytest.raises(ValueError, match="Unknown Forge primitive"):
        ti.algorithms.primitive_capability("missing")
    with pytest.raises(TypeError, match="must be a string"):
        ti.algorithms.primitive_capability(None)


def test_capability_catalog_is_the_method_validation_source_of_truth():
    assert alg_impl._SUPPORTED_SORT_METHODS == _method_names(
        ti.algorithms.primitive_capability("sort")
    )
    assert alg_impl._SUPPORTED_COMPACT_METHODS == _method_names(
        ti.algorithms.primitive_capability("compact")
    )
    assert alg_impl._SUPPORTED_RLE_METHODS == _method_names(
        ti.algorithms.primitive_capability("run_length_encode")
    )
    assert alg_impl._SUPPORTED_RLE_METHODS == _method_names(
        ti.algorithms.primitive_capability("unique")
    )
    assert alg_impl._SUPPORTED_RLE_METHODS == _method_names(
        ti.algorithms.primitive_capability("unique_by_key")
    )
    assert alg_impl._SUPPORTED_SEGMENTED_REDUCE_METHODS == _method_names(
        ti.algorithms.primitive_capability("segmented_reduce")
    )
    assert alg_impl._SUPPORTED_SEGMENTED_SCAN_METHODS == _method_names(
        ti.algorithms.primitive_capability("segmented_scan")
    )
    assert alg_impl._SUPPORTED_REDUCE_METHODS == _method_names(
        ti.algorithms.primitive_capability("reduce")
    )
    assert alg_impl._SUPPORTED_HISTOGRAM_METHODS == _method_names(
        ti.algorithms.primitive_capability("histogram")
    )
    assert alg_impl._SUPPORTED_TRANSFORM_METHODS == _method_names(
        ti.algorithms.primitive_capability("transform")
    )
    indexed_copy_methods = _method_names(
        ti.algorithms.primitive_capability("gather")
    )
    assert alg_impl._SUPPORTED_INDEXED_COPY_METHODS == indexed_copy_methods
    assert (
        _method_names(ti.algorithms.primitive_capability("scatter"))
        == indexed_copy_methods
    )
    assert alg_impl._SUPPORTED_SCATTER_ADD_METHODS == _method_names(
        ti.algorithms.primitive_capability("scatter_add")
    )
    assert alg_impl._SUPPORTED_BUCKET_BUILDER_METHODS == _method_names(
        ti.algorithms.primitive_capability("bucket_builder")
    )
    assert alg_impl._SUPPORTED_GROUPED_REDUCE_METHODS == _method_names(
        ti.algorithms.primitive_capability("grouped_reduce")
    )
    assert alg_impl._SUPPORTED_CHECK_METHODS == _method_names(
        ti.algorithms.primitive_capability("check")
    )
    assert alg_impl._SUPPORTED_METRIC_METHODS == _method_names(
        ti.algorithms.primitive_capability("metric")
    )


def test_resolved_primitive_capability_requires_an_initialized_program():
    ti.reset()
    with pytest.raises(
        ti.TaichiRuntimeError,
        match=r"resolve_primitive_capability\(\) requires ti\.init",
    ):
        ti.algorithms.resolve_primitive_capability("reduce")


def test_fwd_mode_policy_falls_back_or_rejects_explicit_native(monkeypatch):
    monkeypatch.setattr(autodiff_impl, "is_tape_active", lambda: False)
    monkeypatch.setattr(autodiff_impl, "is_fwd_mode_active", lambda: True)

    assert autodiff_impl.native_autodiff_method("transform", "auto") == "kernel"
    assert (
        autodiff_impl.native_autodiff_method("reduce", "auto", op="sum")
        == "field_atomic"
    )
    with pytest.raises(RuntimeError, match=r"cuda_device.*FwdMode"):
        autodiff_impl.native_autodiff_method("transform", "cuda_device")
    with pytest.raises(RuntimeError, match=r"vulkan_native.*FwdMode"):
        autodiff_impl.native_autodiff_method("gather", "vulkan_native")
    with pytest.raises(RuntimeError, match=r"cpu_native.*FwdMode"):
        autodiff_impl.native_autodiff_method("scatter", "cpu_native")
    with pytest.raises(RuntimeError, match=r"grouped_reduce.*FwdMode"):
        autodiff_impl.native_autodiff_method(
            "grouped_reduce",
            "auto",
            op="sum",
        )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_resolved_primitive_capability_matches_active_program_probes():
    expected_backend = {
        ti.cpu: "cpu",
        ti.cuda: "cuda",
        ti.vulkan: "vulkan",
    }[ti.lang.impl.current_cfg().arch]
    prog = ti.lang.impl.get_runtime().prog

    for contract in ti.algorithms.primitive_capabilities():
        resolved = ti.algorithms.resolve_primitive_capability(contract.name)
        assert resolved.schema_version == contract.schema_version
        assert resolved.backend == expected_backend
        assert resolved.contract is contract
        expected_methods = tuple(
            method
            for method in contract.methods
            if expected_backend in method.backends
        )
        assert tuple(method.method for method in resolved.methods) == tuple(
            method.name for method in expected_methods
        )
        for method, expected in zip(resolved.methods, expected_methods):
            assert method.input_dependent is True
            assert method.provider_probes == expected.provider_probes
            assert method.dependency_class == expected.dependency_class
            if method.method == "auto":
                assert method.program_available is any(
                    candidate.program_available
                    for candidate in resolved.methods
                    if candidate.method != "auto"
                )
            elif method.provider_probes:
                assert method.program_available is all(
                    bool(getattr(prog, probe)())
                    if getattr(prog, probe, None) is not None
                    else False
                    for probe in method.provider_probes
                )
            else:
                assert method.program_available is True

        with pytest.raises(FrozenInstanceError):
            resolved.backend = "changed"


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_fwd_mode_auto_uses_kernel_fallback_and_propagates_duals():
    n = 8
    src = ti.field(ti.f32, shape=n, needs_grad=True, needs_dual=True)
    dst = ti.field(ti.f32, shape=n, needs_grad=True, needs_dual=True)
    src.from_numpy(np.arange(n, dtype=np.float32))

    with ti.ad.FwdMode(loss=dst, param=src, seed=[1.0] * n):
        ti.algorithms.experimental_transform(
            src,
            dst,
            scale=3.0,
            bias=-2.0,
            method="auto",
        )

    np.testing.assert_allclose(dst.to_numpy(), np.arange(n) * 3.0 - 2.0)
    np.testing.assert_allclose(dst.dual.to_numpy(), np.full(n, 3.0))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_fwd_mode_kernel_fallback_jvp_for_reduce_and_indexed_ops():
    n = 8
    reverse = np.arange(n - 1, -1, -1, dtype=np.int32)
    duplicate = np.array([0, 0, 1, 1, 1, 3, 3, 7], dtype=np.int32)

    reduce_src = ti.field(ti.f32, shape=n, needs_dual=True)
    reduce_out = ti.field(ti.f32, shape=(), needs_dual=True)
    reduce_src.from_numpy(np.arange(n, dtype=np.float32))
    with ti.ad.FwdMode(loss=reduce_out, param=reduce_src, seed=[1.0] * n):
        ti.algorithms.experimental_reduce(
            reduce_src,
            reduce_out,
            op="sum",
            method="auto",
        )
    np.testing.assert_allclose(reduce_out.dual[None], float(n))

    gather_src = ti.field(ti.f32, shape=n, needs_dual=True)
    gather_dst = ti.field(ti.f32, shape=n, needs_dual=True)
    gather_indices = ti.field(ti.i32, shape=n)
    gather_src.from_numpy(np.arange(n, dtype=np.float32))
    gather_indices.from_numpy(reverse)
    with ti.ad.FwdMode(loss=gather_dst, param=gather_src, seed=[1.0] * n):
        ti.algorithms.experimental_gather(
            gather_src,
            gather_indices,
            gather_dst,
            method="auto",
        )
    np.testing.assert_allclose(gather_dst.dual.to_numpy(), np.ones(n))

    scatter_src = ti.field(ti.f32, shape=n, needs_dual=True)
    scatter_dst = ti.field(ti.f32, shape=n, needs_dual=True)
    scatter_indices = ti.field(ti.i32, shape=n)
    scatter_src.from_numpy(np.arange(n, dtype=np.float32))
    scatter_indices.from_numpy(reverse)
    with ti.ad.FwdMode(loss=scatter_dst, param=scatter_src, seed=[1.0] * n):
        ti.algorithms.experimental_scatter(
            scatter_src,
            scatter_indices,
            scatter_dst,
            method="auto",
        )
    np.testing.assert_allclose(scatter_dst.dual.to_numpy(), np.ones(n))

    add_src = ti.field(ti.f32, shape=n, needs_dual=True)
    add_dst = ti.field(ti.f32, shape=n, needs_dual=True)
    add_indices = ti.field(ti.i32, shape=n)
    add_src.from_numpy(np.arange(n, dtype=np.float32))
    add_dst.fill(0)
    add_indices.from_numpy(duplicate)
    with ti.ad.FwdMode(loss=add_dst, param=add_src, seed=[1.0] * n):
        ti.algorithms.experimental_scatter_add(
            add_src,
            add_indices,
            add_dst,
            method="auto",
        )
    expected = np.bincount(duplicate, minlength=n).astype(np.float32)
    np.testing.assert_allclose(add_dst.dual.to_numpy(), expected)


@test_utils.test(arch=ti.cpu)
def test_fwd_mode_rejects_native_or_discrete_primitive_before_writing():
    n = 4
    param = ti.field(ti.f32, shape=1, needs_dual=True)
    loss = ti.field(ti.f32, shape=1, needs_dual=True)
    src = ti.field(ti.f32, shape=n, needs_dual=True)
    dst = ti.field(ti.f32, shape=n, needs_dual=True)
    keys = ti.field(ti.i32, shape=n)
    src.from_numpy(np.arange(n, dtype=np.float32))
    dst.fill(17)
    keys.from_numpy(np.array([3, 1, 2, 0], dtype=np.int32))

    with pytest.raises(RuntimeError, match=r"method='cpu_native'.*FwdMode"):
        with ti.ad.FwdMode(loss=loss, param=param):
            ti.algorithms.experimental_transform(
                src,
                dst,
                method="cpu_native",
            )
    np.testing.assert_array_equal(dst.to_numpy(), np.full(n, 17))

    with pytest.raises(RuntimeError, match=r"sort\(\).*not differentiable.*FwdMode"):
        with ti.ad.FwdMode(loss=loss, param=param):
            ti.algorithms.sort(keys)
    np.testing.assert_array_equal(
        keys.to_numpy(),
        np.array([3, 1, 2, 0], dtype=np.int32),
    )

    scanner = ti.algorithms.PrefixSumExecutor(n)
    with pytest.raises(RuntimeError, match=r"PrefixSumExecutor\.run.*FwdMode"):
        with ti.ad.FwdMode(loss=loss, param=param):
            scanner.run(keys)
    np.testing.assert_array_equal(
        keys.to_numpy(),
        np.array([3, 1, 2, 0], dtype=np.int32),
    )


@test_utils.test(arch=ti.cpu)
def test_tape_rejects_discrete_primitive_before_writing():
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    keys = ti.field(ti.i32, shape=4)
    keys.from_numpy(np.array([3, 1, 2, 0], dtype=np.int32))

    with pytest.raises(RuntimeError, match=r"sort\(\).*not differentiable.*Tape"):
        with ti.ad.Tape(loss):
            ti.algorithms.sort(keys)
    np.testing.assert_array_equal(
        keys.to_numpy(),
        np.array([3, 1, 2, 0], dtype=np.int32),
    )
