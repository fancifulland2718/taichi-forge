import os
import gc
import weakref

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.hardware._cublaslt import CublasLtProvider
from taichi_forge.hardware._cublaslt import passive_status as cublaslt_passive_status
from taichi_forge.hardware._retained import retained_execution_contract
from taichi_forge.lang import impl
from tests import test_utils


def _provider_or_skip():
    library_path = os.environ.get("TI_FORGE_TEST_CUBLASLT_LIBRARY_PATH")
    try:
        return CublasLtProvider(library_path)
    except RuntimeError as exc:
        if library_path:
            raise
        pytest.skip(f"a compatible user-provided cuBLASLt is unavailable: {exc}")


def test_cublaslt_generic_probe_is_transient():
    library_path = os.environ.get("TI_FORGE_TEST_CUBLASLT_LIBRARY_PATH")
    if not library_path:
        pytest.skip("a user-provided cuBLASLt path is required")
    ti.reset()
    loaded_before = cublaslt_passive_status()["library_loaded"]

    report = ti.hardware.probe("cublaslt", library_path=library_path)
    operation = next(
        item
        for item in report.operations
        if item.descriptor.operation_id == "linalg.matmul.cublaslt_explicit"
    )

    assert operation.discovery == "available"
    assert operation.enablement == "disabled"
    assert operation.selection == "not_considered"
    assert operation.provider_abi == "taichi-forge-cublaslt-provider-c-abi2"
    assert operation.provider_version
    assert operation.native_facts["external_component_probed"]
    assert not operation.native_facts["provider_enablement_changed"]
    assert not operation.native_facts["provider_selection_changed"]
    assert cublaslt_passive_status()["library_loaded"] == loaded_before


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cublaslt_retained_single_matmul_and_graph():
    provider = _provider_or_skip()
    rows, inner, columns = 17, 13, 9
    rng = np.random.default_rng(20260827)
    a_values = (rng.standard_normal((rows, inner)) * 0.25).astype(np.float32)
    b_values = (rng.standard_normal((inner, columns)) * 0.25).astype(np.float32)
    initial = (rng.standard_normal((rows, columns)) * 0.1).astype(np.float32)
    a = ti.ndarray(ti.f32, shape=a_values.shape)
    b = ti.ndarray(ti.f32, shape=b_values.shape)
    output = ti.ndarray(ti.f32, shape=initial.shape)
    a.from_numpy(a_values)
    b.from_numpy(b_values)
    output.from_numpy(initial)

    plan = provider.plan(rows, columns, inner, alpha=1.5, beta=0.25)
    assert plan._plan_info.heuristic_calls == 1
    assert plan._plan_info.restoration_calls == 0
    assert provider._discovery_handle is None
    with pytest.raises(RuntimeError, match="live execution resources"):
        provider._runtime.close()
    with pytest.raises(ValueError, match="size_t"):
        provider.plan(rows, columns, inner, workspace_limit_bytes=1 << 64)
    program = impl.get_runtime().prog
    native_before = program._runtime_statistics_snapshot()["submission"][
        "native_submissions"
    ]
    plan.run(a=a, b=b, output=output)
    native_after = program._runtime_statistics_snapshot()["submission"][
        "native_submissions"
    ]
    assert native_after == native_before + 1
    ti.sync()
    np.testing.assert_allclose(
        output.to_numpy(),
        1.5 * (a_values @ b_values) + 0.25 * initial,
        rtol=2e-5,
        atol=2e-5,
    )

    output.from_numpy(initial)
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan)
    graph = builder.compile()
    graph.run({"a": a, "b": b, "output": output})
    ti.sync()
    np.testing.assert_allclose(
        output.to_numpy(),
        1.5 * (a_values @ b_values) + 0.25 * initial,
        rtol=2e-5,
        atol=2e-5,
    )
    assert graph._debug_info["native_count"] == 1

    contract = retained_execution_contract(plan)
    assert contract.identity.provider_id == "cublaslt"
    assert contract.automatic_selection_policy == "forbidden"
    assert contract.identity.to_dict()["problem_scope"] == {
        "batch_count": 1,
        "k": inner,
        "m": rows,
        "n": columns,
        "transpose_a": False,
        "transpose_b": False,
    }
    assert tuple(
        (item.name, item.amortization_scope) for item in contract.cost_model.fixed_costs
    ) == (
        ("provider_library_load", "process"),
        ("provider_handle", "provider_generation"),
        ("descriptors_heuristic_and_workspace", "provider_generation"),
        ("ctypes_dispatch", "invocation"),
        ("submission_registration", "invocation"),
    )
    assert contract.cost_model.scale_costs[0].dimensions == (
        "batch_count",
        "m",
        "n",
        "k",
    )
    assert plan.workspace_bytes <= plan.workspace_limit_bytes

    square = ti.ndarray(ti.f32, shape=(4, 4))
    square_plan = provider.plan(4, 4, 4)
    with pytest.raises(RuntimeError, match="must not alias"):
        square_plan.run(a=square, b=square, output=square)
    square_plan.close()
    plan.close()
    provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cublaslt_discovery_creation_failure_retains_cleanup(monkeypatch):
    import ctypes

    provider = _provider_or_skip()
    with monkeypatch.context() as failure:
        failure.setattr(provider._library, "create", lambda _out: 3)
        with pytest.raises(RuntimeError, match="discovery handle creation"):
            provider._handle
        assert provider._discovery_handle is None
    assert provider._handle.value  # A null failure can retry without leaking.
    provider.close()

    provider = _provider_or_skip()
    destroyed = []

    def partial_creation(out):
        ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 123
        return 3

    def retire(handle):
        destroyed.append(handle.value)
        return 0

    with monkeypatch.context() as failure:
        failure.setattr(provider._library, "create", partial_creation)
        failure.setattr(provider._library, "destroy", retire)
        with pytest.raises(RuntimeError, match="discovery handle creation"):
            provider._handle
        with pytest.raises(RuntimeError, match="close the provider"):
            provider._handle
        provider.close()
    assert destroyed == [123]
    assert provider.closed


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cublaslt_shared_plan_explicit_adjoint_and_kernel_grad():
    provider = _provider_or_skip()
    m, n, k = 5, 3, 7
    rng = np.random.default_rng(28)
    av = rng.uniform(-0.5, 0.5, (m, k)).astype(np.float32)
    bv = rng.uniform(-0.5, 0.5, (k, n)).astype(np.float32)
    a, b = ti.ndarray(ti.f32, av.shape), ti.ndarray(ti.f32, bv.shape)
    result = ti.ndarray(ti.f32, (m, n), needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    da, db = ti.ndarray(ti.f32, av.shape), ti.ndarray(ti.f32, bv.shape)
    a.from_numpy(av)
    b.from_numpy(bv)
    forward = provider.plan(m, n, k, alpha=0.75)
    adjoint_a = provider.plan(m, k, n, transpose_b=True, alpha=0.75)
    adjoint_b = provider.plan(k, n, m, transpose_a=True, alpha=0.75)

    @ti.kernel
    def objective(values: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        for i, j in values:
            loss[None] += values[i, j] * values[i, j]

    forward.run(a=a, b=b, output=result)
    objective(result)
    loss.grad[None] = 1
    objective.grad(result)
    adjoint_a.run(a=result.grad, b=b, output=da)
    adjoint_b.run(a=a, b=result.grad, output=db)
    expected = 0.75 * (av.astype(np.float64) @ bv.astype(np.float64))
    np.testing.assert_allclose(result.to_numpy(), expected, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(loss[None], np.sum(expected**2), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(
        da.to_numpy(), 1.5 * expected @ bv.T, atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(
        db.to_numpy(), 1.5 * av.T @ expected, atol=2e-6, rtol=2e-5
    )
    adjoint_b.close()
    adjoint_a.close()
    forward.close()
    provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cublaslt_retained_strided_batch_and_transpose():
    provider = _provider_or_skip()
    batch, rows, inner, columns = 3, 6, 4, 5
    rng = np.random.default_rng(20260828)
    a_values = rng.standard_normal((batch, inner, rows)).astype(np.float32)
    b_values = rng.standard_normal((batch, columns, inner)).astype(np.float32)
    a = ti.ndarray(ti.f32, shape=a_values.shape)
    b = ti.ndarray(ti.f32, shape=b_values.shape)
    output = ti.ndarray(ti.f32, shape=(batch, rows, columns))
    a.from_numpy(a_values)
    b.from_numpy(b_values)

    plan = provider.plan(
        rows,
        columns,
        inner,
        batch_count=batch,
        transpose_a=True,
        transpose_b=True,
    )
    plan.run(a=a, b=b, output=output)
    ti.sync()
    expected = np.matmul(np.swapaxes(a_values, -1, -2), np.swapaxes(b_values, -1, -2))
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-5, atol=2e-5)

    plan.close()
    provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cublaslt_reset_closes_runtime_handle_and_plan():
    provider = _provider_or_skip()
    plan = provider.plan(8, 8, 8)
    another = provider.plan(4, 4, 4)
    capture = another._capture()
    a = ti.ndarray(ti.f32, shape=(8, 8))
    b = ti.ndarray(ti.f32, shape=(8, 8))
    output = ti.ndarray(ti.f32, shape=(8, 8))
    plan.run(a=a, b=b, output=output)

    # Distinct resource owners can have identical inherited recording fields.
    # Both must remain in the provider's weak registry.
    assert len(provider._plans) == 2
    plan.close()
    with pytest.raises(RuntimeError, match="plans are live"):
        provider.close()

    ti.reset()

    assert plan.closed
    assert another.closed and capture.plan.closed
    assert provider.closed
    with pytest.raises(RuntimeError, match="previous Taichi runtime generation"):
        plan.run(a=a, b=b, output=output)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("batch", (1, 3))
def test_cublaslt_capture_feedback_and_plan_leases(batch, monkeypatch):
    from taichi_forge.hardware._cublaslt_capture import _MatmulCaptureRecipe

    provider = _provider_or_skip()
    rows, columns, inner = (512, 512, 512) if batch == 1 else (17, 9, 13)
    plan = provider.plan(
        rows, columns, inner, batch_count=batch, transpose_a=True, alpha=0.75, beta=0.25
    )
    rng = np.random.default_rng(781)
    av = rng.standard_normal(plan.a_shape).astype(np.float32)
    bv = rng.standard_normal(plan.b_shape).astype(np.float32)
    initial = rng.standard_normal(plan.output_shape).astype(np.float32)
    a, b, output = [
        ti.ndarray(ti.f32, shape=shape)
        for shape in (plan.a_shape, plan.b_shape, plan.output_shape)
    ]
    a.from_numpy(av)
    b.from_numpy(bv)
    output.from_numpy(initial)
    recording = plan._capture()
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    graph = builder.compile()
    bindings = graph.bind(dict(a=a, b=b, output=output))
    assert graph._spec.runtime_lifetime_leases == ()
    np.testing.assert_array_equal(output.to_numpy(), initial)

    def reject_python_dispatch(*_args, **_kwargs):
        raise AssertionError("replay must not perform Python provider work")

    with monkeypatch.context() as replay:
        replay.setattr(type(plan), "execute", reject_python_dispatch)
        replay.setattr(type(plan), "_validate_array", reject_python_dispatch)
        replay.setattr(type(recording), "execute", reject_python_dispatch)
        replay.setattr(type(provider), "_validate_lifetime", reject_python_dispatch)
        replay.setattr(_MatmulCaptureRecipe, "append_to_graph", reject_python_dispatch)
        for _ in range(4):
            graph.run(bindings)
    # Use an independent higher-precision reference; a second f32 BLAS
    # reduction can differ near cancellation on the larger workspace case.
    expected = initial.astype(np.float64)
    product = 0.75 * (
        np.swapaxes(av.astype(np.float64), -1, -2) @ bv.astype(np.float64)
    )
    for _ in range(4):
        expected = product + 0.25 * expected
    np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-5, atol=2e-5)
    stats = graph._graph_stats[0]
    assert stats["last_path"] == "cuda_exact_replay"
    assert stats["last_fallback_reason"] == "none"
    with pytest.raises(RuntimeError, match="capture leases"):
        plan.close()
    with pytest.raises(RuntimeError, match="plans are live"):
        provider.close()
    plan_ref = weakref.ref(plan)
    del plan, recording, builder
    gc.collect()
    assert plan_ref() is not None  # The compiled Graph retains the real plan.
    graph.run(bindings)
    np.testing.assert_allclose(
        output.to_numpy(), product + 0.25 * expected, rtol=2e-5, atol=2e-5
    )
    retained = plan_ref()
    del graph, bindings
    gc.collect()
    assert retained._capture_leases == 0
    retained.close()
    provider.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cublaslt_capture_rejects_bad_bindings_and_reset_invalidates():
    provider = _provider_or_skip()
    plan = provider.plan(8, 8, 8)
    recording = plan._capture()
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    graph = builder.compile()
    values = ti.ndarray(ti.f32, (8, 8))
    output = ti.ndarray(ti.f32, (8, 8))
    values.from_numpy(np.eye(8, dtype=np.float32))
    # Read/read alias is legal, but output alias and shape mismatch are not.
    with pytest.raises(RuntimeError):
        graph.run(dict(a=values, b=values, output=values))
    wrong = ti.ndarray(ti.f32, (4, 16))
    with pytest.raises(RuntimeError):
        graph.run(dict(a=wrong, b=values, output=output))
    graph.run(dict(a=values, b=values, output=output))
    np.testing.assert_array_equal(output.to_numpy(), np.eye(8, dtype=np.float32))
    ti.reset()
    assert plan.closed and provider.closed
    with pytest.raises(RuntimeError):
        graph.run(dict(a=values, b=values, output=output))


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize(
    "layout,epilogue", (("row_major", "identity"), ("transposed_column_major", "relu"))
)
def test_cublaslt_frozen_configuration_restores_without_heuristic(
    layout, epilogue, monkeypatch
):
    from dataclasses import replace
    from taichi_forge.hardware._cublaslt_algorithms import (
        _AlgorithmChoice,
        _MatmulRecipePlan,
    )

    provider = _provider_or_skip()
    semantics = dict(
        m=65,
        n=49,
        k=97,
        batch_count=2,
        transpose_a=True,
        transpose_b=True,
        alpha=0.75,
        beta=0.25,
        a="a",
        b="b",
        output="output",
    )
    description = _MatmulRecipePlan(
        provider,
        semantics,
        layout=layout,
        epilogue=epilogue,
        workspace_limit_bytes=32 << 20,
        shortlist_size=33,
        preparation_only=True,
    )
    assert description.workspace is None
    with pytest.raises(RuntimeError, match="materialized"):
        description._capture()
    choices = tuple(
        _AlgorithmChoice.from_dict(choice.to_dict()) for choice in description.choices
    )
    assert choices == description.choices
    description.close()
    generator = np.random.default_rng(99)
    host = [
        generator.standard_normal(shape).astype(np.float32)
        for shape in ((2, 97, 65), (2, 49, 97), (2, 65, 49))
    ]
    a, b, output = [ti.ndarray(ti.f32, shape=value.shape) for value in host]
    a.from_numpy(host[0])
    b.from_numpy(host[1])
    expected = (
        0.75
        * (
            host[0].astype(np.float64).transpose(0, 2, 1)
            @ host[1].astype(np.float64).transpose(0, 2, 1)
        )
        + 0.25 * host[2]
    )
    if epilogue == "relu":
        expected = np.maximum(expected, 0)

    def no_heuristic(*_args):
        raise AssertionError(
            "Selected-plan restoration must not rerun heuristic search"
        )

    with monkeypatch.context() as restoration:
        restoration.setattr(provider._library, "heuristic", no_heuristic)
        for choice in dict.fromkeys((choices[0], choices[-1])):
            plan = _MatmulRecipePlan(
                provider,
                semantics,
                layout=layout,
                epilogue=epilogue,
                workspace_limit_bytes=32 << 20,
                choice=choice,
            )
            assert plan.workspace_bytes == choice.workspace_bytes
            assert plan._plan_info.heuristic_calls == 0
            assert plan._plan_info.restoration_calls == 1
            assert plan._algorithm_choice == choice
            recording = plan._capture()
            builder = ti.graph.GraphBuilder()
            builder.append_native(recording)
            graph = builder.compile()
            bindings = graph.bind(dict(a=a, b=b, output=output))
            output.from_numpy(host[2])
            graph.run(bindings)
            np.testing.assert_allclose(
                output.to_numpy(), expected, rtol=2e-5, atol=2e-5
            )
            del graph, bindings, builder, recording
            gc.collect()
            plan.close()
        with pytest.raises(RuntimeError, match="workspace drifted"):
            _MatmulRecipePlan(
                provider,
                semantics,
                layout=layout,
                epilogue=epilogue,
                workspace_limit_bytes=32 << 20,
                choice=replace(
                    choices[0], workspace_bytes=choices[0].workspace_bytes + 256
                ),
            )
    provider.close()
