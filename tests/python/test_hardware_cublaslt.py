import os

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
    assert operation.provider_abi == "cublaslt-dynamic-symbols-v1"
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
        ("provider_handle", "runtime_generation"),
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
    a = ti.ndarray(ti.f32, shape=(8, 8))
    b = ti.ndarray(ti.f32, shape=(8, 8))
    output = ti.ndarray(ti.f32, shape=(8, 8))
    plan.run(a=a, b=b, output=output)

    ti.reset()

    assert plan.closed
    assert provider.closed
    with pytest.raises(RuntimeError, match="previous Taichi runtime generation"):
        plan.run(a=a, b=b, output=output)
