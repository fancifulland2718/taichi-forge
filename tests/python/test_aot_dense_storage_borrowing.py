"""Dense JIT ranges preserve offsets, owners and revocation in AOT consumers."""

import gc
import weakref

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_dense_field_loan_and_derived_alias_retire(tmp_path):
    field = ti.field(ti.i32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 12).place(field)
    tree = fields.finalize()
    field.from_numpy(np.arange(12, dtype=np.int32))
    view = ti.experimental.ndarray_view(field, slices=slice(2, 10))

    @ti.kernel
    def update(x: ti.types.ndarray(ti.i32, 1), out: ti.types.ndarray(ti.i32, 1)):
        for i in x:
            x[i] += 100
            out[i] = x[i]

    exported = ti.aot.Module()
    exported.add_kernel(update)
    artifact = tmp_path / "dense-loan.tcm"
    exported.archive(str(artifact))
    plan = ti.aot.ExecutionPlan(
        [{"kind": "kernel", "entry": "update", "arguments": ["middle", "out"]}],
        resources=[
            {
                "kind": "ndarray_view",
                "name": "middle",
                "array": "input",
                "offset": 2,
                "shape": [4],
            }
        ],
    )
    output = ti.ndarray(ti.i32, 4)
    with ti.aot.borrow_runtime() as runtime:
        imported = runtime.import_dense_view(view)
        target = runtime.import_ndarray(output)
        module = runtime.load_module(artifact)
        domain = runtime.execution_domain() if runtime.arch == "cuda" else None
        call = module.prepare_execution(
            plan, dict(input=imported, out=target), execution_domain=domain
        )
        call.initialize()
        if domain is None:
            call()
            runtime.wait()
        else:
            call.submit().wait()
        np.testing.assert_array_equal(output.to_numpy(), np.arange(4, 8) + 100)
        expected = np.arange(12, dtype=np.int32)
        expected[4:8] += 100
        np.testing.assert_array_equal(field.to_numpy(), expected)
        tree.destroy()
        if domain is not None:
            assert domain.info["closed"]
        with pytest.raises(
            (RuntimeError, ValueError), match="retired|live|closed|unknown"
        ):
            call.submit() if domain is not None else call()
        with pytest.raises(RuntimeError, match="retired|stale"):
            runtime.import_dense_view(view)
        # Revoked native handles remain safe to close, including derived views.
        call.close()
        imported.close()
        module.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_dense_ndarray_loan_outlives_source_and_rejects_unsafe_layout(tmp_path):
    source = ti.ndarray(ti.i32, 12)
    source.from_numpy(np.arange(12, dtype=np.int32))
    view = ti.experimental.ndarray_view(source, slices=slice(3, 7))
    source_ref = weakref.ref(source)
    with ti.aot.borrow_runtime() as runtime:
        with pytest.raises(RuntimeError, match="compact read-write AOS"):
            runtime.import_dense_view(
                ti.experimental.ndarray_view(source, slices=slice(0, 8, 2))
            )
        from taichi_forge.lang._storage_view import describe_storage

        with pytest.raises(RuntimeError, match="compact read-write AOS"):
            runtime._borrower.import_dense_view(
                describe_storage(source, access="read").descriptor
            )
        with (
            runtime._lock,
            pytest.raises(RuntimeError, match="outside a borrowed submission scope"),
        ):
            runtime.import_dense_view(view)
        imported = runtime.import_dense_view(view)
        del source, view
        gc.collect()
        assert source_ref() is None

        @ti.kernel
        def copy(x: ti.types.ndarray(ti.i32, 1), out: ti.types.ndarray(ti.i32, 1)):
            for i in x:
                out[i] = x[i] * 7

        exported = ti.aot.Module()
        exported.add_kernel(copy)
        artifact = tmp_path / "array-loan.tcm"
        exported.archive(str(artifact))
        output = ti.ndarray(ti.i32, 4)
        target = runtime.import_ndarray(output)
        module = runtime.load_module(artifact)
        module.kernel("copy")(imported, target)
        runtime.wait()
        np.testing.assert_array_equal(output.to_numpy(), np.arange(3, 7) * 7)
        imported.close()


@pytest.mark.parametrize("retained", [False, True])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_jit_dense_field_view_complete_instance(retained):
    field = ti.Vector.field(2, ti.i32)
    fields = ti.FieldsBuilder()
    fields.dense(ti.i, 8).place(field)
    tree = fields.finalize()
    field.fill(5)
    view = ti.experimental.ndarray_view(field, slices=slice(2, 6))

    @ti.kernel
    def update(x: ti.types.ndarray(ti.types.vector(2, ti.i32), 1)):
        for i in x:
            x[i] += ti.Vector([i, 7])

    builder = ti.graph.GraphBuilder()
    argument = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.types.vector(2, ti.i32), ndim=1)
    if retained:
        builder._bind_internal_runtime_arg(argument, view)
    builder.dispatch(update, argument)
    graph = builder.compile()
    domain = ti.graph.execution_domain()
    execution = graph.prepare({} if retained else dict(x=view), execution_domain=domain)
    execution.initialize()
    graph.close()
    execution.submit().wait()
    expected = np.full((8, 2), 5, dtype=np.int32)
    expected[2:6] += np.stack([np.arange(4), np.full(4, 7)], axis=1)
    np.testing.assert_array_equal(field.to_numpy(), expected)
    tree.destroy()
    assert domain.info["closed"]
    with pytest.raises(RuntimeError, match="closed|retired|live"):
        execution.submit()
    execution.close()
