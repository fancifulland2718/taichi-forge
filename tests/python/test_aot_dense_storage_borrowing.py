"""Dense JIT ranges preserve offsets, owners and revocation in AOT consumers."""

import gc
import ctypes as C
import weakref

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_finite_loan_transfers_and_interop_preserve_range(tmp_path):
    from taichi_forge_runtime.aot import _c_api as abi

    source = ti.ndarray(ti.i32, 16)
    source.from_numpy(np.arange(16, dtype=np.int32))
    with ti.aot.borrow_runtime() as runtime:
        whole = runtime.import_ndarray(source)
        left = runtime.import_dense_view(
            ti.experimental.ndarray_view(source, slices=slice(2, 6))
        )
        right = runtime.import_dense_view(
            ti.experimental.ndarray_view(source, slices=slice(10, 14))
        )
        np.testing.assert_array_equal(
            np.frombuffer(left.read(), np.int32), np.arange(2, 6)
        )
        left.write(np.arange(102, 106, dtype=np.int32))
        expected = np.arange(16, dtype=np.int32)
        expected[2:6] += 100
        np.testing.assert_array_equal(source.to_numpy(), expected)

        def native(name, restype, *args):
            function = getattr(runtime._api.library, name)
            function.argtypes, function.restype = list(args), restype
            return function

        copy = native(
            "ti_copy_memory_device_to_device",
            None,
            C.c_void_p,
            C.POINTER(abi.MemorySlice),
            C.POINTER(abi.MemorySlice),
        )
        # Nonzero logical slice offsets must compose with each imported offset.
        src = abi.MemorySlice(left._memory._handle, 4, 8)
        dst = abi.MemorySlice(right._memory._handle, 4, 8)
        runtime._api.call(copy, runtime._handle, C.byref(dst), C.byref(src))
        runtime.wait()
        expected[11:13] = expected[3:5]
        np.testing.assert_array_equal(source.to_numpy(), expected)
        with pytest.raises(RuntimeError, match="capacity"):
            runtime._api.call(
                copy,
                runtime._handle,
                C.byref(dst),
                C.byref(abi.MemorySlice(left._memory._handle, 12, 8)),
            )

        if runtime.arch in ("x64", "arm64", "cuda"):

            class PointerInfo(C.Structure):
                _fields_ = [("ptr", C.c_void_p), ("size", C.c_uint64)]

            export = native(
                (
                    "ti_export_cuda_memory"
                    if runtime.arch == "cuda"
                    else "ti_export_cpu_memory"
                ),
                None,
                C.c_void_p,
                C.c_void_p,
                C.POINTER(PointerInfo),
            )
            parent, child = PointerInfo(), PointerInfo()
            runtime._api.call(
                export, runtime._handle, whole._memory._handle, C.byref(parent)
            )
            runtime._api.call(
                export, runtime._handle, left._memory._handle, C.byref(child)
            )
            assert child.ptr == parent.ptr + 8 and child.size == 16
            if runtime.arch != "cuda":
                mapping = native("ti_map_memory", C.c_void_p, C.c_void_p, C.c_void_p)
                unmap = native("ti_unmap_memory", None, C.c_void_p, C.c_void_p)
                pointer = runtime._api.call(
                    mapping, runtime._handle, left._memory._handle
                )
                try:
                    assert pointer == child.ptr
                    C.cast(pointer, C.POINTER(C.c_int32))[1] = 204
                    expected[3] = 204
                finally:
                    runtime._api.call(unmap, runtime._handle, left._memory._handle)
                np.testing.assert_array_equal(source.to_numpy(), expected)
        else:
            from tests.python.aot_external_memory_probe import VkMemoryInfo

            export = native(
                "ti_export_vulkan_memory",
                None,
                C.c_void_p,
                C.c_void_p,
                C.POINTER(VkMemoryInfo),
            )
            with pytest.raises(RuntimeError, match="buffer-relative view offset"):
                runtime._api.call(
                    export,
                    runtime._handle,
                    left._memory._handle,
                    C.byref(VkMemoryInfo()),
                )
            prefix = runtime.import_dense_view(
                ti.experimental.ndarray_view(source, slices=slice(0, 4))
            )
            info = VkMemoryInfo()
            runtime._api.call(
                export, runtime._handle, prefix._memory._handle, C.byref(info)
            )
            assert info.size == 16

        @ti.kernel
        def mutate(x: ti.types.ndarray(ti.i32, 1)):
            for i in x:
                x[i] += 7

        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            mutate, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.i32, ndim=1)
        )
        exported = ti.aot.Module()
        exported.add_graph("mutate", builder.compile())
        artifact = tmp_path / "finite-graph.tcm"
        exported.archive(str(artifact))
        module = runtime.load_module(artifact)
        bound = module.graph("mutate").bind(dict(x=left))
        for operation in (bound, bound.prepare):
            with pytest.raises(RuntimeError, match="Offset views require"):
                operation()
        runtime.wait()
        np.testing.assert_array_equal(source.to_numpy(), expected)


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
    argument = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "x", ti.types.vector(2, ti.i32), ndim=1
    )
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
