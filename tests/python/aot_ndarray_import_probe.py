"""Small shared-storage and explicit-gradient gate, isolated for owner reset."""

import array
import gc
import sys
import weakref

import taichi_forge as ti
from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime


def read(value):
    result = array.array("f")
    result.frombytes(value.read())
    return result.tolist()


def raises(kind, operation):
    try:
        operation()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def main(arch, artifact):
    ti.init(arch=getattr(ti, arch), offline_cache=False)
    source = ti.ndarray(ti.f32, 17, needs_grad=True)
    output = ti.ndarray(ti.f32, 17)
    output.fill(0)
    runtime = ti.aot.borrow_runtime()
    second = ti.aot.borrow_runtime()
    module = runtime.load_module(artifact)
    x = runtime.import_ndarray(source)
    y = runtime.import_ndarray(output)
    dx = runtime.import_ndarray(source.grad)
    shared = second.import_ndarray(source)
    plan = module.execution_plan("differentiate").prepare(module, {"x": x, "y": y, "dx": dx})

    @ti.kernel
    def next_input(a: ti.types.ndarray(dtype=ti.f32, ndim=1), b: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in a:
            a[i] = b[i] + 1

    # Deliberately no host wait between JIT producers and AOT consumers. Later
    # iterations consume previous AOT outputs through the same physical storage.
    for _ in range(3):
        next_input(source, output)
        plan()
    assert read(y) == [675] * 17
    assert source.grad.to_numpy().tolist() == [52] * 17
    assert output.to_numpy().tolist() == read(y)
    shared.write(array.array("f", [7] * 17))
    assert source.to_numpy().tolist() == [7] * 17
    assert read(x) == [7] * 17

    # Native metadata is authoritative even if Python metadata was changed.
    old_shape = source.shape
    source.shape = (1,)
    try:
        metadata_view = runtime.import_ndarray(source)
        assert metadata_view.shape == (17,)
        metadata_view.close()
    finally:
        source.shape = old_shape

    plan()
    x.close()  # Retires its in-flight AOT plan, not the source or second view.
    raises(AotError, plan)
    assert read(shared) == [7] * 17
    next_input(source, output)
    assert source.to_numpy().tolist() == [64] * 17
    second.close()
    source.fill(2)
    assert source.to_numpy().tolist() == [2] * 17

    # The owner lease, not a retained Python ndarray, keeps storage alive. Two
    # imports create no native ndarray and release the source allocation once.
    temporary = ti.ndarray(ti.f32, 5)
    temporary.fill(3)
    ti.sync()
    program = ti.lang.impl.get_runtime().prog
    before = program._debug_ndarray_resource_stats()
    first, last = (runtime.import_ndarray(temporary) for _ in range(2))
    assert program._debug_ndarray_resource_stats()["created_total"] == before["created_total"]
    weak = weakref.ref(temporary)
    del temporary
    gc.collect()
    assert weak() is None
    assert program._debug_ndarray_resource_stats()["released_total"] == before["released_total"]
    first.write(array.array("f", [4] * 5))
    first.close()
    assert read(last) == [4] * 5
    assert program._debug_ndarray_resource_stats()["released_total"] == before["released_total"]
    last.close()
    assert program._debug_ndarray_resource_stats()["released_total"] == before["released_total"] + 1

    tensor = ti.ndarray(ti.types.matrix(2, 2, ti.f32), 3)
    tensor_view = runtime.import_ndarray(tensor)
    assert tensor_view.shape == (3,)
    assert tensor_view.element_shape == (2, 2)
    tensor_view.write(array.array("f", range(12)))
    assert tensor.to_numpy().ravel().tolist() == list(range(12))
    fill = ExecutionPlan([{"kind": "fill32", "destination": "tensor", "value": 1065353216}]).prepare(
        module, {"tensor": tensor_view}
    )
    fill()
    assert tensor.to_numpy().ravel().tolist() == [1] * 12
    tensor_view.close()
    raises(AotError, fill)
    scalar = ti.ndarray(ti.f32, ())
    scalar_view = runtime.import_ndarray(scalar)
    scalar_view.write(array.array("f", [6]))
    assert scalar[None] == 6
    scalar_view.close()
    raises(TypeError, lambda: runtime.import_ndarray([1, 2, 3]))
    struct = ti.ndarray(ti.types.struct(value=ti.f32), 2)
    raises(TypeError, lambda: runtime.import_ndarray(struct))
    if arch == "cpu":
        with Runtime("cpu") as owned:
            raises(RuntimeError, lambda: owned.import_ndarray(source))
    if arch == "cuda" and runtime.probe_native_operation("gemm_f32")["available"]:
        a, b = (ti.ndarray(ti.f32, (3, 3)) for _ in range(2))
        a.fill(2)
        b.fill(3)
        av, bv = (runtime.import_ndarray(value) for value in (a, b))
        result = runtime.ndarray("f32", (3, 3))
        gemm = ExecutionPlan(
            [{"kind": "gemm_f32", "a": "a", "b": "b", "output": "c", "rows": 3, "columns": 3, "inner": 3}]
        )
        multiply = gemm.prepare(module, {"a": av, "b": bv, "c": result})
        multiply()
        assert read(result) == [18] * 9
        # Different import tokens must not hide physical read/write aliasing.
        alias = runtime.import_ndarray(a)
        raises(AotError, lambda: gemm.prepare(module, {"a": av, "b": bv, "c": alias}))
        multiply.close()
        for value in (av, bv, alias, result):
            value.close()

    fresh = runtime.import_ndarray(source)
    pending = module.execution_plan("differentiate").prepare(module, {"x": fresh, "y": y, "dx": dx})
    pending()
    ti.reset()
    raises(RuntimeError, pending)
    raises(RuntimeError, lambda: fresh.read())
    raises(RuntimeError, lambda: runtime.import_ndarray(source))
    for value in (fresh, x, y, dx, pending, plan, module, runtime):
        value.close()
        value.close()

    ti.init(arch=getattr(ti, arch), offline_cache=False)
    new_runtime = ti.aot.borrow_runtime()
    raises(RuntimeError, lambda: new_runtime.import_ndarray(source))
    new_source = ti.ndarray(ti.f32, 3)
    new_source.fill(8)
    imported = new_runtime.import_ndarray(new_source)
    assert read(imported) == [8] * 3
    new_runtime.close()
    new_source.fill(9)
    assert new_source.to_numpy().tolist() == [9] * 3
    ti.reset()
    print("ndarray import OK", arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
