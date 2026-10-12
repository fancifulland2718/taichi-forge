"""Bounded subprocess gate for shared AOT/JIT ownership and threaded teardown."""

import array
import gc
import sys
import threading
import weakref

import taichi_forge as ti
from taichi_forge_runtime.aot import AotError


def read(value, dtype="f"):
    result = array.array(dtype)
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
    marker = ti.field(ti.i32, shape=())
    marker[None] = 100
    jit_result = ti.ndarray(ti.i32, 17)

    @ti.kernel
    def jit_work(out: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in out:
            out[i] = marker[None] + i

    def check_jit():
        jit_work(jit_result)
        assert jit_result.to_numpy().tolist() == list(range(100, 117))

    check_jit()
    runtime = ti.aot.borrow_runtime()
    second = ti.aot.borrow_runtime()
    assert runtime._borrower.device_identity == second._borrower.device_identity
    module = runtime.load_module(artifact)
    other_module = runtime.load_module(artifact)
    unused_module = runtime.load_module(artifact)  # No entry handles are requested.
    x, y, dx, dy = (runtime.ndarray("f32", 17) for _ in range(4))
    values = list(range(1, 18))
    x.write(array.array("f", values))
    y.write(array.array("f", [-99] * 17))
    dx.write(array.array("f", [0] * 17))
    dy.write(array.array("f", [1] * 17))
    forward = module.kernel("forward")
    forward(x, y, gradients={0: dx, 1: dy})
    assert read(y) == [v * v + 2 * v for v in values]
    graph = module.graph("differentiate").prepare({"x": x, "y": y}, gradients={"x": dx, "y": dy})
    graph()
    assert read(dx) == [2 * v + 2 for v in values]
    dx.write(array.array("f", [-88] * 17))
    plan = module.execution_plan("private_gradient").prepare(module, {"x": x, "y": y, "dx": dx})
    assert read(dx) == [-88] * 17
    for _ in range(3):
        check_jit()
        plan()
        ti.sync()  # Shared owner synchronization completes AOT work too.
        assert read(dx) == [2 * v + 2 for v in values]

    counter = runtime.ndarray("i32", 1)
    first_count = module.kernel("next_count").prepare(counter)
    second_count = other_module.kernel("next_count").prepare(counter)
    first_count()
    assert read(counter, "i") == [1]
    second_count()
    assert read(counter, "i") == [1]
    first_count()
    assert read(counter, "i") == [2]
    assert marker[None] == 100
    raises(AotError, lambda: forward.prepare(x, y))
    raises(AotError, lambda: module.kernel("not_exported"))
    plan()
    second.close()
    second.close()
    check_jit()
    assert read(dx) == [2 * v + 2 for v in values]

    # Explicit retirement of in-flight code must leave the Program usable.
    other_module.close()
    raises(AotError, second_count)
    second_count.close()
    check_jit()

    # An unreferenced consumer retires even when pending calls formed a cycle.
    temporary = ti.aot.borrow_runtime()
    temp_module = temporary.load_module(artifact)
    temp_out = temporary.ndarray("i32", 1)
    temp_call = temp_module.kernel("next_count").prepare(temp_out)
    temp_call()
    weak = weakref.ref(temporary)
    del temporary, temp_module, temp_out, temp_call
    gc.collect()
    assert weak() is None
    check_jit()

    # Native finalize waits for an existing scope without holding the GIL.
    # Two finalizers must retire the same borrower only once.
    held, release, began, finished = (threading.Event() for _ in range(4))
    errors = []

    def submitter():
        try:
            with runtime._lock:
                plan()
                held.set()
                assert release.wait(5)
        except BaseException as error:
            errors.append(error)

    program = ti.lang.impl.get_runtime().prog

    def finalize():
        try:
            began.set()
            program.finalize()
            finished.set()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=submitter)
    worker.start()
    assert held.wait(5)
    finalizers = [threading.Thread(target=finalize) for _ in range(2)]
    for thread in finalizers:
        thread.start()
    assert began.wait(5)
    assert not finished.wait(0.02)
    release.set()
    for thread in [worker, *finalizers]:
        thread.join(5)
        assert not thread.is_alive()
    assert not errors, errors
    assert runtime._borrower.closed
    raises(RuntimeError, plan)
    raises(RuntimeError, lambda: x.read())
    raises(RuntimeError, lambda: runtime.load_module(artifact))
    plan.close()
    module.close()
    unused_module.close()
    runtime.close()
    runtime.close()
    ti.reset()

    # Ordinary Python reset uses the same native retirement contract.
    ti.init(arch=getattr(ti, arch), offline_cache=False)
    runtime = ti.aot.borrow_runtime()
    loaded = runtime.load_module(artifact)
    output = runtime.ndarray("i32", 1)
    pending = loaded.kernel("next_count").prepare(output)
    pending()
    ti.reset()
    raises(RuntimeError, pending)
    output.close()
    runtime.close()
    print("borrowed owner OK", arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
