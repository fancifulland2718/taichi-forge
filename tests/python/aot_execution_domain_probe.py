"""Source-free public execution-domain consumer; also callable by a borrower."""

import array
import gc
import importlib.abc
import sys
import weakref


class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError(f"Independent AOT consumer imported {fullname}")


def check(runtime, folder):
    from taichi_forge_runtime.aot import AotError, Scalar

    module = runtime.load_module(folder)
    output = runtime.ndarray("f32", 64)
    other = runtime.ndarray("f32", 64)
    first = runtime.execution_domain(capacity=2, instance_capacity=8)
    second = runtime.execution_domain(capacity=2, instance_capacity=8)
    left = module.prepare_execution("public_stage", dict(out=output, seed=Scalar("f32", 2)), execution_domain=first)
    right = module.prepare_execution("public_stage", dict(out=other, seed=Scalar("f32", 7)), execution_domain=second)
    try:
        left.submit()
    except AotError as error:
        assert "initialized" in str(error)
    else:
        raise AssertionError("Selected-domain submit silently initialized")
    assert left.info["dispatch_count"] == 1 and left.info["execution_kind"] == "cuda_graph"
    left.initialize()
    right.initialize()
    a = left.submit()
    assert a is not None
    b = right.submit(wait_for=[a])
    assert b is not None
    b.wait()
    assert a.done() and b.done()
    assert a.info["owner"] == b.info["owner"] and a.info["sequence"] < b.info["sequence"]
    for target, scale in ((output, 11), (other, 36)):
        values = array.array("f")
        values.frombytes(target.read())
        assert values.tolist() == [scale * 64 * 65 // 2 + i for i in range(64)]

    # Dense SNode roots are actual saved module storage, not a substitute array.
    field = module.prepare_execution(
        "public_field", dict(out=output, seed=Scalar("f32", 2)), execution_domain=first
    )
    field.initialize()
    for _ in range(2):
        ticket = field.submit()
        assert ticket is not None
        ticket.wait()
        ticket.close()
    values = array.array("f")
    values.frombytes(output.read())
    assert values.tolist() == [2 * (11 * 64 * 65 // 2 + i) for i in range(64)]
    field.close()
    field = None

    # Releasing every wrapper does not drop modules/array owners. Native
    # capacity bounds residency until the domain's explicit cold close.
    module_ref, memory_ref = weakref.ref(module), weakref.ref(output._memory)
    left.close()
    right.close()
    del left, right, module, output, other, target
    gc.collect()
    assert module_ref() is not None and memory_ref() is not None
    first.close()
    second.close()
    gc.collect()
    assert module_ref() is None and memory_ref() is None
    assert a.done() and b.done()
    a.close()
    b.close()
    assert not runtime._domain_owners


if __name__ == "__main__":
    sys.meta_path.insert(0, NoFrontend())
    from cuda_cig_test_owner import configure_test_context
    configure_test_context()
    from taichi_forge_runtime.aot import Runtime

    with Runtime("cuda") as owner:
        check(owner, sys.argv[1])
    print("PUBLIC_AOT_DOMAIN_OK")
