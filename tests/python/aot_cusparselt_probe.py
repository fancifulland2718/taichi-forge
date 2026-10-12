"""Stdlib-only consumer; the sparse structure is fixed for each adjoint."""

import struct
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime
from tests.python.aot_cutensor_probe import near, raises

M, N, K = 32, 48, 64


def native(operation, bindings, **parameters):
    return dict(
        kind="native", provider="cusparselt", operation=operation, version=1, parameters=parameters, bindings=bindings
    )


def plan_resource(**options):
    parameters = dict(m=M, n=N, k=K)
    parameters.update(options)
    return dict(native("matmul_plan_f16", {}, **parameters), name="plan")


def compress(a):
    return native("compress_f16", dict(plan="plan", a=a))


def multiply(b, c, d, **parameters):
    return native("matmul_f16", dict(plan="plan", b=b, c=c, d=d), **parameters)


def write(value, entries, dtype="e"):
    value.write(struct.pack("<" + dtype * len(entries), *entries))


def read(value, dtype="e"):
    data = value.read()
    return list(struct.unpack("<" + dtype * (len(data) // struct.calcsize(dtype)), data))


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def inputs(factor, shift=0):
    a = [factor * (1 + (i + k) % 3) / 8 if (k + shift) % 4 < 2 else 0 for i in range(M) for k in range(K)]
    b = [((j * 3 + k) % 5 - 2) / 8 for j in range(N) for k in range(K)]
    c = [(i % 7 - 3) / 16 for i in range(M * N)]
    return a, b, c


def reference(a, b, c):
    product = [sum(a[i * K + k] * b[j * K + k] for k in range(K)) for i in range(M) for j in range(N)]
    d = [half(0.75 * value + 0.25 * seed) for value, seed in zip(product, c)]
    e = [half(-0.5 * value + 0.5 * seed) for value, seed in zip(product, c)]
    weights = [0.75 * x - 0.5 * y for x, y in zip(d, e)]
    da = [
        sum(weights[i * N + j] * b[j * K + k] for j in range(N)) if a[i * K + k] else 0
        for i in range(M)
        for k in range(K)
    ]
    db = [sum(weights[i * N + j] * a[i * K + k] for i in range(M)) for j in range(N) for k in range(K)]
    return dict(d=d, e=e, da=da, db=db, loss=[0.5 * sum(x * x + y * y for x, y in zip(d, e))])


def check(arrays, expected):
    for name, values in expected.items():
        near(read(arrays[name], "e" if name in ("d", "e") else "f"), values)


def main(mode, artifact, adapter, vendor):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        kind = "cusparselt_matmul_f16"
        assert runtime.native_operation_capabilities(kind) == {"cuda_capture"}
        assert not runtime.probe_native_operation(kind)["available"]
        runtime.configure_provider("cusparselt", adapter_path=adapter, library_path=vendor)
        assert runtime.probe_native_operation(kind)["available"]
        module = runtime.load_module(artifact)
        plan = module.execution_plan("shared_a")

        def frame(factor):
            shapes = dict(a=(M, K), b=(N, K), c=(M, N), d=(M, N), e=(M, N), da=(M, K), db=(N, K), loss=(1,))
            arrays = {key: runtime.ndarray("f16" if key in "abcde" else "f32", shape) for key, shape in shapes.items()}
            host = inputs(factor)
            for key, values in zip("abc", host):
                write(arrays[key], values)
            for key in ("d", "e"):
                write(arrays[key], [-77] * (M * N))
            run = plan.prepare(module, arrays)
            assert read(arrays["d"]) == [-77] * (M * N)
            assert run.info["execution_kind"] == "cuda_graph"
            assert run.info["workspace_bytes"] > 0
            return arrays, host, run

        left, host, run = frame(1)
        right, other, another = frame(-0.5)
        for _ in range(3):
            run()
            another()
        check(left, reference(*host))
        check(right, reference(*other))
        changed = inputs(0.5, shift=2)
        write(left["a"], changed[0])
        run()
        check(left, reference(*changed))
        check(right, reference(*other))

        operation = multiply("b", "c", "d", alpha=0.75, beta=0.25)
        for invalid, names in (([operation], "bcd"), ([operation, compress("a")], "abcd")):
            try:
                ExecutionPlan(invalid, resources=[plan_resource()]).prepare(module, {key: left[key] for key in names})
            except AotError as error:
                assert "preceding compression" in str(error)
            else:
                raise AssertionError("Uninitialized compressed data was accepted")
        check(left, reference(*changed))
        one = ExecutionPlan([compress("a"), operation], resources=[plan_resource()]).prepare(
            module, {key: left[key] for key in "abcd"}
        )
        # Extra uses of the same plan do not multiply retained scratch accounting.
        assert one.info["workspace_bytes"] == run.info["workspace_bytes"]
        one()
        near(read(left["d"]), reference(*changed)["d"])
        one.close()

        # Exercise the configured-plan ABI and actual same-shape alias checks.
        small = {name: runtime.ndarray("f16", (16, 16)) for name in "abcd"}
        a = [float(k % 4 < 2) for i in range(16) for k in range(16)]
        b = [float(i == j) for i in range(16) for j in range(16)]
        write(small["a"], a)
        write(small["b"], b)
        write(small["c"], [-2] * 256)
        write(small["d"], [-77] * 256)
        description = ExecutionPlan([compress("a"), operation], resources=[plan_resource(m=16, n=16, k=16, relu=True)])
        raises(lambda: description.prepare(module, dict(small, d=small["b"])))
        assert read(small["b"]) == b
        relu = description.prepare(module, small)
        relu()
        near(read(small["d"]), [max(0, 0.75 * value - 0.5) for value in a])
        relu.close()
        feedback = description.prepare(module, dict(small, d=small["c"]))
        expected = [-2] * 256
        for _ in range(2):
            feedback()
            expected = [half(max(0, 0.75 * value + 0.25 * previous)) for value, previous in zip(a, expected)]
            near(read(small["c"]), expected)
        feedback.close()

        runtime.configure_provider("cusparselt", adapter_path=None)
        assert not runtime.probe_native_operation(kind)["available"]
        raises(lambda: plan.prepare(module, left))
        run()
        check(left, reference(*changed))
        run()
        left["a"].close()
        raises(run)
        run.close()
        another()
        check(right, reference(*other))
        another()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(another)
        another.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
