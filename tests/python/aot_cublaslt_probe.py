"""No-DSL cuBLASLt consumer with scalar forward/adjoint and lifetime checks."""

import array
import copy
import json
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime


def native_plan(name, parameters):
    return dict(
        kind="native",
        name=name,
        provider="cublaslt",
        operation="matmul_plan_f32",
        version=1,
        parameters=parameters,
        bindings={},
    )


def matmul(plan, a, b, output):
    return dict(
        kind="native",
        provider="cublaslt",
        operation="matmul_f32",
        version=1,
        parameters={},
        bindings=dict(plan=plan, a=a, b=b, output=output),
    )


def read(value):
    result = array.array("f")
    result.frombytes(value.read())
    return result.tolist()


def write(value, entries):
    value.write(array.array("f", entries))


def near(actual, expected):
    assert len(actual) == len(expected)
    assert all(abs(a - b) <= 3e-4 * (1 + abs(b)) for a, b in zip(actual, expected)), (actual[:9], expected[:9])


def raises(call):
    try:
        call()
    except (AotError, ValueError, RuntimeError):
        return
    raise AssertionError("Invalid provider configuration or binding succeeded")


def main(mode, artifact, adapter, vendor):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        kind = "cublaslt_matmul_f32"
        assert runtime.native_operation_capabilities(kind) == {"cuda_capture"}
        assert not runtime.native_operation_capabilities("cublaslt_matmul_plan_f32")
        assert not runtime.probe_native_operation(kind)["available"]
        runtime.configure_provider("cublaslt", adapter_path=adapter, library_path=vendor)
        runtime.configure_provider("unused.provider", adapter_path="missing-unused-adapter")
        assert runtime.probe_native_operation(kind)["available"]
        if mode == "borrowed":
            with ti.aot.borrow_runtime() as sibling:
                assert not sibling.probe_native_operation(kind)["available"]
        module = runtime.load_module(artifact)
        plan = module.execution_plan("energy")
        parameters = json.loads(plan.to_json())["resources"][0]["parameters"]
        m, n, k, batch = (parameters[key] for key in ("m", "n", "k", "batch_count"))

        def frame(factor):
            def shape(rows, cols):
                return (rows, cols) if batch == 1 else (batch, rows, cols)

            shapes = dict(
                a=shape(m, k), b=shape(k, n), seed=shape(m, n), c=shape(m, n), da=shape(m, k), db=shape(k, n), loss=(1,)
            )
            arrays = {key: runtime.ndarray("f32", value) for key, value in shapes.items()}
            a = [factor * (i % 11 - 5) / 16 for i in range(batch * m * k)]
            b = [(i % 7 - 3) / 8 for i in range(batch * k * n)]
            c, da, db = [], [], []
            for z in range(batch):
                az, bz = a[z * m * k : (z + 1) * m * k], b[z * k * n : (z + 1) * k * n]
                cz = [
                    0.75 * sum(az[i * k + p] * bz[p * n + j] for p in range(k)) + 0.125
                    for i in range(m)
                    for j in range(n)
                ]
                if parameters["epilogue"] == "relu":
                    cz = [max(value, 0) for value in cz]
                c.extend(cz)
                da.extend(
                    0.75 * sum(cz[i * n + j] * bz[p * n + j] for j in range(n)) for i in range(m) for p in range(k)
                )
                db.extend(
                    0.75 * sum(az[i * k + p] * cz[i * n + j] for i in range(m)) for p in range(k) for j in range(n)
                )
            for key, values in dict(
                a=a, b=b, seed=[0.5] * len(c), c=[-99] * len(c), da=[-99] * len(a), db=[-99] * len(b), loss=[-99]
            ).items():
                write(arrays[key], values)
            run = plan.prepare(module, arrays)
            assert read(arrays["c"]) == [-99] * len(c)
            assert read(arrays["loss"]) == [-99]
            assert read(arrays["seed"]) == [0.5] * len(c)
            assert run.info["execution_kind"] == "cuda_graph"
            return arrays, (c, da, db, [0.5 * sum(value * value for value in c)]), run

        left, expected, run = frame(1)
        right, other, another = frame(-0.5)
        for _ in range(3):
            run()
            another()
        runtime.wait()
        for arrays, values in ((left, expected), (right, other)):
            for name, reference in zip(("c", "da", "db", "loss"), values):
                near(read(arrays[name]), reference)

        # Reconfiguration affects only future preparations, not live plan leases.
        runtime.configure_provider("cublaslt", adapter_path=adapter + ".missing", library_path=vendor)
        assert not runtime.probe_native_operation(kind)["available"]
        raises(lambda: plan.prepare(module, right))
        run()
        near(read(left["da"]), expected[1])
        runtime.configure_provider("cublaslt", adapter_path=None)
        another()
        near(read(right["db"]), other[2])
        runtime.configure_provider("cublaslt", adapter_path=adapter, library_path=vendor)

        square = module.execution_plan("square")
        description = json.loads(square.to_json())
        arrays = {key: runtime.ndarray("f32", (4, 4)) for key in "abc"}
        a = [float(i % 5 - 2) for i in range(16)]
        write(arrays["a"], a)
        write(arrays["b"], [float(i // 4 == i % 4) for i in range(16)])
        write(arrays["c"], [-77] * 16)
        raises(lambda: square.prepare(module, dict(arrays, c=arrays["a"])))
        assert read(arrays["a"]) == a
        for bad in (runtime.ndarray("i32", (4, 4)), runtime.ndarray("f32", (2, 8))):
            raises(lambda: square.prepare(module, dict(arrays, a=bad)))
            bad.close()
        for field in ("provider_version", "workspace", "algorithm_id"):
            wrong = copy.deepcopy(description)
            p = wrong["resources"][0]["parameters"]
            if field == "provider_version":
                p[field] += 1
            elif field == "workspace":
                p["algorithm"]["workspace_bytes"] += 1
            else:
                p["algorithm"]["configuration"]["id"] = 2147483647
            raises(lambda: ExecutionPlan.from_json(json.dumps(wrong)).prepare(module, arrays))
        assert read(arrays["c"]) == [-77] * 16
        call = square.prepare(module, arrays)
        call()
        near(read(arrays["c"]), [0.75 * value - 0.25 * 77 for value in a])
        call.close()
        # Two read-only inputs may alias, while beta reads the previous output.
        write(arrays["c"], [2] * 16)
        call = square.prepare(module, dict(arrays, b=arrays["a"]))
        reference = [2.0] * 16
        for _ in range(2):
            call()
            reference = [
                0.75 * sum(a[i * 4 + p] * a[p * 4 + j] for p in range(4)) + 0.25 * reference[i * 4 + j]
                for i in range(4)
                for j in range(4)
            ]
            near(read(arrays["c"]), reference)
        call.close()

        run()
        left["a"].close()
        raises(run)
        run.close()
        another()
        near(read(right["da"]), other[1])
        if mode == "borrowed":
            another()
            ti.reset()
            raises(another)
        else:
            module.close()
            raises(another)
        another.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
