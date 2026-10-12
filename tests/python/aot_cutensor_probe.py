"""Independent scalar reference and no-DSL/borrowed cuTENSOR AOT consumer."""

import array
import copy
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime


def native_plan(name, shapes, modes, **options):
    parameters = {
        key: {"shape": list(shape), "modes": [ord(mode) for mode in mode_order]}
        for key, shape, mode_order in zip("abcd", shapes, modes)
    }
    parameters.update(options)
    return dict(
        kind="native",
        name=name,
        provider="cutensor",
        operation="contraction_plan_f32",
        version=1,
        parameters=parameters,
        bindings={},
    )


def contract(plan, a, b, c, d, **parameters):
    return dict(
        kind="native",
        provider="cutensor",
        operation="contract_f32",
        version=1,
        parameters=parameters,
        bindings=dict(plan=plan, a=a, b=b, c=c, d=d),
    )


def read(value):
    result = array.array("f")
    result.frombytes(value.read())
    return result.tolist()


def write(value, entries):
    value.write(array.array("f", entries))


def near(actual, expected):
    assert len(actual) == len(expected)
    assert all(abs(a - b) <= 5e-5 * (1 + abs(b)) for a, b in zip(actual, expected)), (actual[:9], expected[:9])


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
        kind = "cutensor_contract_f32"
        assert runtime.native_operation_capabilities(kind) == {"cuda_capture"}
        assert not runtime.native_operation_capabilities("cutensor_contraction_plan_f32")
        assert not runtime.probe_native_operation(kind)["available"]
        runtime.configure_provider("cutensor", adapter_path=adapter, library_path=vendor)
        runtime.configure_provider("unused.provider", adapter_path="missing-unused-adapter")
        assert runtime.probe_native_operation(kind)["available"]
        if mode == "borrowed":
            with ti.aot.borrow_runtime() as sibling:
                assert not sibling.probe_native_operation(kind)["available"]
        module = runtime.load_module(artifact)
        plan = module.execution_plan("contraction_energy")

        def frame(factor):
            shapes = dict(a=(5, 7), b=(7, 3), seed=(5, 3), c=(5, 3), da=(5, 7), db=(7, 3), loss=(1,))
            arrays = {key: runtime.ndarray("f32", shape) for key, shape in shapes.items()}
            a = [factor * (i % 11 - 5) / 8 for i in range(35)]
            b = [(i % 7 - 3) / 4 for i in range(21)]
            c = [1.25 * sum(a[i * 7 + p] * b[p * 3 + j] for p in range(7)) + 0.125 for i in range(5) for j in range(3)]
            da = [1.25 * sum(c[i * 3 + j] * b[p * 3 + j] for j in range(3)) for i in range(5) for p in range(7)]
            db = [1.25 * sum(a[i * 7 + p] * c[i * 3 + j] for i in range(5)) for p in range(7) for j in range(3)]
            for key, values in dict(
                a=a, b=b, seed=[0.5] * 15, c=[-99] * 15, da=[-99] * 35, db=[-99] * 21, loss=[-99]
            ).items():
                write(arrays[key], values)
            run = plan.prepare(module, arrays)
            assert read(arrays["c"]) == [-99] * 15
            assert read(arrays["loss"]) == [-99]
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

        # Configuration belongs to this owner and affects only future preparation.
        runtime.configure_provider("cutensor", adapter_path=adapter + ".missing", library_path=vendor)
        assert not runtime.probe_native_operation(kind)["available"]
        raises(lambda: plan.prepare(module, right))
        run()
        near(read(left["da"]), expected[1])
        runtime.configure_provider("cutensor", adapter_path=None)
        another()
        near(read(right["db"]), other[2])
        runtime.configure_provider("cutensor", adapter_path=adapter, library_path=vendor)

        # Square shapes make alias rejection independent of shape validation.
        shape = (3, 3)
        resource = native_plan("p", [shape] * 4, ["ik", "kj", "ji", "ji"])
        operation = contract("p", "a", "b", "c", "d", alpha=0.75, beta=0.25)
        local_plan = ExecutionPlan([operation], resources=[resource])
        arrays = {key: runtime.ndarray("f32", shape) for key in "abcd"}
        values = [float(i - 4) for i in range(9)]
        write(arrays["a"], values)
        write(arrays["b"], [float(i // 3 == i % 3) for i in range(9)])
        write(arrays["c"], [2] * 9)
        write(arrays["d"], [-77] * 9)
        raises(lambda: local_plan.prepare(module, dict(arrays, d=arrays["a"])))
        assert read(arrays["a"]) == values
        wrong = copy.deepcopy(resource)
        wrong["parameters"]["a"]["modes"] = [ord("i"), ord("i")]
        raises(lambda: ExecutionPlan([operation], resources=[wrong]).prepare(module, arrays))
        wrong = copy.deepcopy(resource)
        wrong["parameters"]["alignment_bytes"] = 3
        raises(lambda: ExecutionPlan([operation], resources=[wrong]).prepare(module, arrays))
        call = local_plan.prepare(module, arrays)
        call()
        near(read(arrays["d"]), [0.75 * values[j * 3 + i] + 0.5 for i in range(3) for j in range(3)])
        call.close()
        same = dict(arrays, d=arrays["c"])
        call = local_plan.prepare(module, same)
        reference = [2.0] * 9
        for _ in range(2):
            call()
            reference = [0.75 * values[j * 3 + i] + 0.25 * reference[i * 3 + j] for i in range(3) for j in range(3)]
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
