"""Consume exported fixed recordings after the entire producer has retired."""

import sys
from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cusolverdn_probe import near, raises, read, solve, write


def main(mode, artifact, adapter, vendor, dtype, phase, count, alias):
    count, alias = int(count), alias == "True"
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    code = "f" if dtype == "f32" else "d"
    shape = (17,) if count == 1 else (count, 17)
    matrix = [4 if i == j else -1 if abs(i - j) == 1 else 0 for i in range(17) for j in range(17)]
    with runtime:
        runtime.configure_provider("cusolverdn", adapter_path=adapter, library_path=vendor)
        module = runtime.load_module(artifact)
        retained = []
        for name in ("cold", "selected"):
            plan = module.execution_plan(name)
            for sign in (1, -0.5):
                frame = {key: runtime.ndarray(dtype, shape) for key in ("source", "rhs", "result")}
                frame["solution"] = frame["rhs"] if alias else runtime.ndarray(dtype, shape)
                frame["a"] = runtime.ndarray(dtype, (17, 17))
                frame.update({key: runtime.ndarray("i32", (2,)) for key in ("status", "before", "after")})
                for key in ("status", "before", "after"):
                    write(frame[key], [-77, -77], "i")
                call = plan.prepare(module, frame)
                assert call.info["execution_kind"] == "cuda_ordered"
                assert read(frame["status"], "i") == [-77, -77]
                for step in range(2):
                    scale = step + 1 if phase == "factor_and_solve" else 1
                    source = [sign * (i + 1 + step) / (17 * count) for i in range(17 * count)]
                    write(frame["source"], source, code)
                    write(frame["a"], [scale * v for v in matrix] if phase == "factor_and_solve" else [0] * 289, code)
                    call()
                    expected = solve([2 * v for v in source], scale)
                    near(read(frame["solution"], code), expected, dtype)
                    near(read(frame["result"], code), [2 * v for v in expected], dtype)
                    assert read(frame["before"], "i") == (
                        ([-1, -1] if phase == "factor_and_solve" else [0, -1]) if step == 0 else [0, 0]
                    )
                    assert read(frame["after"], "i") == [0, 0]
                retained.append((frame, call))
        # Without a public status binding the provider owns one private slot;
        # a full-refactor recording needs no mutable producer snapshot.
        minimal = module.execution_plan("minimal")
        frame = retained[-1][0]
        args = {key: frame[key] for key in minimal.argument_names}
        write(args["a"], matrix if phase == "factor_and_solve" else [0] * 289, code)
        rhs = [(i + 1) / 17 for i in range(17 * count)]
        write(args["rhs"], rhs, code)
        with minimal.prepare(module, args) as call:
            call()
            near(read(args["solution"], code), solve(rhs, 1), dtype)
        retained[0][1]()
        retained[0][1].close()
        retained[-1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for frame, call in retained:
            raises(call)
            call.close()
            for value in set(frame.values()):
                value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
