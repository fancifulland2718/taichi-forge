"""Stdlib-only ordinary cuDSS transaction consumer with an independent oracle."""

import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cudss_probe import near, raises, read, solve, write


def main(mode, artifact, adapter, vendor):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    values = [4, -1, -1, 4, -1, -1, 4, -1, -1, 3]
    with runtime:
        assert runtime.native_operation_capabilities("cudss_refactor_solve_f32") == {"cuda_ordered"}
        runtime.configure_provider("cudss", adapter_path=adapter, library_path=vendor)
        assert runtime.probe_native_operation("cudss_refactor_solve_f32")["available"]
        module = runtime.load_module(artifact)
        retained = []
        for name in ("analysis_only", "explicit_factors", "after_failure"):
            plan = module.execution_plan(name)
            for factor in (1, -0.5):
                frame = {
                    key: runtime.ndarray(
                        "f32", 10 if key in ("values", "live") else 1 if key in ("loss", "dloss") else 4
                    )
                    for key in plan.argument_names
                }
                write(frame["x"], [-99] * 4)
                call = plan.prepare(module, frame)
                assert call.info["execution_kind"] == "cuda_ordered"
                near(read(frame["x"]), [-99] * 4)
                for scale in (1, 2, 0.75):
                    rhs = [factor * (i + 1) for i in range(4)]
                    write(frame["rhs"], rhs)
                    write(frame["values"], [scale * value for value in values])
                    call()
                    x = solve(rhs, scale)
                    near(read(frame["x"]), x)
                    near(read(frame["dx"]), x)
                    near(read(frame["db"]), solve(x, scale))
                    near(read(frame["loss"]), [0.5 * sum(value * value for value in x)])
                alias = dict(frame, db=frame["dx"])
                raises(lambda: plan.prepare(module, alias))
                retained.append((frame, call))
        # Retire immediately after submission; other binding owners remain usable.
        retained[0][1]()
        retained[0][1].close()
        retained[-1][1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for frame, call in retained[1:]:
            raises(call)
            call.close()
        for frame, _call in retained:
            for array in frame.values():
                array.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
