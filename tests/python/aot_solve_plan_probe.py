"""Independent source-free and borrowed consumers of the same Krylov artifact."""

import struct
import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_binding_constraints_probe import failure, imported
from tests.python.aot_graph_recording_probe import read, write


def near(actual, expected, tolerance=2e-5):
    assert len(actual) == len(expected)
    assert all(abs(a - b) <= tolerance * max(1, abs(b)) for a, b in zip(actual, expected)), (actual, expected)


def edges(runtime, module):
    for case in ("zero", "budget", "breakdown"):
        name = case + "_gradient"
        names = module.execution_plan(name).argument_names
        state = next(n for n in names if n.endswith("_state"))
        metrics = next(n for n in names if n.endswith("_metrics"))
        frame = {
            n: runtime.ndarray("i32" if n == state else "f32", (4 if n in (state, metrics) else 12,)) for n in names
        }
        with module.prepare_execution(name, frame) as call:
            for nonzero in (True, False, True):
                write(frame["rhs"], [1 if nonzero else 0] * 12)
                write(frame["output"], [99] * 12)
                call()
                value = 0.3 if case == "budget" and nonzero else 0.0
                near(read(frame["output"]), [value] * 12)
                near(read(frame["energy"]), [value * value] * 12)
                near(read(frame["dx"]), [2 * value] * 12)
                terminal = struct.unpack("<4i", frame[state].read())
                status = (1 if case == "breakdown" else 0) if nonzero else 2
                assert terminal == (status, int(case == "budget" and nonzero), int(status == 1), 1), terminal
                expected_residual = 1.68 if case == "budget" else 12
                near(read(frame[metrics])[:3], [12, expected_residual, 12] if nonzero else [0, 0, 0])
        for array in frame.values():
            array.close()


def main(mode, arch, artifact, method):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        if method == "edges":
            edges(runtime, module)
            module.close()
            if mode == "standalone":
                assert "taichi_forge" not in sys.modules
            return
        names = module.execution_plan("solve").argument_names
        state_name = next(name for name in names if name.endswith("_state"))
        metrics_name = next(name for name in names if name.endswith("_metrics"))
        frames = [
            {
                name: runtime.ndarray(
                    "i32" if name == state_name else "f32", (4 if name in (state_name, metrics_name) else 12,)
                )
                for name in names
            }
            for _ in range(2)
        ]
        calls = [module.prepare_execution("solve", frame) for frame in frames]
        assert calls[0].info["resource_bytes"] > 5 * 12 * 4
        diagonal = [2, 3, 5] * 4

        def check(frame, expected, iterations):
            near(read(frame["output"]), expected)
            terminal = struct.unpack("<4i", frame[state_name].read())
            assert terminal == (2, iterations, 0, 1), terminal
            initial, residual, reference, tolerance = read(frame[metrics_name])
            assert residual <= max(tolerance, 1e-10), (residual, tolerance)
            assert initial >= 0 and reference >= 0

        for step in range(3):
            for index, (frame, call) in enumerate(zip(frames, calls)):
                # The second frame is an explicit adjoint solve A^T lambda=g
                # for this symmetric operator. It does not differentiate CG.
                rhs = [(i - 4 + step) / 8 for i in range(12)] if index == 0 else [1.0] * 12
                exact = [value / d for value, d in zip(rhs, diagonal)]
                write(frame["rhs"], rhs)
                write(frame["initial"], exact if step == 1 else [0] * 12)
                call()
                check(frame, exact, 0 if step == 1 else (3 if method == "cg" else 1))
                if index == 0:
                    # Metrics use the current invocation, not an old terminal.
                    near(read(frame[metrics_name])[2:3], [sum(v * v for v in rhs)])

        frame = frames[0]
        write(frame["output"], [99] * 12)
        wrong = runtime.ndarray("f32", (13,))
        failure(lambda: module.prepare_execution("solve", dict(frame, rhs=wrong)), "exact shape")
        wrong.close()
        alias = imported(runtime, frame["rhs"], arch)
        failure(lambda: module.prepare_execution("solve", dict(frame, output=alias)), "disjoint")
        alias.close()
        bad_terminal = runtime.ndarray("i32", (5,))
        failure(lambda: module.prepare_execution("solve", dict(frame, **{state_name: bad_terminal})), "exact shape")
        bad_terminal.close()
        assert read(frame["output"]) == [99] * 12

        rhs = [float(i - 3) for i in range(12)]
        exact = [v / d for v, d in zip(rhs, diagonal)]
        write(frame["rhs"], rhs)
        write(frame["output"], exact)
        same_view = imported(runtime, frame["output"], arch)
        with module.prepare_execution("again", dict(frame, initial=same_view)) as in_place:
            in_place()
            check(frame, exact, 0)
        same_view.close()
        if arch != "vulkan":
            storage = runtime.ndarray("f32", (24,))
            initial = imported(runtime, storage, arch, size=48)
            output = imported(runtime, storage, arch, offset=4, size=48)
            failure(lambda: module.prepare_execution("solve", dict(frame, initial=initial, output=output)), "same view")
            output.close()
            initial.close()
            storage.close()
        calls[0].close()
        calls[1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        failure(calls[1])
        calls[1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
