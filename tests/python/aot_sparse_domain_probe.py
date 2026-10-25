"""Real sparse Field/control/adjoint consumers on two independently owned domains."""

import os
import sys
import json
import time

if sys.argv[2] == "standalone":
    class NoFrontend:
        def find_spec(self, fullname, *args):
            if fullname == "taichi_forge" or fullname.startswith("taichi_forge."):
                raise AssertionError("Runtime-only sparse consumer imported the compiler")
    sys.meta_path.insert(0, NoFrontend())

from taichi_forge_runtime.aot import Runtime, Scalar
from tests.python.aot_optix_probe import read, write


def rejected(action, *words):
    try:
        action()
    except (RuntimeError, ValueError) as error:
        assert any(word in str(error).lower() for word in words), str(error)
    else:
        raise AssertionError("Expected preparation/access rejection")


def main(arch, mode, artifact):
    assert arch in ("cuda", "vulkan")
    if mode == "borrowed":
        import taichi_forge as ti
        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch, **(dict(device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
                                   if arch == "vulkan" else {}))
    with runtime:
        modules = [runtime.load_module(artifact) for _ in range(2)]
        if arch == "cuda":
            # CUDA tree materialization may replace the source allocator.
            # Complete each module's cold tree closure before lending any
            # managers. This resolves code/resources without executing kernels.
            for module in modules:
                module.kernel("deactivate")
                module.kernel("initialize_only")
        domains = [runtime.execution_domain(capacity=2, instance_capacity=1) for _ in modules]
        frames = []
        for _ in modules:
            frame = {key: runtime.ndarray("f32", size)
                     for key, size in (("out", 3), ("loss", 1), ("gradient", 3))}
            frame["state"] = runtime.ndarray("i32", 2)
            frames.append(frame)
        probe = runtime.ndarray("i32", 1)
        # Failed native recording must return a tree used only by a branch.
        rejected(lambda: modules[0].prepare_execution("unsupported_branch", dict(state=probe),
                 execution_domain=domains[0]), "host", "control", "domain")
        modules[0].kernel("initialize_only")()
        modules[0].kernel("read_initial")(probe)
        assert read(probe, "i") == [71]
        initial = modules[0].prepare_execution("initial_only", dict(state=probe), execution_domain=domains[0])
        rejected(lambda: modules[0].kernel("read_initial")(probe), "loan", "conflict")
        initial.initialize()
        initial.submit().wait()
        initial.close()
        modules[0].kernel("read_initial")(probe)
        assert read(probe, "i") == [71]

        cases = [("loop", 0, 0), ("loop", 2, 2), ("loop", 10, 4), ("zero", 3, 0)]
        if arch == "vulkan":
            cases.append(("nested", 2, 2))
        costs = []
        for name, target, count in cases:
            calls, arguments = [], []
            for lane, (module, domain, frame) in enumerate(zip(modules, domains, frames)):
                write(frame["out"], [-99] * 3)
                args = dict(frame, target=Scalar("i32", target), scale=Scalar("f32", 2 + lane))
                if name == "nested":
                    args["outer_target"] = Scalar("i32", 2)
                arguments.append(args)
                if lane == 1 and name == "loop" and target == 0:
                    # This tree is free, but the target domain's instance pool
                    # is full. Its successfully acquired loan must roll back.
                    rejected(lambda: module.prepare_execution(name, args, execution_domain=domains[0]),
                             "capacity", "instance", "full")
                    module.kernel("deactivate")()
                started = time.perf_counter()
                calls.append(module.prepare_execution(name, args, execution_domain=domain))
                costs.append(dict(plan=name, target=target, lane=lane,
                                  prepare_ms=1000 * (time.perf_counter() - started),
                                  argument_bytes=calls[-1].info["argument_bytes"],
                                  workspace_bytes=calls[-1].info["workspace_bytes"]))
                assert read(frame["out"]) == [-99] * 3, "Prepare ran the user graph"
                rejected(lambda: module.kernel("deactivate")(), "loan", "conflict")
                calls[-1].initialize()
            # An already borrowed module cannot be prepared on another domain.
            # The failed request must not poison its existing reservation.
            rejected(lambda: modules[1].prepare_execution(name, arguments[1], execution_domain=domains[0]),
                     "loan", "conflict", "capacity", "instance")
            for _ in range(2):
                tickets = [call.submit() for call in calls]
                for ticket in reversed(tickets):
                    ticket.wait()
                for lane, frame in enumerate(frames):
                    scale = 2 + lane
                    assert read(frame["out"]) == ([3 * scale, 4 * scale, 2] if count else [0, 0, 0])
                    assert read(frame["loss"]) == [2.5 * scale * scale if count else 0]
                    assert read(frame["gradient"]) == [3 * scale if count else 0, 0, 0]
                    assert read(frame["state"], "i") == [count, (count + 1) * (2 if name == "nested" else 1)]
                for ticket in tickets:
                    ticket.close()
            calls[0].close()
            write(frames[0]["out"], [0] * 3)
            modules[0].kernel("gather")(frames[0]["out"])
            assert read(frames[0]["out"]) == ([6, 8, 2] if count else [0, 0, 0])
            calls[1].submit().wait()  # Peer stays usable after the first closes.
            if name == cases[-1][0]:
                pending = calls[1].submit()
                modules[1].close()  # In-flight nested traversal and adjoint retire.
                pending.wait()
                rejected(calls[1].submit, "retired", "closed", "live")
            calls[1].close()
        modules[0].close()
        modules[1].close()
        for domain in domains:
            domain.close()
        print("SPARSE_DOMAIN_COSTS=" + json.dumps(dict(mode=mode, cases=costs)))
    if mode == "borrowed":
        ti.reset()
    else:
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
