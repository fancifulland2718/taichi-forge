"""Independent runtime-only consumption of nested values and resource members."""
import array
import importlib.abc
import json
import os
import statistics
import sys
import time


class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError(f"Unexpected frontend dependency: {fullname}")


def reject(exception, action):
    try:
        action()
    except exception:
        return
    raise AssertionError(f"Expected {exception}")


def write(value, numbers):
    value.write(array.array("f", numbers).tobytes())


def read(value):
    numbers = array.array("f")
    numbers.frombytes(value.read())
    return numbers.tolist()


def main(arch, artifact):
    sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime import aot

    times = {}
    with aot.Runtime(arch, device_index=(int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0))
                                        if arch == "vulkan" else 0)) as runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        forward, backward = module.kernel("automatic"), module.kernel("backward")
        times["load_lookup_seconds"] = time.perf_counter() - start
        x, dx = runtime.ndarray("f32", 3), runtime.ndarray("f32", 3)
        loss, dloss = runtime.ndarray("f32", 1), runtime.ndarray("f32", 1)
        write(x, [1, -2, 3])
        write(dx, [0, 0, 0])
        write(loss, [0])
        write(dloss, [1])
        values = dict(source=x, basis=[[3, 0], [0, 5]],
                      child=dict(settings=dict(scale=2, bias=[1, -1])))
        gradients = {("source",): dx}
        reject(aot.AotError, lambda: forward.argpack(0, values))
        reject(ValueError, lambda: forward.argpack(0, dict(source=x), gradients=gradients))
        reject(ValueError, lambda: forward.argpack(0, values, gradients={"unknown": dx}))
        start = time.perf_counter()
        pack = forward.argpack(0, values, gradients=gradients)
        times["pack_seconds"] = time.perf_counter() - start
        values["basis"][0][0] = values["child"]["settings"]["scale"] = 99
        call = forward.bind(pack, loss, gradients={1: dloss})
        reverse = backward.bind(pack, loss, gradients={1: dloss})
        start = time.perf_counter()
        call.run()
        reverse.run()
        runtime.wait()
        times["ordinary_forward_reverse_seconds"] = time.perf_counter() - start
        assert read(loss) == [106]
        assert read(dx) == [13, -41, 37]

        plan = aot.ExecutionPlan([
            dict(kind="fill32", destination="total", value=0),
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="kernel", entry="automatic", arguments=["pack", "total"]),
            dict(kind="kernel", entry="backward", arguments=["pack", "total"]),
        ])
        planned = plan.prepare(module, dict(pack=pack, total=loss, dx=dx),
                               gradients={"total": dloss})
        assert read(loss) == [106]  # Recording the fills must not execute them.
        planned.run()
        runtime.wait()
        assert read(loss) == [106]
        assert read(dx) == [13, -41, 37]
        planned.run()
        runtime.wait()
        assert read(loss) == [106]
        assert read(dx) == [13, -41, 37]

        start = time.perf_counter()
        prepared, prepared_reverse = call.prepare(), reverse.prepare()
        times["prepare_forward_reverse_seconds"] = time.perf_counter() - start
        times["prepared_argument_bytes"] = prepared.info["argument_bytes"]
        assert read(loss) == [106]  # Preparation does not execute the kernel.
        write(loss, [0])
        write(dx, [0, 0, 0])
        start = time.perf_counter()
        prepared.run()
        prepared_reverse.run()
        runtime.wait()
        times["prepared_first_forward_reverse_seconds"] = time.perf_counter() - start
        assert read(loss) == [106]
        assert read(dx) == [13, -41, 37]

        for mode, entry in (("ordinary", call), ("prepared", prepared)):
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                entry.run()
                runtime.wait()
                samples.append(time.perf_counter() - start)
            times[mode + "_warm_complete_seconds"] = statistics.median(samples)

        values["basis"][0][0] = 3
        values["child"]["settings"]["scale"] = 3
        replacement = forward.argpack(0, values, gradients=gradients)
        other = runtime.load_module(artifact)
        next_forward, next_reverse = other.kernel("explicit"), other.kernel("backward")
        module.close()
        reject((RuntimeError, aot.AotError), prepared.run)
        reject((RuntimeError, aot.AotError), planned.run)
        write(x, [-1, 4, -2])
        write(dx, [0, 0, 0])
        write(loss, [0])
        next_call = next_forward.prepare(replacement, loss, gradients={1: dloss})
        next_adjoint = next_reverse.prepare(replacement, loss, gradients={1: dloss})
        next_plan = plan.prepare(other, dict(pack=replacement, total=loss, dx=dx),
                                 gradients={"total": dloss})
        next_plan.run()
        runtime.wait()
        assert read(loss) == [278]
        assert read(dx) == [-17, 119, -35]
        write(loss, [0])
        write(dx, [0, 0, 0])
        next_call.run()
        next_adjoint.run()
        # Closing the pack completes its in-flight work; no producer is needed.
        replacement.close()
        reject((RuntimeError, aot.AotError), next_call.run)
        reject((RuntimeError, aot.AotError), next_plan.run)
        assert read(loss) == [278]
        assert read(dx) == [-17, 119, -35]
        stale = next_forward.prepare(pack, loss, gradients={1: dloss})
        stale_plan = plan.prepare(other, dict(pack=pack, total=loss, dx=dx),
                                  gradients={"total": dloss})
        x.close()
        reject(aot.AotError, stale.run)
        reject(aot.AotError, stale_plan.run)
        assert read(loss) == [278]
        print(json.dumps(dict(arch=arch, **times)))
    print("ARGPACK_RUNTIME_PASSED")


if __name__ == "__main__":
    main(*sys.argv[1:])
