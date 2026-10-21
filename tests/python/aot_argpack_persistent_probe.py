"""Consume immutable pack payloads and private arrays after producer retirement."""

import array
import copy
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


def read(value):
    result = array.array("f")
    result.frombytes(value.read())
    return result.tolist()


def write(value, numbers):
    value.write(array.array("f", numbers).tobytes())


def main(mode, arch, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime import aot

    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = aot.Runtime(arch, device_index=(int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0))
                                                 if arch == "vulkan" else 0))
    times = {}
    with runtime:
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        times["load_seconds"] = time.perf_counter() - start
        plan = module.execution_plan("step")
        assert plan.argument_names == ("gradient", "loss")
        frames = []
        start = time.perf_counter()
        for _ in range(2):
            loss, gradient = runtime.ndarray("f32", 1), runtime.ndarray("f32", 3)
            write(loss, [-91])
            write(gradient, [-91] * 3)
            prepared = plan.prepare(module, dict(loss=loss, gradient=gradient))
            assert read(loss) == [-91] and read(gradient) == [-91] * 3
            assert prepared.info["resource_bytes"] == 28
            assert prepared.info["native_resource_bytes"] > 0
            frames.append((loss, gradient, prepared))
        times["two_prepares_seconds"] = time.perf_counter() - start
        times["native_resource_bytes"] = frames[0][2].info["native_resource_bytes"]
        times["argument_bytes"] = frames[0][2].info["argument_bytes"]
        first_loss, first_grad, first = frames[0]
        second_loss, second_grad, second = frames[1]
        start = time.perf_counter()
        first.run()
        runtime.wait()
        times["first_complete_seconds"] = time.perf_counter() - start
        assert read(first_loss) == [30] and read(first_grad) == [5, -7, 13]
        first.run()
        second.run()
        runtime.wait()
        assert read(first_loss) == [47] and read(first_grad) == [9, -3, 17]
        assert read(second_loss) == [30] and read(second_grad) == [5, -7, 13]
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            second.run()
            runtime.wait()
            samples.append(time.perf_counter() - start)
        times["warm_complete_seconds"] = statistics.median(samples)

        # Failed cold reconstruction cannot modify outputs or poison live owners.
        document = json.loads(plan.to_json())
        if arch == "cuda":
            shared = copy.deepcopy(document)
            shared["version"] = max(shared["version"], 10)
            shared["binding_strategy"] = "cuda_immutable_argument_frames"
            a, da = runtime.ndarray("f32", 1), runtime.ndarray("f32", 3)
            b, db = runtime.ndarray("f32", 1), runtime.ndarray("f32", 3)
            family = aot.ExecutionPlan.from_json(json.dumps(shared)).prepare(module, dict(loss=a, gradient=da))
            sibling = family.prepare_binding(dict(loss=b, gradient=db))
            family.run()
            family.close()
            assert read(a) == [30] and read(da) == [5, -7, 13]
            sibling.run()
            sibling.close()
            assert read(b) == [47] and read(db) == [9, -3, 17]
        for patch in (dict(entry="missing"), dict(argument=1),
                      dict(paths=[[99], [1, 0]]), dict(paths=[[0], [0]])):
            invalid = copy.deepcopy(document)
            pack = next(item for item in invalid["resources"] if item["name"] == "pack")
            pack["parameters"].update(patch)
            old_loss, old_grad = read(first_loss), read(first_grad)
            reject((ValueError, aot.AotError), lambda: aot.ExecutionPlan.from_json(json.dumps(invalid)).prepare(
                module, dict(loss=first_loss, gradient=first_grad)))
            assert read(first_loss) == old_loss and read(first_grad) == old_grad

        other = runtime.load_module(artifact)
        changed = other.prepare_execution("other_values", dict(loss=first_loss, gradient=first_grad))
        changed.run()
        runtime.wait()
        assert read(first_loss) == [40] and read(first_grad) == [5, -13, 17]
        copied = runtime.ndarray("f32", 3)
        empty = other.prepare_execution("resource_only", dict(output=copied))
        empty.run()
        runtime.wait()
        assert read(copied) == [1, -2, 3]
        first.run()
        module.close()  # Submitted work retires before reconstructed storage.
        assert read(first_loss) == [76] and read(first_grad) == [13, 1, 21]
        reject(aot.AotError, first.run)
        reject(aot.AotError, second.run)
        changed.run()
        first_loss.close()  # Explicit dependency close completes pending work.
        assert read(first_grad) == [11, -7, 23]
        reject(aot.AotError, changed.run)
        empty.run()  # Borrower close also retires a pending reconstructed pack.
    if mode == "borrowed":
        ti.reset()
    print("PERSISTENT_ARGPACK_PASSED", json.dumps(dict(mode=mode, arch=arch, **times)))


if __name__ == "__main__":
    main(*sys.argv[1:])
