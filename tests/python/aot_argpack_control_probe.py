"""Check immutable uniform reads, actual suffix gradients and owner retirement."""
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


def write(value, numbers, code="f"):
    value.write(array.array(code, numbers).tobytes())


def read(value, code="f"):
    result = array.array(code)
    result.frombytes(value.read())
    return result.tolist()


def main(mode, arch, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime import aot
    if mode == "borrowed":
        import taichi_forge as ti
        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = aot.Runtime(arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0))
                              if arch == "vulkan" else 0)
    with runtime:
        module = runtime.load_module(artifact)
        frames = []
        names = ["flat", "zero", "rebound", "nested"]
        if arch == "vulkan":
            names += ["chained", "conditional"]
        for name in names:
            bindings = dict(loss=runtime.ndarray("f32", 1), gradient=runtime.ndarray("f32", 3),
                            limits=runtime.ndarray("i32", 2), counts=runtime.ndarray("i32", 2))
            write(bindings["loss"], [-91])
            begin = time.perf_counter()
            prepared = module.prepare_execution(name, bindings)
            prepare_seconds = time.perf_counter() - begin
            assert read(bindings["loss"]) == [-91]
            assert prepared.info["launches"] == 0
            frames.append((name, bindings, prepared))
            samples = []
            for limits in ([0, 2], [2, 0], [1, 2], [7, 7], [1, 1]):
                write(bindings["limits"], limits, "i")
                begin = time.perf_counter()
                prepared.run()
                runtime.wait()
                samples.append(time.perf_counter() - begin)
                check(name, bindings, limits)
            print(json.dumps(dict(mode=mode, arch=arch, plan=name, prepare_seconds=prepare_seconds,
                                  complete_seconds=statistics.median(samples[1:]), **prepared.info)))
        # Distinct immutable packs and private arrays cannot contaminate each other.
        for name, bindings, prepared in frames:
            write(bindings["limits"], [1, 2], "i")
            prepared.run()
        runtime.wait()
        for name, bindings, prepared in frames:
            check(name, bindings, [1, 2])
        # Module retirement must cover all recorded commands and uniforms.
        for _, _, prepared in frames:
            prepared.run()
        module.close()
        for name, bindings, prepared in frames:
            check(name, bindings, [1, 2])
            try:
                prepared.run()
            except (RuntimeError, aot.AotError):
                pass
            else:
                raise AssertionError("Retired pack control accepted a launch")
            prepared.close()
            for value in bindings.values():
                value.close()
    if mode == "borrowed":
        ti.reset()
    print("ARGPACK_CONTROL_PASSED", mode, arch)


def check(name, bindings, limits):
    outer = min(limits[0], 2) if name == "nested" else 0
    count = min(limits[1], 3) if name != "zero" else 0
    if name == "nested" and outer == 0:
        count = 0
    rounds = count * outer if name == "nested" else count
    gain, step = (3, 2) if name == "rebound" else (2, 1)
    values = [v + rounds * step for v in (1, -2, 3)]
    assert read(bindings["loss"]) == [gain * sum(v*v for v in values)], (name, limits, read(bindings["loss"]))
    assert read(bindings["gradient"]) == [2 * gain * v for v in values]
    assert read(bindings["counts"], "i") == [count, outer]


if __name__ == "__main__":
    main(*sys.argv[1:])
