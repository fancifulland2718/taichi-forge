"""Runtime-only member reductions: binding, replay, root isolation and retirement."""

import array
import json
import os
from pathlib import Path
import struct
import sys

from aot_struct_consumer_probe import NoFrontend, read, reject


def members(runtime, aot, artifact):
    module = runtime.load_module(artifact)
    layout = module.ndarray_layout(json.loads(Path(artifact).with_suffix(".json").read_text())["layout"])
    stride = layout.element_stride
    offsets = {key: layout.member(key).offset for key in ("tag", "matrix", "leaf", "result", "tail")}
    initial = bytearray(17 * stride)
    for i in range(17):
        base = i * stride
        struct.pack_into("=i", initial, base + offsets["tag"], 37)
        struct.pack_into("=6f", initial, base + offsets["matrix"], *[(i * 6 + j) * .5 for j in range(6)])
        struct.pack_into("=f", initial, base + offsets["leaf"], 1 - i)
        struct.pack_into("=f", initial, base + offsets["result"], -9)
        struct.pack_into("=i", initial, base + offsets["tail"], 91)
    records = runtime.ndarray(layout, 17)
    frame = dict(records=records, out=runtime.ndarray("f32", 17),
                 low=runtime.ndarray("f32", 3), high=runtime.ndarray("f32", 3))

    def reset():
        records.write(initial)
        for key in ("out", "low", "high"):
            count = 17 if key == "out" else 3
            frame[key].write(array.array("f", [-13] * count).tobytes())

    def expected(count):
        return [sum((i * 6 + 5) * .5 for i in range(count))] + [-9] * 16

    def changed(count):
        payload = bytearray(initial)
        struct.pack_into("=f", payload, offsets["result"], expected(count)[0])
        return payload

    reset()
    for name in ("bad_strategy", "bad_extent", "bad_alias"):
        reject((RuntimeError, aot.AotError), lambda: module.prepare_execution(name, frame))
    for parent in (runtime.ndarray(layout, 16), runtime.ndarray(module.ndarray_layout("wrong"), 17)):
        reject((RuntimeError, aot.AotError),
               lambda: module.prepare_execution("cold", dict(frame, records=parent)))
        parent.close()
    assert records.read() == initial and read(frame["out"]) == [-13] * 17
    for name in ("cold", "warm", "parent", "prefix"):
        count = 11 if name == "prefix" else 17
        reset()
        with module.prepare_execution(name, frame) as call:
            call()
            call()
        assert read(frame["out"]) == expected(count)
        assert records.read() == changed(count)
        assert read(frame["low"]) == [2 - count, -13, -13]
        assert read(frame["high"]) == [1, -13, -13]
    with module.prepare_execution("private", {k: v for k, v in frame.items() if k != "records"}) as call:
        call()
    assert read(frame["out"]) == expected(17)
    assert records.read() == changed(11)
    suffix = dict(frame, **{name: runtime.ndarray("f32", 17) for name in ("energy", "dx", "dy")})
    with module.prepare_execution("suffix", suffix) as call:
        call()
    assert read(suffix["dx"]) == expected(17)
    assert read(suffix["energy"]) == [.5 * value * value for value in expected(17)]
    other_module = runtime.load_module(artifact)
    other = other_module.prepare_execution("private", {k: v for k, v in frame.items() if k != "records"})
    call = module.prepare_execution("cold", frame)
    ticket = call.submit()
    records.close()
    ticket.wait()
    ticket.close()
    reject((RuntimeError, aot.AotError), call)
    call.close()
    module.close()
    other()
    assert read(frame["out"]) == expected(17)
    other.close()
    other_module.close()


def fields(runtime, aot, artifact):
    modules = [runtime.load_module(artifact) for _ in range(2)]
    frames = [dict(out=runtime.ndarray("i32", 37)) for _ in modules]

    def check(frame, values):
        actual = array.array("i")
        actual.frombytes(frame["out"].read())
        assert list(actual) == values + list(range(-8, 9)) + [128] * 17

    for module, frame in zip(modules, frames):
        with module.prepare_execution("read", frame) as call:
            call()
        check(frame, [-9] * 3)
    for name in ("cold", "warm", "parent"):
        with modules[0].prepare_execution(name, frames[0]) as call:
            call()
            call()
        check(frames[0], [0, -8, 8])
    with modules[1].prepare_execution("prefix", frames[1]) as call:
        call()
    check(frames[1], [-33, -8, 2])
    call = modules[0].prepare_execution("cold", frames[0])
    ticket = call.submit()
    modules[0].close()
    ticket.wait()
    ticket.close()
    reject((RuntimeError, aot.AotError), call)
    call.close()
    with modules[1].prepare_execution("read", frames[1]) as call:
        call()
    check(frames[1], [-33, -8, 2])
    modules[1].close()


def main(mode, arch, artifact, kind):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot
        runtime = aot.Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0)) if arch == "vulkan" else 0)
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot
        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    with runtime:
        (members if kind == "members" else fields)(runtime, aot, artifact)
    print("Strided reduction, prefix, parent, isolation and retirement passed", mode, arch, kind)


if __name__ == "__main__":
    main(*sys.argv[1:])
