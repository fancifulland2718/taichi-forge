"""A stdlib-only consumer of frozen control and explicitly exported suffix AD."""

import struct
import sys

from taichi_forge_runtime import aot


def frame(runtime, module, name="normal", target=2, scale=1):
    arrays = {n: runtime.ndarray("f32", 17) for n in ("input", "y", "gradient")}
    arrays["state"] = runtime.ndarray("i32", 3)
    arrays["input"].write(struct.pack("<17f", *[scale * (i - 8) for i in range(17)]))
    arrays["y"].write(struct.pack("<17f", *([-91] * 17)))
    bindings = dict(arrays, target=aot.Scalar("i32", target), inverted=aot.Scalar("i32", int(name == "inverted")))
    if name == "nested":
        bindings["outer_target"] = aot.Scalar("i32", 2)
    prepared = module.prepare_execution(name, bindings)
    assert list(struct.unpack("<17f", arrays["y"].read())) == [-91] * 17, "Prepare executed user work"
    assert prepared.info["launches"] == 0
    return arrays, prepared


def expect(arrays, count, target, scale=1, inverted=False, nested=False):
    actual = struct.unpack("<3i", arrays["state"].read())
    assert actual == (count, int(count < target) ^ int(inverted), (count + 1) * 2 if nested else count + 1), actual
    increment = (count + 1) * 2 if nested else count
    x = [scale * (i - 8) + increment for i in range(17)]
    assert list(struct.unpack("<17f", arrays["y"].read())) == [v * v for v in x]
    # The exported gradient is d energy / d final x. This is not an automatic
    # differentiation claim for the enclosing dynamic control structure.
    assert list(struct.unpack("<17f", arrays["gradient"].read())) == [2 * v for v in x]


def check(runtime, arch, path):
    module = runtime.load_module(path)
    if arch == "vulkan":
        try:
            frame(runtime, module)
        except aot.AotError as error:
            assert "selected structured recording" in str(error), str(error)
        else:
            raise AssertionError("Unsupported Vulkan control silently fell back")
        module.close()
        return
    # A changed scalar binding produces an independent prepared executable.
    first, one = frame(runtime, module, target=2)
    second, two = frame(runtime, module, target=4, scale=3)
    for _ in range(3):
        one()
        two()
        expect(first, 2, 2)
        expect(second, 4, 4, scale=3)
    assert one.info["launches"] == 3
    if arch == "cuda":
        assert one.info["execution_kind"] == "cuda_graph", one.info
        assert one.info["workspace_bytes"] == 32
    predicate_plan = aot.ExecutionPlan(
        [
            dict(
                kind="while",
                predicate="p",
                max_iterations=0,
                condition=[dict(kind="fill32", destination="p", value=0)],
                body=[dict(kind="fill32", destination="out", value=0)],
            )
        ]
    )
    for dtype, size in (("f32", 1), ("i32", 2)):
        invalid = runtime.ndarray(dtype, size)
        try:
            module.prepare_execution(predicate_plan, dict(p=invalid, out=first["y"]))
        except aot.AotError as error:
            assert "one aligned scalar i32" in str(error), str(error)
        else:
            raise AssertionError("Invalid predicate layout was accepted")
        invalid.close()
    expect(first, 2, 2)  # Failed preparation did not execute or modify outputs.
    for name, target, count in (
        ("normal", 0, 0),
        ("normal", 10, 4),
        ("zero", 3, 0),
        ("wide", 3, 3),
        ("inverted", 2, 2),
        ("nested", 2, 2),
    ):
        arrays, prepared = frame(runtime, module, name, target)
        prepared()
        expect(arrays, count, target, inverted=name == "inverted", nested=name == "nested")
        prepared.close()
        prepared.close()
        for value in arrays.values():
            value.close()
    # Closing dependencies retires prepared control including work still in flight.
    one()
    first["input"].close()
    try:
        one()
    except aot.AotError as error:
        assert "retired dependency" in str(error), str(error)
    else:
        raise AssertionError("Retired control accepted a closed input")
    one.close()
    two()
    module.close()
    two.close()
    for arrays in (first, second):
        for value in arrays.values():
            value.close()


if __name__ == "__main__":
    mode, arch, path = sys.argv[1:]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        with ti.aot.borrow_runtime() as runtime:
            check(runtime, arch, path)
            if arch != "vulkan":
                module = runtime.load_module(path)
                arrays, prepared = frame(runtime, module)
                prepared()
                ti.reset()
                try:
                    prepared()
                except (RuntimeError, aot.AotError):
                    pass
                else:
                    raise AssertionError("Owner reset did not retire persistent control")
                prepared.close()
    else:
        with aot.Runtime(arch) as runtime:
            check(runtime, arch, path)
        assert "taichi_forge" not in sys.modules
