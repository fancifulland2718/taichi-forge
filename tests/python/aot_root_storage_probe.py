"""Standalone/borrowed Field/native storage, actual AD and owner isolation."""

import array
import json
import os
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime


def write(value, values, code="f"):
    value.write(array.array(code, values))


def read(value, code="f"):
    result = array.array(code)
    result.frombytes(value.read())
    return list(result)


def rejected(call):
    try:
        call()
    except (RuntimeError, ValueError):
        return
    raise AssertionError("Invalid or retired operation was accepted")


def main(mode, arch, artifact, invalid_artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0
        )
    with runtime:
        modules = [runtime.load_module(artifact), runtime.load_module(artifact)]
        keys = [i % 5 - 2 for i in range(17)]
        expected_order = sorted(range(17), key=keys.__getitem__)

        def frame(module, scale):
            arrays = {name: runtime.ndarray("f32", (17,)) for name in ("source", "dx")}
            arrays.update({name: runtime.ndarray("i32", (17,)) for name in ("input_keys", "out_keys", "out_order")})
            arrays["out_loss"] = runtime.ndarray("f32", (1,))
            arrays["out_pair"] = runtime.ndarray("f32", (17,), element_shape=(2,))
            values = [scale * (i - 8) / 16 for i in range(17)]
            write(arrays["source"], values)
            write(arrays["input_keys"], keys, "i")
            write(arrays["out_loss"], [-99])
            call = module.prepare_execution("step", arrays)
            assert call.info["resource_bytes"] == 0, call.info
            assert read(arrays["out_loss"]) == [-99]
            return arrays, values, call

        one, values, first = frame(modules[0], 1)
        two, others, second = frame(modules[1], -0.25)
        for _ in range(3):
            first()
            second()
        for arrays, source in ((one, values), (two, others)):
            assert read(arrays["out_order"], "i") == expected_order
            assert read(arrays["out_keys"], "i") == sorted(keys)
            expected = sum((i + 1) * source[j] ** 2 for i, j in enumerate(expected_order))
            assert abs(read(arrays["out_loss"])[0] - expected) < 2e-4
            gradient = [0.0] * 17
            for i, j in enumerate(expected_order):
                gradient[j] = 2 * (i + 1) * source[j]
            assert max(abs(a - b) for a, b in zip(read(arrays["dx"]), gradient)) < 1e-5
            assert read(arrays["out_pair"]) == [component for value in source for component in (2 * value, 3 * value)]
        tail = runtime.ndarray("i32", (15,))
        call_tail = modules[0].prepare_execution("tail", {"tail_output": tail})
        call_tail()
        assert read(tail, "i") == sorted(keys)[2:]
        invalid = json.loads(modules[0].execution_plan("step").to_json())
        invalid["resources"][0]["view"] = "absent"
        rejected(lambda: ExecutionPlan.from_json(json.dumps(invalid)).prepare(modules[0], one))
        with runtime.load_module(invalid_artifact) as invalid_module:
            try:
                invalid_module.prepare_execution("step", one)
            except RuntimeError as error:
                assert "exceeds its module allocation" in str(error), str(error)
            else:
                raise AssertionError("An out-of-root view was accepted")
        first.close()
        # Releasing an alias handle must not free the shared root or sibling views.
        call_tail()
        assert read(tail, "i") == sorted(keys)[2:]
        modules[0].close()
        rejected(call_tail)
        call_tail.close()
        second()
        if mode == "borrowed":
            ti.reset()
        else:
            modules[1].close()
        rejected(second)
        second.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
