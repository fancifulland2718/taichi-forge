"""Independent no-DXC consumer, stable-sort oracle and actual downstream AD."""

import array
import json
import os
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_binding_constraints_probe import failure, imported
from tests.python.aot_cutensor_probe import near


def write(value, data, dtype="I"):
    value.write(array.array(dtype, data))


def read(value, dtype="I"):
    data = array.array(dtype)
    data.frombytes(value.read())
    return list(data)


def main(mode, artifact):
    assert os.environ["TI_TEST_PARALLEL_SORT_DXC"] == "consumer-has-no-dxc.exe"
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
    with runtime:
        assert runtime.probe_native_operation("vulkan_parallel_sort_u32")["available"]
        runtime.configure_provider("unused", adapter_path="unused-missing-provider.dll")
        module = runtime.load_module(artifact)
        document = json.loads(module.execution_plan("step").to_json())
        native = [r for r in document["resources"] if r["kind"] == "native"]
        parameters = native[0]["parameters"]
        n, payload = parameters["key_count"], parameters["payload"]
        frames, calls = [], []
        for _ in range(2):
            frame = {
                name: runtime.ndarray(dtype, n if name != "loss" else 1)
                for name, dtype in (
                    ("source", "u32"),
                    ("keys", "u32"),
                    ("w", "f32"),
                    ("dw", "f32"),
                    ("loss", "f32"),
                )
            }
            if payload:
                frame["values"] = runtime.ndarray("u32", n)
            write(frame["keys"], [73] * n)
            frames.append(frame)
            calls.append(module.prepare_execution("step", frame))
            assert read(frame["keys"]) == [73] * n

        def check(call, frame, factor):
            inputs = [((n - i) * 7 if factor > 0 else i * 13) % 37 for i in range(n)]
            inputs[0] = 0xFFFFFFFF
            write(frame["source"], inputs)
            write(frame["w"], [factor] * n, "f")
            call()
            order = sorted(range(n), key=inputs.__getitem__)
            expected = [inputs[i] for i in order]
            assert read(frame["keys"]) == expected
            if payload:
                assert read(frame["values"]) == order
            coeff = [(x & 7) + 1 for x in (inputs if payload else expected)]
            near(read(frame["dw"], "f"), [v * factor for v in coeff])
            near(read(frame["loss"], "f"), [0.5 * sum(coeff) * factor * factor])

        for call, frame in zip(calls, frames):
            assert call.info["execution_kind"] == "vulkan_commands"
            groups = min((n + 511) // 512, 1024)
            reduced = groups * 16 if parameters["fused_prefix"] else 16 * ((groups + 511) // 512)
            workspace = n * 4 * (2 if payload else 1) + groups * 16 * 4 + reduced * 4
            assert call.info["workspace_bytes"] == workspace * len(native)
        for _ in range(2):
            check(calls[0], frames[0], 0.5)
            check(calls[1], frames[1], -0.25)
        for key, value, message in (
            ("key_count", n + 1, "u32"),
            ("payload", not payload, "payload"),
            ("shader_sizes", [v + 4 for v in parameters["shader_sizes"]], "blob size"),
        ):
            bad = json.loads(json.dumps(document))
            next(r for r in bad["resources"] if r["kind"] == "native")["parameters"][key] = value
            before = read(frames[0]["keys"])
            failure(
                lambda: module.prepare_execution(ExecutionPlan.from_json(json.dumps(bad)), frames[0]), message
            )
            assert read(frames[0]["keys"]) == before
        wrong = runtime.ndarray("i32", n)
        failure(lambda: module.prepare_execution("step", dict(frames[0], keys=wrong)), "u32")
        wrong.close()
        if payload:
            alias = imported(runtime, frames[0]["keys"], "vulkan")
            alias._dtype, alias._code = "u32", frames[0]["keys"]._code
            failure(lambda: module.prepare_execution("step", dict(frames[0], values=alias)), "overlap")
            alias.close()
        check(calls[0], frames[0], -0.25)
        calls[0].close()
        check(calls[1], frames[1], 0.5)
        calls[1]()  # Retire submitted native commands before the owner closes.
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
