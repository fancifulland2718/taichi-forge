"""Zero policy, initialization ordering and ownership in independent consumers."""

import json
import sys

from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime
from tests.python.aot_cutensor_probe import near, raises
from tests.python.aot_graph_recording_probe import read, write


def check_bytes(runtime, module):
    # The application's typed copy covers odd lengths on Vulkan too, where
    # transfer commands require four-byte alignment. Initialization adds no kernel.
    def copy(source, destination):
        return dict(kind="kernel", entry="copy_bytes", arguments=[source, destination])

    for count in (0, 17):
        output = runtime.ndarray("u8", (count,))
        source = runtime.ndarray("u8", (count,))
        payload = bytes(range(1, count + 1))
        source.write(payload)
        output.write(bytes([93]) * count)
        resource = dict(
            kind="ndarray", name="state", dtype="u8", shape=[dict(array="shape_only", axis=0)], zero_initialize=True
        )
        resources = [
            resource,
            dict(kind="ndarray", name="shape_only", dtype="u8", shape=[count, 123456789], zero_initialize=True),
            dict(kind="ndarray", name="unused", dtype="f64", shape=[2147483647, 2147483647], zero_initialize=True),
        ]
        plan = ExecutionPlan(
            [copy("state", "out"), copy("input", "state")],
            resources=resources,
        )
        one = plan.prepare(module, dict(out=output, input=source))
        assert output.read() == bytes([93]) * count, "Preparation executed work"
        assert one.info["resource_bytes"] == (max(1, count) + 3) // 4 * 4
        assert one.info["dispatch_count"] == 2
        one()
        assert output.read() == bytes(count)
        one.initialize()  # Must not reset state after its first use.
        one()
        assert output.read() == payload

        # Explicit initialization runs after zeros and its input then retires.
        initialized = ExecutionPlan(
            [copy("state", "out")],
            resources=resources,
            initialization=[copy("initial", "state")],
        ).prepare(module, dict(initial=source, out=output))
        initialized.initialize()
        source.close()
        raises(one)
        one.close()
        initialized()
        assert output.read() == payload
        initialized.close()
        output.close()

    # Parse rejection must also happen in the native loader, not just Python.
    output = runtime.ndarray("u8", (17,))
    valid = ExecutionPlan([copy("state", "out")], resources=[dict(resource, shape=[17])])
    document = json.loads(valid.to_json())
    document["resources"][0]["zero_initialize"] = 1
    valid._json = json.dumps(document).encode()
    try:
        valid.prepare(module, dict(out=output))
    except AotError:
        pass
    else:
        raise AssertionError("Native parser accepted a non-boolean policy")
    output.close()


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        check_bytes(runtime, module)
        frames = [{key: runtime.ndarray("f32", (17,)) for key in ("x", "y", "dx", "dy")} for _ in range(2)]
        prepared = [module.prepare_execution("state", arrays) for arrays in frames]
        assert all(item.info["resource_bytes"] == 136 and item.info["dispatch_count"] == 3 for item in prepared)
        for factor, arrays in enumerate(frames, 1):
            write(arrays["x"], [factor * (i - 8) / 8 for i in range(17)])
            write(arrays["dy"], [1] * 17)
        for iteration in range(1, 4):
            for factor, arrays, item in zip((1, 2), frames, prepared):
                item()
                state = [iteration * factor * (i - 8) / 8 for i in range(17)]
                near(read(arrays["y"]), [x * x for x in state])
                near(read(arrays["dx"]), [2 * x for x in state])
        prepared[0].close()
        prepared[1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(prepared[1])
        prepared[1].close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
