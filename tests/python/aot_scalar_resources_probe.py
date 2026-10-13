"""Consume fixed scalar resources without the DSL or producer Python state."""

import json
import struct
import sys

from taichi_forge_runtime.aot import AotError, Runtime
from tests.python.aot_cutensor_probe import near, raises
from tests.python.aot_graph_recording_probe import read, write


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    with runtime:
        module = runtime.load_module(artifact)
        frames = [{key: runtime.ndarray("f32", (17,)) for key in ("x", "y", "dx")} for _ in range(2)]
        for arrays in frames:
            arrays["encoded"] = runtime.ndarray("u64", (3,))
        prepared = [module.prepare_execution(name, arrays) for name, arrays in zip(("negative", "positive"), frames)]
        assert all(item.info["resource_bytes"] == 68 for item in prepared)
        for step in range(3):
            for gain, arrays, item in zip((-1.75, 0.5), frames, prepared):
                values = [(i - 8 + step) / 8 for i in range(17)]
                write(arrays["x"], values)
                item()
                near(read(arrays["y"]), [gain * x * x for x in values])
                near(read(arrays["dx"]), [2 * gain * x for x in values])
                assert struct.unpack("<QQQ", arrays["encoded"].read()) == ((1 << 64) - 1, 1 << 63, 1 << 31)
        bits = module.prepare_execution("bits_only", dict(encoded=frames[0]["encoded"]))
        assert bits.info["resource_bytes"] == 0
        bits()
        assert struct.unpack("<QQQ", frames[0]["encoded"].read()) == ((1 << 64) - 1, 1 << 63, 1 << 31)
        invalid = module.execution_plan("bits_only")
        document = json.loads(invalid.to_json())
        next(value for value in document["resources"] if value["name"] == "wide")["bits"] = "fffffffffffffffff"
        invalid._json = json.dumps(document).encode()
        try:
            invalid.prepare(module, dict(encoded=frames[0]["encoded"]))
        except AotError:
            pass
        else:
            raise AssertionError("Native parser accepted an overflowing scalar payload")
        # A rejected preparation must not affect either already published owner.
        bits()
        assert struct.unpack("<QQQ", frames[0]["encoded"].read()) == ((1 << 64) - 1, 1 << 63, 1 << 31)
        if arch == "cpu":
            output = runtime.ndarray("i64", (6,))
            narrow = module.prepare_execution("narrow", dict(encoded=output))
            assert narrow.info["resource_bytes"] == 0
            narrow()
            assert struct.unpack("<qqqqqq", output.read()) == (-128, -32768, 255, 65535, 12, -(1 << 63))
            narrow.close()
        prepared[0].close()
        prepared[1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for item in (prepared[1], bits):
            raises(item)
            item.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
