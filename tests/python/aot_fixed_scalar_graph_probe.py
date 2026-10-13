"""Execute final fixed Graph values and explicit derivatives without producer state."""

import struct
import sys

from taichi_forge_runtime.aot import Runtime
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
        frames = [{key: runtime.ndarray("f32", (17,)) for key in ("x", "y", "dx", "dy")} for _ in range(2)]
        for arrays in frames:
            arrays["encoded"] = runtime.ndarray("u64", (8,))
        prepared = [module.prepare_execution(name, arrays) for name, arrays in zip(("negative", "positive"), frames)]
        assert all(item.info["resource_bytes"] == 0 for item in prepared)
        for step in range(3):
            for gain, arrays, item in zip((-1.75, 0.5), frames, prepared):
                values = [(i - 8 + step) / 8 for i in range(17)]
                write(arrays["x"], values)
                item()
                near(read(arrays["y"]), [gain * x * x for x in values])
                near(read(arrays["dx"]), [2 * gain * x for x in values])
                assert struct.unpack("<" + "Q" * 8, arrays["encoded"].read()) == tuple(
                    [(1 << 64) - 1, (1 << 64) - 1, 1 << 31, 17] * 2
                )
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
