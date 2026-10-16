"""A private gated artifact cannot be invoked through ordinary public calls."""

import json
from pathlib import Path
import struct
import sys
import time


def main(mode, directory):
    from taichi_forge_runtime.aot import AotError, ExecutionPlan, Runtime

    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, enable_fallback=False, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        begin = time.perf_counter()
        module = runtime.load_module(directory)
        costs = dict(load_ms=(time.perf_counter() - begin) * 1000)
        x, y, dx, dy = [runtime.ndarray("f32", 4) for _ in range(4)]
        x.write(struct.pack("<4f", 1, 2, 3, 4))
        y.write(struct.pack("<4f", -99, -99, -99, -99))
        dx.write(struct.pack("<4f", 0, 0, 0, 0))
        dy.write(struct.pack("<4f", 1, 1, 1, 1))

        def rejected(action):
            try:
                action()
            except AotError:
                pass
            else:
                raise AssertionError("An ordinary call accepted gated code")
            assert struct.unpack("<4f", y.read()) == (-99,) * 4

        begin = time.perf_counter()
        gated = module.kernel("gated_forward")
        costs["gated_load_ms"] = (time.perf_counter() - begin) * 1000
        rejected(lambda: gated(x, y, gradients={0: dx, 1: dy}))
        rejected(lambda: gated.prepare(x, y, gradients={0: dx, 1: dy}))
        plan = ExecutionPlan(
            [dict(kind="kernel", entry="gated_forward", arguments=["x", "y"])],
            gradients=dict(x="dx", y="dy"),
        )
        rejected(lambda: module.prepare_execution(plan, dict(x=x, y=y, dx=dx, dy=dy)))
        ordinary = module.kernel("forward").prepare(x, y, gradients={0: dx, 1: dy})
        ordinary()
        assert struct.unpack("<4f", y.read()) == (1, 4, 9, 16)
        module.kernel("reverse")(x, y, gradients={0: dx, 1: dy})
        assert struct.unpack("<4f", dx.read()) == (2, 4, 6, 8)
        ordinary.close()
        module.close()
        # Test declaration integrity before resource preparation, then prove
        # rejection did not poison the runtime for a subsequent valid load.
        path = Path(directory) / "aot_metadata.json"
        saved = path.read_bytes()
        metadata = json.loads(saved)
        metadata["launch_abi_sha256"] = "0" * 64
        path.write_text(json.dumps(metadata), encoding="utf-8")
        try:
            try:
                runtime.load_module(directory)
            except AotError as error:
                assert "checksum" in str(error), str(error)
            else:
                raise AssertionError("Corrupt gate declaration accepted")
        finally:
            path.write_bytes(saved)
        with runtime.load_module(directory) as reopened:
            reopened.kernel("forward")(x, y, gradients={0: dx, 1: dy})
            assert struct.unpack("<4f", y.read()) == (1, 4, 9, 16)
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    print(json.dumps(costs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
