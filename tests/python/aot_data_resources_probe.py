"""No-DSL and borrowed consumers of lazily verified artifact initial data."""

import json
from pathlib import Path
import shutil
import struct
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_binding_constraints_probe import failure
from tests.python.aot_cutensor_probe import near
from tests.python.aot_graph_recording_probe import read, write


def main(mode, arch, artifact, directory, key):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch)
    weights = [(i % 7 + 1) / 8 for i in range(17)]
    with runtime:
        module = runtime.load_module(artifact)
        frames = [{name: runtime.ndarray("f32", (17,)) for name in ("x", "y", "dx")} for _ in range(2)]
        calls = [module.prepare_execution("step", frame) for frame in frames]
        assert calls[0].info["resource_bytes"] == 3 * 17 * 4
        for step in range(1, 4):
            for index, (frame, call) in enumerate(zip(frames, calls)):
                values = [float(i - 8 + index) for i in range(17)]
                write(frame["x"], values)
                call()
                near(read(frame["y"]), [v * w + w + step for v, w in zip(values, weights)])
                near(read(frame["dx"]), weights)
        words = runtime.ndarray("i32", (3,))
        with module.prepare_execution("bytes", dict(out=words)) as copy_bytes:
            assert copy_bytes.info["resource_bytes"] == 4
            copy_bytes()
            assert struct.unpack("<3i", words.read()) == (1, 128, 255)
        empty = runtime.ndarray("f32", (0,))
        with module.prepare_execution("empty", dict(out=empty)) as empty_call:
            empty_call()
            runtime.wait()
        with module.prepare_execution("initialized", dict(out=frames[0]["y"])) as initialized:
            assert initialized.info["resource_bytes"] == 2 * 17 * 4
            initialized.initialize()
            assert initialized.info["resource_bytes"] == 17 * 4
            initialized()
            near(read(frames[0]["y"]), weights)

        # Entries are read on demand. Failure must not execute kernels or
        # invalidate previously prepared instances and unrelated entries.
        clone = Path(directory).with_name("corrupt-" + mode)
        shutil.copytree(directory, clone)
        loaded = runtime.load_module(clone)
        prepared = loaded.prepare_execution("step", frames[0])
        prepared()
        path = clone / "data" / (key + ".bin")
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        failure(lambda: loaded.prepare_execution("step", frames[0]), "integrity")
        prepared()
        near(read(frames[0]["y"]), [v * w + w + 2 for v, w in zip(range(-8, 9), weights)])
        with loaded.prepare_execution("empty", dict(out=empty)) as unaffected:
            unaffected()
        path.write_bytes(original[:-1])
        failure(lambda: loaded.prepare_execution("step", frames[0]), "size")
        path.unlink()
        failure(lambda: loaded.prepare_execution("step", frames[0]), "missing")
        path.write_bytes(original)
        mismatch = json.loads(loaded.execution_plan("step").to_json())
        next(r for r in mismatch["resources"] if r["name"] == "state")["shape"] = [18]
        failure(lambda: loaded.prepare_execution(ExecutionPlan.from_json(json.dumps(mismatch)), frames[0]), "size")
        prepared.close()
        loaded.close()
        calls[0].close()
        calls[1]()
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
