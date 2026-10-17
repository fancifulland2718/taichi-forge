"""Consume initialized roots without producer code, including failed loads."""

import array
import json
from pathlib import Path
import sys
from zipfile import ZipFile


class NoFrontend:
    def find_spec(self, fullname, *_):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError(f"AOT consumer imported {fullname}")


def read(source):
    values = array.array("f")
    values.frombytes(source.read())
    return values.tolist()


def main(mode, arch, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime import aot

    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        sentinel = ti.field(ti.i32, shape=4)
        sentinel.fill(37)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = aot.Runtime(arch)
    with runtime:

        def expect_failure(call, kind=aot.AotError):
            try:
                call()
            except kind:
                return
            raise AssertionError("Expected AOT initialization/retirement failure")

        frames = [{name: runtime.ndarray("f32", 4) for name in ("y", "dx", "dy")} for _ in range(2)]
        for frame in frames:
            frame["dy"].write(array.array("f", [1] * 4))

        def execute(call, frame, step):
            # Reverse kernels consume output adjoints, so seed each invocation.
            frame["dy"].write(array.array("f", [1] * 4))
            call()
            assert read(frame["y"]) == [(1 + i + step) ** 2 + 10 + i + 2 * step for i in range(4)]
            assert read(frame["dx"]) == [2 * (1 + i + step) for i in range(4)]

        module = runtime.load_module(artifact)
        calls = [module.prepare_execution("step", frame) for frame in frames]
        execute(calls[0], frames[0], 0)
        execute(calls[1], frames[1], 1)  # Prepared calls share module Field state.
        calls[0].close()
        calls[0] = module.prepare_execution("step", frames[0])
        execute(calls[0], frames[0], 2)  # Re-preparation must not reinitialize.
        with runtime.load_module(artifact) as independent:
            with independent.prepare_execution("step", frames[0]) as call:
                execute(call, frames[0], 0)
        execute(calls[1], frames[1], 3)

        if mode == "standalone":
            with ZipFile(artifact) as archive:
                original = {name: archive.read(name) for name in archive.namelist()}
            metadata_name = "metadata.json" if arch == "vulkan" else "aot_metadata.json"
            version_key = "metadata_version" if arch == "vulkan" else "schema_version"
            for failure in ("missing", "corrupt", "size", "duplicate", "absent", "downgrade", "future"):
                files = dict(original)
                document = json.loads(files[metadata_name])
                entry = document["initial_root_data"][-1]
                data_name = "data/" + entry["sha256"] + ".bin"
                if failure == "missing":
                    del files[data_name]
                elif failure == "corrupt":
                    payload = files[data_name]
                    files[data_name] = bytes([payload[0] ^ 1]) + payload[1:]
                elif failure == "size":
                    entry["size"] += 4
                elif failure == "duplicate":
                    document["initial_root_data"].append(dict(entry))
                elif failure == "absent":
                    del document["initial_root_data"]
                elif failure == "downgrade":
                    document[version_key] -= 1
                else:
                    document[version_key] = 0xFFFFFFFF
                files[metadata_name] = json.dumps(document).encode()
                path = Path(artifact).with_name("invalid-" + failure + ".tcm")
                with ZipFile(path, "w") as archive:
                    for name, payload in files.items():
                        archive.writestr(name, payload)
                expect_failure(lambda: runtime.load_module(path))
                with runtime.load_module(artifact) as replacement:
                    with replacement.prepare_execution("step", frames[0]) as call:
                        execute(call, frames[0], 0)
            execute(calls[1], frames[1], 4)  # Failed loads cannot alter live roots.
        else:
            assert sentinel.to_numpy().tolist() == [37] * 4

        calls[0]()  # Close/reset while work may still be pending.
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        expect_failure(calls[1], RuntimeError)
        for call in calls:
            call.close()
    if mode == "standalone":
        assert not {"taichi", "taichi_forge", "numpy"} & set(sys.modules)


if __name__ == "__main__":
    main(*sys.argv[1:])
