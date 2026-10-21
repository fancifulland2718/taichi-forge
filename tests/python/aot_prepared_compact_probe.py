"""Bounded standalone/borrowed consumer of the default native adapter output."""

import array
from contextlib import ExitStack, closing
import importlib.abc
import sys


class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError("Unexpected frontend import: " + fullname)


def main(mode, arch, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime

        runtime = Runtime(arch)
    else:
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    with runtime, runtime.load_module(artifact) as module, ExitStack() as stack:
        n = 17
        types = dict(flags="i", count="i", values_0="i", values_1="Q", output_0="i", output_1="Q")

        def frame():
            return {
                name: stack.enter_context(
                    closing(
                        runtime.ndarray(
                            "u64" if code == "Q" else "i32",
                            3 if name == "count" else n + (3 if name.startswith("output") else 2),
                        )
                    )
                )
                for name, code in types.items()
            }

        frames = [frame(), frame()]
        calls = [stack.enter_context(closing(module.prepare_execution("select", f))) for f in frames]
        assert calls[0].info["workspace_bytes"] == calls[1].info["workspace_bytes"]
        for turn in range(3):
            for frame_index, (bindings, call) in enumerate(zip(frames, calls)):
                if turn == 2 and frame_index == 0:
                    continue
                mask = [0 if turn == 2 else int((i + frame_index + turn) % 3 != 0) for i in range(n)]
                data = dict(flags=[0, *mask, 0], count=[99, 99, 99])
                for column in range(2):
                    data[f"values_{column}"] = [(i + turn + (1 << 45 if column else 0)) for i in range(n + 2)]
                    data[f"output_{column}"] = [91] * (n + 3)
                for name, value in data.items():
                    bindings[name].write(array.array(types[name], value))
                call()
                data["count"][1] = sum(mask)
                for column in range(2):
                    chosen = [value for value, accepted in zip(data[f"values_{column}"][1 : n + 1], mask) if accepted]
                    data[f"output_{column}"][1 : 1 + len(chosen)] = chosen
                for name, expected in data.items():
                    actual = array.array(types[name])
                    actual.frombytes(bindings[name].read())
                    assert actual.tolist() == expected, (name, actual.tolist(), expected)
            if turn == 1:
                calls[0].close()


if __name__ == "__main__":
    main(*sys.argv[1:])
