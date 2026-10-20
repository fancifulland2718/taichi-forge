"""Consume mixed-storage sorting and a real adjoint without producer sources."""

import array
import os
import sys


def main(mode, arch, artifact):
    if mode == "standalone":

        class BlockFrontend:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in ("taichi_forge", "taichi", "numpy", "geophys"):
                    raise AssertionError("Unexpected producer dependency: " + fullname)

        sys.meta_path.insert(0, BlockFrontend())
    from taichi_forge_runtime.aot import Runtime

    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0
        )

    def read(target, code):
        result = array.array(code)
        result.frombytes(target.read())
        return list(result)

    n = 17
    with runtime, runtime.load_module(artifact) as module:
        dtypes = dict(
            source_keys="i32", source="f32", mixed="i32", out_keys="i32", out_ids="i32", dx="f32", out_loss="f32"
        )
        arrays = {name: runtime.ndarray(dtype, 1 if name == "out_loss" else n) for name, dtype in dtypes.items()}
        for name in ("cold", "warm"):
            with module.prepare_execution(name, arrays) as call:
                for iteration in range(3):
                    keys = [(i * 17 + iteration) % 7 - 3 for i in range(n)]
                    source = [(i - 8) * (iteration + 1) / 8 for i in range(n)]
                    arrays["source_keys"].write(array.array("i", keys))
                    arrays["source"].write(array.array("f", source))
                    arrays["dx"].write(array.array("f", [float("nan")] * n))
                    call()
                    expected = sorted(range(n), key=keys.__getitem__)
                    assert read(arrays["out_keys"], "i") == [keys[i] for i in expected]
                    assert read(arrays["out_ids"], "i") == expected
                    gradient = [0.0] * n
                    for rank, index in enumerate(expected):
                        gradient[index] = 2 * (rank + 1) * source[index]
                    assert read(arrays["dx"], "f") == gradient
                    loss = sum((rank + 1) * source[index] ** 2 for rank, index in enumerate(expected))
                    assert abs(read(arrays["out_loss"], "f")[0] - loss) < 1e-4
    if mode == "borrowed":
        ti.reset()
    print("Mixed storage cold/warm replay and downstream adjoint passed:", mode, arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
