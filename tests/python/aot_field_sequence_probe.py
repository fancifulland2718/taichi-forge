"""Source-free consumer checks first-payload semantics and real exported AD."""

import array
import os
import sys

from taichi_forge_runtime.aot import Runtime


def put(target, values, code):
    target.write(array.array(code, values))


def get(target, code):
    data = array.array(code)
    data.frombytes(target.read())
    return list(data)


def main(mode, arch, artifact, native_artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0
        )
    n = 17
    keys = [7, 2, 7, 1, 2, 9, 1, 7, 4, 9, 3, 4, 2, 3, 1, 9, 2]
    sentinel = (1 << 64) - 1
    with runtime, runtime.load_module(artifact) as module:
        dtypes = dict(
            source_keys="u64",
            source_flags="i32",
            source="f32",
            out_keys="u64",
            out_ids="i32",
            stats="i32",
            out_loss="f32",
            dx="f32",
        )
        arrays = {
            name: runtime.ndarray(dtype, 3 if name == "stats" else 1 if name == "out_loss" else n)
            for name, dtype in dtypes.items()
        }
        call = module.prepare_execution("step", arrays)
        assert call.info["resource_bytes"] == n * 4, call.info  # Unique flags only.
        for iteration, flags in enumerate(
            ([i % 3 != 0 for i in range(n)], [1] * n, [0] * n, [i % 2 for i in range(n)])
        ):
            values = [(i - 8) * (iteration + 1) / 16 for i in range(n)]
            put(arrays["source_keys"], keys, "Q")
            put(arrays["source_flags"], [int(x) for x in flags], "i")
            put(arrays["source"], values, "f")
            call()
            selected = sorted((i for i in range(n) if flags[i]), key=keys.__getitem__)
            winners = []
            for i in selected:
                if not winners or keys[i] != keys[winners[-1]]:
                    winners.append(i)
            count = len(winners) + int(len(selected) < n)
            assert get(arrays["stats"], "i") == [len(selected), len(selected), count]
            assert get(arrays["out_keys"], "Q") == [keys[i] for i in winners] + [sentinel] * (n - len(winners))
            assert get(arrays["out_ids"], "i") == winners + [-1] * (n - len(winners))
            expected = sum((rank + 1) * values[index] ** 2 for rank, index in enumerate(winners))
            assert abs(get(arrays["out_loss"], "f")[0] - expected) < 1e-5
            gradient = [0.0] * n
            for rank, index in enumerate(winners):
                gradient[index] = 2 * (rank + 1) * values[index]
            actual_gradient = get(arrays["dx"], "f")
            assert max(abs(a - b) for a, b in zip(actual_gradient, gradient)) < 1e-5, (
                iteration,
                actual_gradient,
                gradient,
            )
        with runtime.load_module(native_artifact) as native:
            outputs = dict(keys=runtime.ndarray("u64", n), ids=runtime.ndarray("i32", n))
            read = native.prepare_execution("read", outputs)
            read()
            assert get(outputs["keys"], "Q") == keys, "Native-only initial Field snapshot was lost"
            assert get(outputs["ids"], "i") == list(range(n))
            sort = native.prepare_execution("sort", {})
            sort()
            read()
            expected_order = sorted(range(n), key=keys.__getitem__)
            assert get(outputs["keys"], "Q") == sorted(keys)
            assert get(outputs["ids"], "i") == expected_order
            sort.close()
            read.close()
        call.close()
    if mode == "borrowed":
        ti.reset()
    else:
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
