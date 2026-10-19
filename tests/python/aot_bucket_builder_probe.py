"""Bucket replay with only the installed runtime API and standard library."""

import array
from contextlib import ExitStack, closing
import json
import os
from pathlib import Path
import struct
import sys

from taichi_forge_runtime.aot import Runtime
from aot_scan_probe import CODES, reject


def exercise(runtime, module, case, arch):
    n, bins = case["size"], case["num_bins"]
    dtype, shape = case["dtype"], case["element_shape"]
    lanes = 1
    for axis in shape:
        lanes *= axis
    width = int(dtype[1:]) // 8
    item_bytes = lanes * width
    prefix = case.get("prefix", 1)
    capacity = case.get("capacity", n + 3)
    offset_capacity = case.get("offset_capacity", bins + 4)
    rows = []
    for i in range(capacity):
        row = struct.pack("<" + CODES[dtype], i + 100)
        for lane in range(1, lanes):
            if dtype == "f32":
                row += struct.pack("<I", 0x7FC01234 if lane % 2 else 0x80000000)
            else:
                row += struct.pack("<" + CODES[dtype], i + lane + 200)
        rows.append(row)
    source = b"".join(rows)
    row_ids = {value: i for i, value in enumerate(rows)}
    sentinel = bytes([0xA5]) * len(source)
    offset_initial = array.array("i", [-77] * offset_capacity).tobytes()
    with ExitStack() as stack:
        keys = stack.enter_context(closing(runtime.ndarray("i32", capacity)))
        values = stack.enter_context(
            closing(runtime.ndarray(dtype, capacity, element_shape=shape))
        )
        output = stack.enter_context(
            closing(runtime.ndarray(dtype, capacity, element_shape=shape))
        )
        offsets = stack.enter_context(closing(runtime.ndarray("i32", offset_capacity)))
        frame = dict(keys=keys, values=values, offsets=offsets, output=output)
        values.write(source)
        output.write(sentinel)
        offsets.write(offset_initial)
        call = stack.enter_context(module.prepare_execution(case["name"], frame))
        peer = stack.enter_context(module.prepare_execution(case["name"], frame))
        assert values.read() == source and output.read() == sentinel
        assert offsets.read() == offset_initial, "Cold preparation wrote offsets"
        scratch = call.info["workspace_bytes"]
        assert (scratch == 0) == (n == 0)
        if arch == "cpu" and n:
            partitions = 4 if n >= 262144 else 1
            assert scratch == partitions * bins * 4
        if arch == "vulkan" and n:
            policy = case["vulkan_policy"]
            private = policy["private_enabled"] and bins <= 4096
            assert scratch == (((n + 2047) // 2048) if private else 1) * bins * 4
        for replay in range(3):
            key_data = (
                [-77] * prefix
                + [
                    -1 if replay == 1 or i % 7 == 0 else (i + replay) % (bins + 2)
                    for i in range(n)
                ]
                + [-77] * (capacity - prefix - n)
            )
            keys_bytes = array.array("i", key_data).tobytes()
            keys.write(keys_bytes)
            output.write(sentinel)
            offsets.write(offset_initial)
            current = peer if replay == 1 else call
            current()
            actual = output.read()
            observed_offsets = array.array("i")
            observed_offsets.frombytes(offsets.read())
            expected_offsets = [-77] * prefix + [0]
            groups = [[] for _ in range(bins)]
            for i, key in enumerate(key_data[prefix : n + prefix], prefix):
                if 0 <= key < bins:
                    groups[key].append(i)
            for group in groups:
                expected_offsets.append(expected_offsets[-1] + len(group))
            expected_offsets += [-77] * (offset_capacity - prefix - bins - 1)
            assert observed_offsets.tolist() == expected_offsets
            accepted = expected_offsets[prefix + bins]
            assert actual[: prefix * item_bytes] == sentinel[: prefix * item_bytes]
            assert (
                actual[(accepted + prefix) * item_bytes :]
                == sentinel[(accepted + prefix) * item_bytes :]
            )
            for b, group in enumerate(groups):
                begin, end = (
                    expected_offsets[b + prefix],
                    expected_offsets[b + prefix + 1],
                )
                ids = [
                    row_ids[
                        actual[
                            (i + prefix) * item_bytes : (i + prefix + 1) * item_bytes
                        ]
                    ]
                    for i in range(begin, end)
                ]
                assert ids == group if arch == "cpu" else sorted(ids) == group
            assert keys.read() == keys_bytes and values.read() == source
            assert current.info["workspace_bytes"] == scratch
        # Separate fixed owners share the bindings, not temporary storage.
        call()
        call.close()
        peer()
        runtime.wait()
        reject(call)
        if n:
            reject(
                lambda: module.prepare_execution(
                    case["name"], dict(frame, output=values)
                )
            )
            if dtype == "i32" and not shape and capacity >= bins + 4:
                reject(
                    lambda: module.prepare_execution(
                        case["name"], dict(frame, offsets=output)
                    )
                )
        else:
            offsets.write(offset_initial)
            with module.prepare_execution(
                case["name"], dict(frame, keys=offsets, values=offsets, output=offsets)
            ) as aliased:
                assert aliased.info["workspace_bytes"] == 0
                aliased()
                expected = array.array(
                    "i",
                    [-77] * prefix
                    + [0] * (bins + 1)
                    + [-77] * (offset_capacity - prefix - bins - 1),
                )
                assert offsets.read() == expected.tobytes()
        with closing(runtime.ndarray("i16", capacity, element_shape=shape)) as wrong:
            reject(
                lambda: module.prepare_execution(
                    case["name"], dict(frame, values=wrong)
                )
            )


def main(mode, arch, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=2)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(
            arch,
            device_index=(
                int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))
                if arch == "vulkan"
                else 0
            ),
        )
    cases = json.loads(Path(artifact).with_suffix(".json").read_text())
    with runtime, runtime.load_module(artifact) as module:
        for case in cases:
            exercise(runtime, module, case, arch)
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules
    else:
        runtime = ti.aot.borrow_runtime()
        module = runtime.load_module(artifact)
        case = cases[0]
        n, bins = case["size"], case["num_bins"]
        capacity = case.get("capacity", n + 3)
        keys, values, output = (runtime.ndarray("i32", capacity) for _ in range(3))
        offsets = runtime.ndarray("i32", case.get("offset_capacity", bins + 4))
        keys.write(array.array("i", [0] * capacity))
        call = module.prepare_execution(
            case["name"], dict(keys=keys, values=values, output=output, offsets=offsets)
        )
        call()
        ti.reset()
        reject(call)
        call.close()
        module.close()
        for obj in (keys, values, output, offsets):
            obj.close()
        runtime.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
