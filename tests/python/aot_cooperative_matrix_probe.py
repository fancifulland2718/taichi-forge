"""Runtime-only matrix reference, rebinding, failure admission and retirement."""

import json
import os
from pathlib import Path
import struct
import sys
from zipfile import ZipFile

from taichi_forge_runtime.aot import Runtime


def values(size, scale, offset):
    return [((i * scale + offset) % 7 - 3) / 8 for i in range(size)]


def write(array, data, code="f"):
    array.write(struct.pack("<" + str(len(data)) + code, *data))


def read(array):
    data = array.read()
    return struct.unpack("<" + str(len(data) // 4) + "f", data)


def near(actual, expected):
    assert len(actual) == len(expected)
    assert max(abs(x - y) for x, y in zip(actual, expected)) < 0.003


def fails(operation, text):
    try:
        operation()
    except Exception as error:
        if text is not None:
            assert text in str(error), str(error)
    else:
        raise AssertionError(f"Expected rejected operation: {text}")


def main(mode, artifact, m, n, k):
    m, n, k = int(m), int(n), int(k)
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.vulkan, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(
            "vulkan",
            device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")),
        )
    with runtime:
        module = runtime.load_module(artifact)
        frames, references, calls = [], [], []
        for offset in (0, 2):
            frame = {
                name: runtime.ndarray(dtype, (1,), element_shape=shape)
                for name, dtype, shape in (
                    ("a", "f16", (m, k)),
                    ("b", "f16", (k, n)),
                    ("c", "f32", (m, n)),
                    ("output", "f32", (m, n)),
                    ("da", "f32", (m, k)),
                    ("db", "f32", (k, n)),
                    ("dc", "f32", (m, n)),
                    ("dy", "f32", (m, n)),
                    ("loss", "f32", ()),
                )
            }
            a, b, c = (
                values(m * k, 3, offset),
                values(k * n, 5, offset),
                values(m * n, 2, offset),
            )
            output = [
                sum(a[i * k + t] * b[t * n + j] for t in range(k)) + c[i * n + j]
                for i in range(m)
                for j in range(n)
            ]
            da = [
                sum(output[i * n + j] * b[t * n + j] for j in range(n))
                for i in range(m)
                for t in range(k)
            ]
            db = [
                sum(a[i * k + t] * output[i * n + j] for i in range(m))
                for t in range(k)
                for j in range(n)
            ]
            reference = dict(
                output=output,
                da=da,
                db=db,
                dc=output,
                loss=[0.5 * sum(value * value for value in output)],
            )
            for name, data, code in (("a", a, "e"), ("b", b, "e"), ("c", c, "f")):
                write(frame[name], data, code)
            for name, data in reference.items():
                write(frame[name], [-37] * len(data))
            call = module.prepare_execution("step", frame)
            assert read(frame["output"]) == tuple([-37] * (m * n))
            frames.append(frame)
            references.append(reference)
            calls.append(call)

        def check(index):
            calls[index]()
            for name, expected in references[index].items():
                near(read(frames[index][name]), expected)

        for index in (0, 1, 0, 1):
            check(index)
        # Ordinary loaded kernel uses the same tuple and binding path.
        with module.kernel("cooperative").prepare(
            *(frames[0][name] for name in ("a", "b", "c", "output"))
        ) as direct:
            write(frames[0]["output"], [-37] * (m * n))
            direct()
            near(read(frames[0]["output"]), references[0]["output"])

        with ZipFile(artifact) as archive:
            original = {name: archive.read(name) for name in archive.namelist()}
        for change, message in (
            ("version", "version-8"),
            ("missing", "requirements"),
            ("subgroup", "subgroup"),
            ("tuple", "dimensions"),
        ):
            metadata = json.loads(original["metadata.json"])
            if change == "version":
                metadata["metadata_version"] = 7
            for kernel in metadata["kernels"]:
                for task in kernel["tasks_attribs"]:
                    requirements = task["cooperative_matrix_requirements"]
                    if change == "missing":
                        task["cooperative_matrix_requirements"] = []
                    elif change in ("subgroup", "tuple"):
                        for requirement in requirements:
                            if change == "subgroup":
                                requirement["subgroup_size"] = 1
                            else:
                                requirement["m"] = 0x7FFFFFFF
            invalid = Path(artifact).with_name(f"invalid-{mode}-{change}.tcm")
            with ZipFile(invalid, "w") as archive:
                for name, data in original.items():
                    archive.writestr(
                        name, json.dumps(metadata) if name == "metadata.json" else data
                    )

            def prepare_invalid():
                with runtime.load_module(invalid) as broken:
                    broken.prepare_execution("step", frames[0])

            fails(prepare_invalid, message)
            check(0)
        calls[0].close()
        check(1)
        ticket = calls[1].submit()
        module.close()
        if ticket is not None:
            ticket.wait()
            ticket.close()
        near(read(frames[1]["output"]), references[1]["output"])
        fails(calls[1], None)
        calls[1].close()
        if mode == "borrowed":
            ti.reset()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
        assert "numpy" not in sys.modules
    print("COOPERATIVE_AOT_OK", mode)


if __name__ == "__main__":
    main(*sys.argv[1:])
