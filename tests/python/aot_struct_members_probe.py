"""Standalone execution of qualified members, private owners and suffix AD."""

import array
import os
import struct
import sys

from aot_struct_consumer_probe import NoFrontend, read, reject


def main(mode, arch, artifact):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot
        runtime = aot.Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0)) if arch == "vulkan" else 0)
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot
        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    with runtime:
        module = runtime.load_module(artifact)
        layout = module.ndarray_layout("records")
        assert layout.element_stride == 16
        assert [layout.member(n).offset for n in ("tag", "value", "other", "tail")] == [0, 4, 8, 12]
        initial = bytearray([0xA5] * (17 * 16))
        for i in range(17):
            for offset, code, value in ((0, "b", -3), (4, "f", i * .5), (8, "f", -9), (12, "b", 11)):
                struct.pack_into("=" + code, initial, i * 16 + offset, value)

        def transformed(data, start, count, inplace=False):
            result = bytearray(data)
            for i in range(start, start + count):
                value = struct.unpack_from("=f", data, i * 16 + (8 if inplace else 4))[0]
                struct.pack_into("=f", result, i * 16 + 8, 3 * value + 7)
            return result

        frames = []
        calls = []
        for _ in range(2):
            frame = {name: runtime.ndarray("f32", 17) for name in ("output", "energy", "dx", "dy")}
            frame["records_out"] = runtime.ndarray(layout, 17)
            frames.append(frame)
            calls.append(module.prepare_execution("frozen_private", frame))
        reject((aot.AotError, ValueError),
               lambda: module.prepare_execution("bad_private", frames[0]))
        expected_bytes = transformed(initial, 0, 16)
        expected = [3 * i * .5 + 7 for i in range(16)] + [-9]
        for call, frame in zip(calls, frames):
            call()
            assert frame["records_out"].read() == expected_bytes
            assert read(frame["output"]) == expected
            assert read(frame["energy"]) == [.5 * x * x for x in expected]
            assert read(frame["dx"]) == expected
        # A separately prepared private plan owns a separate initialized parent.
        with module.prepare_execution("inplace", dict(records_out=frames[0]["records_out"])) as inplace:
            inplace()
            assert frames[0]["records_out"].read() == transformed(initial, 0, 16, True)
        calls[0]()
        assert frames[0]["records_out"].read() == expected_bytes
        records = runtime.ndarray(layout, 17)
        records.write(initial)
        external = module.prepare_execution("frozen_external", dict(records=records))
        external()
        assert records.read() == transformed(initial, 1, 16)
        alias = runtime.ndarray(module.ndarray_layout("records_alias"), 17)
        alias.write(initial)
        with module.prepare_execution("frozen_external", dict(records=alias)) as call:
            call()
            assert alias.read() == transformed(initial, 1, 16)
        invalid = [
            runtime.ndarray(module.ndarray_layout("other_records"), 17),
            runtime.ndarray(layout, 18),
            runtime.ndarray(layout, (1, 17)),
            runtime.ndarray("f32", 17 * 4),
        ]
        for candidate in invalid:
            before = candidate.read()
            reject((aot.AotError, ValueError), lambda: module.prepare_execution(
                "frozen_external", dict(records=candidate)))
            assert candidate.read() == before
        if arch == "cuda":
            with module.prepare_execution("frozen_family", dict(records=records)) as family:
                with family.prepare_binding(dict(records=alias)) as frame:
                    frame()
                    assert alias.read() == transformed(initial, 1, 16)
                for candidate in invalid:
                    reject((aot.AotError, ValueError),
                           lambda: family.prepare_binding(dict(records=candidate)))
        empty_parent = runtime.ndarray(layout, 0)
        empty_output = runtime.ndarray(layout, 0)
        with module.prepare_execution("frozen_empty", dict(records=empty_parent, output=empty_output)) as empty:
            empty()
            assert empty_output.read() == b""
        packed = runtime.ndarray("f32", 16)
        with module.prepare_execution("packed_output", dict(records=records, output=packed)) as copy:
            copy()
            assert read(packed) == [3 * i * .5 + 7 for i in range(1, 17)]
        for name in ("oversize", "wrong_path", "packed_rejects_member", "member_copy"):
            before = records.read()
            bindings = dict(records=records, **({"output": packed} if name == "member_copy" else {}))
            reject((aot.AotError, ValueError), lambda: module.prepare_execution(name, bindings))
            assert records.read() == before
        wrong = runtime.ndarray("f32", 17 * 4)
        reject((aot.AotError, ValueError),
               lambda: module.prepare_execution("external", dict(records=wrong)))
        result = runtime.ndarray("i32", 1)
        with module.prepare_execution("empty", dict(result=result)) as empty:
            empty()
            assert result.read() == array.array("i", [0]).tobytes()
        ticket = external.submit()
        records.close()
        ticket.wait()
        ticket.close()
        reject((RuntimeError, aot.AotError), external)
        external.close()
        calls[1]()
        ticket = calls[0].submit()
        module.close()
        ticket.wait()
        ticket.close()
        for call in calls:
            reject((RuntimeError, aot.AotError), call)
            call.close()
    print("Typed private resources, affine members, downstream AD and retirement passed", mode, arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
