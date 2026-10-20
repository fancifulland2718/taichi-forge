"""Source-free scalar tensor projections, strict binding, AD suffix and retirement."""

import array
import os
import struct
import sys

from aot_struct_consumer_probe import NoFrontend, read, reject


def main(mode, arch, artifact, layout_name):
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
        layout = module.ndarray_layout(layout_name)
        stride = layout.element_stride
        offsets = {name: layout.member(name).offset for name in ("tag", "inner.matrix", "inner.vector", "tail")}
        initial = bytearray(17 * stride)
        for i in range(17):
            struct.pack_into("=i", initial, i * stride + offsets["tag"], 37)
            struct.pack_into("=6f", initial, i * stride + offsets["inner.matrix"], *[(i * 6 + j) * .5 for j in range(6)])
            struct.pack_into("=3f", initial, i * stride + offsets["inner.vector"], -7, -9, -11)
            struct.pack_into("=i", initial, i * stride + offsets["tail"], 91)
        expected = [6 * ((i * 6 + 5) * .5) + 13 for i in range(17)]

        def changed(count):
            payload = bytearray(initial)
            for i in range(count):
                struct.pack_into("=f", payload, i * stride + offsets["inner.vector"] + 4, expected[i])
            return payload

        records, output = runtime.ndarray(layout, 17), runtime.ndarray("f32", 17)
        records.write(initial)
        output.write(array.array("f", [-13] * 17).tobytes())
        arguments = dict(records=records, output=output)
        marker = runtime.ndarray("i32", 1)
        marker.write(array.array("i", [-19]).tobytes())
        for name in ("bad_rank", "bad_index", "bad_scalar"):
            reject((RuntimeError, aot.AotError), lambda: module.prepare_execution(name, dict(records=records, out=marker)))
        reject((RuntimeError, aot.AotError), lambda: module.prepare_execution("bad_copy", arguments))
        wrong = runtime.ndarray(module.ndarray_layout("wrong"), 17)
        short = runtime.ndarray(layout, 16)
        for parent in (wrong, short):
            reject((RuntimeError, aot.AotError), lambda: module.prepare_execution("cold", dict(records=parent, output=output)))
        assert records.read() == initial and read(output) == [-13] * 17
        assert marker.read() == array.array("i", [-19]).tobytes()
        empty = runtime.ndarray(layout, 0)
        with module.prepare_execution("empty", dict(records=empty, out=marker)) as call:
            call()
        assert marker.read() == array.array("i", [0]).tobytes()
        for name in ("cold", "warm", "parent"):
            records.write(initial)
            with module.prepare_execution(name, arguments) as call:
                call()
                call()
            assert read(output) == expected and records.read() == changed(17)
        records.write(initial)
        output.write(array.array("f", [-13] * 17).tobytes())
        with module.prepare_execution("prefix", arguments) as call:
            call()
        assert read(output) == expected[:11] + [-13] * 6
        assert records.read() == changed(11)
        with module.prepare_execution("private", dict(output=output)) as call:
            call()
        assert read(output) == expected
        assert records.read() == changed(11)  # Private state did not write the external parent.
        frame = dict(arguments, **{name: runtime.ndarray("f32", 17) for name in ("energy", "dx", "dy")})
        with module.prepare_execution("suffix", frame) as call:
            call()
        assert read(frame["dx"]) == expected and read(frame["energy"]) == [.5 * value * value for value in expected]
        other_module = runtime.load_module(artifact)
        other = other_module.prepare_execution("private", dict(output=output))
        call = module.prepare_execution("cold", arguments)
        ticket = call.submit()
        records.close()
        ticket.wait()
        ticket.close()
        reject((RuntimeError, aot.AotError), call)
        call.close()
        module.close()
        other()
        assert read(output) == expected
        other.close()
        other_module.close()
    print("Tensor components, frozen types, prefix, private state, parent, suffix AD and retirement passed", mode, arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
