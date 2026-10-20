"""Consume only the emitted parent/member recipes, without a DSL in standalone."""

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
        assert layout.element_stride == 12
        initial = b"".join(struct.pack("=iff", -7, i * .5, -9) for i in range(17))
        expected = [3 * i + 13 for i in range(17)]
        records = runtime.ndarray(layout, 17)
        output = runtime.ndarray("f32", 17)
        records.write(initial)
        for name in ("cold", "warm", "parent"):
            with module.prepare_execution(name, dict(records=records, output=output)) as call:
                call()
                assert read(output) == expected
                actual = list(struct.iter_unpack("=iff", records.read()))
                assert actual == [(-7, i * .5, 1.5 * i + 7) for i in range(17)]
                call()
                assert read(output) == expected
        frame = dict(records=records, output=output)
        frame.update({name: runtime.ndarray("f32", 17) for name in ("energy", "dx", "dy")})
        with module.prepare_execution("suffix", frame) as call:
            call()
            assert read(frame["dx"]) == expected
            assert read(frame["energy"]) == [.5 * x * x for x in expected]
        # A second prepared binding and module retain separate external parents.
        second = runtime.load_module(artifact)
        second_records = runtime.ndarray(second.ndarray_layout(layout_name), 17)
        second_records.write(b"".join(struct.pack("=iff", 2, 1, 0) for _ in range(17)))
        second_output = runtime.ndarray("f32", 17)
        other = second.prepare_execution("parent", dict(records=second_records, output=second_output))
        call = module.prepare_execution("parent", dict(records=records, output=output))
        ticket = call.submit()
        module.close()
        ticket.wait()
        ticket.close()
        reject((RuntimeError, aot.AotError), call)
        call.close()
        other()
        assert read(second_output) == [19] * 17
        other.close()
        second.close()
    print("Graph structure members, parent composition, suffix AD and retirement passed", mode, arch)


if __name__ == "__main__":
    main(*sys.argv[1:])
