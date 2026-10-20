"""Private snapshot instances retain independent state and complete raw records."""

import os
import struct
import sys

from aot_struct_consumer_probe import NoFrontend, read, reject


def main(mode, arch, artifact, layout_name, with_field):
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
        assert layout.element_stride == 16
        initial = [i * .5 for i in range(17)]
        def expected(rounds):
            values = initial
            for _ in range(rounds):
                values = [3 * value + 7 for value in values]
            return [value + (5 if int(with_field) else 0) for value in values]

        output = runtime.ndarray("f32", 17)
        independent = runtime.ndarray("f32", 17)
        call = module.prepare_execution("step", dict(output=output))
        other = module.prepare_execution("step", dict(output=independent))
        for count in (1, 2):
            call()
            assert read(output) == expected(count)
        other()
        assert read(independent) == expected(1)
        with module.prepare_execution("warm", dict(output=independent)) as warm:
            warm()
            assert read(independent) == expected(2)
        frame = {name: runtime.ndarray("f32", 17) for name in ("output", "energy", "dx", "dy")}
        frame["records_out"] = runtime.ndarray(layout, 17)
        with module.prepare_execution("raw_and_ad", frame) as full:
            full()
            assert read(frame["dx"]) == expected(1)
            raw = bytearray([0xA5] * (17 * 16))
            for i in range(17):
                for offset, code, value in ((0, "b", -3), (4, "f", 1.5 * i + 7),
                                            (8, "f", -9), (12, "b", 11)):
                    struct.pack_into("=" + code, raw, i * 16 + offset, value)
            assert frame["records_out"].read() == raw
        ticket = call.submit()
        module.close()
        ticket.wait()
        ticket.close()
        reject((RuntimeError, aot.AotError), call)
        call.close()
        other.close()
    print("Private snapshots, raw padding, alias reuse, independent state and suffix AD passed",
          mode, arch, with_field)


if __name__ == "__main__":
    main(*sys.argv[1:])
