"""A warm-cache AOT producer must remain safe to destroy after Program reset."""

import gc
from pathlib import Path
import sys

import taichi_forge as ti


def main():
    arch, directory, phase = sys.argv[1:]
    root = Path(directory)
    ti.init(
        arch=getattr(ti, arch),
        offline_cache=True,
        offline_cache_file_path=str(root / "cache"),
    )
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)

    @ti.kernel
    def compute(value: ti.f32) -> ti.f32:
        return 3 * value - 2

    module = ti.aot.Module()
    module.add_kernel(compute)
    module.add_kernel(compute, name="alias")
    module.save(str(root / phase))
    module.archive(str(root / (phase + ".tcm")))
    stats = runtime.debug_kernel_executable_lifecycle_stats()
    assert stats["compiler_invocations"] == (1 if phase == "cold" else 0), stats
    assert stats["disk_loads"] == (0 if phase == "cold" else 1), stats
    ti.reset()
    del module
    gc.collect()
    print("CACHE_PRODUCER_RETIRED", arch, phase, flush=True)


if __name__ == "__main__":
    main()
