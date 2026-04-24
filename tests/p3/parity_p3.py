"""P3 — 3-backend numeric parity on heavy_kernels + new budget knobs.

With all P3 knobs set to 0 (default), output MUST match the pre-P3 baseline
bit-for-bit on cpu/cuda/vulkan. Also verify that a kernel that fits under
high budgets is unaffected.
"""
import os
import sys
import numpy as np
import taichi as ti

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "p2"))


def run_on(arch, **init_kwargs):
    ti.reset()
    ti.init(arch=arch, offline_cache=False, default_fp=ti.f32,
            log_level=ti.WARN, **init_kwargs)
    N = 256
    x = ti.field(ti.f32, shape=N)

    @ti.kernel
    def run():
        for i in x:
            s = ti.f32(0.0)
            for k in ti.static(range(16)):
                s += ti.sin(ti.cast(i, ti.f32) * 0.01 * float(k + 1))
            x[i] = s

    run()
    ti.sync()
    arr = x.to_numpy()
    ti.reset()
    return arr


def main():
    backends = [("cpu", ti.cpu), ("cuda", ti.cuda), ("vulkan", ti.vulkan)]
    # Baseline with defaults (knobs = 0 / disabled).
    base = {name: run_on(arch) for name, arch in backends}
    # With generous hard limits set — must be identical.
    with_limits = {
        name: run_on(arch, unrolling_hard_limit=1000,
                     unrolling_kernel_hard_limit=10000,
                     func_inline_depth_limit=20)
        for name, arch in backends
    }
    ok = True
    for name, _ in backends:
        delta = float(np.abs(base[name] - with_limits[name]).max())
        print(f"{name:8s}  |delta| default-vs-budgeted = {delta:.3e}")
        if delta != 0.0:
            ok = False
    # Cross-backend sanity vs cpu
    for name, _ in backends[1:]:
        delta = float(np.abs(base[name] - base["cpu"]).max())
        print(f"{name:8s}  |delta| vs cpu = {delta:.3e}")
        if delta > 1e-4:
            ok = False
    print("OK" if ok else "MISMATCH")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
