"""P5.b parity — parallel batch compile must produce same results as serial launches."""

import taichi_forge as ti

ti.init(arch=ti.cpu, offline_cache=False, num_compile_threads=4)


kernels = []
for idx in range(8):
    a = idx + 1

    @ti.kernel
    def _k(x: ti.f32) -> ti.f32:
        return ti.sin(x * a) + a * 0.25

    _k.__name__ = f"_k_{idx}"
    kernels.append(_k)


# Parallel pre-compile (pass args tuple because kernels take an f32).
n = ti.compile_kernels([(k, (0.5,)) for k in kernels])
assert n == 8, n

# Now call each kernel — must return correct result (cache hit expected).
out = [k(0.5) for k in kernels]

# Reference: serial on a fresh program.
ti.reset()
ti.init(arch=ti.cpu, offline_cache=False)
ref_kernels = []
for idx in range(8):
    a = idx + 1

    @ti.kernel
    def _r(x: ti.f32) -> ti.f32:
        return ti.sin(x * a) + a * 0.25

    _r.__name__ = f"_r_{idx}"
    ref_kernels.append(_r)

ref = [k(0.5) for k in ref_kernels]

for i, (a_, b_) in enumerate(zip(out, ref)):
    assert abs(a_ - b_) < 1e-6, (i, a_, b_)

print(f"[p5b_parity] ok: parallel-compiled {len(out)} kernels match serial, max_delta<1e-6")
