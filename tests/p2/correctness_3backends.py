"""Numeric correctness check for the heavy P2 kernels (mat14, sph_force).

For each backend (cpu / cuda / vulkan), run the kernel once on a fresh init
and snapshot the output. Then compare the three snapshots elementwise.

Used as a stability gate after each P2.x sub-stage IR change.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import taichi_forge as ti

ARCHES = [("cpu", ti.cpu), ("cuda", ti.cuda), ("vulkan", ti.vulkan)]


def run_mat14(arch):
    ti.init(arch=arch, offline_cache=False, log_level="error", random_seed=42,
            unrolling_limit=0)
    N = 4
    M = 6  # 6x6 instead of 14x14 to keep cold-compile under ~1s on each backend
    A = ti.Matrix.field(M, M, dtype=ti.f32, shape=(N, N))
    B = ti.Matrix.field(M, M, dtype=ti.f32, shape=(N, N))
    C = ti.Matrix.field(M, M, dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def init_mat():
        for p, q in A:
            for i, j in ti.static(ti.ndrange(M, M)):
                A[p, q][i, j] = ti.cast((p * 13 + q * 7 + i * 3 + j) % 11, ti.f32) * 0.01
                B[p, q][i, j] = ti.cast((p * 11 + q * 5 + i * 2 + j * 9) % 13, ti.f32) * 0.01

    @ti.kernel
    def matmul14():
        for p, q in C:
            C[p, q] = A[p, q] @ B[p, q] + A[p, q].transpose()

    init_mat()
    matmul14()
    out = C.to_numpy().reshape(-1)
    ti.reset()
    return out


def run_sph_force(arch):
    ti.init(arch=arch, offline_cache=False, log_level="error", random_seed=42)
    N = 64
    K = 8  # smaller stencil for fast correctness check
    pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
    rho = ti.field(ti.f32, shape=N)
    pressure = ti.field(ti.f32, shape=N)
    nlist = ti.field(ti.i32, shape=(N, K))
    force = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_sph():
        for i in pos:
            pos[i] = ti.Vector([ti.cast(i % 8, ti.f32) * 0.1,
                                ti.cast((i // 8) % 8, ti.f32) * 0.1,
                                ti.cast(i // 64, ti.f32) * 0.1])
            vel[i] = ti.Vector([0.0, 0.0, 0.0])
            rho[i] = 1.0 + ti.cast(i % 7, ti.f32) * 0.01
            pressure[i] = ti.cast(i % 5, ti.f32) * 0.1
            for k in ti.static(range(K)):
                nlist[i, k] = (i + k + 1) % N

    @ti.kernel
    def compute_force():
        for p in range(N):
            f = ti.Vector([0.0, 0.0, 0.0])
            for k in ti.static(range(K)):
                q = nlist[p, k]
                rij = pos[p] - pos[q]
                r2 = rij.dot(rij) + 1e-9
                w = ti.exp(-r2 * 4.0)
                pterm = (pressure[p] / (rho[p] * rho[p]) +
                         pressure[q] / (rho[q] * rho[q]))
                f -= rij * (pterm * w)
            force[p] = f

    init_sph()
    compute_force()
    out = force.to_numpy().reshape(-1)
    ti.reset()
    return out


def main():
    for kernel_name, runner in [("mat14", run_mat14), ("sph_force", run_sph_force)]:
        outs = {}
        for name, arch in ARCHES:
            try:
                outs[name] = runner(arch)
                print(f"[{kernel_name}/{name}] head={outs[name][:4].tolist()}")
            except Exception as exc:
                print(f"[{kernel_name}/{name}] FAILED: {exc}")
                return 1
        # Cross-backend agreement (relative tolerance)
        ref = outs["cpu"]
        for name in ("cuda", "vulkan"):
            diff = np.abs(outs[name] - ref)
            scale = np.maximum(np.abs(ref), 1e-6)
            rel = (diff / scale).max()
            tag = "OK" if rel < 1e-4 else "MISMATCH"
            print(f"[{kernel_name}/{name} vs cpu] max_rel_err={rel:.3e} [{tag}]")
            if rel >= 1e-4:
                return 1
    print("\nAll 3 backends agree numerically (max rel err < 1e-4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
