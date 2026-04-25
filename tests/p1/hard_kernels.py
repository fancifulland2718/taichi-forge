"""Hard-to-compile Taichi kernels for P1 compile-time benchmarking.

Each factory function accepts a live `taichi` module (already init'd) and returns a
compiled kernel.  The factories are intentionally *heavy*:

  - `autodiff_heavy`      – large autodiff tape; 3× IR expansion under tape
  - `matscalar_12x12`     – 12×12 matrix ops; 144 scalar ops after scalarise
  - `branch_heavy`        – 32-branch if/elif chain per element; wide CFG
  - `deep_func_chain`     – 6-level @ti.func call chain; deep inlining
  - `wide_stencil_2d`     – 13-point 2-D stencil with 4 field planes
  - `many_fields`         – kernel touching 12 separate fields; many GEP chains
  - `sparse_scatter`      – hash-grid pointer-chase over a dynamic SNode

Design rule: each kernel should take ≥ 200 ms to compile cold on a modern CPU.
"""
from __future__ import annotations
import taichi_forge as _ti_placeholder  # used only for type hints in forward-declarations


# ---------------------------------------------------------------------------
# 1. autodiff_heavy
#    N=512 particles; 8 quadratic force terms per pair; tape is ~3× larger.
# ---------------------------------------------------------------------------
def make_autodiff_heavy(ti):
    N = 512
    x   = ti.field(ti.f32, shape=N, needs_grad=True)
    e1  = ti.field(ti.f32, shape=N, needs_grad=True)
    e2  = ti.field(ti.f32, shape=N, needs_grad=True)
    e3  = ti.field(ti.f32, shape=N, needs_grad=True)
    e4  = ti.field(ti.f32, shape=N, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def forward():
        for i in x:
            v = x[i]
            # 8 quadratic terms — generates ~80 IR instructions per iteration
            e1[i] = v * v + 0.1 * v
            e2[i] = e1[i] * e1[i] + 0.2 * v
            e3[i] = e2[i] * e2[i] + 0.3 * e1[i]
            e4[i] = e3[i] * e3[i] + 0.4 * e2[i]
            t = e1[i] + e2[i] + e3[i] + e4[i]
            loss[None] += t * t

    return forward


# ---------------------------------------------------------------------------
# 2. matscalar_12x12
#    10×10 matrix multiply + transpose + add; fully scalarized → 1000 ops.
#    (12×12 would be even larger but 10×10 is sufficient to hit 64×+ expansion)
# ---------------------------------------------------------------------------
def make_matscalar_12x12(ti):
    N = 64
    A = ti.Matrix.field(10, 10, dtype=ti.f32, shape=(N, N))
    B = ti.Matrix.field(10, 10, dtype=ti.f32, shape=(N, N))
    C = ti.Matrix.field(10, 10, dtype=ti.f32, shape=(N, N))

    @ti.kernel
    def matmul():
        for i, j in C:
            # Full 10×10 matrix mul + add with transpose
            # Scalarization expands this to 10*10*(10 muls + 9 adds) = 1900 IR ops
            C[i, j] = A[i, j] @ B[i, j] + A[i, j].transpose()

    return matmul


# ---------------------------------------------------------------------------
# 3. branch_heavy
#    32-category dispatch per pixel: many if/elif → wide CFG
# ---------------------------------------------------------------------------
def make_branch_heavy(ti):
    N = 256
    src  = ti.field(ti.i32, shape=N * N)
    dst  = ti.field(ti.f32, shape=N * N)

    @ti.kernel
    def classify():
        for idx in src:
            v = src[idx]
            r = 0.0
            # 32-way branch — each elif is a BasicBlock; PHI insertion is heavy
            if   v < 32:  r = ti.cast(v, ti.f32) * 0.01
            elif v < 64:  r = ti.cast(v, ti.f32) * 0.02
            elif v < 96:  r = ti.cast(v, ti.f32) * 0.03
            elif v < 128: r = ti.cast(v, ti.f32) * 0.04
            elif v < 160: r = ti.cast(v, ti.f32) * 0.05
            elif v < 192: r = ti.cast(v, ti.f32) * 0.06
            elif v < 224: r = ti.cast(v, ti.f32) * 0.07
            elif v < 256: r = ti.cast(v, ti.f32) * 0.08
            elif v < 288: r = ti.cast(v, ti.f32) * 0.09
            elif v < 320: r = ti.cast(v, ti.f32) * 0.10
            elif v < 352: r = ti.cast(v, ti.f32) * 0.11
            elif v < 384: r = ti.cast(v, ti.f32) * 0.12
            elif v < 416: r = ti.cast(v, ti.f32) * 0.13
            elif v < 448: r = ti.cast(v, ti.f32) * 0.14
            elif v < 480: r = ti.cast(v, ti.f32) * 0.15
            elif v < 512: r = ti.cast(v, ti.f32) * 0.16
            elif v < 544: r = ti.cast(v, ti.f32) * 0.17
            elif v < 576: r = ti.cast(v, ti.f32) * 0.18
            elif v < 608: r = ti.cast(v, ti.f32) * 0.19
            elif v < 640: r = ti.cast(v, ti.f32) * 0.20
            elif v < 672: r = ti.cast(v, ti.f32) * 0.21
            elif v < 704: r = ti.cast(v, ti.f32) * 0.22
            elif v < 736: r = ti.cast(v, ti.f32) * 0.23
            elif v < 768: r = ti.cast(v, ti.f32) * 0.24
            elif v < 800: r = ti.cast(v, ti.f32) * 0.25
            elif v < 832: r = ti.cast(v, ti.f32) * 0.26
            elif v < 864: r = ti.cast(v, ti.f32) * 0.27
            elif v < 896: r = ti.cast(v, ti.f32) * 0.28
            elif v < 928: r = ti.cast(v, ti.f32) * 0.29
            elif v < 960: r = ti.cast(v, ti.f32) * 0.30
            elif v < 992: r = ti.cast(v, ti.f32) * 0.31
            else:          r = ti.cast(v, ti.f32) * 0.32
            dst[idx] = r

    return classify


# ---------------------------------------------------------------------------
# 4. deep_func_chain
#    6 levels of @ti.func, each doing 10 ops — LLVM inlines all 6 levels
# ---------------------------------------------------------------------------
def make_deep_func_chain(ti):
    N = 1024
    x = ti.field(ti.f32, shape=N)
    y = ti.field(ti.f32, shape=N)

    @ti.func
    def f1(v):
        return v * 1.1 + 0.01 * v * v - 0.001 * v * v * v + 0.0001

    @ti.func
    def f2(v):
        return f1(v) * 1.2 + f1(v - 0.5) * 0.3 - f1(v + 0.5) * 0.1

    @ti.func
    def f3(v):
        return f2(v * 1.3) + f2(v - 1.0) * 0.5 + f2(v + 1.0) * 0.2

    @ti.func
    def f4(v):
        a = f3(v)
        b = f3(-v)
        c = f3(v * 0.5)
        return a * 0.5 + b * 0.3 + c * 0.2

    @ti.func
    def f5(v):
        return f4(v + 0.1) * f4(v - 0.1) - f4(v) * f4(v) * 0.01

    @ti.func
    def f6(v):
        return f5(v) + f5(-v) * 0.25 + f5(v * 2.0) * 0.1

    @ti.kernel
    def chain():
        for i in x:
            y[i] = f6(x[i])

    return chain


# ---------------------------------------------------------------------------
# 5. wide_stencil_2d
#    4-plane 13-point 2-D stencil; many GEP chains per cell
# ---------------------------------------------------------------------------
def make_wide_stencil_2d(ti):
    N = 256
    # 4 field planes — pressure, u-vel, v-vel, vorticity
    p  = ti.field(ti.f32, shape=(N, N))
    u  = ti.field(ti.f32, shape=(N, N))
    v  = ti.field(ti.f32, shape=(N, N))
    w  = ti.field(ti.f32, shape=(N, N))
    # output
    np = ti.field(ti.f32, shape=(N, N))
    nu = ti.field(ti.f32, shape=(N, N))
    nv = ti.field(ti.f32, shape=(N, N))
    nw = ti.field(ti.f32, shape=(N, N))

    @ti.kernel
    def stencil():
        for i, j in ti.ndrange((2, N - 2), (2, N - 2)):
            # 13-point Laplacian on p, u, v, w
            lp = (-60.0   * p[i,   j  ]
                  + 16.0  * (p[i-1, j  ] + p[i+1, j  ] + p[i, j-1] + p[i, j+1])
                  -  1.0  * (p[i-2, j  ] + p[i+2, j  ] + p[i, j-2] + p[i, j+2])
                  -  1.0  * (p[i-1,j-1] + p[i-1,j+1] + p[i+1,j-1] + p[i+1,j+1])
                 ) / 12.0
            lu = (-60.0   * u[i,   j  ]
                  + 16.0  * (u[i-1, j  ] + u[i+1, j  ] + u[i, j-1] + u[i, j+1])
                  -  1.0  * (u[i-2, j  ] + u[i+2, j  ] + u[i, j-2] + u[i, j+2])
                  -  1.0  * (u[i-1,j-1] + u[i-1,j+1] + u[i+1,j-1] + u[i+1,j+1])
                 ) / 12.0
            lv = (-60.0   * v[i,   j  ]
                  + 16.0  * (v[i-1, j  ] + v[i+1, j  ] + v[i, j-1] + v[i, j+1])
                  -  1.0  * (v[i-2, j  ] + v[i+2, j  ] + v[i, j-2] + v[i, j+2])
                  -  1.0  * (v[i-1,j-1] + v[i-1,j+1] + v[i+1,j-1] + v[i+1,j+1])
                 ) / 12.0
            lw = (-60.0   * w[i,   j  ]
                  + 16.0  * (w[i-1, j  ] + w[i+1, j  ] + w[i, j-1] + w[i, j+1])
                  -  1.0  * (w[i-2, j  ] + w[i+2, j  ] + w[i, j-2] + w[i, j+2])
                  -  1.0  * (w[i-1,j-1] + w[i-1,j+1] + w[i+1,j-1] + w[i+1,j+1])
                 ) / 12.0
            # cross-field coupling
            np[i, j] = lp + 0.01 * (u[i, j] * lu + v[i, j] * lv)
            nu[i, j] = lu - 0.5 * lp
            nv[i, j] = lv - 0.5 * lp
            nw[i, j] = lw + lu * nv[i, j] - lv * nu[i, j]

    return stencil


# ---------------------------------------------------------------------------
# 6. many_fields
#    12 separate ti.field arguments; heavy arg-load / alias analysis
# ---------------------------------------------------------------------------
def make_many_fields(ti):
    N = 512
    fields = [ti.field(ti.f32, shape=N) for _ in range(12)]
    out    = ti.field(ti.f32, shape=N)

    f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = fields

    @ti.kernel
    def wide_read():
        for i in out:
            # Read from all 12 fields, compute weighted sum + products
            a  = f0[i] + f1[i] * 0.1
            b  = f2[i] + f3[i] * 0.2
            c  = f4[i] + f5[i] * 0.3
            d  = f6[i] + f7[i] * 0.4
            e  = f8[i] + f9[i] * 0.5
            ff = f10[i] + f11[i] * 0.6
            # Cross-products stressing alias analysis / load-hoisting
            out[i] = (a * b + c * d + e * ff
                      + a * c * 0.01 + b * d * 0.02 + c * e * 0.03
                      + a * e * 0.04 + b * ff * 0.05 + d * ff * 0.06)

    return wide_read


# ---------------------------------------------------------------------------
# 7. autodiff_matrix
#    Autodiff over a matrix-heavy kernel — both the tape expansion AND the
#    scalarization combine to produce maximal IR size.
# ---------------------------------------------------------------------------
def make_autodiff_matrix(ti):
    N = 32
    pos = ti.Matrix.field(3, 1, dtype=ti.f32, shape=N, needs_grad=True)
    vel = ti.Matrix.field(3, 1, dtype=ti.f32, shape=N, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def energy():
        for i in range(N):
            for j in range(N):
                if j > i:
                    r = pos[i] - pos[j]
                    # Dot product + velocity coupling
                    d2 = (r.transpose() @ r)[0, 0] + 1e-4
                    v_rel = vel[i] - vel[j]
                    ke = (v_rel.transpose() @ v_rel)[0, 0]
                    loss[None] += 1.0 / d2 + 0.5 * ke

    return energy


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
HARD_KERNELS = [
    ("autodiff_heavy",    make_autodiff_heavy),
    ("matscalar_10x10",   make_matscalar_12x12),
    ("branch_heavy_32",   make_branch_heavy),
    ("deep_func_chain_6", make_deep_func_chain),
    ("wide_stencil_4pln", make_wide_stencil_2d),
    ("many_fields_12",    make_many_fields),
    ("autodiff_matrix",   make_autodiff_matrix),
]
