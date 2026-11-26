"""Field-explicit kernels shared by the one-shot and prepared Krylov solvers.

Only scalar precision belongs to these definitions. Concrete field templates
are supplied by the caller and can also be bound by an AOT module.
"""

from functools import lru_cache

import taichi_forge as ti
from taichi_forge.lang.kernel_impl import kernel


@kernel
def clear_scratch(fields: ti.template()):
    for I in ti.grouped(fields[0]):
        for k in ti.static(range(len(fields))):
            fields[k][I] = 0


@lru_cache(maxsize=2)
def dot_kernel(solver_dtype):
    @ti.kernel
    def reduce(p: ti.template(), q: ti.template()) -> solver_dtype:
        result = solver_dtype(0.0)
        for I in ti.grouped(p):
            result += p[I] * q[I]
        return result

    return reduce


@lru_cache(maxsize=2)
def cg_kernels(solver_dtype):
    @ti.kernel
    def init(
        x: ti.template(), b: ti.template(), Ax: ti.template(), r: ti.template(), p: ti.template(), Ap: ti.template()
    ):
        for I in ti.grouped(x):
            r[I] = b[I] - Ax[I]
            p[I] = r[I]
            Ap[I] = 0.0

    @ti.kernel
    def update_x(x: ti.template(), p: ti.template(), alpha: solver_dtype):
        for I in ti.grouped(x):
            x[I] += alpha * p[I]

    @ti.kernel
    def update_r(r: ti.template(), Ap: ti.template(), alpha: solver_dtype):
        for I in ti.grouped(r):
            r[I] -= alpha * Ap[I]

    @ti.kernel
    def update_p(p: ti.template(), r: ti.template(), beta: solver_dtype):
        for I in ti.grouped(p):
            p[I] = r[I] + beta * p[I]

    return init, update_x, update_r, update_p


@lru_cache(maxsize=2)
def bicgstab_kernels(solver_dtype):
    @ti.kernel
    def init(
        x: ti.template(),
        b: ti.template(),
        Ax: ti.template(),
        r: ti.template(),
        r_tld: ti.template(),
        p: ti.template(),
        p_hat: ti.template(),
        Ap: ti.template(),
        t: ti.template(),
    ):
        for I in ti.grouped(x):
            r[I] = b[I] - Ax[I]
            r_tld[I] = b[I]
            p[I] = r[I]
            p_hat[I] = p[I]
            Ap[I] = 0.0
            t[I] = 0.0

    @ti.kernel
    def squared_norm_and_dot(v: ti.template(), w: ti.template()) -> ti.types.vector(2, solver_dtype):
        norm_sq = solver_dtype(0.0)
        dot = solver_dtype(0.0)
        for I in ti.grouped(v):
            norm_sq += v[I] * v[I]
            dot += v[I] * w[I]
        return ti.Vector([norm_sq, dot])

    @ti.kernel
    def update_p(
        p: ti.template(), r: ti.template(), Ap: ti.template(), p_hat: ti.template(), beta: solver_dtype, omega: solver_dtype
    ):
        for I in ti.grouped(p):
            p[I] = r[I] + beta * (p[I] - omega * Ap[I])
            p_hat[I] = p[I]

    @ti.kernel
    def update_s(s: ti.template(), r: ti.template(), Ap: ti.template(), s_hat: ti.template(), alpha: solver_dtype):
        for I in ti.grouped(s):
            s[I] = r[I] - alpha * Ap[I]
            s_hat[I] = s[I]

    @ti.kernel
    def update_x(
        x: ti.template(), p_hat: ti.template(), s_hat: ti.template(), alpha: solver_dtype, omega: solver_dtype
    ):
        for I in ti.grouped(x):
            x[I] += alpha * p_hat[I] + omega * s_hat[I]

    @ti.kernel
    def update_r(r: ti.template(), s: ti.template(), t: ti.template(), omega: solver_dtype):
        for I in ti.grouped(r):
            r[I] = s[I] - omega * t[I]

    return init, update_p, update_s, update_x, update_r, squared_norm_and_dot
