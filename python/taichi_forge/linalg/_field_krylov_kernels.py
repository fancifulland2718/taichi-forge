"""Field-explicit kernels shared by the one-shot and prepared Krylov solvers.

Fields and loop-domain choices are explicit templates. AOT callers can bind
them at compilation just as prepared solves do.
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
    def update_solution(
        x: ti.template(), p: ti.template(), r: ti.template(), Ap: ti.template(),
        alpha: solver_dtype, same_domain: ti.template(),
    ) -> solver_dtype:
        norm_sq = solver_dtype(0.0)
        if ti.static(not same_domain):
            for I in ti.grouped(x):
                x[I] += alpha * p[I]
        for I in ti.grouped(r):
            if ti.static(same_domain):
                x[I] += alpha * p[I]
            value = r[I] - alpha * Ap[I]
            r[I] = value
            norm_sq += value * value
        return norm_sq

    @ti.kernel
    def update_p(p: ti.template(), r: ti.template(), beta: solver_dtype):
        for I in ti.grouped(p):
            p[I] = r[I] + beta * p[I]

    return init, update_solution, update_p


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

    return init, update_p, update_s, squared_norm_and_dot


@lru_cache(maxsize=4)
def bicgstab_solution_kernel(solver_dtype, pair_dots):
    result_type = ti.types.vector(2, solver_dtype) if pair_dots else solver_dtype

    @ti.kernel
    def update_solution(
        x: ti.template(), p_hat: ti.template(), s_hat: ti.template(), r: ti.template(),
        s: ti.template(), t: ti.template(), shadow: ti.template(), alpha: solver_dtype,
        omega: solver_dtype, same_domain: ti.template(),
    ) -> result_type:
        norm_sq = solver_dtype(0.0)
        dot = solver_dtype(0.0)
        if ti.static(not same_domain):
            for I in ti.grouped(x):
                x[I] += alpha * p_hat[I] + omega * s_hat[I]
        for I in ti.grouped(r):
            if ti.static(same_domain):
                x[I] += alpha * p_hat[I] + omega * s_hat[I]
            value = s[I] - omega * t[I]
            r[I] = value
            norm_sq += value * value
            if ti.static(pair_dots):
                dot += value * shadow[I]
        if ti.static(pair_dots):
            return ti.Vector([norm_sq, dot])
        else:
            return norm_sq

    return update_solution
