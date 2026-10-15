"""Small nested control payload; each region is compiled only once."""

import os

import taichi_forge as ti


ti.init(arch=ti.vulkan, offline_cache=False)
if ti.lang.impl.current_cfg().arch != ti.vulkan:
    raise SystemExit(0)


@ti.kernel
def condition(state: ti.types.ndarray(ti.i32, 1), base: ti.i32):
    state[base + 5] += 1
    state[base] = ti.cast(state[base + 1] < state[base + 4], ti.i32)
    state[base + 2] = 2 - state[base]


@ti.kernel
def reset(state: ti.types.ndarray(ti.i32, 1), base: ti.i32):
    state[base + 1] = 0


@ti.kernel
def advance(state: ti.types.ndarray(ti.i32, 1), base: ti.i32):
    state[3] *= 2
    state[7] += 1
    state[base + 1] += 1


@ti.kernel
def outer_advance(state: ti.types.ndarray(ti.i32, 1), base: ti.i32):
    state[base + 1] += 1


@ti.kernel
def adjoint(state: ti.types.ndarray(ti.i32, 1), base: ti.i32):
    # Explicit derivative of the executed doublings followed by a square.
    state[base + 6] = 2 * state[3] * (1 << state[7])


module = ti.aot.Module()
for kernel in (condition, reset, advance, outer_advance, adjoint):
    module.add_kernel(kernel)
module.save(os.environ["TAICHI_AOT_FOLDER_PATH"])
