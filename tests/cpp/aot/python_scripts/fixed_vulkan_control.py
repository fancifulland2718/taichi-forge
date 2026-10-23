"""Small precompiled payload for native Vulkan control/retirement tests."""

import os

import taichi_forge as ti


ti.init(arch=ti.vulkan, offline_cache=False)
if ti.lang.impl.current_cfg().arch != ti.vulkan:
    raise SystemExit(0)


@ti.kernel
def condition(state: ti.types.ndarray(ti.i32, 1)):
    state[5] += 1
    state[0] = ti.cast(state[1] < state[4], ti.i32)
    state[2] = 2 - state[0]


@ti.kernel
def advance(state: ti.types.ndarray(ti.i32, 1)):
    state[3] *= 2
    state[1] += 1


@ti.kernel
def adjoint(state: ti.types.ndarray(ti.i32, 1)):
    # Explicit adjoint of the executed doubling steps and squared output.
    state[6] = 2 * state[3] * (1 << state[1])


@ti.kernel
def condition_at(state: ti.types.ndarray(ti.i32, 1), base: ti.i32):
    state[base] = ti.cast(state[base + 1] < 2, ti.i32)
    state[base + 2] = 2 - state[base]


@ti.kernel
def advance_at(state: ti.types.ndarray(ti.i32, 1), base: ti.i32):
    state[base + 1] += 1


module = ti.aot.Module()
for name, kernel in (("condition", condition), ("advance", advance), ("adjoint", adjoint),
                     ("condition_at", condition_at), ("advance_at", advance_at)):
    module.add_kernel(kernel, name=name)
module.save(os.environ["TAICHI_AOT_FOLDER_PATH"])
