import numpy as np

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_private_accumulator_matches_observable_atomics():
    width, height = 19, 11
    output = ti.field(ti.f32, shape=(2, width, height))
    reference = ti.field(ti.f32, shape=(width, height))
    previous = ti.field(ti.f32, shape=(width, height, 3))
    value = ti.field(ti.f32, shape=3)
    scale = ti.field(ti.f32, shape=3)

    @ti.kernel
    def accumulate(plane: ti.i32):
        for cell in range(width * height):
            for k in range(3):
                row = cell // width
                col = cell - row * width
                output[plane, col, row] += value[k] * scale[k]

    @ti.kernel
    def observed():
        for cell in range(width * height):
            for k in range(3):
                row, col = cell // width, cell % width
                previous[col, row, k] = ti.atomic_add(reference[col, row], value[k] * scale[k])

    # 4097**2 rounds before adding. Contraction would leave 1 instead of 0.
    for initial, terms in ((-16785408.0, [4097.0, 0, 0]), (0.0, [0, 2**-127, 2**-127])):
        output.fill(initial)
        reference.fill(initial)
        value.from_numpy(np.array(terms, dtype=np.float32))
        scale.from_numpy(np.array([4097, 1, 1], dtype=np.float32))
        for _ in range(2):
            accumulate(1)
            observed()
            np.testing.assert_array_equal(output.to_numpy()[1].view(np.uint32), reference.to_numpy().view(np.uint32))
        assert np.all(output.to_numpy()[0] == initial)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_shared_and_observed_accumulators():
    width, height = 17, 9
    shared = ti.field(ti.f32, shape=width)
    cell = ti.field(ti.f32, shape=(height, width))

    @ti.kernel
    def update():
        for i in range(height * width):
            for k in range(7):
                shared[i % width] += 1
        for i in range(height * width):
            for k in range(4):
                row, col = i // width, i % width
                cell[row, col] += cell[row, col] + 1

    cell.fill(1)
    update()
    assert np.all(shared.to_numpy() == height * 7)
    assert np.all(cell.to_numpy() == 31)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_private_primal_with_shared_adjoint():
    width, height = 7, 5
    output = ti.field(ti.f32, shape=(height, width), needs_grad=True)
    weights = ti.field(ti.f32, shape=4, needs_grad=True)

    @ti.kernel
    def accumulate():
        for i in range(height * width):
            for k in range(4):
                output[i // width, i % width] += weights[k] * (k + 1)

    weights.fill(2)
    output.grad.fill(1)
    accumulate()
    accumulate.grad()
    assert np.all(output.to_numpy() == 20)
    np.testing.assert_array_equal(weights.grad.to_numpy(), width * height * np.arange(1, 5))
