import pytest

import taichi_forge as ti
from tests import test_utils


def seed_for_maximum_first_word(arch):
    # Invert the current xorshift/multiply initialization in integer arithmetic.
    # This targets the rounding boundary without a probabilistic sampling test.
    modulus = 1 << 32
    mask = modulus - 1
    value = (mask * pow(1000000007, -1, modulus)) & mask
    value ^= 88675123 ^ (88675123 >> 19)
    mixed = value ^ (value >> 8) ^ (value >> 16) ^ (value >> 24)
    initial_x = (mixed ^ (mixed << 11) ^ (mixed << 22)) & mask
    multiplier = 123456789 * 1000000007
    if arch == ti.vulkan:
        counter = (initial_x * pow(7654321 * multiplier, -1, modulus) - 1234567)
        counter *= pow(9723451, -1, modulus)
    else:
        counter = initial_x * pow(multiplier, -1, modulus)
    seed = (counter * pow(1048391, -1, modulus)) & mask
    return seed if seed < (1 << 31) else seed - modulus


@pytest.mark.parametrize("dtype", [ti.f16, ti.f32])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], cpu_max_num_threads=1)
def test_random_float_excludes_one(dtype):
    arch = ti.lang.impl.current_cfg().arch
    seed = seed_for_maximum_first_word(arch)

    @ti.kernel
    def integer_sample() -> ti.u32:
        return ti.random(ti.u32)

    @ti.kernel
    def float_sample() -> ti.f32:
        return ti.cast(ti.random(dtype), ti.f32)

    ti.init(arch=arch, random_seed=seed, cpu_max_num_threads=1)
    assert integer_sample() == 0xFFFFFFFF, "boundary seed no longer targets the maximum word"
    ti.init(arch=arch, random_seed=seed, cpu_max_num_threads=1)
    assert 0.0 <= float_sample() < 1.0
