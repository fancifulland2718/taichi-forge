import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("word_bits", [32, 64])
@pytest.mark.parametrize("fixed", [False, True])
@test_utils.test(arch=ti.vulkan, vulkan_quant_experimental=True, offline_cache=False)
def test_parallel_quantized_stores_preserve_other_cells(word_bits, fixed):
    if word_bits == 64 and not ti.lang.impl.get_runtime().prog.get_device_caps().get("spirv_has_atomic_int64"):
        pytest.skip("64-bit packed stores require device integer atomics")
    dtype = ti.types.quant.fixed(7, scale=0.25) if fixed else ti.types.quant.int(7, signed=True)
    x = ti.field(dtype)
    cells = 4 if word_bits == 32 else 8
    count = 2 * cells
    ti.root.dense(ti.i, 2).quant_array(ti.i, cells, max_num_bits=word_bits).place(x)
    out = ti.field(ti.f32, shape=count)

    @ti.kernel
    def seed(offset: ti.i32):
        for i in range(count):
            x[i] = (i + offset) * (0.25 if ti.static(fixed) else 1)

    @ti.kernel
    def update_even():
        for i in range(count // 2):
            x[2 * i] = (10 - i) * (0.25 if ti.static(fixed) else 1)

    @ti.kernel
    def gather():
        for i in x:
            out[i] = x[i]

    factor = 0.25 if fixed else 1
    for offset in (-8, 3, -1):
        seed(offset)
        gather()
        assert out.to_numpy().tolist() == [(i + offset) * factor for i in range(count)]
        update_even()
        gather()
        assert out.to_numpy().tolist() == [(10 - i // 2 if i % 2 == 0 else i + offset) * factor for i in range(count)]
