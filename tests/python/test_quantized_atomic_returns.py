import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("word_bits", [32, 64])
@pytest.mark.parametrize("kind", ["signed", "unsigned", "fixed"])
@pytest.mark.parametrize("layout", ["bitpacked", "quant_array"])
@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    vulkan_quant_experimental=True,
    quant_opt_atomic_demotion=False,
    offline_cache=False,
)
def test_quantized_atomic_add_returns_each_previous_value(word_bits, kind, layout):
    if (
        ti.lang.impl.current_cfg().arch == ti.vulkan
        and word_bits == 64
        and not ti.lang.impl.get_runtime().prog.get_device_caps().get("spirv_has_atomic_int64")
    ):
        pytest.skip("64-bit packed atomics require device integer atomics")
    dtype = ti.types.quant.fixed(7, scale=0.25) if kind == "fixed" else ti.types.quant.int(7, signed=kind == "signed")
    x = ti.field(dtype)
    neighbor = (
        ti.field(ti.types.quant.int(32 if word_bits == 64 else 9, signed=False)) if layout == "bitpacked" else None
    )
    if layout == "bitpacked":
        packed = ti.BitpackedFields(max_num_bits=word_bits)
        packed.place(neighbor, x)
        ti.root.place(packed)
    else:
        ti.root.quant_array(ti.i, 8 if word_bits == 64 else 4, max_num_bits=word_bits).place(x)
    factor = 0.25 if kind == "fixed" else 1
    target = () if layout == "bitpacked" else (5 if word_bits == 64 else 1,)
    old = ti.field(ti.f32, shape=8)

    @ti.kernel
    def seed(first: ti.i32):
        x[target] = first * factor
        if ti.static(layout == "bitpacked"):
            neighbor[None] = 113
        else:
            x[0] = 7 * factor
            x[2] = 13 * factor

    @ti.kernel
    def add(step: ti.i32):
        for i in range(8):
            old[i] = ti.atomic_add(x[target], step * factor)

    for first in ((3, 124) if kind == "unsigned" else (-4, 60)):
        seed(first)
        add(1)

        def expected(value):
            value %= 128
            return (value - 128 if kind != "unsigned" and value >= 64 else value) * factor

        assert sorted(old.to_numpy().tolist()) == sorted(expected(first + i) for i in range(8))
        assert x[target] == expected(first + 8)
        add(-1)
        assert sorted(old.to_numpy().tolist()) == sorted(expected(first + i) for i in range(1, 9))
        assert x[target] == expected(first)
        if layout == "bitpacked":
            assert neighbor[None] == 113
        else:
            assert x[0] == 7 * factor
            assert x[2] == 13 * factor


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], vulkan_quant_experimental=True, offline_cache=False)
def test_quantized_signed_load_and_atomic_return_sign_extend():
    if (
        ti.lang.impl.current_cfg().arch == ti.vulkan
        and not ti.lang.impl.get_runtime().prog.get_device_caps().get("spirv_has_int64")
    ):
        pytest.skip("64-bit compute type requires device integer support")
    x = ti.field(ti.types.quant.int(7, signed=True, compute=ti.i64))
    ti.root.quant_array(ti.i, 4, max_num_bits=32).place(x)
    old = ti.field(ti.i64, shape=4)
    x[1] = -2
    assert x[1] == -2

    @ti.kernel
    def add():
        for i in range(4):
            old[i] = ti.atomic_add(x[1], 1)

    add()
    assert sorted(old.to_numpy().tolist()) == [-2, -1, 0, 1]
    assert x[1] == 2
