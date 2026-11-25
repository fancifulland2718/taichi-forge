import shutil
import subprocess

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


# A partial final subgroup also exercises election among the active lanes.
COUNT = 65539


def shader_text(kernel, directory, *args):
    manifest = ti.tools.export_vulkan_kernel(
        kernel, *args, directory=directory, validate=bool(shutil.which("spirv-val"))
    )
    dis = shutil.which("spirv-dis")
    if dis is None:
        return None
    return "\n".join(
        subprocess.check_output([dis, str(directory / shader["file"])], text=True)
        for shader in manifest["shaders"]
    )


@pytest.mark.parametrize("minimum", [True, False])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_scalar_minmax_default(minimum, tmp_path):
    target = ti.field(ti.f32, shape=())
    inputs = ti.field(ti.u32, shape=COUNT)
    op = ti.atomic_min if minimum else ti.atomic_max

    @ti.kernel
    def reduce(n: ti.template()):
        for i in range(n):
            op(target[None], ti.bit_cast(inputs[i], ti.f32))

    @ti.kernel
    def seed(bits: ti.u32):
        target[None] = ti.bit_cast(bits, ti.f32)

    @ti.kernel
    def read() -> ti.u32:
        return ti.bit_cast(target[None], ti.u32)

    caps = ti.lang.impl.get_runtime().prog.get_device_caps()
    text = shader_text(reduce, tmp_path / "large", COUNT)
    group_op = "OpGroupNonUniformFMin" if minimum else "OpGroupNonUniformFMax"
    if text is not None:
        if caps.get("spirv_has_atomic_float_minmax"):
            assert ("OpAtomicFMinEXT" if minimum else "OpAtomicFMaxEXT") in text
            assert group_op not in text
        elif (
            caps.get("spirv_has_subgroup_basic")
            and caps.get("spirv_has_subgroup_arithmetic")
            and caps.get("spirv_version") >= 0x10300
        ):
            assert group_op in text
    small = shader_text(reduce, tmp_path / "small", 1)
    if small is not None:
        assert group_op not in small

    values = np.linspace(-7, 13, COUNT, dtype=np.float32)
    # Make the final partial subgroup solely responsible for the winner.
    values[-1] = -17 if minimum else 23
    inputs.from_numpy(values.view(np.uint32))
    seed(0x3F800000)
    reduce(COUNT)
    assert read() == values[-1:].view(np.uint32)[0]

    # One exceptional lane in an otherwise normal subgroup must still update
    # the destination, including the sign of the winning zero.
    initial = 0x40000000 if minimum else 0xC0000000
    mixed = np.full(COUNT, 0x3F800000 if minimum else 0xBF800000, np.uint32)
    mixed[1] = 0x80000000
    inputs.from_numpy(mixed)
    seed(initial)
    reduce(2)
    expected = read()
    seed(initial)
    reduce(COUNT)
    assert read() == expected

    # Compare raw bits against this driver's scalar operation, including a
    # NaN payload, the two zero signs, infinities and a subnormal on each side.
    for bits in (0, 0x80000000, 1, 0x80000001, 0x7F800000, 0xFF800000, 0x7FC12345):
        for initial, operand in ((bits, 0x3F800000), (0x3F800000, bits)):
            inputs.fill(operand)
            seed(initial)
            reduce(1)
            expected = read()
            seed(initial)
            reduce(COUNT)
            assert read() == expected


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_observable_atomic_old_value(tmp_path):
    target = ti.field(ti.f32, shape=())
    previous = ti.field(ti.f32, shape=COUNT)

    @ti.kernel
    def record_old():
        for i in range(COUNT):
            previous[i] = ti.atomic_min(target[None], 1.0)

    text = shader_text(record_old, tmp_path / "observable")
    if text is not None:
        assert "OpGroupNonUniformFMin" not in text
    target[None] = 2.0
    record_old()
    old = previous.to_numpy()
    assert np.count_nonzero(old == 2.0) == 1
    assert np.count_nonzero(old == 1.0) == COUNT - 1


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_indexed_atomic_min_keeps_destinations_separate(tmp_path):
    buckets = ti.field(ti.f32, shape=2)

    @ti.kernel
    def scatter():
        for i in range(COUNT):
            ti.atomic_min(buckets[i % 2], ti.cast(i, ti.f32))

    text = shader_text(scatter, tmp_path / "indexed")
    if text is not None:
        assert "OpGroupNonUniformFMin" not in text
    buckets.fill(float(COUNT))
    scatter()
    np.testing.assert_array_equal(buckets.to_numpy(), [0.0, 1.0])


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_explicit_subgroup_arithmetic_caps(tmp_path):
    caps = ti.lang.impl.get_runtime().prog.get_device_caps()
    if not (caps.get("spirv_has_subgroup_basic") and caps.get("spirv_has_subgroup_arithmetic")):
        pytest.skip("device lacks subgroup arithmetic or election")

    @ti.kernel
    def explicit(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(137):
            reduced = ti.simt.subgroup.reduce_min(i)
            if ti.simt.subgroup.elect():
                out[reduced] = 1

    out = ti.ndarray(ti.i32, shape=137)
    out.fill(0)
    text = shader_text(explicit, tmp_path / "explicit", out)
    if text is not None:
        assert "OpCapability GroupNonUniformArithmetic" in text
    explicit(out)
    assert out.to_numpy()[0] == 1

    module = ti.aot.Module(ti.vulkan, caps=["spirv_has_subgroup_basic", "spirv_has_subgroup_arithmetic"])
    module.add_kernel(explicit)
    module.save(str(tmp_path))
    shaders = list(tmp_path.glob("*.spv"))
    assert shaders
    validator = shutil.which("spirv-val")
    if validator:
        for shader in shaders:
            subprocess.run([validator, "--target-env", "vulkan1.2", str(shader)], check=True)
