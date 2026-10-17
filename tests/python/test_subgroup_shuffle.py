"""Exercise subgroup exchange without assuming a 32-lane Vulkan subgroup."""

import shutil
import subprocess

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang.simt import subgroup
from tests import test_utils


OPERATIONS = ("shuffle", "shuffle_up", "shuffle_down", "shuffle_xor")


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("index_kind", ["literal", "signed", "unsigned"])
@test_utils.test(arch=ti.vulkan, offline_cache=False, compile_tier="fast")
def test_shuffle_integer_indices(operation, index_kind, tmp_path, monkeypatch):
    if not subgroup.is_shuffle_supported(operation):
        pytest.skip("device lacks the required shuffle capability")
    monkeypatch.setenv("TI_VULKAN_VALIDATE_SPIRV", "1")
    out = ti.field(ti.i32, shape=32)
    exchange = getattr(subgroup, operation)

    @ti.kernel
    def run(offset: ti.i32):
        ti.loop_config(block_dim=32)
        for i in range(32):
            # Uniform values let this test exercise plain zero for all four
            # operations without assuming a hardware subgroup width.
            if ti.static(index_kind == "literal"):
                out[i] = exchange(37, 0)
            elif ti.static(index_kind == "signed"):
                out[i] = exchange(37, offset)
            else:
                out[i] = exchange(37, ti.u32(offset))

    ti.tools.export_vulkan_kernel(run, 0, directory=tmp_path / "jit", validate=True)
    run(0)
    np.testing.assert_array_equal(out.to_numpy(), np.full(32, 37, np.int32))


@pytest.mark.parametrize("index", [-1, 2**32, 0.5])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_shuffle_rejects_invalid_python_indices(index):
    if not subgroup.is_shuffle_supported():
        pytest.skip("device lacks shuffle support")

    @ti.kernel
    def run() -> ti.i32:
        return subgroup.shuffle(1, ti.static(index))

    with pytest.raises(ti.TaichiCompilationError, match="index must be"):
        run()


@pytest.mark.parametrize("dtype", [ti.i32, ti.u32, ti.f32])
@pytest.mark.parametrize("block_dim", [32, 64, 128])
@pytest.mark.parametrize("divergent", [False, True])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_shuffle_values_and_spirv(dtype, block_dim, divergent, tmp_path):
    if not all(subgroup.is_shuffle_supported(op) for op in OPERATIONS):
        pytest.skip("device lacks a required shuffle capability")
    validator = shutil.which("spirv-val")
    if validator is None:
        pytest.skip("spirv-val is required before shader execution")
    # A partial final workgroup and subgroup, in addition to divergent lanes.
    count = 137
    values = ti.field(dtype, shape=(count, 4))
    metadata = ti.field(ti.i32, shape=(count, 2))

    @ti.kernel
    def exchange(delta: ti.u32):
        ti.loop_config(block_dim=block_dim)
        for i in range(count):
            lane = subgroup.invocation_id()
            width = subgroup.group_size()
            metadata[i, 0] = lane
            metadata[i, 1] = width
            if ti.static(not divergent) or lane % 2 == 0:
                value = ti.cast(i * 3 + 17, dtype)
                if ti.static(dtype == ti.u32):
                    value += ti.u32(0x80000000)
                elif ti.static(dtype == ti.i32):
                    value -= 1000
                else:
                    value -= 1000.25
                source = ti.cast(lane, ti.u32) ^ delta
                values[i, 0] = subgroup.shuffle(value, source)
                values[i, 1] = subgroup.shuffle_up(value, delta)
                values[i, 2] = subgroup.shuffle_down(value, delta)
                values[i, 3] = subgroup.shuffle_xor(value, delta)

    module = ti.aot.Module(
        ti.vulkan,
        caps=[
            "spirv_has_subgroup_basic",
            "spirv_has_subgroup_shuffle",
            "spirv_has_subgroup_shuffle_relative",
        ],
    )
    module.add_kernel(exchange)
    module.save(str(tmp_path))
    shaders = list(tmp_path.glob("*.spv"))
    assert shaders
    for shader in shaders:
        result = subprocess.run(
            [validator, "--target-env", "vulkan1.2", str(shader)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    # Large offsets exercise undefined sources without asserting their values.
    for delta in (0, 2, 128, 0xFFFFFFFF):
        exchange(delta)
        actual = values.to_numpy()
        info = metadata.to_numpy()
        checked = 0
        for i, (lane, width) in enumerate(info):
            assert 0 <= lane < width
            if divergent and lane % 2:
                continue
            sources = (
                int(lane) ^ delta,
                int(lane) - delta,
                int(lane) + delta,
                int(lane) ^ delta,
            )
            for operation, source in enumerate(sources):
                origin = i - int(lane) + source
                if 0 <= source < width and 0 <= origin < count:
                    if not divergent or source % 2 == 0:
                        adjustment = 0x80000000 if dtype == ti.u32 else (-1000 if dtype == ti.i32 else -1000.25)
                        assert actual[i, operation] == origin * 3 + 17 + adjustment
                        checked += 1
        if delta in (0, 2):
            assert checked > 0


@pytest.mark.parametrize("operation", OPERATIONS)
@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_shuffle_rejects_unsupported_backend(operation):
    assert not subgroup.is_shuffle_supported(operation)
    op = getattr(subgroup, operation)

    @ti.kernel
    def invoke() -> ti.i32:
        return op(1, ti.u32(0))

    with pytest.raises(ti.TaichiCompilationError, match="requires Vulkan device capability"):
        invoke()


@test_utils.test(arch=ti.vulkan)
def test_shuffle_missing_capability_and_invalid_dtype(monkeypatch):
    if not subgroup.is_shuffle_supported():
        pytest.skip("device lacks shuffle capability")

    @ti.kernel
    def requires_shuffle() -> ti.i32:
        return subgroup.shuffle(1, ti.u32(0))

    # AOT targets have their own declared capabilities; current-device support
    # must not silently enable an undeclared target requirement.
    with pytest.raises(RuntimeError, match="requires device capability"):
        ti.aot.Module(ti.vulkan).add_kernel(requires_shuffle)

    @ti.kernel
    def invalid_dtype() -> ti.i64:
        return subgroup.shuffle(ti.i64(1), ti.u32(0))

    with pytest.raises(RuntimeError, match="supports scalar i32, u32, or f32"):
        invalid_dtype()

    monkeypatch.setattr(subgroup, "is_shuffle_supported", lambda operation: False)

    @ti.kernel
    def unavailable() -> ti.i32:
        return subgroup.shuffle(1, ti.u32(0))

    with pytest.raises(ti.TaichiCompilationError, match="spirv_has_subgroup_shuffle"):
        unavailable()

    monkeypatch.undo()
    with pytest.raises(ValueError, match="Unknown subgroup shuffle"):
        subgroup.is_shuffle_supported("unknown")
