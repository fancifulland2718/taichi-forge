"""Basic HIP backend contracts; no Graph or vendor-library qualification."""

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import misc


def test_amdgpu_selection_respects_fallback(monkeypatch):
    monkeypatch.setattr(misc._ti_core, "with_amdgpu", lambda: False)
    with pytest.raises(RuntimeError, match="not supported"):
        misc.adaptive_arch_select(ti.amdgpu, enable_fallback=False)
    assert misc.adaptive_arch_select(ti.amdgpu, enable_fallback=True) == ti.cpu
    assert misc.adaptive_arch_select([ti.amdgpu, ti.cpu], enable_fallback=False) == ti.cpu


def test_amdgpu_selection_preserves_requested_backend(monkeypatch):
    monkeypatch.setattr(misc._ti_core, "with_amdgpu", lambda: True)
    assert misc.adaptive_arch_select(ti.amdgpu, enable_fallback=False) == ti.amdgpu


@pytest.fixture
def amdgpu_runtime():
    # The general backend matrix intentionally does not include AMDGPU yet.
    # This bounded basic test must still run on a HIP-capable machine.
    if not misc.is_arch_supported(ti.amdgpu):
        pytest.skip("HIP runtime / AMD device unavailable")
    ti.init(arch=ti.amdgpu, enable_fallback=False, offline_cache=False)
    try:
        yield
    finally:
        ti.reset()


def test_amdgpu_dense_field_ndarray_and_device_math(amdgpu_runtime):
    # Non-block-aligned length covers the tail of range kernels as well.
    n = 4099
    field = ti.field(ti.f32, shape=n)
    output = ti.ndarray(ti.f32, shape=n)
    source = np.arange(n, dtype=np.float32) / n

    @ti.kernel
    def run(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            field[i] = ti.sin(values[i]) + 2.0 * values[i]
            output[i] = field[i] + 1.0

    run(source)
    ti.sync()
    expected = np.sin(source) + 2.0 * source
    np.testing.assert_allclose(field.to_numpy(), expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(output.to_numpy(), expected + 1, rtol=2e-5, atol=2e-6)
    # Reuse device bindings with new data, not a second initialization.
    source *= 0.5
    run(source)
    np.testing.assert_allclose(
        output.to_numpy(), np.sin(source) + 2.0 * source + 1, rtol=2e-5, atol=2e-6
    )
