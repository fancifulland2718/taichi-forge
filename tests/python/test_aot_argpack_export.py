"""ArgPack producer specialization; consumer ownership is tested separately."""

import json
import time

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_argpack_nested_resources_and_explicit_adjoint(tmp_path):
    settings = ti.types.struct(scale=ti.f32, bias=ti.types.vector(2, ti.f32))
    child = ti.types.argpack(settings=settings)
    pack_type = ti.types.argpack(
        source=ti.types.ndarray(ti.f32, 1, needs_grad=True),
        basis=ti.types.matrix(2, 2, ti.f32), child=child,
    )
    source = ti.ndarray(ti.f32, 3, needs_grad=True)
    source.from_numpy(np.array([1, -2, 3], dtype=np.float32))
    loss = ti.ndarray(ti.f32, 1, needs_grad=True)
    pack = pack_type(source=source, basis=[[3, 0], [0, 5]],
                     child=child(settings=settings(scale=2, bias=[1, -1])))

    @ti.kernel
    def energy(p: pack_type, total: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.source:
            total[0] += (p.child.settings.scale * p.basis[i % 2, i % 2] * p.source[i] ** 2
                         + p.child.settings.bias[i % 2] * p.source[i])

    loss.fill(0)
    loss.grad.fill(1)
    source.grad.fill(0)
    energy(pack, loss)
    energy.grad(pack, loss)
    np.testing.assert_allclose(loss.to_numpy(), [106])
    np.testing.assert_allclose(source.grad.to_numpy(), [13, -41, 37])

    started = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(energy, name="automatic")
    module.add_kernel(energy, template_args={"p": pack, "total": loss}, name="explicit")
    module.add_kernel(energy.grad, name="backward")
    assert module._exported_kernels["automatic"] is module._exported_kernels["explicit"]
    module.archive(str(tmp_path / "packs.tcm"))
    np.testing.assert_allclose(loss.to_numpy(), [106])  # Export does not execute.
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(tmp_path / "packs.tcm")
        for name in ("automatic", "explicit", "backward"):
            entry = loaded.kernel(name)
            assert 0 not in entry._argument_layout.encoders  # A pack is a resource.
    print(json.dumps(dict(argpack_export_seconds=time.perf_counter() - started,
                          artifact_bytes=(tmp_path / "packs.tcm").stat().st_size)))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_argpack_template_specialization_and_errors(tmp_path):
    concrete = ti.types.argpack(offset=ti.i32, values=ti.types.ndarray(ti.f32, 1))
    generic = ti.types.argpack(child=ti.types.argpack(values=ti.types.ndarray()))

    @ti.kernel
    def fixed(p: concrete, scale: ti.template()) -> ti.f32:
        return scale * p.values[0] + p.offset

    @ti.kernel
    def inferred(p: generic, scale: ti.template()) -> ti.f32:
        return scale * p.child.values[0]

    values = ti.ndarray(ti.f32, 1)
    values.fill(3)
    exemplar = concrete(offset=7, values=values)
    nested = generic(child=generic.members["child"](values=values))
    assert fixed(exemplar, 2) == 13
    assert inferred(nested, 2) == 6
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiCompilationError, match="p.child.values.*template_args"):
        module.add_kernel(inferred, template_args={"scale": 2})
    with pytest.raises(ti.TaichiCompilationError, match="missing.*scale.*unexpected.*typo"):
        module.add_kernel(fixed, template_args={"typo": 2})
    with pytest.raises(ti.TaichiRuntimeTypeError, match="argument pack"):
        module.add_kernel(fixed, template_args={"p": 0, "scale": 2})
    module.add_kernel(fixed, template_args={"scale": 2}, name="fixed")
    module.add_kernel(inferred, template_args={"p": nested, "scale": 2}, name="inferred")
    with module.add_kernel_template(fixed) as family:
        family.instantiate(scale=2)
        family.instantiate(p=exemplar, scale=2)
        assert len(family._instantiated_keys) == 1
    with module.add_kernel_template(inferred) as family:
        family.instantiate(p=nested, scale=2)
        assert len(family._instantiated_keys) == 1
    module.archive(str(tmp_path / "specialized.tcm"))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(tmp_path / "specialized.tcm")
        for name in module._exported_kernels:
            assert loaded.kernel(name)._return_layout.has_values
    exemplar.offset = 11
    values.fill(5)
    assert fixed(exemplar, 2) == 21
    assert inferred(nested, 2) == 10
