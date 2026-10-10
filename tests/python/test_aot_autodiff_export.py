import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_exports_explicit_adjoint_and_rejects_name_collision(tmp_path):
    x = ti.field(ti.f32, shape=4, needs_grad=True)
    y = ti.field(ti.f32, shape=4, needs_grad=True)

    @ti.kernel
    def square():
        for i in x:
            y[i] = x[i] * x[i]

    module = ti.aot.Module()
    module.add_kernel(square)
    module.add_kernel(square.grad)
    module.add_kernel(square.grad)  # Identical exports are idempotent.
    with pytest.raises(ti.TaichiCompilationError, match="already bound"):
        module.add_kernel(square.grad, name="square")
    module.save(tmp_path)
    assert (tmp_path / "__content__").read_text().splitlines() == ["kernel:square", "kernel:square_grad"]
    assert "square_grad" in (tmp_path / "metadata.json").read_text()


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_aot_bound_adjoint_and_template_export(tmp_path):
    @ti.data_oriented
    class Energy:
        def __init__(self):
            self.x = ti.field(ti.f32, shape=4, needs_grad=True)
            self.y = ti.field(ti.f32, shape=4, needs_grad=True)

        @ti.kernel
        def evaluate(self, factor: ti.template()):
            for i in self.x:
                self.y[i] = factor * self.x[i] * self.x[i]

    energy = Energy()
    module = ti.aot.Module()
    module.add_kernel(energy.evaluate, template_args={"factor": 2})
    module.add_kernel(energy.evaluate.grad, template_args={"factor": 2})
    with pytest.raises(ti.TaichiCompilationError, match="owner"):
        module.add_kernel(energy.evaluate, template_args={"self": energy, "factor": 2})
    with module.add_kernel_template(energy.evaluate.grad) as template:
        template.instantiate(factor=3)
        template.instantiate(factor=3)
    module.save(tmp_path)
    metadata = (tmp_path / "metadata.json").read_text()
    assert "evaluate_grad__tmpl__factor=3__" in metadata
    assert "evaluate_grad" in metadata
