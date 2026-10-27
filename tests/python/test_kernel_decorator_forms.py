import pytest

import taichi_forge as ti
from tests import test_utils


@ti.kernel
def module_bare(value: ti.i32) -> ti.i32:
    return value + 1


@ti.kernel(opt_level="full")
def module_configured(value: ti.i32) -> ti.i32:
    return value + 2


@pytest.mark.parametrize("form", ["bare", None, "fast", "balanced", "full"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_kernel_decorator_forms_preserve_class_and_module_binding(form):
    decorate = ti.kernel if form == "bare" else ti.kernel(opt_level=form)

    @ti.data_oriented
    class Model:
        def __init__(self, gain):
            self.gain = gain
            self.values = ti.field(ti.i32, shape=17)

        @decorate
        def write(self, factor: ti.template(), offset: ti.i32):
            for i in self.values:
                self.values[i] = i * factor + self.gain + offset

    first, second = Model(3), Model(11)
    assert first.write._primal.classkernel
    assert first.write._primal.opt_level == (None if form == "bare" else form)
    first.write(2, 5)
    assert first.values[16] == 40
    builder = ti.graph.GraphBuilder()
    offset = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "offset", ti.i32)
    for model in (first, second):
        builder.dispatch(model.write, offset, template_args={"self": model, "factor": 4})
    builder.compile().run(dict(offset=7))
    assert first.values[16] == 74
    assert second.values[16] == 82
    for kernel, expected in ((module_bare, 4), (module_configured, 5)):
        assert not kernel._primal.classkernel
        assert kernel(3) == expected


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_configured_methods_keep_real_function_tiers_separate():
    @ti.real_func
    def add_one(value: ti.i32) -> ti.i32:
        return value + 1

    @ti.data_oriented
    class Model:
        @ti.kernel(opt_level="fast")
        def fast(self, value: ti.i32) -> ti.i32:
            return add_one(value)

        @ti.kernel(opt_level="full")
        def full(self, value: ti.i32) -> ti.i32:
            return add_one(value)

    model = Model()
    assert model.fast(3) == 4
    assert len(add_one.func.taichi_functions) == 1
    assert model.full(5) == 6
    assert len(add_one.func.taichi_functions) == 2
