import ast
import textwrap

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.lang import kernel_impl, _definition
from taichi_forge.lang._compiled_template import bind
from taichi_forge.tools._compiled_host import lower_module
from tests import test_utils

SOURCE = """
import taichi_forge as ti
GLOBAL = 2.0

def make(scale):
    @ti.func
    def affine(x, offset=1.0):
        return x * scale + offset + GLOBAL

    @ti.kernel
    def run(x: ti.types.ndarray(ti.f32, 1), y: ti.types.ndarray(ti.f32, 1)):
        for i in x:
            y[i] = affine(x[i])

    def update(value):
        nonlocal scale
        scale = value
    return run, update

@ti.data_oriented
class Model:
    def __init__(self, width):
        self.width = width

    @ti.func
    def square(self, x):
        return x * x

    @ti.kernel
    def run(self, x: ti.types.ndarray(ti.f32, 1), y: ti.types.ndarray(ti.f32, 1)):
        for i in x:
            value = 0.0
            for j in ti.static(range(self.width)):
                value += self.square(x[i] + j)
            y[i] = value

    @staticmethod
    @ti.kernel
    def static_run(x: ti.i32) -> ti.i32:
        return x + 1

@ti.pyfunc
def increment(x):
    return x + 1

@ti.kernel
def call_increment(x: ti.i32) -> ti.i32:
    return increment(x)
"""


def load(source=SOURCE):
    converted, entries = lower_module(textwrap.dedent(source), "compiled_test")
    namespace = {"__name__": "compiled_test"}
    # No backing source file exists, as with a producer-compiled extension.
    exec(compile(converted, "<producer-output>", "exec"), namespace)
    return namespace, entries


@pytest.mark.parametrize("definition_kind", ["source", "compiled"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_compiled_host_specialization_grad_and_scope(monkeypatch, tmp_path, definition_kind):
    if definition_kind == "compiled":
        module, entries = load()
        assert {e["kind"] for e in entries} == {"kernel", "func", "pyfunc"}
    else:
        source = tmp_path / "source_definition.py"
        source.write_text(SOURCE, encoding="utf8")
        module = {"__name__": "source_test", "__file__": str(source)}
        exec(compile(SOURCE, str(source), "exec"), module)
    original = _definition.getsourcefile

    def no_private_source(function):
        assert function.__module__ != "compiled_test", "compiled definitions must not query source"
        return original(function)

    monkeypatch.setattr(_definition, "getsourcefile", no_private_source)
    values = np.arange(7, dtype=np.float32) / 8
    x, y = ti.ndarray(ti.f32, 7, needs_grad=True), ti.ndarray(ti.f32, 7, needs_grad=True)
    x.from_numpy(values)
    models = []
    for width in (2, 4):
        model = module["Model"](width)
        models.append(model)
        model.run(x, y)
        np.testing.assert_allclose(y.to_numpy(), sum((values + j) ** 2 for j in range(width)))
        x.grad.fill(0)
        y.grad.fill(1)
        model.run.grad(x, y)
        np.testing.assert_allclose(x.grad.to_numpy(), sum(2 * (values + j) for j in range(width)))
        assert model.static_run(4) == 5
    assert module["increment"](4) == module["call_increment"](4) == 5
    run, update = module["make"](2)
    update(3)
    run(x, y)
    np.testing.assert_allclose(y.to_numpy(), 3 * values + 3)
    run2, _ = module["make"](5)
    module["GLOBAL"] = 4
    run2(x, y)
    np.testing.assert_allclose(y.to_numpy(), 5 * values + 5)
    run(x, y)
    np.testing.assert_allclose(y.to_numpy(), 3 * values + 3)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_compiled_host_uncached_real_func_and_diagnostic(monkeypatch):
    monkeypatch.setattr(kernel_impl, "_SOURCE_TEMPLATE_CACHE", False)
    module, _ = load("""
        import taichi_forge as ti
        @ti.real_func
        def square(x: ti.i32) -> ti.i32:
            return x * x
        @ti.kernel
        def run(x: ti.i32) -> ti.i32:
            return square(x)
        @ti.kernel
        def bad() -> ti.i32:
            return missing_value
    """)
    assert module["run"](7) == 49
    with pytest.raises(ti.TaichiNameError, match="compiled:compiled_test:bad"):
        module["bad"]()


def test_plain_host_does_not_acquire_compiler_dependency():
    source = "def public_api(x):\n    return x + 1\n"
    assert lower_module(source, "plain") == (source, ())


def test_compiled_template_rejects_wrong_constructor():
    @bind(
        lambda: ast.parse("def wrong(): pass"),
        globals_getter=lambda: {},
        closure_getter=lambda: {},
        is_class=False,
        filename="<compiled:test>",
        annotations={},
    )
    def actual():
        pass

    with pytest.raises(ValueError, match="undecorated function"):
        actual._taichi_compiled_template.instantiate("actual")
