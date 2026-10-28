import gc
import weakref

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.aot._component import Array, Component, RuntimeValue, StaticValue
from taichi_forge.lang import kernel_impl
from tests import test_utils


@ti.data_oriented
class _Model:
    def __init__(self, n, gravity=(1, 2, 3), scale=True):
        self.n = n
        self.gravity = ti.Vector(gravity)
        self.scale = scale
        self.out = ti.ndarray(ti.f32, n)

    @ti.func
    def value(self, i):
        value = self.gravity[1] + i
        if ti.static(self.scale):
            value *= 2
        return value

    @ti.kernel
    def evaluate(self, extra: ti.f32):
        for i in range(self.n):
            self.out[i] = self.value(i) + extra

    @ti.kernel
    def illegal_static(self):
        for i in ti.static(range(self.n)):
            self.out[i] = i

    @ti.kernel
    def missing(self):
        self.out[0] = self.undeclared

    @ti.kernel
    def conditional(self, flag: ti.i32) -> ti.f32:
        value = 0.0
        if flag:
            value = self.gravity[0]
        return value + self.gravity[1]


def _declarations(**extra):
    return Component(
        n=RuntimeValue(ti.i32),
        gravity=RuntimeValue(ti.types.vector(3, ti.f32)),
        scale=StaticValue(),
        out=Array(ti.f32, 1),
        **extra,
    )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_used_abi_single_lowering_and_independent_bindings(monkeypatch):
    lowerings = []
    original = kernel_impl.transform_tree

    def observe(tree, ctx):
        if ctx.is_kernel:
            lowerings.append(ctx.func.func.__name__)
        return original(tree, ctx)

    monkeypatch.setattr(kernel_impl, "transform_tree", observe)
    first = _Model(7)
    second = _Model(37, gravity=(0, -4, 0))
    env = _declarations()
    expanded = _declarations(unused_array=Array(ti.f64, 3), unused_option=StaticValue())
    code = env.compile(first.evaluate, 5.0)
    assert expanded.compile(second.evaluate, -1.0) is code
    assert lowerings == ["evaluate"]
    assert {item[0] for item in code.contract["captures"]} == {"n", "out", "gravity"}
    assert "unused" not in str(code.contract)

    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)
    try:
        code.run(env, first, 5.0)
        code.run(expanded, second, -1.0)
        code.run(env, first, 3.0)
        assert runtime.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 1
    finally:
        runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
    np.testing.assert_array_equal(first.out.to_numpy(), (np.arange(7) + 2) * 2 + 3)
    np.testing.assert_array_equal(second.out.to_numpy(), (np.arange(37) - 4) * 2 - 1)

    # Ordinary JIT remains on the original numerical definition/cache path.
    normal = _Model(7)
    normal.out = ti.field(ti.f32, shape=7)
    normal.evaluate(3.0)
    np.testing.assert_array_equal(normal.out.to_numpy(), (np.arange(7) + 2) * 2 + 3)
    second.scale = False
    changed = expanded.compile(second.evaluate, 1.0)
    assert changed.semantic_key != code.semantic_key
    changed.run(expanded, second, 1.0)
    np.testing.assert_array_equal(second.out.to_numpy(), np.arange(37) - 3)
    with pytest.raises(ValueError, match="static dependencies"):
        code.run(expanded, second, 1.0)
    owner_ref = weakref.ref(first)
    del first
    gc.collect()
    assert owner_ref() is None


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_capture_failure_is_not_published_and_retry_is_valid():
    owner = _Model(7)
    owner.undeclared = 42
    env = _declarations()
    kernel = owner.missing._primal
    runtime = ti.lang.impl.get_runtime()
    count = runtime._compiled_specialization_count
    with pytest.raises(ti.TaichiCompilationError, match="undeclared capture self.undeclared"):
        env.compile(owner.missing)
    assert not kernel.compiled_kernels
    assert not kernel._component_candidates
    assert kernel._deployment_context is None
    assert runtime._compiled_specialization_count == count
    corrected = _declarations(undeclared=RuntimeValue(ti.i32))
    code = corrected.compile(owner.missing)
    code.run(corrected, owner)
    assert owner.out[0] == 42
    with pytest.raises(ti.TaichiCompilationError, match="runtime capture self.n"):
        env.compile(owner.illegal_static)
    assert not owner.illegal_static._primal.compiled_kernels
    ti.reset()
    with pytest.raises(RuntimeError, match="retired Runtime"):
        code.run(corrected, owner)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_nested_definition_guard_and_conditional_value_scope(monkeypatch):
    owner = _Model(7)
    env = _declarations()
    first = env.compile(owner.evaluate, 0.0)

    @ti.func
    def replacement(self, i):
        return self.gravity[2] - i

    monkeypatch.setattr(_Model, "value", replacement)
    second = env.compile(owner.evaluate, 0.0)
    assert second is not first
    assert second.semantic_key != first.semantic_key
    second.run(env, owner, 0.0)
    np.testing.assert_array_equal(owner.out.to_numpy(), 3 - np.arange(7))
    conditional = env.compile(owner.conditional, 0)
    assert conditional.run(env, owner, 0) == 2
    assert conditional.run(env, owner, 1) == 3
    assert {item[0] for item in conditional.contract["captures"]} == {"gravity"}


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_uncategorized_closure_and_field_bindings_are_rejected():
    scale = 17

    @ti.data_oriented
    class Model:
        @ti.kernel
        def read(self) -> ti.i32:
            return scale

    with pytest.raises(ti.TaichiCompilationError, match="undeclared global/closure capture 'scale'"):
        Component().compile(Model().read)
    owner = _Model(7)
    owner.out = ti.field(ti.f32, shape=7)
    with pytest.raises(ti.TaichiCompilationError, match="SNode layout contract"):
        _declarations().compile(owner.evaluate, 0.0)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_internal_matrix_functions_and_stable_definition_contents():
    from taichi_forge.aot._component import _definition

    @ti.data_oriented
    class Model:
        @ti.kernel
        def cross(self, x: ti.f32) -> ti.f32:
            return ti.Vector([x, 2.0, 3.0]).cross(ti.Vector([0.0, 1.0, 0.0]))[2]

    owner = Model()
    function = owner.cross._primal.func
    original = _definition(function)
    # Holding references to constants must not change a content-based key.
    retained_constants = list(function.__code__.co_consts)
    env = Component()
    code = env.compile(owner.cross, 7.0)
    assert _definition(function) == original
    assert env.compile(Model().cross, 19.0) is code
    assert code.run(env, owner, 19.0) == 19.0
    assert retained_constants == list(function.__code__.co_consts)
