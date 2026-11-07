import gc
import importlib.util
import weakref
from types import ModuleType

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


def _pose_factory(fields):
    pose_fields = tuple(fields)
    field_count = len(pose_fields)
    pose_type = ti.types.vector(field_count * 3, ti.f32)

    @ti.kernel
    def upload(pose: pose_type):
        for field_index in ti.static(range(field_count)):
            base = field_index * 3
            pose_fields[field_index][None] = ti.Vector([pose[base], pose[base + 1], pose[base + 2]])

    return upload


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_lexical_resources_use_declared_fields_and_independent_instance_bindings():
    from types import MappingProxyType

    class Camera:
        def __init__(self):
            builder = ti.FieldsBuilder()
            scope = ti.experimental.ResourceScope(self, builder=builder)
            scope.position = scope.vector_field(3, dtype=ti.f32, shape=())
            scope.direction = scope.vector_field(3, dtype=ti.f32, shape=())
            self.tree = builder.finalize()

    first, second = Camera(), Camera()
    upload = _pose_factory((first.position, first.direction))
    function = upload._primal.func
    environment = Component.from_owner(first)
    environment._lexical_bindings = MappingProxyType(
        {
            (function, "pose_fields"): ("sequence", (("path", ("position",)), ("path", ("direction",)))),
            (function, "field_count"): ("static", ("int", 2)),
        }
    )
    values = ti.Vector([1, 2, 3, 4, 5, 6])
    code = environment.compile(upload, values, owner=first)
    assert len(code.contract["groups"]) == 1
    assert {name for name, _ in code.contract["captures"]} == {"position", "direction"}
    code.run(environment, second, values)
    np.testing.assert_array_equal(first.position.to_numpy(), [0, 0, 0])
    np.testing.assert_array_equal(second.position.to_numpy(), [1, 2, 3])
    np.testing.assert_array_equal(second.direction.to_numpy(), [4, 5, 6])
    assert environment.compile(upload, values, owner=second) is code
    # The original closure and JIT path are unchanged, even after AOT lowering.
    upload(values * -1)
    np.testing.assert_array_equal(first.position.to_numpy(), [-1, -2, -3])
    np.testing.assert_array_equal(second.position.to_numpy(), [1, 2, 3])


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_lexical_bindings_are_lazy_and_reject_changed_used_paths():
    from types import MappingProxyType, SimpleNamespace

    first, second = ti.ndarray(ti.i32, 7), ti.ndarray(ti.i32, 13)
    fields = (first, second)

    @ti.kernel
    def fill(value: ti.i32):
        for i in fields[0]:
            fields[0][i] = value

    owner = SimpleNamespace(left=first, right=second)
    environment = Component(left=Array(ti.i32, 1), right=Array(ti.i32, 1))
    function = fill._primal.func
    environment._lexical_bindings = MappingProxyType(
        {
            (function, "fields"): ("sequence", (("path", ("left",)), ("path", ("right",)))),
        }
    )
    code = environment.compile(fill, 3, owner=owner)
    assert [name for name, _ in code.contract["captures"]] == ["left"]
    environment._lexical_bindings = MappingProxyType(
        {
            (function, "fields"): ("sequence", (("path", ("left",)), ("path", ("missing_unused",)))),
        }
    )
    assert environment.compile(fill, 4, owner=owner) is code
    code.run(environment, owner, 4)
    np.testing.assert_array_equal(first.to_numpy(), np.full(7, 4))
    environment._lexical_bindings = MappingProxyType(
        {
            (function, "fields"): ("sequence", (("path", ("right",)), ("path", ("left",)))),
        }
    )
    with pytest.raises(ValueError, match="bindings do not match"):
        code.run(environment, owner, 5)
    other = environment.compile(fill, 5, owner=owner)
    assert other is not code
    other.run(environment, owner, 5)
    np.testing.assert_array_equal(second.to_numpy(), np.full(13, 5))
    environment._lexical_bindings = MappingProxyType({})
    with pytest.raises(ti.TaichiCompilationError, match="undeclared global/closure capture"):
        environment.compile(fill, 1, owner=owner)


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


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_legacy_import_of_same_frontend_is_a_dsl_module():
    spec = importlib.util.spec_from_file_location("taichi", ti.__file__, submodule_search_locations=list(ti.__path__))
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    unrelated = ModuleType("taichi")
    unrelated.cast, unrelated.f32 = ti.cast, ti.f32

    @ti.data_oriented
    class Model:
        @ti.kernel
        def cast(self, value: ti.i32) -> ti.f32:
            return compat.cast(value, compat.f32)

        @ti.kernel
        def invalid(self, value: ti.i32) -> ti.f32:
            return unrelated.cast(value, unrelated.f32)

    owner, environment = Model(), Component()
    code = environment.compile(owner.cast, 7)
    assert code.run(environment, owner, 13) == 13.0
    with pytest.raises(ti.TaichiCompilationError, match="undeclared global/closure capture 'unrelated'"):
        environment.compile(owner.invalid, 7)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False, fast_math=False)
def test_svd_intrinsic_results_are_inline_arguments_not_model_captures():
    @ti.data_oriented
    class Model:
        def __init__(self, matrices):
            self.source = ti.Matrix.ndarray(3, 3, ti.f32, len(matrices))
            self.u = ti.Matrix.ndarray(3, 3, ti.f32, len(matrices))
            self.s = ti.Matrix.ndarray(3, 3, ti.f32, len(matrices))
            self.v = ti.Matrix.ndarray(3, 3, ti.f32, len(matrices))
            self.source.from_numpy(matrices)

        @ti.kernel
        def decompose(self):
            for i in range(self.source.shape[0]):
                self.u[i], self.s[i], self.v[i] = ti.svd(self.source[i])

    matrix = ti.types.matrix(3, 3, ti.f32)
    environment = Component(**{name: Array(matrix, 1) for name in ("source", "u", "s", "v")})
    matrices = np.array(
        [[[1, 0.3, -0.2], [0, 1.2, 0.1], [0.1, 0, 0.8]], np.diag([-0.5, 2.0, 1.5]), np.diag([0.0, 1.0, 1.0])],
        dtype=np.float32,
    )
    owners = [Model(matrices[:1]), Model(matrices)]
    code = environment.compile(owners[0].decompose)
    assert environment.compile(owners[1].decompose) is code
    assert {capture[0] for capture in code.contract["captures"]} == {"source", "u", "s", "v"}
    for owner in owners:
        code.run(environment, owner)
        u, s, v = (getattr(owner, name).to_numpy() for name in ("u", "s", "v"))
        source = owner.source.to_numpy()
        np.testing.assert_allclose(u @ s @ v.transpose(0, 2, 1), source, atol=2e-5, rtol=2e-5)
        identity = np.broadcast_to(np.eye(3), u.shape)
        np.testing.assert_allclose(u.transpose(0, 2, 1) @ u, identity, atol=2e-5)
        np.testing.assert_allclose(v.transpose(0, 2, 1) @ v, identity, atol=2e-5)
        expected = np.linalg.svd(source, compute_uv=False)
        actual = np.sort(np.abs(np.diagonal(s, axis1=1, axis2=2)), axis=1)[:, ::-1]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
