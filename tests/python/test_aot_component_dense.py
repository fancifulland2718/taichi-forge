import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.aot._component import Component, DenseField, Dimension, RuntimeValue
from tests import test_utils


@pytest.mark.parametrize("soa", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_model_tensor_field_indices_share_code_across_element_extents(tmp_path, soa):
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap
    from taichi_forge.aot._component import _ModelTensorField
    from taichi_forge_runtime.aot import Runtime

    if soa and ti.cfg.arch == ti.vulkan:
        pytest.skip("SOA model tensor qualification currently covers CPU/CUDA")

    @ti.data_oriented
    class Model:
        def __init__(self, rows, dofs):
            self.rows, self.dofs = rows, dofs
            self.j = ti.Vector.field(dofs, ti.f32)
            self.alias = self.j
            self.fixed = ti.Vector.field(3, ti.f32)
            self.h = ti.Matrix.field(2, dofs, ti.f32)
            builder = ti.FieldsBuilder()
            if soa:
                builder.dense(ti.i, rows).place(self.fixed)
                for field in (self.j, self.h):
                    field._aot_tensor_soa = True
                    for i in range(field.n):
                        for j in range(field.m):
                            builder.dense(ti.i, rows).place(field.get_scalar_field(i, j))
            else:
                builder.dense(ti.i, rows).place(self.j, self.fixed)
                builder.dense(ti.i, rows).place(self.h)
            self.tree = builder.finalize()

        @ti.func
        def read(self, field: ti.template(), row, column):
            return field[row][column]

        @ti.kernel
        def update(self, scale: ti.f32):
            for i in self.j:
                for d in range(self.j.n):
                    self.j[i][d] += scale + i * 100 + d
                self.fixed[i] += ti.Vector([1.0, 2.0, 3.0])
            for I in ti.grouped(self.h):
                for a in ti.static(range(self.h.n)):
                    for d in range(getattr(self.h, "m")):
                        self.h[I][a, d] = self.read(self.alias, I[0], d) + a * 10

        @ti.kernel
        def local_copy(self):
            for i in self.j:
                local = self.j[i]
                local[0] = 42.0

        def record(self, builder):
            builder.dispatch(self.update, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32))

    env = Component(rows=Dimension(), dofs=Dimension(),
                    j=_ModelTensorField(ti.f32, ("rows",), "state", ("dofs",)),
                    alias=_ModelTensorField(ti.f32, ("rows",), "state", ("dofs",)),
                    fixed=DenseField(ti.types.vector(3, ti.f32), ("rows",), "state"),
                    h=_ModelTensorField(ti.f32, ("rows",), "state", (2, "dofs")))
    first, second = Model(2, 5), Model(3, 65)
    try:
        code = env.compile(first.update, 1.0)
        assert env.compile(second.update, 3.0) is code
        assert set(dict(code.contract["dense_access"]).values()) == {"affine"}
        assert code.contract["members"]["j"] == code.contract["members"]["alias"]
        assert sum(name in ("j", "alias") for name in code.dynamic) == 1
        conflict = Component(**{**env.declarations, "alias": DenseField(ti.types.vector(5, ti.f32), ("rows",), "state")})
        with pytest.raises(ti.TaichiCompilationError, match="conflicting tensor dimensions"):
            conflict.compile(first.update, 1.0)
        local_copy = env.compile(first.local_copy)
        code.run(env, second, 3.0)
        expected = 3 + np.arange(3)[:, None] * 100 + np.arange(65)
        np.testing.assert_array_equal(second.j.to_numpy(), expected)
        np.testing.assert_array_equal(second.h.to_numpy(), expected[:, None, :] + np.arange(2)[None, :, None] * 10)
        np.testing.assert_array_equal(second.fixed.to_numpy(), np.tile([1, 2, 3], (3, 1)))
        local_copy.run(env, second)
        np.testing.assert_array_equal(second.j.to_numpy(), expected)
        np.testing.assert_array_equal(first.j.to_numpy(), np.zeros((2, 5)))
        # Deployment lowering leaves ordinary JIT and its fixed tensor types intact.
        first.update(2.0)
        np.testing.assert_array_equal(first.j.to_numpy(), 2 + np.arange(2)[:, None] * 100 + np.arange(5))
        module = ti.aot.Module()
        code.export(module, "update")
        graph = env.compile_graph(first.record)
        graph.export(module, "step")
        archive = tmp_path / "model_tensor.tcm"
        module.archive(str(archive))
        with Runtime(ti.cfg.arch.name) as runtime, runtime.load_module(archive) as loaded:
            with runtime.dense_snode_group(code.contract["groups"]["state"], {"rows": 4, "dofs": 19}) as group:
                views = {}
                for name in code.dynamic:
                    declaration = env.declarations[name]
                    if isinstance(declaration, DenseField):
                        _, selector = code.contract["members"][name]
                        element = ((19,) if name in ("j", "alias") else (2, 19) if name == "h" else (3,))
                        views[name] = group.schema_member(selector[1], "f32", element_shape=element,
                                                         scalar_axes=isinstance(declaration, _ModelTensorField))
                loaded.kernel("update")(4.0, *(views[name] for name in code.dynamic))
                expected = 4 + np.arange(4)[:, None] * 100 + np.arange(19)
                np.testing.assert_array_equal(views["j"].to_numpy(), expected)
                np.testing.assert_array_equal(views["h"].to_numpy(), expected[:, None, :] + np.arange(2)[None, :, None] * 10)
                np.testing.assert_array_equal(views["fixed"].to_numpy(), np.tile([1, 2, 3], (4, 1)))
                for view in views.values():
                    view.close()
            component = loaded.component("step")
            instances = [component.instantiate(dict(rows=n, dofs=d)) for n, d in ((2, 19), (3, 65))]
            calls = []
            try:
                for instance in instances:
                    instance.initialize(zero=component.contract["resources"])
                    calls.append(instance.prepare(arguments=dict(scale=7.0)))
                calls[0]()
                calls[1]()
                calls[0]()
                for instance, n, d, times in zip(instances, (2, 3), (19, 65), (2, 1)):
                    expected = times * (7 + np.arange(n)[:, None] * 100 + np.arange(d))
                    np.testing.assert_array_equal(instance.resource("j").to_numpy(), expected)
                    np.testing.assert_array_equal(instance.resource("fixed").to_numpy(), np.tile([1, 2, 3], (n, 1)) * times)
            finally:
                for call in calls:
                    call.close()
                for instance in instances:
                    instance.close()
        graph._graph.close()
        consumer = textwrap.dedent("""
            import importlib.abc, sys
            class NoFrontend(importlib.abc.MetaPathFinder):
                def find_spec(self, name, *args):
                    if name.split('.')[0] in ('taichi', 'taichi_forge', 'tests'):
                        raise AssertionError('consumer attempted frontend import: ' + name)
            sys.meta_path.insert(0, NoFrontend())
            import numpy as np
            from taichi_forge_runtime.aot import Runtime
            with Runtime(sys.argv[1]) as runtime, runtime.load_module(sys.argv[2]) as module:
                component = module.component('step')
                with component.instantiate(dict(rows=3, dofs=129)) as instance:
                    instance.initialize(zero=component.contract['resources'])
                    with instance.prepare(arguments=dict(scale=9.0)) as call:
                        call()
                    expected = 9 + np.arange(3)[:, None] * 100 + np.arange(129)
                    np.testing.assert_array_equal(instance.resource('j').to_numpy(), expected)
                    np.testing.assert_array_equal(instance.resource('h').to_numpy(),
                        expected[:, None, :] + np.arange(2)[None, :, None] * 10)
                    np.testing.assert_array_equal(instance.resource('fixed').to_numpy(), np.tile([1, 2, 3], (3, 1)))
            print('MODEL_TENSOR_COMPONENT_SOURCE_FREE_OK')
        """)
        result = subprocess.run([sys.executable, "-B", "-c", consumer, ti.cfg.arch.name, str(archive)],
                                cwd=tmp_path, capture_output=True, text=True, timeout=60,
                                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "packaging/runtime/src")})
        assert result.returncode == 0, result.stdout + result.stderr
        assert "MODEL_TENSOR_COMPONENT_SOURCE_FREE_OK" in result.stdout
    finally:
        first.tree.destroy()
        second.tree.destroy()


@ti.data_oriented
class _DenseModel:
    def __init__(self, capacity, active, gravity=(2, 3, 4)):
        self.capacity = capacity
        self.active = active
        self.gravity = ti.Vector(gravity)
        self.x = ti.field(ti.f32)
        self.y = ti.field(ti.f32)
        first = ti.FieldsBuilder()
        first.dense(ti.i, capacity).place(self.x)
        first.dense(ti.i, capacity).place(self.y)
        self.first = first.finalize()
        self.alias_x = self.x
        self.v = ti.Vector.field(3, ti.f32)
        self.m = ti.Matrix.field(2, 2, ti.f32)
        second = ti.FieldsBuilder()
        second.dense(ti.i, capacity).place(self.v)
        second.dense(ti.i, capacity).place(self.m)
        self.second = second.finalize()
        self.x.fill(-11)
        self.y.fill(-17)
        self.v.fill(-23)
        self.m.fill(-29)

    @ti.func
    def value(self, i):
        return self.alias_x[i] + self.gravity[0]

    @ti.kernel
    def active_pass(self):
        for i in range(self.active):
            self.x[i] = self.value(i)
            self.y[i] = 2 * self.x[i]
            self.v[i] = ti.Vector([self.x[i], self.y[i], self.gravity[1]])
            self.m[i] = ti.Matrix([[self.x[i], 0.0], [0.0, self.gravity[2]]])

    @ti.kernel
    def capacity_pass(self):
        for i in range(self.y.shape[0]):
            self.y[i] = i
        for i in self.v:
            self.v[i] = ti.Vector([i, i + 1, i + 2])

    def record_repeated(self, builder):
        for _ in range(4):
            builder.dispatch(self.active_pass)
            builder.dispatch(self.capacity_pass)

    def record_changed_alias(self, builder):
        builder.dispatch(self.active_pass)
        previous = self.alias_x
        try:
            self.alias_x = self.y
            builder.dispatch(self.active_pass)
        finally:
            self.alias_x = previous


def _environment():
    return Component(
        capacity=Dimension(),
        active=Dimension("active", "capacity"),
        gravity=RuntimeValue(ti.types.vector(3, ti.f32)),
        x=DenseField(ti.f32, ("capacity",), "state"),
        alias_x=DenseField(ti.f32, ("capacity",), "state"),
        y=DenseField(ti.f32, ("capacity",), "state"),
        v=DenseField(ti.types.vector(3, ti.f32), ("capacity",), "work"),
        m=DenseField(ti.types.matrix(2, 2, ti.f32), ("capacity",), "work"),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_original_fields_preserve_groups_aliases_and_dynamic_ranges():
    producer = _DenseModel(7, 7)
    env = _environment()
    code = env.compile(producer.active_pass)
    capacity_code = env.compile(producer.capacity_pass)
    assert set(dict(code.contract["dense_access"]).values()) == {"compact"}
    assert set(dict(capacity_code.contract["dense_access"]).values()) == {"compact"}
    assert len(code.contract["groups"]) == 2
    assert code.contract["members"]["x"] == code.contract["members"]["alias_x"]
    assert sum(name in ("x", "alias_x") for name in code.dynamic) == 1
    bigger = _DenseModel(41, 37, gravity=(5, 7, 9))
    empty = _DenseModel(129, 0)
    assert env.compile(bigger.active_pass) is code
    assert env.compile(empty.active_pass) is code
    assert env.compile(bigger.capacity_pass) is capacity_code
    assert env.compile(empty.capacity_pass) is capacity_code
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)
    try:
        code.run(env, bigger)
        code.run(env, empty)
        code.run(env, producer)
        assert runtime.debug_kernel_executable_lifecycle_stats()["compiler_invocations"] == 1
    finally:
        runtime.set_kernel_executable_lifecycle_telemetry_enabled(False)
    assert (producer.x.to_numpy() == -9).all()
    x = bigger.x.to_numpy()
    np.testing.assert_array_equal(x[:37], np.full(37, -6))
    np.testing.assert_array_equal(x[37:], np.full(4, -11))
    np.testing.assert_array_equal(bigger.y.to_numpy()[37:], np.full(4, -17))
    assert (empty.x.to_numpy() == -11).all()
    np.testing.assert_array_equal(bigger.v.to_numpy()[36], [-6, -12, 7])
    np.testing.assert_array_equal(bigger.m.to_numpy()[36], [[-6, 0], [0, 9]])
    assert (bigger.m.to_numpy()[37:] == -29).all()

    # Same numerical definition and normal Field construction on ordinary JIT.
    ordinary = _DenseModel(41, 37, gravity=(5, 7, 9))
    ordinary.active_pass()
    for name in ("x", "y", "v", "m"):
        np.testing.assert_array_equal(getattr(bigger, name).to_numpy(), getattr(ordinary, name).to_numpy())
    capacity_code.run(env, empty)
    np.testing.assert_array_equal(empty.y.to_numpy(), np.arange(129))
    np.testing.assert_array_equal(empty.v.to_numpy()[-1], [128, 129, 130])

    before = bigger.x.to_numpy()
    bigger.active = 42
    with pytest.raises(ti.TaichiCompilationError, match="exceeds capacity"):
        code.run(env, bigger)
    np.testing.assert_array_equal(bigger.x.to_numpy(), before)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_group_cannot_be_split_or_infer_dimension_from_equal_values():
    model = _DenseModel(7, 7)
    env = _environment()
    code = env.compile(model.active_pass)
    replacement = _DenseModel(7, 7)
    model.y = replacement.y
    with pytest.raises(ti.TaichiCompilationError, match="one actual SNode root"):
        code.run(env, model)

    model = _DenseModel(7, 7)
    model.same_number = model.capacity
    declarations = dict(env.declarations)
    declarations["same_number"] = Dimension()
    declarations["alias_x"] = DenseField(ti.f32, ("same_number",), "state")
    with pytest.raises(ti.TaichiCompilationError, match="Conflicting dimension provenance"):
        Component(**declarations).compile(model.active_pass)

    @ti.data_oriented
    class Scalars:
        def __init__(self):
            self.a = ti.field(ti.i32)
            self.b = ti.field(ti.i32)
            builder = ti.FieldsBuilder()
            builder.place(self.a, self.b)
            self.tree = builder.finalize()

        @ti.kernel
        def step(self):
            self.a[None] += self.b[None]

    scalar = Scalars()
    with pytest.raises(ti.TaichiCompilationError, match="cannot split the same SNode root"):
        Component(a=DenseField(ti.i32, (), "a"), b=DenseField(ti.i32, (), "b")).compile(scalar.step)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_dense_validation_projects_groups_once_per_operation(monkeypatch):
    from collections import Counter
    from taichi_forge.aot import _component, _component_graph

    model, env = _DenseModel(7, 5), _environment()
    projections = Counter()
    project = _component._dense_group

    def counted(component, owner, group):
        projections[group] += 1
        return project(component, owner, group)

    monkeypatch.setattr(_component, "_dense_group", counted)
    code = env.compile(model.active_pass)
    # One snapshot for lowering, another for publication; five Field/alias
    # captures must not each re-project their complete group.
    assert projections == {"state": 2, "work": 2}
    assert code.contract["members"]["x"] == code.contract["members"]["alias_x"]
    env.compile(model.capacity_pass)
    finish = _component_graph._GraphCapture.finish
    finished = Counter()

    def checked_finish(capture, builder):
        projections.clear()
        result = finish(capture, builder)
        finished.update(projections)
        return result

    monkeypatch.setattr(_component_graph._GraphCapture, "finish", checked_finish)
    graph = env.compile_graph(model.record_repeated)
    try:
        assert finished == {"state": 1, "work": 1}
    finally:
        graph._graph.close()

    # Reusing the same declarations on a new owner still validates live roots,
    # extents and aliases; no physical snapshot survives the operation.
    other = _DenseModel(19, 17)
    assert env.compile(other.active_pass) is code
    code.run(env, other)
    np.testing.assert_array_equal(other.x.to_numpy()[:17], np.full(17, -9))
    np.testing.assert_array_equal(other.x.to_numpy()[17:], np.full(2, -11))


@test_utils.test(arch=ti.cpu, offline_cache=False)
@pytest.mark.parametrize("replace_whole_groups", [False, True])
def test_dense_snapshot_revalidates_lowering_and_each_recorded_dispatch(monkeypatch, replace_whole_groups):
    from taichi_forge.aot import _component

    model, replacement, env = _DenseModel(7, 5), _DenseModel(7, 5), _environment()
    original = _component._Capture.attribute
    fields = ("x", "alias_x", "y", "v", "m") if replace_whole_groups else ("y",)
    previous = {name: getattr(model, name) for name in fields}
    changed = False

    def mutate_after_capture(capture, ctx, name, **kwargs):
        nonlocal changed
        result = original(capture, ctx, name, **kwargs)
        if name == "m" and not changed:
            # Mutation after all Field captures must still be rejected at
            # publication. Failed captures cannot populate reusable candidates.
            for field in fields:
                setattr(model, field, getattr(replacement, field))
            changed = True
        return result

    with monkeypatch.context() as patch:
        patch.setattr(_component._Capture, "attribute", mutate_after_capture)
        with pytest.raises(ti.TaichiCompilationError, match="changed during compilation|one actual SNode root"):
            env.compile(model.active_pass)
    for field, value in previous.items():
        setattr(model, field, value)
    code = env.compile(model.active_pass)
    assert env.compile(model.active_pass) is code

    # Host recording changes an alias and then restores it. The intermediate
    # dispatch must not reuse the first dispatch's layout validation snapshot.
    with pytest.raises(ti.TaichiCompilationError, match="dependencies changed while recording"):
        env.compile_graph(model.record_changed_alias)
    assert env.compile(model.active_pass) is code
    model.capacity += 1
    with pytest.raises(ti.TaichiCompilationError, match="shape differs"):
        env.compile(model.active_pass)
