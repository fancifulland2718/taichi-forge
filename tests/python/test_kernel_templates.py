import gc
import weakref

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test()
def test_kernel_template_basic():
    x = ti.field(ti.i32)
    y = ti.field(ti.f32)

    n = 16

    ti.root.dense(ti.i, n).place(x, y)

    @ti.kernel
    def inc(a: ti.template(), b: ti.template()):
        for i in a:
            a[i] += b

    inc(x, 1)
    inc(y, 2)

    for i in range(n):
        assert x[i] == 1
        assert y[i] == 2

    @ti.kernel
    def inc2(z: ti.i32, a: ti.template(), b: ti.i32):
        for i in a:
            a[i] += b + z

    inc2(10, x, 1)
    for i in range(n):
        assert x[i] == 12


@test_utils.test()
def test_kernel_template_gradient():
    x = ti.field(ti.f32)
    y = ti.field(ti.f32)
    z = ti.field(ti.f32)
    loss = ti.field(ti.f32)

    ti.root.dense(ti.i, 16).place(x, y, z)
    ti.root.place(loss)
    ti.root.lazy_grad()

    @ti.kernel
    def double(a: ti.template(), b: ti.template()):
        for i in range(16):
            b[i] = a[i] * 2 + 1

    @ti.kernel
    def compute_loss():
        for i in range(16):
            ti.atomic_add(loss[None], z[i])

    for i in range(16):
        x[i] = i

    with ti.ad.Tape(loss):
        double(x, y)
        double(y, z)
        compute_loss()

    for i in range(16):
        assert z[i] == i * 4 + 3
        assert x.grad[i] == 4


@test_utils.test()
def test_func_template():
    a = [ti.field(dtype=ti.f32) for _ in range(2)]
    b = [ti.field(dtype=ti.f32) for _ in range(2)]

    for l in range(2):
        ti.root.dense(ti.ij, 16).place(a[l], b[l])

    @ti.func
    def sample(x: ti.template(), l: ti.template(), I):
        return x[l][I]

    @ti.kernel
    def fill(l: ti.template()):
        for I in ti.grouped(a[l]):
            a[l][I] = l

    @ti.kernel
    def aTob(l: ti.template()):
        for I in ti.grouped(b[l]):
            b[l][I] = sample(a, l, I)

    for l in range(2):
        fill(l)
        aTob(l)

    for l in range(2):
        for i in range(16):
            for j in range(16):
                assert b[l][i, j] == l


@test_utils.test()
def test_func_template2():
    a = ti.field(dtype=ti.f32)
    b = ti.field(dtype=ti.f32)

    ti.root.dense(ti.ij, 16).place(a, b)

    @ti.func
    def sample(x: ti.template(), I):
        return x[I]

    @ti.kernel
    def fill():
        for I in ti.grouped(a):
            a[I] = 1.0

    @ti.kernel
    def aTob():
        for I in ti.grouped(b):
            b[I] = sample(a, I)

    for l in range(2):
        fill()
        aTob()

    for i in range(16):
        for j in range(16):
            assert b[i, j] == 1.0


@test_utils.test(arch=ti.cpu)
def test_dead_kernel_definitions_leave_runtime_registry():
    runtime = ti.lang.impl.get_runtime()
    baseline = len(runtime.kernels)

    @ti.kernel
    def temporary(value: ti.i32) -> ti.i32:
        return value

    primal_ref = weakref.ref(temporary._primal)
    adjoint_ref = weakref.ref(temporary._adjoint)
    assert len(runtime.kernels) == baseline + 2

    del temporary
    gc.collect()

    assert primal_ref() is None
    assert adjoint_ref() is None
    assert len(runtime.kernels) == baseline


@test_utils.test(arch=ti.cpu, kernel_specialization_limit=2)
def test_kernel_specialization_budget_preserves_cache_hits():
    @ti.kernel
    def identity(value: ti.template()) -> ti.i32:
        return value

    assert identity(11) == 11
    assert identity(22) == 22
    with pytest.raises(ti.TaichiRuntimeError, match="kernel_specialization_limit=2"):
        identity(33)

    # Reaching the budget blocks only a new cache miss. Existing compiled
    # specializations remain valid for Graph and asynchronous launch users.
    assert identity(22) == 22


@test_utils.test(arch=ti.cpu, kernel_specialization_limit=1)
def test_dead_kernel_definition_cannot_reclaim_native_specialization_budget():
    @ti.kernel
    def temporary() -> ti.i32:
        return 1

    assert temporary() == 1
    del temporary
    gc.collect()

    @ti.kernel
    def replacement() -> ti.i32:
        return 2

    with pytest.raises(ti.TaichiRuntimeError, match="kernel_specialization_limit=1"):
        replacement()


@test_utils.test(arch=ti.cpu, kernel_specialization_limit=1)
def test_destroyed_dense_snode_specialization_reuses_resident_template():
    @ti.kernel
    def initialize(dst: ti.template(), value: ti.i32):
        for i in dst:
            dst[i] = value

    for value in (3, 7):
        field = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, 8).place(field)
        tree = builder.finalize()
        initialize(field, value)
        assert field[4] == value
        tree.destroy()
        del field
        gc.collect()

    stats = ti.lang.impl.get_runtime().debug_kernel_executable_lifecycle_stats()
    assert stats["historical_materializations"] == 1
    assert stats["resident_specializations"] == 0
    assert stats["specialization_reclaims"] == 1
    assert stats["relocatable_templates"] == 1


@test_utils.test(arch=ti.cpu, kernel_specialization_limit=1)
def test_stale_graph_handle_pins_template_but_allows_same_layout_rebind():
    @ti.kernel
    def initialize(dst: ti.template()):
        for i in dst:
            dst[i] = i

    first = ti.field(ti.i32)
    first_builder = ti.FieldsBuilder()
    first_builder.dense(ti.i, 8).place(first)
    first_tree = first_builder.finalize()
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(initialize, template_args={"dst": first})
    graph = graph_builder.compile()
    graph.run({})
    first_tree.destroy()
    del first
    gc.collect()

    with pytest.raises(ti.TaichiRuntimeError, match="stale|destroyed|retired"):
        graph.run({})

    second = ti.field(ti.i32)
    second_builder = ti.FieldsBuilder()
    second_builder.dense(ti.i, 8).place(second)
    second_tree = second_builder.finalize()
    initialize(second)
    assert second[5] == 5

    stats = ti.lang.impl.get_runtime().debug_kernel_executable_lifecycle_stats()
    assert stats["historical_materializations"] == 1
    assert stats["relocatable_templates"] == 1

    second_tree.destroy()
    del graph
    del graph_builder


@test_utils.test(arch=ti.cpu, offline_cache=False, kernel_specialization_limit=1)
def test_dense_snode_template_reuses_code_across_serial_generations():
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)

    @ti.kernel
    def initialize(dst: ti.template(), base: ti.i32) -> ti.i32:
        for i in dst:
            dst[i] = base + i
        return dst[7]

    for generation in range(12):
        field = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, 16).place(field)
        tree = builder.finalize()
        assert initialize(field, generation) == generation + 7
        tree.destroy()

    stats = runtime.debug_kernel_executable_lifecycle_stats()
    assert stats["compiler_invocations"] == 1
    assert stats["historical_materializations"] == 1
    assert stats["relocatable_templates"] == 1
    assert stats["relocatable_template_hits"] == 11
    assert stats["relocatable_bindings_created"] == 11
    assert stats["budget_rejections"] == 0


@test_utils.test(arch=ti.cpu, offline_cache=False, kernel_specialization_limit=2)
def test_dense_snode_template_key_distinguishes_layouts_and_reuses_each():
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)

    @ti.kernel
    def initialize(dst: ti.template(), base: ti.i32) -> ti.i32:
        for i in dst:
            dst[i] = base + i
        return dst[0]

    for generation, shape in enumerate((8, 16, 8, 16, 8, 16)):
        field = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, shape).place(field)
        tree = builder.finalize()
        assert initialize(field, generation) == generation
        tree.destroy()

    stats = runtime.debug_kernel_executable_lifecycle_stats()
    assert stats["compiler_invocations"] == 2
    assert stats["historical_materializations"] == 2
    assert stats["relocatable_templates"] == 2
    assert stats["relocatable_template_hits"] == 4


@test_utils.test(arch=ti.cpu, offline_cache=False, kernel_specialization_limit=8)
def test_simultaneously_live_dense_trees_never_share_generation_binding():
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)

    @ti.kernel
    def initialize(dst: ti.template(), base: ti.i32) -> ti.i32:
        for i in dst:
            dst[i] = base + i
        return dst[5]

    first = ti.field(ti.i32)
    first_builder = ti.FieldsBuilder()
    first_builder.dense(ti.i, 8).place(first)
    first_tree = first_builder.finalize()

    second = ti.field(ti.i32)
    second_builder = ti.FieldsBuilder()
    second_builder.dense(ti.i, 8).place(second)
    second_tree = second_builder.finalize()

    assert initialize(first, 10) == 15
    assert initialize(second, 20) == 25
    assert initialize(first, 30) == 35

    stats = runtime.debug_kernel_executable_lifecycle_stats()
    assert stats["compiler_invocations"] == 2
    assert stats["relocatable_template_hits"] == 0

    first_tree.destroy()
    second_tree.destroy()


@test_utils.test(arch=ti.cpu, offline_cache=False, kernel_specialization_limit=1)
def test_dense_graph_rebuild_reuses_template_while_old_graph_stays_stale():
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)

    @ti.kernel
    def initialize(
        dst: ti.template(),
        base: ti.i32,
        out: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        for i in dst:
            dst[i] = base + i
        out[None] = dst[7]

    def make_graph(field):
        base = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "base", ti.i32)
        out = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=0)
        builder = ti.graph.GraphBuilder()
        builder.dispatch(initialize, base, out, template_args={"dst": field})
        return builder, builder.compile()

    previous = None
    retained_builders = []
    for generation in range(4):
        field = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, 16).place(field)
        tree = builder.finalize()
        graph_builder, graph = make_graph(field)
        retained_builders.append(graph_builder)
        out = ti.ndarray(ti.i32, shape=())
        graph.run({"base": generation, "out": out})
        ti.sync()
        assert out.to_numpy().item() == generation + 7
        if previous is not None:
            with pytest.raises(ti.TaichiRuntimeError, match="stale|destroyed|retired"):
                previous.run({"base": generation, "out": out})
        tree.destroy()
        previous = graph

    stats = runtime.debug_kernel_executable_lifecycle_stats()
    assert stats["compiler_invocations"] == 1
    assert stats["relocatable_template_hits"] == 3


@test_utils.test(arch=ti.cpu, offline_cache=False, kernel_specialization_limit=8)
def test_sparse_snode_template_remains_generation_bound():
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)

    @ti.kernel
    def initialize(dst: ti.template(), base: ti.i32) -> ti.i32:
        index = base & 7
        dst[index] = base
        return dst[index]

    for generation in range(3):
        field = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.pointer(ti.i, 8).dense(ti.i, 4).place(field)
        tree = builder.finalize()
        assert initialize(field, generation) == generation
        tree.destroy()

    stats = runtime.debug_kernel_executable_lifecycle_stats()
    assert stats["compiler_invocations"] == 3
    assert stats["relocatable_templates"] == 0
    assert stats["relocatable_template_hits"] == 0


@test_utils.test(arch=ti.cpu, offline_cache=False, kernel_specialization_limit=8)
def test_hidden_captured_field_prevents_direct_template_rebind():
    captured = ti.field(ti.i32, shape=())
    captured[None] = 9
    runtime = ti.lang.impl.get_runtime()
    runtime.set_kernel_executable_lifecycle_telemetry_enabled(True)
    runtime.debug_kernel_executable_lifecycle_stats(True)

    @ti.kernel
    def initialize(dst: ti.template(), base: ti.i32) -> ti.i32:
        for i in dst:
            dst[i] = base + captured[None]
        return dst[3]

    for generation in range(2):
        field = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, 8).place(field)
        tree = builder.finalize()
        assert initialize(field, generation) == generation + 9
        tree.destroy()

    stats = runtime.debug_kernel_executable_lifecycle_stats()
    assert stats["compiler_invocations"] == 2
    assert stats["relocatable_template_hits"] == 0


@test_utils.test(arch=ti.cpu, offline_cache=False, kernel_specialization_limit=1)
def test_relocatable_template_eviction_falls_back_without_dangling_candidate():
    @ti.kernel
    def initialize(dst: ti.template(), base: ti.i32) -> ti.i32:
        for i in dst:
            dst[i] = base + i
        return dst[0]

    field = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 8).place(field)
    tree = builder.finalize()
    assert initialize(field, 3) == 3
    tree.destroy()

    runtime = ti.lang.impl.get_runtime()
    assert runtime.prog._reclaim_relocatable_kernel_templates(0) == 1

    replacement = ti.field(ti.i32)
    replacement_builder = ti.FieldsBuilder()
    replacement_builder.dense(ti.i, 8).place(replacement)
    replacement_tree = replacement_builder.finalize()
    assert initialize(replacement, 7) == 7
    replacement_tree.destroy()
