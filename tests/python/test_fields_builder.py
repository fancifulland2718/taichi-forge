import numpy as np
import pytest
from taichi_forge.lang.exception import TaichiRuntimeError

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan, ti.metal])
def test_fields_with_shape():
    shape = 5
    x = ti.field(ti.f32, shape=shape)

    @ti.kernel
    def assign_field_single():
        for i in range(shape):
            x[i] = i

    assign_field_single()
    for i in range(shape):
        assert x[i] == i

    y = ti.field(ti.f32, shape=shape)

    @ti.kernel
    def assign_field_multiple():
        for i in range(shape):
            y[i] = i * 2
        for i in range(shape):
            x[i] = i * 3

    assign_field_multiple()
    for i in range(shape):
        assert x[i] == i * 3
        assert y[i] == i * 2

    assign_field_single()
    for i in range(shape):
        assert x[i] == i


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan, ti.dx11, ti.metal])
def test_fields_builder_dense():
    shape = 5
    fb1 = ti.FieldsBuilder()
    x = ti.field(ti.f32)
    fb1.dense(ti.i, shape).place(x)
    fb1.finalize()

    @ti.kernel
    def assign_field_single():
        for i in range(shape):
            x[i] = i * 3

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3

    fb2 = ti.FieldsBuilder()
    y = ti.field(ti.f32)
    fb2.dense(ti.i, shape).place(y)
    z = ti.field(ti.f32)
    fb2.dense(ti.i, shape).place(z)
    fb2.finalize()

    @ti.kernel
    def assign_field_multiple():
        for i in range(shape):
            x[i] = i * 2
        for i in range(shape):
            y[i] = i + 5
        for i in range(shape):
            z[i] = i + 10

    assign_field_multiple()
    for i in range(shape):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_fields_builder_pointer():
    shape = 5
    fb1 = ti.FieldsBuilder()
    x = ti.field(ti.f32)
    fb1.pointer(ti.i, shape).place(x)
    fb1.finalize()

    @ti.kernel
    def assign_field_single():
        for i in range(shape):
            x[i] = i * 3

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3

    fb2 = ti.FieldsBuilder()
    y = ti.field(ti.f32)
    fb2.pointer(ti.i, shape).place(y)
    z = ti.field(ti.f32)
    fb2.pointer(ti.i, shape).place(z)
    fb2.finalize()

    @ti.kernel
    def assign_field_multiple_range_for():
        for i in range(shape):
            x[i] = i * 2
        for i in range(shape):
            y[i] = i + 5
        for i in range(shape):
            z[i] = i + 10

    assign_field_multiple_range_for()
    for i in range(shape):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    @ti.kernel
    def assign_field_multiple_struct_for():
        for i in y:
            y[i] += 5
        for i in z:
            z[i] -= 5

    assign_field_multiple_struct_for()
    for i in range(shape):
        assert y[i] == i + 10
        assert z[i] == i + 5

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3


# We currently only consider data types that all platforms support.
# See https://docs.taichi-lang.org/docs/type#primitive-types for more details.
@pytest.mark.parametrize("test_1d_size", [1, 10, 100])
@pytest.mark.parametrize("field_type", [ti.f32, ti.i32])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan, ti.dx11, ti.metal])
def test_fields_builder_destroy(test_1d_size, field_type):
    def test_for_single_destroy_multi_fields():
        fb = ti.FieldsBuilder()
        for create_field_idx in range(10):
            field = ti.field(field_type)
            fb.dense(ti.i, test_1d_size).place(field)
        fb_snode_tree = fb.finalize()
        fb_snode_tree.destroy()

    def test_for_multi_destroy_multi_fields():
        fb0 = ti.FieldsBuilder()
        fb1 = ti.FieldsBuilder()

        for create_field_idx in range(10):
            field0 = ti.field(field_type)
            field1 = ti.field(field_type)

            fb0.dense(ti.i, test_1d_size).place(field0)
            fb1.pointer(ti.i, test_1d_size).place(field1)

        fb0_snode_tree = fb0.finalize()
        fb1_snode_tree = fb1.finalize()

        fb0_snode_tree.destroy()
        fb1_snode_tree.destroy()

    def test_for_raise_destroy_twice():
        fb = ti.FieldsBuilder()
        a = ti.field(ti.f32)
        fb.dense(ti.i, test_1d_size).place(a)
        c = fb.finalize()

        with pytest.raises(TaichiRuntimeError):
            c.destroy()
            c.destroy()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan, ti.dx11])
def test_field_initialize_zero():
    fb0 = ti.FieldsBuilder()
    a = ti.field(ti.i32)
    fb0.dense(ti.i, 1).place(a)
    c = fb0.finalize()
    a[0] = 5
    c.destroy()
    fb1 = ti.FieldsBuilder()
    b = ti.field(ti.i32)
    fb1.dense(ti.i, 1).place(b)
    d = fb1.finalize()
    assert b[0] == 0


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_destroyed_field_accessors_do_not_accumulate_frontend_state():
    prog = ti.lang.impl.get_runtime().prog
    baseline_kernels = prog._debug_kernel_definition_count()
    baseline_registrations = prog._debug_kernel_registration_count()
    baseline_fields = prog._debug_snode_field_mapping_count()

    for value in range(3):
        fb = ti.FieldsBuilder()
        field = ti.field(ti.i32)
        fb.dense(ti.i, 1).place(field)
        tree = fb.finalize()
        assert prog._debug_snode_field_mapping_count() == baseline_fields + 1

        field[0] = value
        assert field[0] == value
        assert prog._debug_kernel_definition_count() == baseline_kernels + 2
        assert (
            prog._debug_kernel_registration_count()
            == baseline_registrations + 2
        )

        tree.destroy()
        assert prog._debug_kernel_definition_count() == baseline_kernels
        assert (
            prog._debug_kernel_registration_count() == baseline_registrations
        )
        assert prog._debug_snode_field_mapping_count() == baseline_fields


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_finalized_roots_only_include_active_snode_trees():
    prog = ti.lang.impl.get_runtime().prog

    first_builder = ti.FieldsBuilder()
    first = ti.field(ti.f32, needs_grad=True)
    first_builder.dense(ti.i, 4).place(first, first.grad)
    first_tree = first_builder.finalize()
    first_id = first_tree.id

    second_builder = ti.FieldsBuilder()
    second = ti.field(ti.f32, needs_grad=True)
    second_builder.dense(ti.i, 4).place(second, second.grad)
    second_tree = second_builder.finalize()

    assert {
        root._snode_tree_id for root in ti.FieldsBuilder._finalized_roots()
    } == {first_id, second_tree.id}

    first_tree.destroy()
    with pytest.raises(RuntimeError, match="no longer active"):
        prog.get_snode_root(first_id)
    assert [
        root._snode_tree_id for root in ti.FieldsBuilder._finalized_roots()
    ] == [second_tree.id]

    replacement_builder = ti.FieldsBuilder()
    replacement = ti.field(ti.f32, needs_grad=True)
    replacement_builder.dense(ti.i, 4).place(replacement, replacement.grad)
    replacement_tree = replacement_builder.finalize()
    assert replacement_tree.id == first_id

    ti.ad.clear_all_gradients()
    active_ids = {
        root._snode_tree_id for root in ti.FieldsBuilder._finalized_roots()
    }
    assert {replacement_tree.id, second_tree.id} <= active_ids
    assert sum(
        root._snode_tree_id == first_id
        for root in ti.FieldsBuilder._finalized_roots()
    ) == 1


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_first_kernel_reuses_explicit_fields_builder_layouts():
    builder = ti.FieldsBuilder()
    value = ti.field(ti.i32)
    builder.dense(ti.i, 4).place(value)
    tree = builder.finalize()
    prog = ti.lang.impl.get_runtime().prog

    assert prog.get_active_snode_tree_ids() == [tree.id]

    @ti.kernel
    def initialize():
        for i in value:
            value[i] = i + 1

    initialize()

    assert prog.get_active_snode_tree_ids() == [tree.id]
    assert value.to_numpy().tolist() == [1, 2, 3, 4]


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_field_free_first_kernel_materializes_default_root():
    prog = ti.lang.impl.get_runtime().prog
    assert prog.get_active_snode_tree_ids() == []

    @ti.kernel
    def constant() -> ti.i32:
        return 7

    assert constant() == 7
    assert prog.get_active_snode_tree_ids() == [0]


@test_utils.test(exclude=[ti.opengl, ti.gles])
def test_field_builder_place_grad():
    @ti.kernel
    def mul(arr: ti.template(), out: ti.template()):
        for i in arr:
            out[i] = arr[i] * 2.0

    @ti.kernel
    def calc_loss(arr: ti.template(), loss: ti.template()):
        for i in arr:
            loss[None] += arr[i]

    arr = ti.field(ti.f32, needs_grad=True)
    fb0 = ti.FieldsBuilder()
    fb0.dense(ti.i, 10).place(arr, arr.grad)
    snode0 = fb0.finalize()
    out = ti.field(ti.f32)
    fb1 = ti.FieldsBuilder()
    fb1.dense(ti.i, 10).place(out, out.grad)
    snode1 = fb1.finalize()
    loss = ti.field(ti.f32)
    fb2 = ti.FieldsBuilder()
    fb2.place(loss, loss.grad)
    snode2 = fb2.finalize()
    arr.fill(1.0)
    mul(arr, out)
    calc_loss(out, loss)
    loss.grad[None] = 1.0
    calc_loss.grad(out, loss)
    mul.grad(arr, out)
    for i in range(10):
        assert arr.grad[i] == 2.0


@test_utils.test(arch=ti.cpu)
def test_fields_builder_numpy_dimension():
    shape = np.int32(5)
    fb = ti.FieldsBuilder()
    x = ti.field(ti.f32)
    y = ti.field(ti.i32)
    fb.dense(ti.i, shape).place(x)
    fb.pointer(ti.j, shape).place(y)
    fb.finalize()


def test_snode_tree_from_previous_runtime_drops_native_references(monkeypatch):
    from taichi_forge._snode.snode_tree import SNodeTree
    from taichi_forge.lang import impl

    class StalePointer:
        def id(self):
            raise AssertionError("stale native pointer was dereferenced")

    class Runtime:
        prog = object()

    tree = object.__new__(SNodeTree)
    tree.prog = object()
    tree.ptr = StalePointer()
    tree.destroyed = False
    monkeypatch.setattr(impl, "get_runtime", lambda: Runtime())

    tree.destroy()

    assert tree.destroyed
    assert tree.ptr is None
    assert tree.prog is None
    with pytest.raises(TaichiRuntimeError, match="destroyed"):
        _ = tree.id


def test_snode_tree_registers_for_pre_finalize_runtime_invalidation(monkeypatch):
    from taichi_forge._snode.snode_tree import SNodeTree
    from taichi_forge.lang import impl

    class Runtime:
        prog = object()

        def __init__(self):
            self.registered = []

        def register_runtime_object(self, obj):
            self.registered.append(obj)

    runtime = Runtime()
    pointer = object()
    monkeypatch.setattr(impl, "get_runtime", lambda: runtime)

    tree = SNodeTree(pointer)
    assert runtime.registered == [tree]

    tree._invalidate_runtime()
    assert tree.destroyed
    assert tree.ptr is None
    assert tree.prog is None


def test_snode_tree_stays_destroyed_if_cache_cleanup_fails(monkeypatch):
    from taichi_forge._snode.snode_tree import SNodeTree
    from taichi_forge.lang import impl

    class Pointer:
        @staticmethod
        def id():
            return 3

        @staticmethod
        def generation():
            return 5

        @staticmethod
        def destroy_snode_tree(prog):
            assert prog is runtime.prog

    class Runtime:
        prog = object()

        @staticmethod
        def begin_snode_tree_destroy(dependency):
            assert dependency == (3, 5)
            return []

        @staticmethod
        def cancel_snode_tree_destroy(dependency, notified):
            raise AssertionError("successful native destroy must not roll back")

        @staticmethod
        def clear_compiled_functions():
            raise RuntimeError("injected cache cleanup failure")

    runtime = Runtime()
    tree = object.__new__(SNodeTree)
    tree.prog = runtime.prog
    tree.ptr = Pointer()
    tree.destroyed = False
    monkeypatch.setattr(impl, "get_runtime", lambda: runtime)

    with pytest.raises(RuntimeError, match="injected cache cleanup failure"):
        tree.destroy()

    assert tree.destroyed
    assert tree.ptr is None
    assert tree.prog is None
