import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], advanced_optimization=False, offline_cache=False)
def test_fast_field_loads_observe_atomic_updates():
    value = ti.field(ti.i32, shape=8)
    observed = ti.Vector.field(3, ti.i32, shape=8)

    @ti.kernel
    def update():
        for i in range(8):
            value[i] = i + 2
            before = value[i]
            old = ti.atomic_add(value[i], 5)
            after = value[i]
            observed[i] = ti.Vector([before, old, after])

    update()
    for i in range(8):
        assert list(observed[i]) == [i + 2, i + 2, i + 7]


@test_utils.test(
    arch=[ti.cpu, ti.cuda], require=ti.extension.sparse, advanced_optimization=False, offline_cache=False
)
def test_fast_sparse_lookup_observes_branch_deactivation_and_reactivation():
    x = ti.field(ti.i32)
    y = ti.field(ti.i32)
    pointer = ti.root.pointer(ti.i, 4)
    pointer.dense(ti.i, 4).place(x, y)

    @ti.kernel
    def update(clear: ti.i32) -> ti.types.vector(3, ti.i32):
        x[0] = 3
        y[0] = 4
        before = x[0] + y[0]
        if clear:
            ti.deactivate(pointer, [0])
        inactive = x[0] + y[0]
        x[0] = 5
        y[0] = 6
        return ti.Vector([before, inactive, x[0] + y[0]])

    for clear in (0, 1, 0, 1):
        assert list(update(clear)) == [7, 0 if clear else 7, 11]


@test_utils.test(arch=[ti.cpu, ti.cuda], advanced_optimization=False, offline_cache=False)
def test_fast_field_loads_observe_real_function_effects():
    value = ti.field(ti.i32, shape=1)

    @ti.real_func
    def mutate(index: ti.i32):
        value[index] += 4

    @ti.kernel
    def update() -> ti.types.vector(2, ti.i32):
        value[0] = 7
        before = value[0]
        mutate(0)
        return ti.Vector([before, value[0]])

    assert list(update()) == [7, 11]
