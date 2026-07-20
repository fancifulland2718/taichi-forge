import taichi_forge as ti
from tests import test_utils


@test_utils.test(require=ti.extension.sparse)
def test_pointer():
    x = ti.field(ti.f32)
    s = ti.field(ti.i32)

    n = 16

    ptr = ti.root.pointer(ti.i, n)
    ptr.dense(ti.i, n).place(x)
    ti.root.place(s)

    s[None] = 0

    @ti.kernel
    def activate():
        ti.activate(ptr, ti.rescale_index(x, ptr, [1]))
        ti.activate(ptr, ti.rescale_index(x, ptr, [32]))

    @ti.kernel
    def func():
        for i in x:
            s[None] += 1

    activate()
    func()
    assert s[None] == 32


@test_utils.test(require=ti.extension.sparse)
def test_non_dfs_snode_order():
    x = ti.field(dtype=ti.i32)
    y = ti.field(dtype=ti.i32)

    grid1 = ti.root.dense(ti.i, 1)
    grid2 = ti.root.dense(ti.i, 1)
    ptr = grid1.pointer(ti.i, 1)
    ptr.place(x)
    grid2.place(y)
    """
    This SNode tree has node ids that do not follow DFS order:
    S0root
      S1dense
        S3pointer
          S4place<i32>
      S2dense
        S5place<i32>
    """

    @ti.kernel
    def foo():
        ti.activate(ptr, [0])

    foo()  # Just make sure it doesn't crash
    ti.sync()


@test_utils.test(
    arch=ti.cuda,
    offline_cache=False,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
    cuda_pointer_deterministic_slot=True,
)
def test_cuda_duplicate_pointer_and_bitmasked_activation():
    value = ti.field(ti.i32)
    pointer = ti.root.pointer(ti.i, 4)
    bitmasked = pointer.bitmasked(ti.i, 32)
    bitmasked.place(value)

    workers = 8192
    active_cells = 16

    @ti.kernel
    def scatter():
        for worker in range(workers):
            ti.atomic_add(value[worker % active_cells], 1)

    @ti.kernel
    def reduce() -> ti.i32:
        total = 0
        for i in value:
            total += value[i]
        return total

    scatter()
    assert reduce() == workers

    # Topology-stable writes take the read-only active fast path.
    scatter()
    assert reduce() == 2 * workers

    # Exercise both leaf-only and whole-pointer reactivation.
    bitmasked.deactivate_all()
    scatter()
    assert reduce() == workers

    pointer.deactivate_all()
    scatter()
    assert reduce() == workers
