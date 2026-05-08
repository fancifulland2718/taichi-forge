"""Multi-level sparse listgen coverage (S4 prerequisite).

Establishes correctness baseline for 2-level and 3-level mixed sparse
trees (`pointer.bitmasked.dense`, `pointer.dense.dense`,
`pointer.bitmasked.dense.dense`, etc.). S4 (LLVM listgen 跨层融合) will
need to validate against this once the runtime refactor lands.

Activation is performed via direct host-side field writes (instead of
nested @ti.kernel) to avoid Taichi's source-inspection failures on
nested closures across pytest cwd boundaries.
"""
import taichi_forge as ti
from tests import test_utils


@test_utils.test(require=ti.extension.sparse)
def test_listgen_pointer_bitmasked_dense_2d():
    N = 32
    x = ti.field(ti.f32)
    ti.root.pointer(ti.ij, N // 4).bitmasked(ti.ij, 2).dense(ti.ij, 2).place(x)

    for i in range(N):
        for j in range(N):
            if (i * 31 + j * 17) % 5 == 0:
                x[i, j] = float(i * 1000 + j)

    @ti.kernel
    def reduce() -> ti.f32:
        s = 0.0
        for i, j in x:
            s += x[i, j]
        return s

    expected = sum(i * 1000 + j for i in range(N) for j in range(N)
                    if (i * 31 + j * 17) % 5 == 0)
    got = reduce()
    assert abs(got - expected) / max(abs(expected), 1.0) < 1e-5


@test_utils.test(require=ti.extension.sparse)
def test_listgen_pointer_dense_dense_2d():
    N = 32
    x = ti.field(ti.f32)
    ti.root.pointer(ti.ij, N // 4).dense(ti.ij, 2).dense(ti.ij, 2).place(x)

    for i in range(N):
        for j in range(N):
            if (i * 7 + j * 11) % 3 == 0:
                x[i, j] = float(i + j * 100)

    @ti.kernel
    def reduce() -> ti.f32:
        s = 0.0
        for i, j in x:
            s += x[i, j]
        return s

    expected = sum(i + j * 100 for i in range(N) for j in range(N)
                    if (i * 7 + j * 11) % 3 == 0)
    got = reduce()
    assert abs(got - expected) / max(abs(expected), 1.0) < 1e-5


@test_utils.test(require=ti.extension.sparse)
def test_listgen_pointer_bitmasked_dense_dense_2d():
    """4-level path: candidate target for future S4 fusion."""
    N = 32
    x = ti.field(ti.f32)
    (ti.root
        .pointer(ti.ij, N // 8)
        .bitmasked(ti.ij, 2)
        .dense(ti.ij, 2)
        .dense(ti.ij, 2)
        .place(x))

    for i in range(N):
        for j in range(N):
            if (i + j) % 4 == 0:
                x[i, j] = float((i ^ j) + 1)

    @ti.kernel
    def reduce() -> ti.f32:
        s = 0.0
        for i, j in x:
            s += x[i, j]
        return s

    expected = sum((i ^ j) + 1 for i in range(N) for j in range(N)
                    if (i + j) % 4 == 0)
    got = reduce()
    assert abs(got - expected) / max(abs(expected), 1.0) < 1e-5


@test_utils.test(require=ti.extension.sparse)
def test_listgen_pointer_bitmasked_dense_3d():
    """3D version exercises pcoord refinement on multiple axes."""
    N = 16
    x = ti.field(ti.f32)
    ti.root.pointer(ti.ijk, N // 4).bitmasked(ti.ijk, 2).dense(ti.ijk, 2).place(x)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if (i + j * 3 + k * 5) % 7 == 0:
                    x[i, j, k] = float(i * 10000 + j * 100 + k)

    @ti.kernel
    def reduce() -> ti.f32:
        s = 0.0
        for i, j, k in x:
            s += x[i, j, k]
        return s

    expected = 0.0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if (i + j * 3 + k * 5) % 7 == 0:
                    expected += i * 10000 + j * 100 + k
    got = reduce()
    assert abs(got - expected) / max(abs(expected), 1.0) < 1e-5


@test_utils.test(require=ti.extension.sparse)
def test_listgen_pointer_pointer_dense_2d():
    """Two-level pointer chain: both levels need activation masks."""
    N = 32
    x = ti.field(ti.f32)
    ti.root.pointer(ti.ij, N // 8).pointer(ti.ij, 2).dense(ti.ij, 4).place(x)

    for i in range(N):
        for j in range(N):
            if i % 6 == 0 and j % 6 == 0:
                x[i, j] = 1.0

    @ti.kernel
    def reduce() -> ti.f32:
        s = 0.0
        for i, j in x:
            s += x[i, j]
        return s

    # struct-for visits all dense leaves in active pointer.pointer blocks.
    # Unwritten cells = 0, so sum equals number of writes.
    expected = sum(1.0 for i in range(N) for j in range(N)
                    if i % 6 == 0 and j % 6 == 0)
    got = reduce()
    assert abs(got - expected) / max(abs(expected), 1.0) < 1e-5
