"""P3.c matrix-path smoke & correctness (cpu only).

Exercises MatrixInitStmt + MatrixPtrStmt + MatrixOfGlobalPtrStmt paths
that the P3.c pre-scan MUST detect so the full scalarize pipeline still
runs. Validates against hand-computed references.
"""
import sys
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, offline_cache=False, default_fp=ti.f32,
        log_level=ti.WARN)


def test_matrix_init_and_arith():
    out = ti.Vector.field(3, ti.f32, shape=4)

    @ti.kernel
    def run():
        for i in out:
            v = ti.Vector([ti.f32(i + 1), ti.f32(2.0), ti.f32(3.0)])
            u = ti.Vector([ti.f32(1.0), ti.f32(1.0), ti.f32(1.0)])
            out[i] = v + u * ti.f32(i)

    run()
    ti.sync()
    got = out.to_numpy()
    ref = np.array([[i + 1 + i, 2 + i, 3 + i] for i in range(4)],
                   dtype=np.float32)
    d = float(np.abs(got - ref).max())
    print(f"matrix_init_and_arith |delta|={d:.3e}  "
          f"{'OK' if d < 1e-6 else 'FAIL'}")
    return d < 1e-6


def test_matrix_field_matmul():
    N = 16
    A = ti.Matrix.field(3, 3, ti.f32, shape=N)
    B = ti.Matrix.field(3, 3, ti.f32, shape=N)
    C = ti.Matrix.field(3, 3, ti.f32, shape=N)

    @ti.kernel
    def run():
        for i in A:
            a = ti.Matrix([[ti.f32(1.0), ti.f32(0.0), ti.f32(0.0)],
                           [ti.f32(0.0), ti.f32(1.0), ti.f32(0.0)],
                           [ti.f32(0.0), ti.f32(0.0), ti.f32(1.0)]])
            b = ti.Matrix([[ti.f32(i), ti.f32(1.0), ti.f32(0.0)],
                           [ti.f32(0.0), ti.f32(i), ti.f32(1.0)],
                           [ti.f32(1.0), ti.f32(0.0), ti.f32(i)]])
            A[i] = a
            B[i] = b
            C[i] = a @ b

    run()
    ti.sync()
    B_np = B.to_numpy()
    C_np = C.to_numpy()
    d = float(np.abs(C_np - B_np).max())
    print(f"matrix_field_matmul   |delta|={d:.3e}  "
          f"{'OK' if d < 1e-5 else 'FAIL'}")
    return d < 1e-5


def test_matrix_of_global_ptr():
    N = 8
    M = ti.Matrix.field(2, 2, ti.f32, shape=N)

    @ti.kernel
    def seed():
        for i in M:
            M[i][0, 0] = ti.f32(i)
            M[i][0, 1] = ti.f32(i + 1)
            M[i][1, 0] = ti.f32(i + 2)
            M[i][1, 1] = ti.f32(i + 3)

    @ti.kernel
    def run():
        for i in M:
            v = M[i]
            M[i] = v + v

    seed()
    run()
    ti.sync()
    got = M.to_numpy()
    ref = np.array([[[2 * i, 2 * (i + 1)],
                     [2 * (i + 2), 2 * (i + 3)]] for i in range(N)],
                   dtype=np.float32)
    d = float(np.abs(got - ref).max())
    print(f"matrix_of_global_ptr  |delta|={d:.3e}  "
          f"{'OK' if d < 1e-6 else 'FAIL'}")
    return d < 1e-6


def main():
    ok = all([
        test_matrix_init_and_arith(),
        test_matrix_field_matmul(),
        test_matrix_of_global_ptr(),
    ])
    print("OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
