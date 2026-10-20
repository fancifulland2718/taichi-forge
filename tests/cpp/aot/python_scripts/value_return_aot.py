"""Ordinary value returns and an explicit ndarray adjoint, shared by consumers."""

import time

import taichi_forge as ti


def export(arch, path):
    vector = ti.types.vector(3, ti.f32)
    column = ti.types.matrix(3, 1, ti.f32)
    large_vector = ti.types.vector(65, ti.f32)
    basis = ti.types.matrix(9, 9, ti.f32)
    inner = ti.types.struct(code=ti.u16, weight=ti.f32)
    record = ti.types.struct(
        tag=ti.i8, wide=ti.u64, inner=inner, values=large_vector, basis=basis
    )

    widths_record = ti.types.struct(
        flag=ti.u1, flags=ti.types.vector(2, ti.u1), half=ti.f16, precise=ti.f64
    )

    @ti.kernel
    def scalar(x: ti.i32, effects: ti.types.ndarray(ti.i32, ndim=1)) -> ti.i32:
        effects[0] += 1
        return 3 * x - 2

    @ti.kernel
    def vec(x: ti.f32) -> vector:
        return vector(x, x + 0.5, -x)

    @ti.kernel
    def col(x: ti.f32) -> column:
        return column([[x], [x + 0.5], [-x]])

    @ti.kernel
    def structured(x: ti.f32) -> record:
        return record(
            tag=-7,
            wide=ti.u64(0xFEDCBA9876543210),
            inner=inner(code=513, weight=x),
            values=large_vector([x + i for i in range(65)]),
            basis=basis([[x + i * 9 + j for j in range(9)] for i in range(9)]),
        )

    @ti.kernel
    def multiple(x: ti.i32) -> (ti.i32, ti.f32):
        return x - 3, ti.cast(x, ti.f32) * 0.5

    @ti.kernel
    def widths(x: ti.i32) -> widths_record:
        return widths_record(
            flag=x > 0,
            flags=ti.Vector([x > 0, x < 0]),
            half=ti.cast(x, ti.f16) * 0.5,
            precise=ti.cast(x, ti.f64) + ti.f64(2.0**-40),
        )

    @ti.kernel
    def energy(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in range(4):
            y[0] += x[i] * x[i]

    @ti.kernel
    def read_energy(y: ti.types.ndarray(ti.f32, ndim=1)) -> ti.f32:
        return y[0]

    @ti.kernel
    def checked(x: ti.i32) -> ti.i32:
        assert x >= 0, "negative result input"
        return x + 5

    started = time.perf_counter()
    # Verify the top-level shape change against JIT independently of the AOT
    # decoder: a vector and a single-column matrix must remain distinguishable.
    assert vec(2).to_list() == [2, 2.5, -2]
    assert col(2).to_list() == [[2], [2.5], [-2]]
    caps = (
        [
            "spirv_has_int8",
            "spirv_has_int16",
            "spirv_has_int64",
            "spirv_has_float16",
            "spirv_has_float64",
            "spirv_has_int8_storage",
            "spirv_has_int16_storage",
        ]
        if arch == "vulkan"
        else []
    )
    module = ti.aot.Module(caps=caps)
    for kernel in (scalar, vec, col, structured, multiple, widths, energy, read_energy):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    if arch != "vulkan":
        module.add_kernel(checked)
    module.save(str(path))
    return time.perf_counter() - started
