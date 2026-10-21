"""Aggregate values, mixed widths and constant parameters in an explicit adjoint."""

import time

import taichi_forge as ti


def export(arch, path, *, include_backward=True):
    vector = ti.types.vector(3, ti.f32)
    column = ti.types.matrix(3, 1, ti.f32)
    inner = ti.types.struct(scale=ti.f32, code=ti.i32)
    record = ti.types.struct(tag=ti.i32, inner=inner, bias=vector,
                             basis=ti.types.matrix(2, 2, ti.f32), column=column)
    large = ti.types.vector(65, ti.f32)
    matrix = ti.types.matrix(9, 9, ti.f32)
    widths = ti.types.struct(flag=ti.u1, other=ti.u1, flags=ti.types.vector(2, ti.u1),
                             half=ti.f16, precise=ti.f64, tag=ti.i8, code=ti.u16, wide=ti.u64)

    @ti.kernel
    def write_values(v: vector, r: record, out: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(3):
            out[i] = (r.inner.scale * v[i] + r.bias[i] + r.basis[i % 2, i % 2]
                      + r.column[i, 0] + r.tag + r.inner.code)

    @ti.kernel
    def value_return(v: vector, c: column) -> vector:
        return vector(v[0] + c[2, 0], v[1] - c[1, 0], v[2] * c[0, 0])

    @ti.kernel
    def wide(v: large, m: matrix, out: ti.types.ndarray(ti.f32, ndim=1)):
        out[0] = v[0] + v[32] + v[64] + m[0, 0] + m[4, 4] + m[8, 8]

    @ti.kernel
    def mixed_widths(r: widths, integers: ti.types.ndarray(ti.i64, ndim=1),
                     floats: ti.types.ndarray(ti.f64, ndim=1),
                     unsigned: ti.types.ndarray(ti.u64, ndim=1)):
        integers[0] = r.tag
        integers[1] = r.code
        integers[2] = r.flag + 2 * r.other
        integers[3] = r.flags[0] + 2 * r.flags[1]
        floats[0] = ti.cast(r.half, ti.f64) + r.precise
        unsigned[0] = r.wide

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
               v: vector, r: record,
               loss: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)):
        for i in range(3):
            loss[0] += r.inner.scale * v[i] * x[i] * x[i] + r.bias[i] * x[i]

    caps = (["spirv_has_int8", "spirv_has_int16", "spirv_has_int64",
             "spirv_has_float16", "spirv_has_float64", "spirv_has_int8_storage",
             "spirv_has_int16_storage"] if arch == "vulkan" else [])
    started = time.perf_counter()
    module = ti.aot.Module(caps=caps)
    for kernel in (write_values, value_return, wide, mixed_widths, energy):
        module.add_kernel(kernel)
    if include_backward:
        module.add_kernel(energy.grad, name="backward")
    module.save(str(path))
    module.archive(str(path / "values.tcm"))
    return time.perf_counter() - started
