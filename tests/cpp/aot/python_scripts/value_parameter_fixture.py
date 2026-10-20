"""Shared producer for logical value metadata and unchanged physical launches."""
import taichi_forge as ti


def add_value_parameter_kernels(module):
    vec3 = ti.types.vector(3, ti.f32)
    column = ti.types.matrix(3, 1, ti.f32)
    matrix = ti.types.matrix(2, 2, ti.f32)
    record = ti.types.struct(tag=ti.i32, vector=vec3, matrix=matrix, column=column)
    vec65 = ti.types.vector(65, ti.f32)
    mat9 = ti.types.matrix(9, 9, ti.f32)

    @ti.kernel
    def value_parameters(v: vec3, r: record) -> ti.f32:
        return (v[0] + 2 * v[1] + 3 * v[2] + r.tag + r.vector.sum()
                + r.matrix[0, 0] + 2 * r.matrix[0, 1]
                + 3 * r.matrix[1, 0] + 4 * r.matrix[1, 1] + r.column[2, 0])

    @ti.kernel
    def wide_parameters(v: vec65, m: mat9) -> ti.f32:
        return v[0] + v[64] + m[0, 0] + m[8, 8]

    module.add_kernel(value_parameters)
    module.add_kernel(value_parameters, name="value_parameters_alias")
    module.add_kernel(wide_parameters)
