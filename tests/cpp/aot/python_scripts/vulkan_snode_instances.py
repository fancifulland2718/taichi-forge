"""Fixed sparse tree recipes for shared-code, private-state native tests."""

import os

import taichi_forge as ti


def export_instances():
    ti.init(arch=ti.vulkan, offline_cache=False)
    assert ti.lang.impl.current_cfg().arch == ti.vulkan
    builder = ti.FieldsBuilder()
    grid, sibling = ti.field(ti.i32), ti.field(ti.f64)
    leaf = builder.pointer(ti.i, 4).dense(ti.i, 4)
    leaf.place(grid, sibling)
    builder.finalize()
    total = ti.field(ti.i32, shape=())
    unused_builder = ti.FieldsBuilder()
    unused = ti.field(ti.i32)
    unused_builder.place(unused)
    unused_builder.finalize()

    @ti.kernel
    def activate(active: ti.i32, base: ti.i32):
        for i in range(active):
            grid[i] = base + i
        total[None] += active

    @ti.kernel
    def update(delta: ti.i32):
        for i in grid:
            grid[i] += delta
        total[None] += delta

    @ti.kernel
    def read(out: ti.types.ndarray(ti.i32, 1)):
        for i in range(16):
            out[i] = grid[i]
        out[16] = total[None]

    @ti.kernel
    def drop():
        ti.deactivate(grid.parent().parent(), 0)

    @ti.kernel
    def unrelated():
        unused[None] += 1

    module = ti.aot.Module(ti.vulkan)
    for kernel in (activate, update, read, drop, unrelated):
        module.add_kernel(kernel)
    module.add_kernel(activate, name="activate_alias")
    module.save(os.environ["TAICHI_AOT_FOLDER_PATH"])


if __name__ == "__main__":
    export_instances()
