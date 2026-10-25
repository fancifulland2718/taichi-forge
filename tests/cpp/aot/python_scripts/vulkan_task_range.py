"""Small compiled payloads consumed by native per-task recording tests."""
import json
import os
from pathlib import Path

import taichi_forge as ti


def main():
    ti.init(arch=ti.vulkan, offline_cache=False)

    @ti.kernel
    def sequence(x: ti.types.ndarray(dtype=ti.f32, ndim=1),
                 y: ti.types.ndarray(dtype=ti.f32, ndim=1),
                 limits: ti.types.ndarray(dtype=ti.i32, ndim=1),
                 status: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        status[0] = 0
        for i in range(limits[0], limits[1]):
            y[i + 3] = x[i + 3] * x[i + 3]
        status[0] = 17
        for i in range(limits[0] + 1, limits[1] - 1):
            y[i + 3] += 2 * x[i + 3]
        status[1] = status[0] + 5

    @ti.kernel
    def square(x: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
               y: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True),
               limits: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(limits[0], limits[1]):
            y[i + 3] = x[i + 3] * x[i + 3]

    x = ti.ndarray(ti.f32, shape=65, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=65, needs_grad=True)
    limits = ti.ndarray(ti.i32, shape=2)
    status = ti.ndarray(ti.i32, shape=2)
    module = ti.aot.Module(ti.vulkan)
    module.add_kernel(sequence, template_args=dict(x=x, y=y, limits=limits, status=status))
    module.add_kernel(square, template_args=dict(x=x, y=y, limits=limits))
    module.add_kernel(square.grad, name="square_grad", template_args=dict(x=x, y=y, limits=limits))
    destination = Path(os.environ["TAICHI_VULKAN_RANGE_AOT_DIR"])
    module.save(str(destination))
    document = json.loads((destination / "metadata.json").read_text())
    assert document["metadata_version"] == 13
    for kernel in document["kernels"]:
        qualified = [t for t in kernel["tasks_attribs"]
                     if t["device_range_dispatch_version"] == 1]
        assert len(qualified) == (2 if kernel["name"] == "sequence" else 1), kernel["name"]
    print("COMPILED-RANGE-QUALIFICATION", [(k["name"], len(k["tasks_attribs"])) for k in document["kernels"]])


if __name__ == "__main__":
    main()
