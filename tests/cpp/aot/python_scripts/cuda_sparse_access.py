"""Small real-code fixture for executor-wide CUDA sparse admission."""

import os

import taichi_forge as ti

ti.init(arch=ti.cuda, offline_cache=False, debug=True)
left = ti.field(ti.i32)
right = ti.field(ti.i32)
for value in (left, right):
    builder = ti.FieldsBuilder()
    builder.pointer(ti.i, 4).dense(ti.i, 4).place(value)
    builder.finalize()


@ti.kernel
def initialize():
    for i in range(16):
        left[i] = i + 3
        right[i] = i + 7


@ti.kernel
def bump_left():
    for i in left:
        left[i] += 1


@ti.kernel
def bump_right():
    for i in right:
        right[i] += 1


@ti.kernel
def cold_bump_left():
    for i in left:
        left[i] += 10


@ti.kernel
def peek_left() -> ti.i32:
    return left[0]


@ti.kernel
def peek_right() -> ti.i32:
    return right[0]


@ti.kernel
def fail_left():
    assert left[0] < 0, "left sparse diagnostic"


artifact_format = os.environ.get("TI_TEST_SPARSE_ARTIFACT_FORMAT", "bc")
caps = (
    [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
    if artifact_format == "cubin"
    else None
)
module = ti.aot.Module(ti.cuda, artifact_format=artifact_format, caps=caps)
module.add_field("left", left)
module.add_field("right", right)
for kernel in (initialize, bump_left, bump_right, cold_bump_left, peek_left, peek_right, fail_left):
    module.add_kernel(kernel)
graph = ti.graph.GraphBuilder()
graph.dispatch(bump_right)
graph.dispatch(bump_left)
module.add_graph("right_then_left", graph.compile())
module.save(os.environ["TAICHI_CUDA_SPARSE_ACCESS_FIXTURE"])
