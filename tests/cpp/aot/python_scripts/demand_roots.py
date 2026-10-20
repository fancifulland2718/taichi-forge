"""Three independent snapshotted roots; consumers need only the selected roots."""

import argparse
import os

import taichi_forge as ti


def main(arch, artifact_format):
    ti.init(arch=arch, offline_cache=False)
    fields = []
    for seed in (11, 23, 37):
        field = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.dense(ti.i, 4).place(field)
        builder.finalize()
        field.fill(seed)
        fields.append(field)
    left, right, dormant = fields

    @ti.kernel
    def bump_left():
        for i in left:
            left[i] += 1

    @ti.kernel
    def bump_right():
        for i in right:
            right[i] += 1

    @ti.kernel
    def bump_dormant():
        for i in dormant:
            dormant[i] += 1

    @ti.kernel
    def peek_left() -> ti.i32:
        return left[0]

    @ti.kernel
    def peek_right() -> ti.i32:
        return right[0]

    caps = (
        [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
        if artifact_format == "cubin"
        else None
    )
    module = ti.aot.Module(artifact_format=artifact_format, caps=caps)
    module.add_field("right", right, as_array=True)
    for kernel in (bump_left, bump_right, bump_dormant, peek_left, peek_right):
        module.add_kernel(kernel)
    for name, kernel in (("left", bump_left), ("right", bump_right), ("dormant", bump_dormant)):
        graph = ti.graph.GraphBuilder()
        graph.dispatch(kernel)
        module.add_execution_plan(name, graph.compile(), snapshot_state=True)
    module.save(os.environ["TAICHI_AOT_FOLDER_PATH"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", choices=("cpu", "cuda"))
    parser.add_argument("--artifact-format", choices=("bc", "ptx", "cubin"), default="bc")
    args = parser.parse_args()
    main(getattr(ti, args.arch), args.artifact_format)
