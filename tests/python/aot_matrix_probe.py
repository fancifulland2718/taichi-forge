"""Pure-Python reference for half-input MMA; independent consumer needs no DSL."""

import json
import struct
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from taichi_forge_runtime.aot._runtime import NdArray
from tests.python.aot_binding_constraints_probe import failure, imported
from tests.python.aot_cutensor_probe import near


def write(array, values, dtype="f"):
    array.write(struct.pack("<" + str(len(values)) + dtype, *values))


def read(array):
    data = array.read()
    return struct.unpack("<" + str(len(data) // 4) + "f", data)


def matmul(a, b, batch, transpose_a=False, transpose_b=False):
    result = []
    for t in range(batch):
        for i in range(16):
            for j in range(16):
                result.append(
                    sum(
                        a[t * 256 + (k * 16 + i if transpose_a else i * 16 + k)]
                        * b[t * 256 + (j * 16 + k if transpose_b else k * 16 + j)]
                        for k in range(16)
                    )
                )
    return result


def main(mode, artifact, batch):
    batch = int(batch)
    shape = (16, 16) if batch == 1 else (batch, 16, 16)
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        assert runtime.native_operation_capabilities("cuda_matrix_mma_f16_f32") == {"cuda_capture"}
        assert runtime.probe_native_operation("cuda_matrix_mma_f16_f32")["available"]
        runtime.configure_library("unused", "not-needed-for-driver-mma.dll")
        module = runtime.load_module(artifact)
        frames = []
        expected = []
        for index in range(2):
            arrays = {
                name: runtime.ndarray(dtype, shape)
                for name, dtype in (("a", "f16"), ("b", "f16"), ("c", "f32"), ("da", "f32"), ("db", "f32"))
            }
            arrays["loss"] = runtime.ndarray("f32", (1,))
            a = [((i * 3 + index) % 7 - 3) / 8 for i in range(batch * 256)]
            b = [((i * 5 + index) % 7 - 3) / 8 for i in range(batch * 256)]
            write(arrays["a"], a, "e")
            write(arrays["b"], b, "e")
            write(arrays["c"], [-91] * (batch * 256))
            c = matmul(a, b, batch)
            expected.append(
                (
                    c,
                    matmul(c, b, batch, transpose_b=True),
                    matmul(a, c, batch, transpose_a=True),
                    [0.5 * sum(v * v for v in c)],
                )
            )
            frames.append(arrays)
        calls = [module.prepare_execution("step", frame) for frame in frames]
        assert all(read(frame["c"]) == tuple([-91] * (batch * 256)) for frame in frames)

        def check(call, frame, reference):
            call()
            for name, values in zip(("c", "da", "db", "loss"), reference):
                near(read(frame[name]), values)

        for _ in range(2):
            for call, frame, reference in zip(calls, frames, expected):
                assert call.info["execution_kind"] == "cuda_graph"
                assert call.info["workspace_bytes"] == 0
                check(call, frame, reference)

        description = json.loads(module.execution_plan("step").to_json())
        description.update(version=10, binding_strategy="cuda_immutable_argument_frames")
        family = module.prepare_execution(ExecutionPlan.from_json(json.dumps(description)), frames[0])
        sibling = family.prepare_binding(frames[1])
        assert family.binding_family_info["identity"] == sibling.binding_family_info["identity"]
        assert family.binding_family_info["executables"] == 1
        check(family, frames[0], expected[0])
        check(sibling, frames[1], expected[1])
        family.close()
        check(sibling, frames[1], expected[1])

        bad = runtime.ndarray("f32", shape)
        failure(lambda: module.prepare_execution("step", dict(frames[0], a=bad)), "f16 inputs")
        oversized = runtime.ndarray("f16", (batch * 256,))
        failure(lambda: module.prepare_execution("step", dict(frames[0], b=oversized)), "shape")
        # Distinct imported allocation IDs must not hide physical aliasing.
        alias = imported(runtime, frames[0]["c"], "cuda", size=batch * 512)
        alias = NdArray(alias._memory, "f16", frames[0]["a"]._code, shape, ())
        failure(lambda: module.prepare_execution("step", dict(frames[0], a=alias)), "overlap")
        alias.close()
        storage = runtime.ndarray("f16", (batch * 256 + 16,))
        unaligned = imported(runtime, storage, "cuda", offset=2, size=batch * 512)
        unaligned = NdArray(unaligned._memory, "f16", frames[0]["a"]._code, shape, ())
        failure(lambda: module.prepare_execution("step", dict(frames[0], a=unaligned)), "alignment")
        unaligned.close()
        storage.close()
        check(calls[0], frames[0], expected[0])
        calls[0].close()
        check(calls[1], frames[1], expected[1])
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        failure(calls[1])
        failure(sibling)
        calls[1].close()
        sibling.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
