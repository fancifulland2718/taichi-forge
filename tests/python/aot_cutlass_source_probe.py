"""Independent CUTLASS AOT consumer with a pure-Python matrix/adjoint reference."""

import json
import os
import struct
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from taichi_forge_runtime.aot._runtime import NdArray
from tests.python.aot_binding_constraints_probe import failure, imported
from tests.python.aot_cutensor_probe import near
from tests.python.aot_matrix_probe import read, write


def main(mode, artifact, transpose_a, transpose_b, activation):
    ta, tb = bool(int(transpose_a)), bool(int(transpose_b))
    m, n, k = 7, 9, 129
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider(
            "cutlass_source",
            adapter_path=os.environ["TI_CUTLASS_SOURCE_TEST_ADAPTER"],
            library_path=os.environ["TI_CUTLASS_SOURCE_TEST_LIBRARY"],
        )
        assert runtime.probe_native_operation("cutlass_source_execute")["available"]
        module = runtime.load_module(artifact)
        frames, references, inputs = [], [], []
        for index in range(2):
            shapes = dict(
                a=(k, m) if ta else (m, k),
                b=(n, k) if tb else (k, n),
                output=(m, n),
                da=(k, m) if ta else (m, k),
                db=(n, k) if tb else (k, n),
                dprevious=(m, n),
                loss=(1,),
            )
            frame = {name: runtime.ndarray("f32", shape) for name, shape in shapes.items()}
            a = [((i * 3 + index) % 11 - 5) / 16 for i in range(m * k)]
            b = [((i * 7 + index) % 13 - 6) / 16 for i in range(n * k)]
            previous = [((i + index) % 5 - 2) / 8 for i in range(m * n)]
            for name, values in (("a", a), ("b", b), ("output", previous)):
                write(frame[name], values)
            for name in ("da", "db", "dprevious", "loss"):
                write(
                    frame[name], [-91] * (shapes[name][0] * (shapes[name][1] if len(shapes[name]) > 1 else 1))
                )
            frames.append(frame)
            references.append(previous)
            inputs.append((a, b))
        calls = [module.prepare_execution("step", frame) for frame in frames]
        assert all(read(frame["output"]) == tuple(reference) for frame, reference in zip(frames, references))

        def check(call, index, iterations=1):
            frame, (a, b) = frames[index], inputs[index]
            amap = lambda i, j: j * m + i if ta else i * k + j
            bmap = lambda i, j: j * k + i if tb else i * n + j
            previous = references[index]
            for _ in range(iterations):
                output = [
                    0.75 * sum(a[amap(i, t)] * b[bmap(t, j)] for t in range(k)) + 0.25 * previous[i * n + j]
                    for i in range(m)
                    for j in range(n)
                ]
                if activation == "relu":
                    output = [max(v, 0) for v in output]
                previous = output
            call()
            if not iterations:
                near(read(frame["output"]), previous)
                return
            references[index] = output
            dy = [v if activation != "relu" or v > 0 else 0 for v in output]
            da, db = [0] * (m * k), [0] * (k * n)
            for i in range(m):
                for t in range(k):
                    da[amap(i, t)] = 0.75 * sum(dy[i * n + j] * b[bmap(t, j)] for j in range(n))
            for t in range(k):
                for j in range(n):
                    db[bmap(t, j)] = 0.75 * sum(a[amap(i, t)] * dy[i * n + j] for i in range(m))
            for name, expected in (
                ("output", output),
                ("da", da),
                ("db", db),
                ("dprevious", [0.25 * v for v in dy]),
                ("loss", [0.5 * sum(v * v for v in output)]),
            ):
                near(read(frame[name]), expected)

        for _ in range(2):
            for index, call in enumerate(calls):
                assert call.info["execution_kind"] == "cuda_graph"
                check(call, index)
        inputs[0] = ([-v for v in inputs[0][0]], inputs[0][1])
        write(frames[0]["a"], inputs[0][0])
        check(calls[0], 0)
        document = json.loads(module.execution_plan("step").to_json())
        for key, value, message in (
            ("binary_sha256", "0" * 64, "identity"),
            ("cutlass_version", 1, "identity"),
            ("workspace_bytes", 1, "workspace"),
            ("target_code", ["sm_999"], "target code"),
            ("minimum_driver_api_version", 999999, "driver"),
        ):
            bad = json.loads(json.dumps(document))
            next(r for r in bad["resources"] if r["kind"] == "native")["parameters"][key] = value
            failure(
                lambda: module.prepare_execution(ExecutionPlan.from_json(json.dumps(bad)), frames[0]), message
            )
        wrong = runtime.ndarray("i32", frames[0]["a"].shape)
        failure(lambda: module.prepare_execution("step", dict(frames[0], a=wrong)), "f32")
        wrong.close()
        wrong = runtime.ndarray("f32", (m * k,))
        failure(lambda: module.prepare_execution("step", dict(frames[0], a=wrong)), "shape")
        wrong.close()
        alias = imported(runtime, frames[0]["a"], "cuda", size=m * n * 4)
        alias = NdArray(alias._memory, "f32", frames[0]["output"]._code, (m, n), ())
        failure(lambda: module.prepare_execution("step", dict(frames[0], output=alias)), "overlap")
        alias.close()
        document.update(version=10, binding_strategy="cuda_immutable_argument_frames")
        family = module.prepare_execution(ExecutionPlan.from_json(json.dumps(document)), frames[0])
        sibling = family.prepare_binding(frames[1])
        assert family.binding_family_info["identity"] == sibling.binding_family_info["identity"]
        assert family.binding_family_info["executables"] == 1
        check(family, 0)
        check(sibling, 1)
        family.close()
        check(sibling, 1)
        target, iterations = (runtime.ndarray("i32", 1) for _ in range(2))
        loop = module.prepare_execution("loop", dict(frames[0], target=target, iterations=iterations))
        for limit in (0, 1, 3, 5):
            target.write(struct.pack("<i", limit))
            check(loop, 0, min(limit, 3))
            assert struct.unpack("<i", iterations.read())[0] == min(limit, 3)
        loop.close()
        check(calls[0], 0)
        calls[0].close()
        check(calls[1], 1)
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
