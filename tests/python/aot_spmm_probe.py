"""SpMM prepared state and numeric updates, without importing the producer DSL."""

import json
import sys
from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from taichi_forge_runtime.aot._runtime import NdArray
from tests.python.aot_binding_constraints_probe import failure, imported
from tests.python.aot_dot_probe import near, read, write


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        assert runtime.probe_native_operation("cusparse_spmm_f32")["available"]
        module = runtime.load_module(artifact)
        frames = []
        for name in ("row", "direct", "optional", "prepared", "single"):
            rhs = 1 if name == "single" else 3
            shapes = dict(
                x=(4, rhs), y0=(3, rhs), y=(3, rhs), dx=(4, rhs), values=(4,), transpose_values=(4,), loss=(1,)
            )
            frame = {k: runtime.ndarray("f32", shape) for k, shape in shapes.items()}
            write(frame["y"], [99] * (3 * rhs))
            call = module.prepare_execution(name, frame)
            assert read(frame["y"]) == [99] * (3 * rhs), "Preparing must execute no SpMM"
            assert call.info["execution_kind"] == "cuda_graph"
            frames.append((name, rhs, frame, call))
        for step in range(3):
            for _, rhs, frame, call in frames:
                scale = float(step - 1)
                x = [(i - 3) / 4 for i in range(4 * rhs)]
                y0 = [(i + 1) / 8 for i in range(3 * rhs)]
                write(frame["x"], x)
                write(frame["y0"], y0)
                write(frame["values"], [v * scale for v in (2, -1, 3, 0.5)])
                write(frame["transpose_values"], [v * scale for v in (2, 3, -1, 0.5)])
                call()
                a = [[2 * scale, 0, -scale, 0], [0, 0, 0, 0], [0, 3 * scale, 0, 0.5 * scale]]
                y = [
                    1.5 * sum(a[i][k] * x[k * rhs + j] for k in range(4)) - 0.25 * y0[i * rhs + j]
                    for i in range(3)
                    for j in range(rhs)
                ]
                dx = [1.5 * sum(a[i][k] * y[i * rhs + j] for i in range(3)) for k in range(4) for j in range(rhs)]
                near(read(frame["y"]), y)
                near(read(frame["loss"]), [0.5 * sum(v * v for v in y)])
                near(read(frame["dx"]), dx)

        _, _, frame, _ = frames[0]
        write(frame["y"], [99] * 9)
        wrong = runtime.ndarray("f32", (9,))
        failure(lambda: module.prepare_execution("row", dict(frame, y=wrong)))
        wrong.close()
        alias = imported(runtime, frame["x"], "cuda", size=36)
        typed_alias = NdArray(alias._memory, "f32", alias._code, (3, 3), ())
        failure(lambda: module.prepare_execution("row", dict(frame, y=typed_alias)), "overlaps")
        alias.close()
        document = json.loads(module.execution_plan("row").to_json())
        for patch in (
            dict(rhs_count=0),
            dict(rhs_count=True),
            dict(rhs_count=0x7FFFFFFFFFFFFFFF),
            dict(algorithm="automatic"),
            dict(preprocess=True),
            dict(preprocess="maybe"),
            dict(expected_workspace_bytes=0x7FFFFFFFFFFFFFFF),
        ):
            changed = dict(document)
            changed["operations"] = [
                dict(op, parameters=dict(op["parameters"], **patch)) if op.get("operation") == "spmm_f32" else op
                for op in document["operations"]
            ]
            failure(lambda: ExecutionPlan.from_json(json.dumps(changed)).prepare(module, frame))
        assert read(frame["y"]) == [99] * 9
        frames[0][3].close()
        frames[1][3]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        failure(frames[1][3])
        for _, _, _, call in frames[1:]:
            call.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
