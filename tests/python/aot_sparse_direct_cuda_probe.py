"""Source-free CUDA capture, ordered numerical phases and factor retirement."""

import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from aot_sparse_direct_probe import dense_solve, near, raises, read, resource_spec, solve_op, write


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        assert runtime.probe_native_resource("sparse_direct_plan")["available"]
        assert runtime.native_operation_capabilities("sparse_direct_solve") == {"cuda_capture"}
        assert runtime.native_operation_capabilities("sparse_direct_ordered_solve") == {"cuda_ordered"}
        module = runtime.load_module(artifact)
        rhs = [6, 10, 8]
        spd = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]

        def frame(scale):
            buffers = {key: runtime.ndarray("f32", (3,)) for key in ("b", "x", "db")}
            buffers["loss"] = runtime.ndarray("f32", (1,))
            write(buffers["b"], [scale * v for v in rhs])
            write(buffers["x"], [-99] * 3)
            return buffers

        left, right = frame(1), frame(-0.5)
        run = module.prepare_execution("fixed", left)
        other = module.prepare_execution("fixed", right)
        assert read(left["x"]) == [-99] * 3
        assert run.info["execution_kind"] == "cuda_graph"
        for _ in range(2):
            run()
            other()
        for buffers, scale in ((left, 2), (right, -1)):
            expected = [scale * v for v in (1, 2, 3)]
            near(read(buffers["x"]), expected)
            near(read(buffers["db"]), dense_solve(spd, expected))
            near(read(buffers["loss"]), [sum(v * v for v in expected) / 2])
        for method in ("LLT", "LU"):
            for prefix in ("exported_", "parent_"):
                exported = module.prepare_execution(prefix + method, dict(b=left["b"], x=left["x"]))
                exported()
                near(read(left["x"]), [1, 2, 3])
                exported.close()
        # Fixed-factor replay can use an independent stream and completion.
        domain = runtime.execution_domain()
        selected = module.prepare_execution("fixed", left, execution_domain=domain)
        selected.initialize()
        completion = selected.submit()
        assert completion is not None
        completion.wait()
        completion.close()
        near(read(left["x"]), [2, 4, 6])
        selected.close()
        domain.close()

        for method in ("LLT", "LU"):
            general = spd if method == "LLT" else [[4, 2, 0], [1, 3, 1], [0, 0.5, 2]]
            values = [1, 4, 1, 1, 3, 2, 1] if method == "LLT" else [2, 4, 1, 1, 3, 2, 0.5]
            dynamic = {key: runtime.ndarray("f32", (3,)) for key in ("b", "b2", "x", "x2", "scaled")}
            dynamic["values"] = runtime.ndarray("f32", (7,))
            write(dynamic["b"], rhs)
            write(dynamic["b2"], [-v for v in rhs])
            active = module.prepare_execution("dynamic_" + method, dynamic)
            assert active.info["execution_kind"] == "cuda_ordered"
            for scale in (1, 2, 0.5):
                write(dynamic["values"], [scale * v for v in values])
                active()
                expected = [v / scale for v in dense_solve(general, rhs)]
                near(read(dynamic["x"]), expected)
                near(read(dynamic["x2"]), [-v for v in expected])
                near(read(dynamic["scaled"]), [2 * v for v in expected])
            write(dynamic["values"], [0] * 7)
            write(dynamic["x"], [-99] * 3)
            raises(active)
            near(read(dynamic["x"]), [-99] * 3, 0)
            # A failed prepared CUDA sequence retires; recovery uses a new
            # prepared owner rather than replaying a partially failed sequence.
            active.close()
            write(dynamic["values"], values)
            active = module.prepare_execution("dynamic_" + method, dynamic)
            active()
            near(read(dynamic["x"]), dense_solve(general, rhs))
            raises(lambda: module.prepare_execution("dynamic_" + method, dict(dynamic, x=dynamic["b"])))
            domain = runtime.execution_domain()
            raises(lambda: module.prepare_execution("dynamic_" + method, dynamic, execution_domain=domain))
            domain.close()
            active.close()

        # Construction copies seed arrays. A shared plan serializes workspace
        # within one execution domain and rejects concurrent cross-domain use.
        seed = {
            key: runtime.ndarray(dtype, (size,))
            for key, dtype, size in (("offsets", "i32", 4), ("indices", "i32", 7), ("seed", "f32", 7))
        }
        write(seed["offsets"], [0, 2, 5, 7], "i")
        write(seed["indices"], [0, 1, 0, 1, 2, 1, 2], "i")
        write(seed["seed"], [4, 1, 1, 3, 1, 1, 2])
        spec = resource_spec(implementation="cusolver_sp")
        bindings = {key: seed[name] for key, name in spec["bindings"].items()}
        for invalid in (dict(method="LDLT"), dict(dtype="f64"), dict(implementation="eigen")):
            raises(
                lambda: runtime.native_resource(
                    "sparse_direct", "plan", parameters=dict(spec["parameters"], **invalid), bindings=bindings
                )
            )
        write(seed["indices"], [0, 0, 0, 1, 2, 1, 2], "i")
        raises(
            lambda: runtime.native_resource("sparse_direct", "plan", parameters=spec["parameters"], bindings=bindings)
        )
        write(seed["indices"], [0, 1, 0, 1, 2, 1, 2], "i")
        owner = runtime.native_resource("sparse_direct", "plan", parameters=spec["parameters"], bindings=bindings)
        for value in seed.values():
            value.close()
        plan = ExecutionPlan([solve_op([("b", "x")])])
        public_bindings = dict(solver=owner, b=left["b"], x=left["x"])
        public = plan.prepare(module, public_bindings)
        domain = runtime.execution_domain()
        raises(lambda: plan.prepare(module, public_bindings, execution_domain=domain))
        domain.close()
        public.close()
        domain = runtime.execution_domain()
        public = plan.prepare(module, public_bindings, execution_domain=domain)
        public.initialize()
        ticket = public.submit()
        assert ticket is not None
        # No application wait before closing the factor owner.
        owner.close()
        ticket.wait()
        ticket.close()
        near(read(left["x"]), [1, 2, 3])
        raises(public)
        public.close()
        domain.close()
        run.close()
        other()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(other)
        other.close()
    if mode != "borrowed":
        assert "taichi_forge" not in sys.modules
        assert "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
