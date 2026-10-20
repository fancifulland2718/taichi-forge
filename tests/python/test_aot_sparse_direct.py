import array
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_sparse_direct_probe import resource_spec, solve_op


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_sparse_direct_aot_cpu_reconstruction_and_explicit_adjoint(tmp_path):
    @ti.kernel
    def energy(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), loss: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        for i in x:
            loss[0] += 0.5 * x[i] * x[i]

    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")

    def seed(dtype, entries):
        code = {"i32": "i", "f32": "f", "f64": "d"}[dtype]
        return dict(
            kind="ndarray", dtype=dtype, shape=[len(entries)], initial_data=module.add_data(array.array(code, entries))
        )

    def resources(spec, indices, values):
        return [
            spec,
            dict(name="offsets", **seed("i32", [0, 2, 5, 7])),
            dict(name="indices", **seed("i32", indices)),
            dict(name="seed", **seed(spec["parameters"]["dtype"], values)),
        ]

    fixed = resources(resource_spec(), [0, 1, 0, 1, 2, 1, 2], [4, 1, 1, 3, 1, 1, 2])
    fixed += [
        dict(kind="ndarray", name="dx", dtype="f32", shape=[3]),
        dict(kind="ndarray", name="dloss", dtype="f32", shape=[1]),
    ]
    module.add_execution_plan(
        "fixed",
        ti.aot.ExecutionPlan(
            [
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="fill32", destination="loss", value=0),
                dict(kind="fill32", destination="dloss", value=1065353216),
                solve_op([("b", "x")]),
                dict(kind="kernel", entry="energy", arguments=["x", "loss"]),
                dict(kind="kernel", entry="energy_grad", arguments=["x", "loss"]),
                solve_op([("dx", "db")]),
            ],
            resources=fixed,
            gradients={"x": "dx", "loss": "dloss"},
        ),
    )
    dynamic = resources(
        resource_spec(dtype="f64", method="LU", ordering="COLAMD", format="csr", refactor=True),
        [1, 0, 2, 0, 1, 2, 1],
        [2, 4, 1, 1, 3, 2, 0.5],
    )
    module.add_execution_plan(
        "dynamic", ti.aot.ExecutionPlan([solve_op([("b", "x"), ("b2", "x2")], "values")], resources=dynamic)
    )
    for storage_format, dtype in (("col_major", ti.f32), ("row_major", ti.f32), ("row_major", ti.f64)):
        builder = ti.linalg.SparseMatrixBuilder(3, 3, max_num_triplets=7, dtype=dtype, storage_format=storage_format)

        @ti.kernel
        def build(a: ti.types.sparse_matrix_builder()):
            for i in range(3):
                a[i, i] += 4 - i
                if i:
                    a[i, i - 1] += 1
                if i < 2:
                    a[i, i + 1] += 1

        build(builder)
        matrix = builder.build()
        solver = ti.linalg.SparseSolver(dtype=dtype, solver_type="LLT", ordering="COLAMD")
        solver.compute(matrix)
        recording = solver.record_solve(rhs="b", solution="x")
        graph_builder = ti.graph.GraphBuilder()
        graph_builder.append_native(recording)
        graph_builder.append_native(recording)
        graph = graph_builder.compile()
        rhs, output = ti.ndarray(dtype, 3), ti.ndarray(dtype, 3)
        rhs.from_numpy(np.asarray([6, 10, 8], dtype=np.float32 if dtype == ti.f32 else np.float64))
        graph.run(dict(b=rhs, x=output))
        np.testing.assert_allclose(output.to_numpy(), [1, 2, 3], rtol=1e-6)
        suffix = storage_format + ("_f32" if dtype == ti.f32 else "_f64")
        module.add_execution_plan("from_graph_" + suffix, graph)
        # Exported factors are independent of later source refactors.
        solver.factorize(matrix)
        with pytest.raises(RuntimeError, match="factor changed"):
            recording.execute(dict(b=rhs, x=output))
        with pytest.raises(RuntimeError, match="factor changed"):
            module.add_execution_plan("stale_" + suffix, graph)
    artifact = tmp_path / "sparse_direct.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_sparse_direct_probe.py")), mode, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Failed to retire" not in result.stdout + result.stderr


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_sparse_direct_cpu_factory_does_not_advertise_gpu_support():
    with ti.aot.borrow_runtime() as runtime:
        assert runtime.probe_native_resource("sparse_direct_plan")["available"] is False
        assert not runtime.native_operation_capabilities("sparse_direct_solve")
