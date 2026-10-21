"""Legacy CUDA factors share the runtime core without a compiler in consumers."""

import array
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from tests import test_utils
from tests.python.aot_sparse_direct_probe import resource_spec, solve_op


class _RetainedSparseGraph(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, graph, bindings):
        self.graph = graph
        self.bindings = graph.bind(bindings)

    def compile(self):
        return self

    def run(self):
        self.graph.run(self.bindings)

    def _export_aot(self, context):
        context.bound_graph(self.graph, self.bindings)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_sparse_direct_cuda_export_capture_ordered_and_adjoint(tmp_path):
    @ti.kernel
    def energy(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in x:
            loss[0] += 0.5 * x[i] * x[i]

    @ti.kernel
    def scale(source: ti.types.ndarray(ti.f32, 1), target: ti.types.ndarray(ti.f32, 1)):
        for i in source:
            target[i] = 2 * source[i]

    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    module.add_kernel(scale)

    def seeds(spec, indices, values):
        resources = [spec]
        for name, dtype, code, entries in (
            ("offsets", "i32", "i", [0, 2, 5, 7]),
            ("indices", "i32", "i", indices),
            ("seed", "f32", "f", values),
        ):
            resources.append(
                dict(
                    kind="ndarray",
                    name=name,
                    dtype=dtype,
                    shape=[len(entries)],
                    initial_data=module.add_data(array.array(code, entries)),
                )
            )
        return resources

    fixed = seeds(
        resource_spec(implementation="cusolver_sp", ordering="COLAMD"),
        [0, 1, 0, 1, 2, 1, 2],
        [4, 1, 1, 3, 1, 1, 2],
    )
    fixed += [
        dict(kind="ndarray", name=name, dtype="f32", shape=[size])
        for name, size in (("rhs", 3), ("dx", 3), ("dloss", 1))
    ]
    module.add_execution_plan(
        "fixed",
        ti.aot.ExecutionPlan(
            [
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="fill32", destination="loss", value=0),
                dict(kind="fill32", destination="dloss", value=1065353216),
                dict(kind="kernel", entry="scale", arguments=["b", "rhs"]),
                solve_op([("rhs", "x")]),
                dict(kind="kernel", entry="energy", arguments=["x", "loss"]),
                dict(kind="kernel", entry="energy_grad", arguments=["x", "loss"]),
                solve_op([("dx", "db")]),
            ],
            resources=fixed,
            gradients={"x": "dx", "loss": "dloss"},
        ),
    )
    for method in ("LLT", "LU"):
        # Unsorted CSR; numeric updates preserve this input order.
        values = [1, 4, 1, 1, 3, 2, 1] if method == "LLT" else [2, 4, 1, 1, 3, 2, 0.5]
        resources = seeds(
            resource_spec(implementation="cusolver_sp", method=method, format="csr", refactor=True),
            [1, 0, 2, 0, 1, 2, 1],
            values,
        )
        module.add_execution_plan(
            "dynamic_" + method,
            ti.aot.ExecutionPlan(
                [
                    solve_op([("b", "x"), ("b2", "x2")], "values", ordered=True),
                    dict(kind="kernel", entry="scale", arguments=["x", "scaled"]),
                ],
                resources=resources,
            ),
        )
        builder = ti.linalg.SparseMatrixBuilder(3, 3, max_num_triplets=7)

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
        solver = ti.linalg.SparseSolver(solver_type=method, ordering="COLAMD", provider="cusolver_sp")
        solver.compute(matrix)
        recording = solver.record_solve(rhs="b", solution="x")
        assert recording._definition is solver._definition
        graph_builder = ti.graph.GraphBuilder()
        graph_builder.append_native(recording)
        graph_builder.append_native(recording)
        graph = graph_builder.compile()
        b, x = ti.ndarray(ti.f32, 3), ti.ndarray(ti.f32, 3)
        b.from_numpy(np.asarray([6, 10, 8], dtype=np.float32))
        graph.run(dict(b=b, x=x))
        np.testing.assert_allclose(x.to_numpy(), [1, 2, 3], rtol=1e-5)
        module.add_execution_plan("exported_" + method, graph)
        parent = ti.graph.GraphBuilder()
        parent.append_native(_RetainedSparseGraph(graph, dict(b=b, x=x)))
        parent_graph = parent.compile()
        parent_graph.run({})
        np.testing.assert_allclose(x.to_numpy(), [1, 2, 3], rtol=1e-5)
        module.add_execution_plan("parent_" + method, parent_graph, source_bindings=dict(b=b, x=x))
        solver.factorize(matrix)
        with pytest.raises(RuntimeError, match="factor changed"):
            recording.execute(dict(b=b, x=x))
        with pytest.raises(RuntimeError, match="factor changed"):
            module.add_execution_plan("stale_" + method, graph)
    artifact = tmp_path / "sparse_cuda.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_sparse_direct_cuda_probe.py")), mode, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Failed to retire" not in result.stdout + result.stderr
