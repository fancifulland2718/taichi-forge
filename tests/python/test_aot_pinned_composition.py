"""Compound operator epochs retain coefficients, code, and private scratch."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_pinned_operator import operator_graph
from tests.python.test_linear_operator_graph_action import _rectangular_operator


def diagonal_with_adjoint(values):
    count = len(values)
    topology, numeric = ti.ndarray(ti.i32, count), ti.ndarray(ti.f32, count)
    topology.from_numpy(np.arange(count, dtype=np.int32))
    numeric.from_numpy(np.array(values, np.float32))

    @ti.kernel
    def apply_diagonal(
        size: ti.i32,
        topology: ti.types.ndarray(ti.i32, ndim=1),
        numeric: ti.types.ndarray(ti.f32, ndim=1),
        x: ti.types.ndarray(ti.f32, ndim=1),
        y: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in range(size):
            y[i] = numeric[i] * x[topology[i]]

    return ti.linalg.LinearOperator.from_kernel(
        apply_diagonal, count, topology, numeric=numeric, adjoint=apply_diagonal
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_pinned_composition_exports_numeric_and_coefficient_epochs(tmp_path, monkeypatch):
    a = np.array([[2, -1, 0.5], [3, 4, -2]], np.float32)
    b = np.array([[1, 0.25, -2], [0.5, -1, 3]], np.float32)
    left, right = _rectangular_operator(a), _rectangular_operator(b)
    diagonal = diagonal_with_adjoint([1, 2, 3])
    affine = left.parameterized_affine(right, alpha=0.5, beta=-0.25, alpha_range=(-2, 2), beta_range=(-2, 2))
    operator = (affine.adjoint().compose(affine).scaled(0.5) + diagonal).shifted(0.25)
    before = ti.runtime.stats().transfer.device_to_host_bytes
    old = operator_graph(operator)
    independent = operator_graph(operator)
    rectangle = operator_graph(affine)
    assert ti.runtime.stats().transfer.device_to_host_bytes == before
    old_affine = 0.5 * a - 0.25 * b
    old_matrix = 0.5 * old_affine.T @ old_affine + np.diag([1.25, 2.25, 3.25])
    new_values = ti.ndarray(ti.f32, 6)
    new_values.from_numpy((2 * a).reshape(-1))
    left.update_numeric(new_values, expected_topology_version=1, expected_numeric_version=1)
    new_values.fill(0)
    affine.update_parameters(alpha=1.25, beta=0.5, expected_version=1)
    current = operator_graph(operator)
    new_affine = 2.5 * a + 0.5 * b
    new_matrix = 0.5 * new_affine.T @ new_affine + np.diag([1.25, 2.25, 3.25])
    arrays = {name: ti.ndarray(ti.f32, 3) for name in ("x", "y", "dx")}
    other_arrays = {name: ti.ndarray(ti.f32, 3) for name in arrays}
    first, second, updated = old.bind(arrays), independent.bind(other_arrays), current.bind(arrays)

    from taichi_forge.lang._ndarray import ScalarNdarray
    from taichi_forge.linalg._runtime import _LinearOperatorCompositionGraphExecutable, _LinearOperatorGraphExecutable

    def scratch(graph):
        return {
            value._runtime_allocation_identity
            for value in graph._instance._fixed_runtime_args.values()
            if isinstance(value, ScalarNdarray)
        }

    assert scratch(old) and scratch(old).isdisjoint(scratch(independent))

    def forbidden(*args, **kwargs):
        raise AssertionError("Pinned replay or export reselected a published generation")

    module = ti.aot.Module()
    with monkeypatch.context() as fixed:
        fixed.setattr(_LinearOperatorGraphExecutable, "_bind_provider_generation", forbidden)
        fixed.setattr(_LinearOperatorCompositionGraphExecutable, "_bind_provider_generation", forbidden)
        for graph, frame, target, matrix, x in (
            (old, first, arrays, old_matrix, [0.25, -0.5, 2]),
            (independent, second, other_arrays, old_matrix, [-1, 0.5, 0.25]),
            (current, updated, arrays, new_matrix, [1, 0.5, -0.25]),
            (old, first, arrays, old_matrix, [2, -0.5, 0.75]),
        ):
            target["x"].from_numpy(np.array(x, np.float32))
            graph.run(frame)
            np.testing.assert_allclose(target["y"].to_numpy(), matrix @ x, rtol=3e-6, atol=1e-5)
            np.testing.assert_allclose(target["dx"].to_numpy(), matrix.T @ matrix @ x, rtol=3e-6, atol=2e-4)
        before = ti.runtime.stats().transfer.device_to_host_bytes
        plan = module.add_execution_plan("old", old)
        assert ti.runtime.stats().transfer.device_to_host_bytes - before == 120
        description = json.loads(plan.to_json())
        resources = description["resources"]
        assert len([r for r in resources if "initial_data" in r]) == 6
        scratch_bytes = sum(r["shape"][0] * 4 for r in resources if r.get("zero_initialize"))
        assert scratch_bytes > 0
        assert len(module._data_payloads) == 5
        module.add_execution_plan("current", current)
        module.add_execution_plan("rectangle", rectangle)
        assert len(module._data_payloads) == 6, "Repeated composition leaves duplicated immutable payloads"
    for graph in (old, independent, current, rectangle):
        graph.close()
    del first, second, updated, old, independent, current, rectangle, graph, frame
    del operator, affine, left, right, diagonal
    gc.collect()
    artifact = tmp_path / "composition.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_pinned_composition_probe.py")),
                mode,
                "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name,
                str(artifact),
                str(scratch_bytes),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
