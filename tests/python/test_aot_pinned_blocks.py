"""Nested block views reuse the common AOT resource closure without copies."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_pinned_composition import diagonal_with_adjoint
from tests.python.test_aot_pinned_from_graph import provider
from tests.python.test_linear_operator_graph_action import _rectangular_operator


def graph_for(operator, private=False):
    builder = ti.graph.GraphBuilder()
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y", "dx")}
    if private:
        symbols["y"] = builder.private_ndarray("y", ti.f32, (operator.shape[0],))
    builder.append_native(operator.graph_action(symbols["x"], symbols["y"], update_policy="pinned"))
    builder.append_native(operator.graph_action(symbols["y"], symbols["dx"], adjoint=True, update_policy="pinned"))
    return builder.compile()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_pinned_blocks_reject_private_symbol_impostors_and_missing_parent_dependencies():
    from types import MappingProxyType
    from taichi_forge.graph._native import NativeGraphNode

    operator = ti.linalg.block_diagonal([diagonal_with_adjoint([2]), diagonal_with_adjoint([3])])
    builder = ti.graph.GraphBuilder()
    builder.private_ndarray("y", ti.f32, (2,))
    x, y = (ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y"))
    with pytest.raises(ti.TaichiRuntimeError, match="symbol returned by private_ndarray"):
        builder.append_native(operator.graph_action(x, y, update_policy="pinned"))

    executable = operator.adjoint().compose(operator).graph_action(x, y, update_policy="pinned").compile()
    executable._action._binding_dependencies = MappingProxyType({})

    class MissingDependencies(NativeGraphNode):
        def compile(self):
            return executable

    with pytest.raises(ti.TaichiRuntimeError, match="Recordable action bindings must match"):
        ti.graph.GraphBuilder().append_native(MissingDependencies())


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_pinned_nested_blocks_export_views_over_public_and_private_parents(tmp_path, monkeypatch):
    right, matrix = provider(stateful=False)
    left = _rectangular_operator(matrix)
    diagonal = diagonal_with_adjoint([2])
    operator = ti.linalg.block_diagonal([left, ti.linalg.block_diagonal([right, diagonal])])
    normal = operator.adjoint().compose(operator).scaled(0.5)
    old, private, gram = graph_for(operator), graph_for(operator, True), graph_for(normal)
    parent = ti.graph.GraphBuilder()
    sequence = parent.create_sequential()
    ports = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y", "dx")}
    sequence.append_native(operator.graph_action(ports["x"], ports["y"], update_policy="pinned"))
    sequence.append_native(operator.graph_action(ports["y"], ports["dx"], adjoint=True, update_policy="pinned"))
    parent.append(sequence)
    parent.append(sequence)
    parent = parent.compile()
    expected = np.zeros((5, 7), np.float32)
    expected[:2, :3] = matrix
    expected[2:4, 3:6] = matrix
    expected[4, 6] = 2
    normal_matrix = 0.5 * expected.T @ expected
    x = np.arange(7, dtype=np.float32) * 0.25 - 0.5
    arrays = {name: ti.ndarray(ti.f32, n) for name, n in (("x", 7), ("y", 5), ("dx", 7))}
    arrays["x"].from_numpy(x)
    updated = ti.ndarray(ti.f32, 6)
    updated.from_numpy((2 * matrix).reshape(-1))
    left.update_numeric(updated, expected_topology_version=1, expected_numeric_version=1)
    current = graph_for(operator)
    field = ti.field(ti.f32, shape=(1, 7))
    field.from_numpy(x.reshape(1, 7))
    module = ti.aot.Module()
    from taichi_forge.linalg._runtime import _LinearOperatorGraphExecutable, _LinearOperatorCompositionGraphExecutable

    def forbidden(*args, **kwargs):
        raise AssertionError("Pinned block replay/export queried a provider generation")

    with monkeypatch.context() as check:
        check.setattr(_LinearOperatorGraphExecutable, "_bind_provider_generation", forbidden)
        check.setattr(_LinearOperatorCompositionGraphExecutable, "_bind_provider_generation", forbidden)
        old.run(arrays)
        np.testing.assert_allclose(arrays["y"].to_numpy(), expected @ x)
        np.testing.assert_allclose(arrays["dx"].to_numpy(), expected.T @ expected @ x)
        private.run({"x": field, "dx": arrays["dx"]})
        np.testing.assert_allclose(arrays["dx"].to_numpy(), expected.T @ expected @ x)
        square = dict(arrays, y=ti.ndarray(ti.f32, 7))
        gram.run(square)
        np.testing.assert_allclose(square["dx"].to_numpy(), normal_matrix.T @ normal_matrix @ x)
        before = ti.runtime.stats().transfer.device_to_host_bytes
        description = module.add_execution_plan("old", old)
        assert ti.runtime.stats().transfer.device_to_host_bytes - before == 88
        document = json.loads(description.to_json())
        assert len([r for r in document["resources"] if r["kind"] == "ndarray_view"]) == 16
        assert set(description.argument_names) == {"x", "y", "dx"}
        module.add_execution_plan("private", private)
        module.add_execution_plan("normal", gram)
        module.add_execution_plan("current", current)
        repeated = module.add_execution_plan("repeated", parent)
        assert sum(r["kind"] == "ndarray_view" for r in json.loads(repeated.to_json())["resources"]) == 16
    with pytest.raises(ti.TaichiRuntimeError, match="exactly"):
        old.bind(dict(arrays, x=ti.ndarray(ti.f32, 8)))
    for item in (old, private, gram, current, parent):
        item.close()
    del old, private, gram, current, parent, sequence, item, operator, normal, left, right, diagonal
    gc.collect()
    artifact = tmp_path / "blocks.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_pinned_blocks_probe.py")),
                mode,
                "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
