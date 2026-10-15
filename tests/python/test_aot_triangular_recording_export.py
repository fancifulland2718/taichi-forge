"""Triangular recordings preserve selected Graph order and one CSR snapshot."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.graph._recipes.map_fusion import GraphMapFusionRecipeProvider
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("rhs_count", (1, 2))
def test_triangular_recording_and_selected_recipe_export(tmp_path, rhs_count):
    probe = (
        ti.hardware.linalg.cusparse_spsv_is_available
        if rhs_count == 1
        else ti.hardware.linalg.cusparse_spsm_is_available
    )
    if not probe():
        pytest.skip("External cuSPARSE triangular solve is unavailable")
    arrays = []
    for data, dtype in (([0, 1, 3, 5], ti.i32), ([0, 0, 1, 1, 2], ti.i32), ([2, 1, 4, -1, 2], ti.f32)):
        array = ti.ndarray(dtype, len(data))
        array.from_numpy(np.asarray(data, np.float32 if dtype == ti.f32 else np.int32))
        arrays.append(array)
    matrix = ti.linalg.SparsePattern.csr(3, 3, *arrays[:2]).matrix(arrays[2])
    constructor = (
        ti.hardware.linalg.CusparseSpsvRecording if rhs_count == 1 else ti.hardware.linalg.CusparseSpsmRecording
    )
    arguments = (matrix,) if rhs_count == 1 else (matrix, rhs_count)
    shape = (3,) if rhs_count == 1 else (3, rhs_count)
    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=len(shape))
        for name in ("x", "y", "dx", "scratch", "result")
    }
    frames = {name: ti.ndarray(ti.f32, shape) for name in symbols}
    host = np.arange(3 * rhs_count, dtype=np.float32).reshape(shape) / 4 - 0.5
    frames["x"].from_numpy(host)

    if rhs_count == 1:

        @ti.kernel
        def scale(source: ti.types.ndarray(ti.f32, 1), output: ti.types.ndarray(ti.f32, 1)):
            for i in range(3):
                output[i] = source[i] * 2

        @ti.kernel
        def shift(source: ti.types.ndarray(ti.f32, 1), output: ti.types.ndarray(ti.f32, 1)):
            for i in range(3):
                output[i] = source[i] + 0.25

    else:

        @ti.kernel
        def scale(source: ti.types.ndarray(ti.f32, 2), output: ti.types.ndarray(ti.f32, 2)):
            for i, j in ti.ndrange(3, 2):
                output[i, j] = source[i, j] * 2

        @ti.kernel
        def shift(source: ti.types.ndarray(ti.f32, 2), output: ti.types.ndarray(ti.f32, 2)):
            for i, j in ti.ndrange(3, 2):
                output[i, j] = source[i, j] + 0.25

    builder = ti.graph.GraphBuilder()
    builder.append_native(constructor(*arguments, input="x", output="y"))
    builder.append_native(constructor(*arguments, transpose=True, input="y", output="dx"))
    builder.dispatch(scale, symbols["dx"], symbols["scratch"])
    builder.dispatch(shift, symbols["scratch"], symbols["result"])
    graph = builder.compile()
    module = ti.aot.Module()

    def export(name, source):
        before = matrix._debug_runtime_stats()["transfers"]["device_to_host_bytes"]
        plan = module.add_execution_plan(name, source)
        after = matrix._debug_runtime_stats()["transfers"]["device_to_host_bytes"]
        assert after - before == 56, "Both triangular plans must share one cold CSR snapshot"
        resources = json.loads(plan.to_json())["resources"]
        assert sum(r.get("operation") == "csr_f32" for r in resources) == 1
        assert sum(r.get("operation") == "triangular_plan_f32" for r in resources) == 2

    prefix = "spsv" if rhs_count == 1 else "spsm"
    export("cold", graph)
    assert matrix._debug_runtime_stats()["operations"][prefix + "_plan_builds"] == 0
    graph.run(graph.bind(frames))
    dense = np.array([[2, 0, 0], [1, 4, 0], [0, -1, 2]], np.float32)
    expected = 2 * np.linalg.solve(dense.T, np.linalg.solve(dense, host)) + 0.25
    np.testing.assert_allclose(frames["result"].to_numpy(), expected, rtol=2e-5, atol=2e-5)
    export("warm", graph)
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(), GraphMapFusionRecipeProvider()))
    catalog.build_compatible_stage(candidate_limit=16)
    # Only the one-dimensional maps meet the current fusion domain. Both
    # layouts still exercise selected materialization and native barriers.
    assert len(catalog.entries()) == (2 if rhs_count == 1 else 1)
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        for index, entry in enumerate(catalog.entries()):
            with context.materialize(entry.recipe) as result:
                selected = result.executor
                selected.run(selected.bind(frames))
                np.testing.assert_allclose(frames["result"].to_numpy(), expected, rtol=2e-5, atol=2e-5)
                export(f"selected_{index}", selected)
    changed = ti.ndarray(ti.f32, 5)
    changed.from_numpy(np.array([4, 2, 8, -2, 4], np.float32))
    matrix.update_values(changed)
    export("changed", graph)
    assert len(module._data_payloads) == 4

    class Unqualified(constructor):
        pass

    bad = ti.graph.GraphBuilder()
    bad.append_native(Unqualified(*arguments))
    bad_graph = bad.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="unavailable"):
        module.add_execution_plan("bad", bad_graph)
    bad_graph.close()
    graph.close()
    artifact = tmp_path / "triangular-recording.tcm"
    module.archive(str(artifact))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(Path(__file__).with_name("aot_triangular_recording_probe.py")),
            str(artifact),
            str(rhs_count),
        ],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
