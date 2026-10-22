"""Explicit consumer matrices preserve shared and independent sparse owners."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_public_sparse_matrix_with_private_triangular_analysis(tmp_path):
    if not ti.hardware.linalg.cusparse_spsv_is_available():
        pytest.skip("External cuSPARSE triangular solve required")
    arrays = []
    for dtype, data in ((ti.i32, [0, 1, 3]), (ti.i32, [0, 0, 1]), (ti.f32, [2, 1, 4])):
        array = ti.ndarray(dtype, len(data))
        array.from_numpy(np.asarray(data, np.int32 if dtype == ti.i32 else np.float32))
        arrays.append(array)
    pattern = ti.linalg.SparsePattern.csr(2, 2, *arrays[:2])
    left, right = pattern.matrix(arrays[2]), pattern.matrix(arrays[2])
    builder = ti.graph.GraphBuilder()
    builder.append_native(ti.hardware.linalg.CusparseSpmvRecording(left, input="x", output="product"))
    solve = ti.hardware.linalg.CusparseSpsvRecording(left, input="product", output="recovered")
    builder.append_native(solve)
    builder.append_native(ti.hardware.linalg.CusparseSpmvRecording(right, input="x", output="other"))
    graph = builder.compile()
    module = ti.aot.Module()
    before = [m._debug_runtime_stats()["transfers"]["device_to_host_bytes"] for m in (left, right)]
    for name in ("first", "again"):
        plan = module.add_execution_plan(name, graph, resource_bindings={left: "left", right: "right"})
        doc = json.loads(plan.to_json())
        resources = doc["resources"]
        assert len(resources) == 1 and resources[0]["operation"] == "triangular_plan_f32"
        assert resources[0]["parameters"]["mutable_values"]
        assert resources[0]["bindings"]["matrix"] == "left"
        assert [op["operation"] for op in doc["operations"]] == [
            "spmv_f32",
            "triangular_update_f32",
            "triangular_solve_f32",
            "spmv_f32",
        ]
    after = [m._debug_runtime_stats()["transfers"]["device_to_host_bytes"] for m in (left, right)]
    assert after == before and not module._data_payloads
    # An external matrix owns its coefficient storage. A second values port
    # would otherwise promise a relationship the consumer cannot validate.
    failed = ti.aot.Module()
    with pytest.raises(ValueError, match="owns its values"):
        failed.add_execution_plan(
            "ambiguous",
            graph,
            resource_bindings={left: "left", right: "right"},
            source_bindings=dict(coefficients=left),
        )
    assert not failed._execution_plans and not failed._data_payloads
    # A shared plan must expose the matrix identity checked by its operation.
    with pytest.raises(ValueError, match="matrix as a public resource"):
        failed.add_execution_plan("plan", graph, resource_bindings={solve: "solve"})
    block_arrays = []
    for dtype, data in ((ti.i32, [0, 1]), (ti.i32, [0]), (ti.f32, [2, 0, 1, 4])):
        array = ti.ndarray(dtype, len(data))
        array.from_numpy(np.asarray(data, np.int32 if dtype == ti.i32 else np.float32))
        block_arrays.append(array)
    block = ti.linalg.SparsePattern.bsr(1, 1, 2, *block_arrays[:2]).matrix(block_arrays[2])
    block_builder = ti.graph.GraphBuilder()
    block_builder.append_native(ti.hardware.linalg.CusparseSpmvRecording(block, input="x", output="y"))
    block_graph = block_builder.compile()
    block_plan = module.add_execution_plan("bsr", block_graph, resource_bindings={block: "block"})
    assert not json.loads(block_plan.to_json()).get("resources", []) and not module._data_payloads
    artifact = tmp_path / "public-sparse.tcm"
    module.archive(str(artifact))
    graph.close()
    block_graph.close()
    for mode in ("standalone", "borrowed"):
        run = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_sparse_public_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert run.returncode == 0, run.stdout + run.stderr
