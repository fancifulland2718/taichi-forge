"""Public triangular resources retain checked matrix and dense-port identity."""

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
def test_public_triangular_plan_contract(tmp_path):
    if not ti.hardware.linalg.cusparse_spsv_is_available() or not ti.hardware.linalg.cusparse_spsm_is_available():
        pytest.skip("External cuSPARSE triangular solves required")
    rows, columns, values = ti.ndarray(ti.i32, 4), ti.ndarray(ti.i32, 9), ti.ndarray(ti.f32, 9)
    rows.from_numpy(np.array([0, 3, 6, 9], np.int32))
    columns.from_numpy(np.tile(np.arange(3, dtype=np.int32), 3))
    values.from_numpy(np.array([2, 0.2, -0.1, 1, 4, 0.3, 0.5, -1, 3], np.float32))
    matrix = ti.linalg.SparsePattern.csr(3, 3, rows, columns).matrix(values)
    module = ti.aot.Module()
    for rhs, fill, unit, transpose in ((1, "lower", False, False), (2, "upper", True, True)):
        args = dict(fill_mode=fill, unit_diagonal=unit, transpose=transpose, input="x", output="y")
        recording = (
            ti.hardware.linalg.CusparseSpsvRecording(matrix, **args)
            if rhs == 1
            else ti.hardware.linalg.CusparseSpsmRecording(matrix, rhs, **args)
        )
        builder = ti.graph.GraphBuilder()
        builder.append_native(recording)
        graph = builder.compile()
        exported = module.add_execution_plan(str(rhs), graph, resource_bindings={matrix: "matrix", recording: "solve"})
        doc = json.loads(exported.to_json())
        assert not doc.get("resources") and not module._data_payloads
        assert [op["version"] for op in doc["operations"]] == [2, 2]
        assert all(set(op["bindings"]) == {"plan", "matrix", "input", "output"} for op in doc["operations"])
        assert all(op["parameters"]["mutable_values"] for op in doc["operations"])
        graph.close()
    artifact = tmp_path / "public-plans.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_triangular_public_plan_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
