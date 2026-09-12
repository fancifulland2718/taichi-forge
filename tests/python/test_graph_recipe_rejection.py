import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.lang._offload_execution_plan import (
    _OffloadExecutionPlan,
    _bind_offload_execution_plan,
)
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_expected_fusion_rejection_keeps_search_success_and_unexpected_error_logs():
    # Native Windows logging owns OS handles created at module import; pytest
    # capfd cannot reliably redirect those handles. Capture a fresh process
    # from startup, using the same source/native pair as this test invocation.
    bootstrap = f"""
import importlib.abc
import importlib.util
import runpy
import sys
sys.path[:] = {sys.path!r}
class NativePair(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'taichi_forge._lib.core.taichi_python':
            return importlib.util.spec_from_file_location(fullname, {core.__file__!r})
sys.meta_path.insert(0, NativePair())
runpy.run_path(sys.argv[1], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, str(Path(__file__).resolve())],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    logs = result.stdout + result.stderr
    assert result.returncode == 0, logs
    assert "EXPECTED_SEARCH_COMPLETED" in logs
    assert "offload phase fusion rejected" not in logs
    assert "UNEXPECTED_FAILURE_PRESERVED" in logs
    assert "offload execution plan topology mismatch" in logs


def _exercise_rejection_and_unexpected_failure():
    count = 67
    data = ti.field(ti.i32, shape=count)
    output = ti.field(ti.i32, shape=count)

    @ti.kernel
    def phases():
        for i in range(count):
            data[i] = i + 1
        for i in range(count):
            output[i] = data[(i + 1) % count] * 3

    builder = ti.graph.GraphBuilder()
    builder.dispatch(phases)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    explanation = next(
        item["provider_explanation"]
        for item in catalog.discovery_report()["providers"]
        if item["provider_namespace"] == "taichi_forge.graph.offload_phase_fusion"
    )
    reason = explanation["sources"][0]["generation_rejections"][0]
    assert "non-pointwise dense field access" in reason["reason"]
    assert reason["category"] == "fusion_legality_rejection"
    assert reason["source_task_indices"] == [0, 1]
    assert len(catalog.entries()) == 1

    session = definition.search_recipes(
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("physical_tasks", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=2),
    )

    def evaluate(graph, _recipe):
        graph.run(graph.bind({}))
        np.testing.assert_array_equal(
            output.to_numpy(), ((np.arange(count, dtype=np.int32) + 1) % count + 1) * 3
        )
        return {"physical_tasks": float(len(graph.task_manifest()))}

    decision = session.run(evaluate)
    assert decision.status == "selected"
    print("EXPECTED_SEARCH_COMPLETED", flush=True)

    # A malformed plan is an unexpected compiler contract failure. Its native
    # ERROR and exception must survive the change to expected candidate logs.
    baseline = definition._runtime_spec._graph_offload_fusion_sources[0].baseline_plan
    malformed = _OffloadExecutionPlan(
        baseline.semantic_kernel_identity, baseline.tasks[:-1]
    )
    with pytest.raises(RuntimeError, match="topology mismatch"):
        _bind_offload_execution_plan(phases, malformed).report()
    print("UNEXPECTED_FAILURE_PRESERVED", flush=True)


if __name__ == "__main__":
    ti.init(arch=ti.cuda, offline_cache=False, enable_fallback=False)
    _exercise_rejection_and_unexpected_failure()
