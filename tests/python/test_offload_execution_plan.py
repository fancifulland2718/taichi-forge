from dataclasses import dataclass

import pytest

from taichi_forge.lang._offload_execution_plan import (
    _OffloadExecutionPlan,
    _TaskOptimizationSpec,
)


@dataclass(frozen=True)
class _Manifest:
    logical_task_id: str
    task_index: int
    task_type: str


def _manifests():
    return (
        _Manifest("tfl:kernel:stable:0:serial", 0, "serial"),
        _Manifest("tfl:kernel:stable:1:range_for", 1, "range_for"),
        _Manifest("tfl:kernel:stable:2:range_for", 2, "range_for"),
    )


def test_offload_execution_plan_has_complete_deterministic_identity():
    first = _OffloadExecutionPlan.from_task_manifests(_manifests())
    second = _OffloadExecutionPlan.from_task_manifests(_manifests())

    assert first == second
    assert first.is_baseline
    assert first.identity == second.identity
    assert first.compilation_identity == second.compilation_identity
    assert first.recipe_id.startswith("kernel-execution:offload-plan:v1:")
    assert len(first.tasks) == len(_manifests())


def test_launch_only_task_policy_does_not_manufacture_compilation_identity():
    baseline = _OffloadExecutionPlan.from_task_manifests(_manifests())
    launch = baseline.replace_task(
        1,
        grid_residency_waves=2,
        range_work_per_thread_target=4,
    )
    compiled = baseline.replace_task(1, workgroup_size=128, thread_local="off")

    assert launch.identity != baseline.identity
    assert launch.compilation_identity == baseline.compilation_identity
    assert compiled.compilation_identity != baseline.compilation_identity


def test_task_plan_rejects_topology_drift_and_non_range_controls():
    baseline = _OffloadExecutionPlan.from_task_manifests(_manifests())
    with pytest.raises(ValueError, match="only tunes physical range_for"):
        baseline.replace_task(0, workgroup_size=64)
    with pytest.raises(ValueError, match="topology"):
        baseline.validate_topology(_manifests()[:-1])
    with pytest.raises(ValueError, match="logical task identity"):
        _OffloadExecutionPlan(
            "kernel:stable",
            (
                _TaskOptimizationSpec(
                    "tfl:other:0:range_for",
                    0,
                    "range_for",
                ),
            ),
        )


def test_task_plan_requires_complete_contiguous_order():
    tasks = (
        _TaskOptimizationSpec(
            "tfl:kernel:stable:1:range_for",
            1,
            "range_for",
        ),
    )
    with pytest.raises(ValueError, match="contiguous physical ordinal"):
        _OffloadExecutionPlan("kernel:stable", tasks)
