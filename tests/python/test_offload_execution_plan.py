from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from taichi_forge.lang import _offload_execution_plan as offload_execution_plan
from taichi_forge.lang import kernel_impl
from taichi_forge.lang._offload_execution_plan import (
    _OffloadExecutionPlan,
    _TaskOptimizationSpec,
)
from taichi_forge.lang.enums import AutodiffMode


@dataclass(frozen=True)
class _Manifest:
    logical_task_id: str
    task_index: int
    task_type: str


@dataclass(frozen=True)
class _MaterializedManifest:
    logical_task_id: str
    task_index: int
    task_type: str
    optimization_spec_id: str
    selected_block_size: object
    requested_thread_local_mode: str
    requested_cuda_min_blocks_per_sm: int
    requested_cuda_max_registers: object
    requested_grid_residency_waves: object
    requested_range_work_per_thread_target: int
    requested_memory_strategy: str


def _manifests():
    return (
        _Manifest("tfl:kernel:stable:0:serial", 0, "serial"),
        _Manifest("tfl:kernel:stable:1:range_for", 1, "range_for"),
        _Manifest("tfl:kernel:stable:2:range_for", 2, "range_for"),
    )


def _materialized_manifests(plan):
    return tuple(
        _MaterializedManifest(
            logical_task_id=task.logical_task_id,
            task_index=task.task_index,
            task_type=task.task_kind,
            optimization_spec_id=plan.compilation_identity,
            selected_block_size=task.workgroup_size,
            requested_thread_local_mode=task.thread_local,
            requested_cuda_min_blocks_per_sm=task.cuda_min_blocks_per_sm,
            requested_cuda_max_registers=task.cuda_max_registers,
            requested_grid_residency_waves=task.grid_residency_waves,
            requested_range_work_per_thread_target=(task.range_work_per_thread_target),
            requested_memory_strategy=task.memory_strategy,
        )
        for task in plan.tasks
    )


def _fake_offload_kernel(plan):
    counters = {
        "ensure": 0,
        "validate": 0,
        "raw_launch": 0,
        "retained_launch": 0,
        "retained_build": 0,
    }
    registered = set()

    def fake_kernel():
        pass

    runtime = SimpleNamespace(
        target_tape=None,
        fwd_mode_manager=None,
        grad_replaced=False,
    )
    kernel = SimpleNamespace(
        func=fake_kernel,
        autodiff_mode=AutodiffMode.NONE,
        runtime=runtime,
        compiled_kernels={},
        mapper=SimpleNamespace(lookup=lambda args: (int(args[0]), ())),
        materialized_manifests=_materialized_manifests(plan),
    )

    def specialization_key(args):
        return (
            kernel.func,
            kernel.mapper.lookup(args)[0],
            kernel.autodiff_mode,
            ("offload_execution_plan", plan.identity),
            False,
        )

    def ensure(candidate, *args):
        assert candidate is plan
        counters["ensure"] += 1
        key = specialization_key(args)
        kernel.compiled_kernels.setdefault(key, object())
        return key

    def validate(key, candidate):
        assert candidate is plan
        counters["validate"] += 1
        candidate.validate_materialization(kernel.materialized_manifests)
        return kernel.materialized_manifests

    def raw_launch(kernel_cpp, *args):
        counters["raw_launch"] += 1
        registered.add(kernel_cpp)
        return ("raw", tuple(args))

    def build_retained(
        key,
        args,
        *,
        allow_snode_tree_dependencies,
        reuse_gpu_context,
    ):
        assert allow_snode_tree_dependencies
        assert not reuse_gpu_context
        counters["retained_build"] += 1
        kernel_cpp = kernel.compiled_kernels[key]
        if kernel_cpp not in registered:
            return None, False
        return (
            SimpleNamespace(matches=lambda active_runtime, values: True),
            False,
        )

    def retained_launch(retained, args):
        assert retained.matches(kernel.runtime, args)
        counters["retained_launch"] += 1
        return ("retained", tuple(args))

    kernel._task_launch_backend_kind = lambda: ("cuda", "native")
    kernel._ensure_compiled_with_offload_execution_plan = ensure
    kernel._validate_offload_execution_plan_specialization = validate
    kernel.launch_kernel = raw_launch
    kernel._build_ordinary_launch_plan = build_retained
    kernel._launch_with_ordinary_plan = retained_launch
    return kernel, counters, registered


def test_offload_execution_plan_has_complete_deterministic_identity():
    first = _OffloadExecutionPlan.from_task_manifests(_manifests())
    second = _OffloadExecutionPlan.from_task_manifests(_manifests())

    assert first == second
    assert first.is_baseline
    assert first.identity == second.identity
    assert first.compilation_identity == second.compilation_identity
    assert first.recipe_id.startswith("kernel-execution:offload-plan:v1:")
    assert len(first.tasks) == len(_manifests())


def test_offload_execution_plan_caches_identity_and_validates_cached_topology(
    monkeypatch,
):
    plan = _OffloadExecutionPlan.from_task_manifests(_manifests())
    expected = (plan.identity, plan.compilation_identity)
    materialized = _materialized_manifests(plan)

    def unexpected_identity(*_args, **_kwargs):
        raise AssertionError("immutable offload identity was recomputed")

    monkeypatch.setattr(offload_execution_plan, "_identity", unexpected_identity)
    for _ in range(100):
        assert (plan.identity, plan.compilation_identity) == expected
        assert plan.validate_materialization(materialized)

    malformed = list(materialized)
    malformed[1] = replace(malformed[1], optimization_spec_id="oep1c:wrong")
    with pytest.raises(ValueError, match="compilation identity"):
        plan.validate_materialization(malformed)


def test_offload_binding_validates_once_and_reuses_retained_launch_plan(
    monkeypatch,
):
    plan = _OffloadExecutionPlan.from_task_manifests(_manifests())
    kernel, counters, registered = _fake_offload_kernel(plan)
    monkeypatch.setattr(
        kernel_impl,
        "_process_args",
        lambda unused_kernel, args, kwargs: tuple(args),
    )
    binding = kernel_impl._OffloadExecutionPlanBinding(kernel, plan)

    assert binding(0) == ("raw", (0,))
    assert binding(0) == ("retained", (0,))
    assert counters["ensure"] == 1
    assert counters["validate"] == 1
    assert counters["raw_launch"] == 1
    assert counters["retained_launch"] == 1

    # A different mapper specialization must compile and validate once.
    assert binding(1) == ("raw", (1,))
    assert binding(1) == ("retained", (1,))
    assert counters["ensure"] == 2
    assert counters["validate"] == 2

    # Runtime replacement invalidates both the validation certificate and the
    # retained native plan even when the mapper specialization is unchanged.
    kernel.runtime = SimpleNamespace(
        target_tape=None,
        fwd_mode_manager=None,
        grad_replaced=False,
    )
    kernel.compiled_kernels = {}
    registered.clear()
    assert binding(1) == ("raw", (1,))
    assert counters["ensure"] == 3
    assert counters["validate"] == 3


def test_offload_binding_rejects_malformed_manifest_before_launch(monkeypatch):
    plan = _OffloadExecutionPlan.from_task_manifests(_manifests())
    kernel, counters, _ = _fake_offload_kernel(plan)
    malformed = list(kernel.materialized_manifests)
    malformed[0] = replace(malformed[0], optimization_spec_id="oep1c:wrong")
    kernel.materialized_manifests = tuple(malformed)
    monkeypatch.setattr(
        kernel_impl,
        "_process_args",
        lambda unused_kernel, args, kwargs: tuple(args),
    )
    binding = kernel_impl._OffloadExecutionPlanBinding(kernel, plan)

    with pytest.raises(ValueError, match="compilation identity"):
        binding(0)
    assert counters["validate"] == 1
    assert counters["raw_launch"] == 0


def test_launch_only_task_policy_does_not_manufacture_compilation_identity():
    baseline = _OffloadExecutionPlan.from_task_manifests(_manifests())
    launch = baseline.replace_task(
        1,
        grid_residency_waves=2,
        range_work_per_thread_target=4,
    )
    compiled = baseline.replace_task(1, workgroup_size=128, thread_local="off")
    shared_staged = baseline.replace_task(
        1,
        workgroup_size=128,
        memory_strategy="shared_staged_1d",
    )

    assert launch.identity != baseline.identity
    assert launch.compilation_identity == baseline.compilation_identity
    assert compiled.compilation_identity != baseline.compilation_identity
    assert shared_staged.compilation_identity != baseline.compilation_identity
    assert shared_staged.compilation_identity != compiled.compilation_identity
    assert shared_staged.requires_graph_memory
    assert not baseline.requires_graph_memory


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
