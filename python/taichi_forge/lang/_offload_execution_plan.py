"""Immutable task-indexed execution plans for one semantic Taichi kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Optional


_TASK_PLAN_SCHEMA = "taichi-forge.offload-execution-plan.v1"
_TASK_PLAN_RECIPE_PREFIX = "kernel-execution:offload-plan:v1:"
_THREAD_LOCAL_MODES = ("auto", "on", "off")
_CUDA_MIN_BLOCKS_PER_SM = (1, 2, 4)
_CUDA_MAX_REGISTERS = (None, 0, 24, 48)
_CUDA_GRID_RESIDENCY_WAVES = (None, 1, 2, 4)
_RANGE_WORK_PER_THREAD_TARGETS = (1, 2, 4, 8)
_MEMORY_STRATEGIES = ("direct", "shared_staged_1d")


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _identity(prefix, payload):
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return prefix + digest


def _validate_workgroup_size(value):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("workgroup_size must be an integer or None")
    if not 1 <= value <= 1024:
        raise ValueError("workgroup_size must be in [1, 1024]")
    if value % 32 != 0 and value & (value - 1):
        raise ValueError("workgroup_size must be a power of two or a multiple of 32")


def _validated_manifest_topology(manifests):
    manifests = tuple(manifests)
    if not manifests:
        raise ValueError("cannot build an execution plan without task manifests")
    semantic_identity = None
    topology = []
    for expected_index, manifest in enumerate(manifests):
        task_index = manifest.task_index
        task_kind = manifest.task_type
        logical_task_id = manifest.logical_task_id
        if task_index != expected_index:
            raise ValueError("task manifests are not in physical ordinal order")
        suffix = f":{task_index}:{task_kind}"
        if not logical_task_id.startswith("tfl:") or not logical_task_id.endswith(
            suffix
        ):
            raise ValueError("task manifest has no stable logical task identity")
        current_semantic_identity = logical_task_id[4 : -len(suffix)]
        if not current_semantic_identity:
            raise ValueError("task manifest has an empty semantic kernel identity")
        if semantic_identity is None:
            semantic_identity = current_semantic_identity
        elif semantic_identity != current_semantic_identity:
            raise ValueError("task manifests do not belong to one semantic kernel")
        topology.append((logical_task_id, task_index, task_kind))
    return manifests, semantic_identity, tuple(topology)


@dataclass(frozen=True)
class _TaskOptimizationSpec:
    """One complete topology-preserving policy for one physical task."""

    logical_task_id: str
    task_index: int
    task_kind: str
    workgroup_size: Optional[int] = None
    thread_local: str = "auto"
    cuda_min_blocks_per_sm: int = 2
    cuda_max_registers: Optional[int] = None
    grid_residency_waves: Optional[int] = None
    range_work_per_thread_target: int = 1
    memory_strategy: str = "direct"
    memory_source_arg_indices: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.logical_task_id, str) or not self.logical_task_id:
            raise ValueError("logical_task_id must be a nonempty string")
        if (
            isinstance(self.task_index, bool)
            or not isinstance(self.task_index, int)
            or self.task_index < 0
        ):
            raise ValueError("task_index must be a nonnegative integer")
        if not isinstance(self.task_kind, str) or not self.task_kind:
            raise ValueError("task_kind must be a nonempty string")
        _validate_workgroup_size(self.workgroup_size)
        if self.thread_local not in _THREAD_LOCAL_MODES:
            raise ValueError("thread_local must be 'auto', 'on', or 'off'")
        if self.cuda_min_blocks_per_sm not in _CUDA_MIN_BLOCKS_PER_SM:
            raise ValueError("cuda_min_blocks_per_sm must be 1, 2, or 4")
        if self.cuda_max_registers not in _CUDA_MAX_REGISTERS:
            raise ValueError("cuda_max_registers must be None, 0, 24, or 48 in v1")
        if self.grid_residency_waves not in _CUDA_GRID_RESIDENCY_WAVES:
            raise ValueError("grid_residency_waves must be None, 1, 2, or 4")
        if self.range_work_per_thread_target not in (_RANGE_WORK_PER_THREAD_TARGETS):
            raise ValueError("range_work_per_thread_target must be 1, 2, 4, or 8")
        if self.memory_strategy not in _MEMORY_STRATEGIES:
            raise ValueError("memory_strategy must be 'direct' or 'shared_staged_1d'")
        memory_source_arg_indices = tuple(self.memory_source_arg_indices)
        if (
            any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in memory_source_arg_indices
            )
            or tuple(sorted(set(memory_source_arg_indices)))
            != memory_source_arg_indices
        ):
            raise ValueError(
                "memory_source_arg_indices must be sorted unique nonnegative integers"
            )
        object.__setattr__(
            self,
            "memory_source_arg_indices",
            memory_source_arg_indices,
        )
        if self.memory_strategy == "direct" and memory_source_arg_indices:
            raise ValueError("memory_source_arg_indices require shared_staged_1d")
        if self.memory_strategy == "shared_staged_1d" and (
            self.task_kind != "range_for"
            or self.workgroup_size is None
            or self.grid_residency_waves is not None
            or self.range_work_per_thread_target != 1
        ):
            raise ValueError(
                "shared_staged_1d requires a range_for task, an exact "
                "workgroup_size, automatic grid residency, and one item per thread"
            )
        if self.task_kind != "range_for" and (
            self.workgroup_size is not None
            or self.thread_local != "auto"
            or self.cuda_min_blocks_per_sm != 2
            or self.cuda_max_registers is not None
            or self.grid_residency_waves is not None
            or self.range_work_per_thread_target != 1
            or self.memory_strategy != "direct"
        ):
            raise ValueError(
                "the v1 execution plan only tunes physical range_for tasks"
            )

    @property
    def is_baseline(self):
        return (
            self.workgroup_size is None
            and self.thread_local == "auto"
            and self.cuda_min_blocks_per_sm == 2
            and self.cuda_max_registers is None
            and self.grid_residency_waves is None
            and self.range_work_per_thread_target == 1
            and self.memory_strategy == "direct"
            and not self.memory_source_arg_indices
        )

    @property
    def compilation_payload(self):
        payload = asdict(self)
        payload["grid_residency_waves"] = None
        payload["range_work_per_thread_target"] = 1
        return payload


@dataclass(frozen=True)
class _OffloadExecutionPlan:
    """A complete opaque-recipe candidate for one semantic kernel.

    The plan is intentionally a full ordered task topology.  CompileIQ sees
    only :attr:`recipe_id`; Forge retains every internal policy and validates
    it again after materialization.
    """

    semantic_kernel_identity: str
    tasks: tuple
    fusion_groups: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.semantic_kernel_identity, str)
            or not self.semantic_kernel_identity
        ):
            raise ValueError("semantic_kernel_identity must be a nonempty string")
        tasks = tuple(self.tasks)
        object.__setattr__(self, "tasks", tasks)
        if not tasks:
            raise ValueError("an offload execution plan must contain every task")
        if any(not isinstance(task, _TaskOptimizationSpec) for task in tasks):
            raise TypeError("tasks must contain _TaskOptimizationSpec values")
        logical_ids = set()
        for expected_index, task in enumerate(tasks):
            if task.task_index != expected_index:
                raise ValueError(
                    "task specs must be ordered by contiguous physical ordinal"
                )
            expected_id = (
                f"tfl:{self.semantic_kernel_identity}:{expected_index}:"
                f"{task.task_kind}"
            )
            if task.logical_task_id != expected_id:
                raise ValueError(
                    "logical task identity does not match kernel, ordinal, and kind"
                )
            if task.logical_task_id in logical_ids:
                raise ValueError("logical task identities must be unique")
            logical_ids.add(task.logical_task_id)

        fusion_groups = tuple(tuple(group) for group in self.fusion_groups)
        object.__setattr__(self, "fusion_groups", fusion_groups)
        occupied = set()
        previous_start = -1
        for group in fusion_groups:
            if not 2 <= len(group) <= 4:
                raise ValueError("offload fusion groups must contain two to four tasks")
            if any(
                isinstance(index, bool) or not isinstance(index, int) for index in group
            ):
                raise TypeError("offload fusion task indices must be integers")
            if tuple(range(group[0], group[0] + len(group))) != group:
                raise ValueError("offload fusion groups must be contiguous and ordered")
            if group[0] < 0:
                raise IndexError(
                    "offload fusion task index is outside this execution plan"
                )
            if group[0] <= previous_start or occupied.intersection(group):
                raise ValueError("offload fusion groups must be ordered and disjoint")
            if group[-1] >= len(tasks):
                raise IndexError(
                    "offload fusion task index is outside this execution plan"
                )
            selected = tuple(tasks[index] for index in group)
            if any(task.task_kind != "range_for" for task in selected):
                raise ValueError(
                    "offload fusion supports only consecutive range_for tasks"
                )
            if any(not task.is_baseline for task in selected):
                raise ValueError(
                    "offload fusion cannot be combined with per-task fixed-axis controls"
                )
            occupied.update(group)
            previous_start = group[0]

        stable_payload = {
            "schema": _TASK_PLAN_SCHEMA,
            "semantic_kernel_identity": self.semantic_kernel_identity,
            "tasks": tuple(asdict(task) for task in tasks),
        }
        compilation_payload = {
            "schema": _TASK_PLAN_SCHEMA,
            "semantic_kernel_identity": self.semantic_kernel_identity,
            "tasks": tuple(task.compilation_payload for task in tasks),
        }
        if fusion_groups:
            topology = {
                "operation": "fuse_exact_pointwise_range_tasks",
                "source_task_groups": fusion_groups,
            }
            stable_payload["topology_transform"] = topology
            compilation_payload["topology_transform"] = topology
        # The plan and every task are frozen, so all derived launch identities
        # and topology facts are immutable for the plan's lifetime.
        object.__setattr__(
            self,
            "_is_baseline",
            not fusion_groups and all(task.is_baseline for task in tasks),
        )
        object.__setattr__(
            self,
            "_requires_graph_memory",
            any(task.memory_strategy != "direct" for task in tasks),
        )
        object.__setattr__(
            self,
            "_topology_signature",
            tuple(
                (task.logical_task_id, task.task_index, task.task_kind)
                for task in tasks
            ),
        )
        materialized_tasks = []
        materialized_lineage = []
        group_by_start = {group[0]: group for group in fusion_groups}
        grouped_members = {index for group in fusion_groups for index in group[1:]}
        for source_index, task in enumerate(tasks):
            if source_index in grouped_members:
                continue
            group = group_by_start.get(source_index, (source_index,))
            physical_index = len(materialized_tasks)
            materialized_tasks.append(
                replace(
                    task,
                    logical_task_id=(
                        f"tfl:{self.semantic_kernel_identity}:{physical_index}:"
                        f"{task.task_kind}"
                    ),
                    task_index=physical_index,
                )
            )
            materialized_lineage.append(
                tuple(tasks[index].logical_task_id for index in group)
            )
        object.__setattr__(self, "_materialized_tasks", tuple(materialized_tasks))
        object.__setattr__(
            self, "_materialized_task_lineage", tuple(materialized_lineage)
        )
        object.__setattr__(self, "_identity", _identity("oep1:", stable_payload))
        object.__setattr__(
            self,
            "_compilation_identity",
            _identity("oep1c:", compilation_payload),
        )

    @classmethod
    def from_task_manifests(cls, manifests):
        _, semantic_identity, topology = _validated_manifest_topology(manifests)
        return cls(
            semantic_identity,
            tuple(
                _TaskOptimizationSpec(
                    logical_task_id=logical_task_id,
                    task_index=task_index,
                    task_kind=task_kind,
                )
                for logical_task_id, task_index, task_kind in topology
            ),
        )

    @property
    def is_baseline(self):
        return self._is_baseline

    @property
    def requires_graph_memory(self):
        return self._requires_graph_memory

    @property
    def stable_payload(self):
        payload = {
            "schema": _TASK_PLAN_SCHEMA,
            "semantic_kernel_identity": self.semantic_kernel_identity,
            "tasks": tuple(asdict(task) for task in self.tasks),
        }
        if self.fusion_groups:
            payload["topology_transform"] = {
                "operation": "fuse_exact_pointwise_range_tasks",
                "source_task_groups": self.fusion_groups,
            }
        return payload

    @property
    def compilation_payload(self):
        payload = {
            "schema": _TASK_PLAN_SCHEMA,
            "semantic_kernel_identity": self.semantic_kernel_identity,
            "tasks": tuple(task.compilation_payload for task in self.tasks),
        }
        if self.fusion_groups:
            payload["topology_transform"] = {
                "operation": "fuse_exact_pointwise_range_tasks",
                "source_task_groups": self.fusion_groups,
            }
        return payload

    @property
    def materialized_tasks(self):
        return self._materialized_tasks

    @property
    def materialized_task_lineage(self):
        return self._materialized_task_lineage

    @property
    def identity(self):
        return self._identity

    @property
    def compilation_identity(self):
        return self._compilation_identity

    @property
    def recipe_id(self):
        return _TASK_PLAN_RECIPE_PREFIX + self.identity.removeprefix("oep1:")

    def replace_task(self, task_index, **changes):
        if (
            isinstance(task_index, bool)
            or not isinstance(task_index, int)
            or not 0 <= task_index < len(self.tasks)
        ):
            raise IndexError("task_index is outside this execution plan")
        tasks = list(self.tasks)
        tasks[task_index] = replace(tasks[task_index], **changes)
        return type(self)(
            self.semantic_kernel_identity,
            tuple(tasks),
            fusion_groups=self.fusion_groups,
        )

    def with_fused_task_groups(self, *groups):
        if self.fusion_groups:
            raise ValueError(
                "offload execution plan already contains a topology transform"
            )
        return type(self)(
            self.semantic_kernel_identity,
            self.tasks,
            fusion_groups=tuple(groups),
        )

    def validate_topology(self, manifests):
        _, semantic_identity, topology = _validated_manifest_topology(manifests)
        expected = tuple(
            (task.logical_task_id, task.task_index, task.task_kind)
            for task in self.materialized_tasks
        )
        if semantic_identity != self.semantic_kernel_identity or topology != expected:
            raise ValueError("materialized offload topology does not match the plan")
        return True

    def validate_materialization(self, manifests):
        manifests = tuple(manifests)
        self.validate_topology(manifests)
        if len(manifests) != len(self.materialized_tasks):
            raise ValueError("materialized task count does not match the plan")
        compilation_identity = self.compilation_identity
        for spec, manifest in zip(self.materialized_tasks, manifests):
            if manifest.optimization_spec_id != compilation_identity:
                raise ValueError(
                    "materialized compilation identity does not match the plan"
                )
            if (
                spec.workgroup_size is not None
                and manifest.selected_block_size != spec.workgroup_size
            ):
                raise ValueError(
                    f"task {spec.task_index} did not materialize workgroup_size"
                )
            expected_max_registers = spec.cuda_max_registers
            if (
                manifest.requested_thread_local_mode != spec.thread_local
                or manifest.requested_cuda_min_blocks_per_sm
                != spec.cuda_min_blocks_per_sm
                or manifest.requested_cuda_max_registers != expected_max_registers
                or manifest.requested_grid_residency_waves != spec.grid_residency_waves
                or manifest.requested_range_work_per_thread_target
                != spec.range_work_per_thread_target
                or manifest.requested_memory_strategy != spec.memory_strategy
            ):
                raise ValueError(
                    f"task {spec.task_index} materialized different controls"
                )
            if (
                spec.memory_source_arg_indices
                and tuple(manifest.staged_external_arg_indices)
                != spec.memory_source_arg_indices
            ):
                raise ValueError(
                    f"task {spec.task_index} materialized different staged sources"
                )
        return True

    @property
    def native_arguments(self):
        """Return exact parallel vectors for the private C++ boundary."""

        return (
            self.compilation_identity,
            self.identity,
            [task.task_index for task in self.tasks],
            [task.task_kind for task in self.tasks],
            [
                0 if task.workgroup_size is None else task.workgroup_size
                for task in self.tasks
            ],
            [task.thread_local for task in self.tasks],
            [task.cuda_min_blocks_per_sm for task in self.tasks],
            [
                -1 if task.cuda_max_registers is None else task.cuda_max_registers
                for task in self.tasks
            ],
            [
                0 if task.grid_residency_waves is None else task.grid_residency_waves
                for task in self.tasks
            ],
            [task.range_work_per_thread_target for task in self.tasks],
            [task.memory_strategy for task in self.tasks],
            [list(task.memory_source_arg_indices) for task in self.tasks],
            [list(group) for group in self.fusion_groups],
        )


def _bind_offload_execution_plan(kernel, plan):
    if not isinstance(plan, _OffloadExecutionPlan):
        raise TypeError("plan must be an _OffloadExecutionPlan")
    from taichi_forge.lang.kernel_impl import _OffloadExecutionPlanBinding
    from taichi_forge.lang.task_launch import TaskLaunchPolicy

    view = kernel.with_launch_policy(TaskLaunchPolicy.auto())
    return _OffloadExecutionPlanBinding(view._kernel, plan, bound_args=view._bound_args)


__all__ = [
    "_OffloadExecutionPlan",
    "_TaskOptimizationSpec",
    "_bind_offload_execution_plan",
]
