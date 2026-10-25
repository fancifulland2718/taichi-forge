"""Lazy, builder-scoped compiler facts shared by dispatch recipe families."""

from dataclasses import dataclass
import threading

from taichi_forge.lang._range_dispatch import device_range_dispatch_rejection


@dataclass(frozen=True)
class _TaskFacts:
    manifests: tuple
    fusible_ranges: bool
    bounded_sparse_parent: bool
    device_range: bool


class GraphKernelTaskFacts:
    """One compiled kernel's facts, without occurrence bindings or a global cache.

    Sources retain this entry rather than its builder's whole cache. Failed
    queries publish nothing, so later discovery can retry normally.
    """

    def __init__(self, kernel_cpp):
        self._kernel_cpp = kernel_cpp
        self._snapshot = None
        self._lock = threading.Lock()

    @property
    def snapshot(self):
        if self._snapshot is not None:
            return self._snapshot
        with self._lock:
            if self._snapshot is None:
                from taichi_forge.graph._graph import _kernel_task_manifests

                tasks = _kernel_task_manifests(self._kernel_cpp)
                listgens = tuple(task for task in tasks if task.task_type == "listgen")
                bounded_sparse_parent = bool(listgens) and not any(
                    task.may_mutate_sparse_topology for task in tasks
                ) and all(
                    task.sparse_list_op == 2
                    and task.sparse_list_snode_id >= 0
                    and task.sparse_list_parent_snode_id >= 0
                    and task.sparse_list_parent_grid_bound is not None
                    and task.sparse_list_parent_grid_bound > 0
                    and task.selected_grid_size is not None
                    and task.selected_grid_size > 0
                    for task in listgens
                ) and any(
                    task.sparse_list_parent_grid_bound < task.selected_grid_size
                    for task in listgens
                )
                self._snapshot = _TaskFacts(
                    manifests=tasks,
                    fusible_ranges=any(
                        first.task_type == second.task_type == "range_for"
                        for first, second in zip(tasks, tasks[1:])
                    ),
                    bounded_sparse_parent=bounded_sparse_parent,
                    device_range=any(device_range_dispatch_rejection(task) is None for task in tasks),
                )
        return self._snapshot
