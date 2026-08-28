"""Private bounded Forge kernel-variant materialization."""

from dataclasses import dataclass
from types import MappingProxyType

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._kernel_optimization import (
    _BackendCodegenOptions,
    _IrOptimizationOptions,
    _KernelOptimizationSpec,
    _LaunchOptions,
    _bind_kernel_optimization_spec,
)


_BLOCK_DIMS = (64, 128, 256, 512)
_GRID_RESIDENCY_WAVES = (None, 1, 2, 4)
_MAX_VARIANTS = 32


@dataclass(frozen=True)
class _KernelVariant:
    variant_id: str
    compilation_id: str
    spec: _KernelOptimizationSpec
    logical_task_id: str
    baseline_thread_local_bytes: int
    physical_equivalence_key: tuple


@dataclass(frozen=True)
class _KernelCompilationGroup:
    compilation_id: str
    representative_variant_id: str
    variant_ids: tuple


@dataclass(frozen=True)
class _KernelVariantRejection:
    block_dim: int
    reason: str


class _KernelVariantSession:
    """Materialize a bounded, explicit-only CUDA variant search space."""

    def __init__(self, kernel, args, *, block_dims=_BLOCK_DIMS):
        if impl.current_cfg().arch != _ti_core.Arch.cuda:
            raise RuntimeError("kernel variant sessions require the CUDA backend")
        requested = tuple(block_dims)
        if not requested or any(value not in _BLOCK_DIMS for value in requested):
            raise ValueError(
                "block_dims must be a non-empty subset of (64, 128, 256, 512)"
            )
        if len(set(requested)) != len(requested):
            raise ValueError("block_dims must not contain duplicates")

        self._kernel = kernel
        self._args = tuple(args)
        variants = []
        rejections = []
        for block_dim in requested:
            baseline_spec = self._spec(block_dim, "auto", None)
            baseline_binding = _bind_kernel_optimization_spec(
                kernel, baseline_spec
            )
            try:
                report = baseline_binding.report(*self._args)
                snapshot = baseline_binding._gpu_semantics_snapshot(*self._args)
                task, dimensions = self._eligible_range_task(report, snapshot)
            except (RuntimeError, TypeError, ValueError) as error:
                rejections.append(_KernelVariantRejection(block_dim, str(error)))
                continue

            from taichi_forge.lang._gpu_semantics import _GpuPhysicalEffect
            from taichi_forge.lang._gpu_semantics_tuning import (
                _RESIDENCY_DIMENSION,
                _TLS_DIMENSION,
                _WORKGROUP_DIMENSION,
                _dimension_by_name,
                _gpu_physical_equivalence_key,
            )
            workgroup = _dimension_by_name(dimensions, _WORKGROUP_DIMENSION)
            if block_dim not in workgroup.legal_values:
                rejections.append(
                    _KernelVariantRejection(block_dim, workgroup.status.reason)
                )
                continue
            tls_modes = _dimension_by_name(
                dimensions, _TLS_DIMENSION
            ).legal_values
            residency_values = _dimension_by_name(
                dimensions, _RESIDENCY_DIMENSION
            ).legal_values
            for thread_local in tls_modes:
                for waves in residency_values:
                    spec = self._spec(block_dim, thread_local, waves)
                    selections = {
                        _WORKGROUP_DIMENSION: block_dim,
                        _TLS_DIMENSION: thread_local,
                        _RESIDENCY_DIMENSION: waves,
                    }
                    variants.append(
                        _KernelVariant(
                            variant_id=spec.identity,
                            compilation_id=spec.compilation_identity,
                            spec=spec,
                            logical_task_id=task.logical_task_id,
                            baseline_thread_local_bytes=int(task.thread_local_bytes),
                            physical_equivalence_key=(
                                _gpu_physical_equivalence_key(
                                    dimensions,
                                    selections,
                                    _GpuPhysicalEffect.ARTIFACT,
                                )
                            ),
                        )
                    )

        if len(variants) > _MAX_VARIANTS:
            raise RuntimeError("kernel variant search exceeded its bounded budget")
        self._variants = MappingProxyType(
            {variant.variant_id: variant for variant in variants}
        )
        self._rejections = tuple(rejections)
        members = {}
        for variant in variants:
            group = members.setdefault(
                variant.physical_equivalence_key,
                (variant.compilation_id, []),
            )
            if group[0] != variant.compilation_id:
                raise RuntimeError(
                    "semantic artifact equivalence disagrees with compilation identity"
                )
            group[1].append(variant.variant_id)
        self._compilation_groups = tuple(
            _KernelCompilationGroup(
                compilation_id=compilation_id,
                representative_variant_id=variant_ids[0],
                variant_ids=tuple(variant_ids),
            )
            for compilation_id, variant_ids in members.values()
        )

    @staticmethod
    def _spec(block_dim, thread_local, waves):
        return _KernelOptimizationSpec(
            ir=_IrOptimizationOptions(thread_local=thread_local),
            backend=_BackendCodegenOptions(workgroup_size=block_dim),
            launch=_LaunchOptions(block_mode="require", grid_residency_waves=waves),
        )

    @staticmethod
    def _eligible_range_task(report, snapshot):
        if report.backend != "cuda" or report.status != "applied":
            raise RuntimeError("variant baseline did not apply on CUDA")
        from taichi_forge.lang._gpu_semantics import _GpuAvailability
        from taichi_forge.lang._gpu_semantics_tuning import (
            _WORKGROUP_DIMENSION,
            _derive_gpu_tuning_dimensions,
            _dimension_by_name,
        )

        dimensions = _derive_gpu_tuning_dimensions(
            snapshot,
            max_threads=impl.current_cfg().max_block_dim,
            canonical_workgroup_sizes=_BLOCK_DIMS,
            residency_values=_GRID_RESIDENCY_WAVES,
        )
        workgroup = _dimension_by_name(dimensions, _WORKGROUP_DIMENSION)
        if workgroup.status.availability != _GpuAvailability.PROVEN:
            raise RuntimeError(workgroup.status.reason)
        task_by_id = {task.task_id: task for task in report.tasks}
        range_dispatch = next(
            dispatch
            for dispatch in snapshot.dispatches
            if dispatch.task_kind == "range_for"
        )
        return task_by_id[range_dispatch.physical_dispatch_id], dimensions

    @property
    def rejections(self):
        return self._rejections

    @property
    def compilation_groups(self):
        return self._compilation_groups

    def variant_ids(self):
        return tuple(self._variants)

    def compilation_variant_ids(self):
        return tuple(group.compilation_id for group in self._compilation_groups)

    def variant(self, variant_id):
        try:
            return self._variants[variant_id]
        except KeyError as error:
            raise KeyError(f"unknown Forge kernel variant {variant_id!r}") from error

    def bind(self, variant_id):
        return _bind_kernel_optimization_spec(
            self._kernel, self.variant(variant_id).spec
        )


__all__ = [
    "_KernelCompilationGroup",
    "_KernelVariant",
    "_KernelVariantRejection",
    "_KernelVariantSession",
]
