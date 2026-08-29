"""Private bounded Forge kernel-variant materialization."""

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang._kernel_optimization import (
    _ArtifactOptions,
    _BackendCodegenOptions,
    _IrOptimizationOptions,
    _KernelOptimizationSpec,
    _LaunchOptions,
    _bind_kernel_optimization_spec,
)


_BLOCK_DIMS = (64, 128, 256, 512)
_GRID_RESIDENCY_WAVES = (None, 1, 2, 4)
_RANGE_WORK_PER_THREAD_TARGETS = (1, 2, 4, 8)
_COMPILE_TIERS = ("inherit", "full")
_DEFAULT_COMPILE_TIERS = ("inherit",)
_CUDA_MIN_BLOCKS_PER_SM = (1, 2, 4)
_CUDA_MAX_REGISTERS = (None, 24, 48)
_MAX_VARIANTS = 512


@dataclass(frozen=True)
class _KernelVariant:
    variant_id: str
    compilation_id: str
    spec: _KernelOptimizationSpec
    logical_task_id: str
    logical_task_ids: tuple
    baseline_thread_local_bytes: int
    physical_equivalence_key: tuple
    selections: tuple
    resource_envelope: object
    tiling_recipe_id: str


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

    def __init__(
        self,
        kernel,
        args,
        *,
        block_dims=_BLOCK_DIMS,
        range_work_per_thread_targets=_RANGE_WORK_PER_THREAD_TARGETS,
        compile_tiers=_DEFAULT_COMPILE_TIERS,
        cuda_min_blocks_per_sm=_CUDA_MIN_BLOCKS_PER_SM,
        cuda_max_registers=_CUDA_MAX_REGISTERS,
        structural_mode="one_axis",
    ):
        if impl.current_cfg().arch != _ti_core.Arch.cuda:
            raise RuntimeError("kernel variant sessions require the CUDA backend")
        if structural_mode not in ("one_axis", "cartesian"):
            raise ValueError("structural_mode must be 'one_axis' or 'cartesian'")
        requested = tuple(block_dims)
        if not requested or any(value not in _BLOCK_DIMS for value in requested):
            raise ValueError(
                "block_dims must be a non-empty subset of (64, 128, 256, 512)"
            )
        if len(set(requested)) != len(requested):
            raise ValueError("block_dims must not contain duplicates")
        requested_work = tuple(range_work_per_thread_targets)
        if (
            not requested_work
            or requested_work[0] != 1
            or any(
                value not in _RANGE_WORK_PER_THREAD_TARGETS
                for value in requested_work
            )
        ):
            raise ValueError(
                "range_work_per_thread_targets must start with 1 and be a "
                "non-empty subset of (1, 2, 4, 8)"
            )
        if len(set(requested_work)) != len(requested_work):
            raise ValueError(
                "range_work_per_thread_targets must not contain duplicates"
            )
        requested_tiers = tuple(compile_tiers)
        if (
            not requested_tiers
            or requested_tiers[0] != "inherit"
            or any(value not in _COMPILE_TIERS for value in requested_tiers)
        ):
            raise ValueError(
                "compile_tiers must start with 'inherit' and be a non-empty "
                "subset of ('inherit', 'full')"
            )
        requested_min_blocks = tuple(cuda_min_blocks_per_sm)
        if (
            not requested_min_blocks
            or 2 not in requested_min_blocks
            or any(
                value not in _CUDA_MIN_BLOCKS_PER_SM for value in requested_min_blocks
            )
        ):
            raise ValueError(
                "cuda_min_blocks_per_sm must contain 2 and be a non-empty "
                "subset of (1, 2, 4)"
            )
        requested_registers = tuple(cuda_max_registers)
        if (
            not requested_registers
            or requested_registers[0] is not None
            or any(value not in _CUDA_MAX_REGISTERS for value in requested_registers)
        ):
            raise ValueError(
                "cuda_max_registers must start with None and be a non-empty "
                "subset of (None, 24, 48)"
            )
        for name, values in (
            ("compile_tiers", requested_tiers),
            ("cuda_min_blocks_per_sm", requested_min_blocks),
            ("cuda_max_registers", requested_registers),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        if impl.current_cfg().compile_tier == "full":
            requested_tiers = ("inherit",)
        if structural_mode == "cartesian" and len(requested) != 1:
            raise ValueError(
                "cartesian structural refinement requires exactly one block_dim"
            )

        self._kernel = kernel
        self._args = tuple(args)
        self._block_dims = requested
        self._range_work_per_thread_targets = requested_work
        self._compile_tiers = requested_tiers
        self._cuda_min_blocks_per_sm = requested_min_blocks
        self._cuda_max_registers = requested_registers
        self._structural_mode = structural_mode
        self._kernel_wide = False
        kernel_wide_baseline = self._kernel_wide_spec(
            "auto", "inherit", 2, None
        )
        kernel_wide_binding = _bind_kernel_optimization_spec(
            kernel, kernel_wide_baseline
        )
        try:
            kernel_wide_report = kernel_wide_binding.report(*self._args)
            kernel_wide_snapshot = kernel_wide_binding._gpu_semantics_snapshot(
                *self._args
            )
        except (RuntimeError, TypeError, ValueError):
            kernel_wide_report = None
            kernel_wide_snapshot = None
        if kernel_wide_snapshot is not None:
            range_dispatches = tuple(
                dispatch
                for dispatch in kernel_wide_snapshot.dispatches
                if dispatch.task_kind == "range_for"
            )
            if len(range_dispatches) != 1:
                self._initialize_kernel_wide(
                    kernel_wide_report,
                    kernel_wide_snapshot,
                    compile_tiers=requested_tiers,
                    cuda_min_blocks_per_sm=requested_min_blocks,
                    cuda_max_registers=requested_registers,
                    structural_mode=structural_mode,
                )
                return
        variants = []
        rejections = []
        tiling_recipes = {}
        dimension_contract = None
        for block_dim in requested:
            baseline_spec = self._spec(
                block_dim,
                "auto",
                1,
                None,
                "inherit",
                2,
                None,
            )
            baseline_binding = _bind_kernel_optimization_spec(
                kernel, baseline_spec
            )
            try:
                report = baseline_binding.report(*self._args)
                snapshot = baseline_binding._gpu_semantics_snapshot(*self._args)
                task, dimensions, resource_envelope = self._eligible_range_task(
                    report,
                    snapshot,
                    compile_tiers=requested_tiers,
                    cuda_min_blocks_per_sm=requested_min_blocks,
                    cuda_max_registers=requested_registers,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                rejections.append(_KernelVariantRejection(block_dim, str(error)))
                continue

            if dimension_contract is None:
                dimension_contract = dimensions
            elif dimension_contract != dimensions:
                raise RuntimeError(
                    "GPU tuning dimension contract changed across block candidates"
                )

            from taichi_forge.lang._gpu_semantics import (
                _GpuAvailability,
                _GpuPhysicalEffect,
                _GpuTileStrategy,
            )
            from taichi_forge.lang._gpu_semantics_tuning import (
                _COMPILE_TIER_DIMENSION,
                _CUDA_MAX_REGISTERS_DIMENSION,
                _CUDA_MIN_BLOCKS_DIMENSION,
                _RESIDENCY_DIMENSION,
                _RANGE_WORK_PER_THREAD_DIMENSION,
                _TLS_DIMENSION,
                _WORKGROUP_DIMENSION,
                _derive_gpu_tiling_recipes,
                _dimension_by_name,
                _gpu_physical_equivalence_key,
            )
            block_tiling_recipes = _derive_gpu_tiling_recipes(
                snapshot, dimensions
            )
            for recipe in block_tiling_recipes:
                previous = tiling_recipes.setdefault(recipe.recipe_id, recipe)
                if previous != recipe:
                    raise RuntimeError(
                        "tiling recipe identity changed across block candidates"
                    )
            executable_tiling_by_work = {
                recipe.work_per_thread: recipe
                for recipe in block_tiling_recipes
                if recipe.status.availability == _GpuAvailability.PROVEN
                and recipe.strategy
                in (
                    _GpuTileStrategy.BASELINE,
                    _GpuTileStrategy.THREAD_COARSENED,
                )
            }
            workgroup = _dimension_by_name(dimensions, _WORKGROUP_DIMENSION)
            if block_dim not in workgroup.legal_values:
                rejections.append(
                    _KernelVariantRejection(block_dim, workgroup.status.reason)
                )
                continue
            tls_modes = _dimension_by_name(dimensions, _TLS_DIMENSION).legal_values
            compile_tier_values = _dimension_by_name(
                dimensions, _COMPILE_TIER_DIMENSION
            ).legal_values
            min_blocks_values = _dimension_by_name(
                dimensions, _CUDA_MIN_BLOCKS_DIMENSION
            ).legal_values
            max_register_values = _dimension_by_name(
                dimensions, _CUDA_MAX_REGISTERS_DIMENSION
            ).legal_values
            residency_dimension = _dimension_by_name(
                dimensions, _RESIDENCY_DIMENSION
            )
            residency_values = residency_dimension.legal_values
            if not residency_values:
                residency_values = (None,)
            work_dimension = _dimension_by_name(
                dimensions, _RANGE_WORK_PER_THREAD_DIMENSION
            )
            work_values = tuple(
                value
                for value in work_dimension.legal_values
                if value in requested_work
            )
            if not work_values:
                if 1 in requested_work:
                    work_values = (1,)
                else:
                    rejections.append(
                        _KernelVariantRejection(
                            block_dim, work_dimension.status.reason
                        )
                    )
                    continue
            baseline_structural = {
                _WORKGROUP_DIMENSION: block_dim,
                _TLS_DIMENSION: "auto",
                _COMPILE_TIER_DIMENSION: "inherit",
                _CUDA_MIN_BLOCKS_DIMENSION: 2,
                _CUDA_MAX_REGISTERS_DIMENSION: None,
            }
            if structural_mode == "cartesian":
                structural_selections = [
                    {
                        _WORKGROUP_DIMENSION: block_dim,
                        _TLS_DIMENSION: thread_local,
                        _COMPILE_TIER_DIMENSION: compile_tier,
                        _CUDA_MIN_BLOCKS_DIMENSION: min_blocks,
                        _CUDA_MAX_REGISTERS_DIMENSION: max_registers,
                    }
                    for (
                        thread_local,
                        compile_tier,
                        min_blocks,
                        max_registers,
                    ) in product(
                        tls_modes,
                        compile_tier_values,
                        min_blocks_values,
                        max_register_values,
                    )
                ]
            else:
                structural_selections = [baseline_structural]
                for dimension_name, values in (
                    (_TLS_DIMENSION, tls_modes),
                    (_COMPILE_TIER_DIMENSION, compile_tier_values),
                    (_CUDA_MIN_BLOCKS_DIMENSION, min_blocks_values),
                    (_CUDA_MAX_REGISTERS_DIMENSION, max_register_values),
                ):
                    for value in values:
                        if value == baseline_structural[dimension_name]:
                            continue
                        selection = dict(baseline_structural)
                        selection[dimension_name] = value
                        structural_selections.append(selection)

            for structural in structural_selections:
                for work_per_thread in work_values:
                    tiling_recipe = executable_tiling_by_work.get(
                        work_per_thread
                    )
                    if tiling_recipe is None:
                        raise RuntimeError(
                            "executable work-per-thread variant has no proven "
                            "tiling recipe"
                        )
                    for waves in residency_values:
                        spec = self._spec(
                            block_dim,
                            structural[_TLS_DIMENSION],
                            work_per_thread,
                            waves,
                            structural[_COMPILE_TIER_DIMENSION],
                            structural[_CUDA_MIN_BLOCKS_DIMENSION],
                            structural[_CUDA_MAX_REGISTERS_DIMENSION],
                        )
                        selections = {
                            **structural,
                            _RANGE_WORK_PER_THREAD_DIMENSION: work_per_thread,
                            _RESIDENCY_DIMENSION: waves,
                        }
                        variants.append(
                            _KernelVariant(
                                variant_id=spec.identity,
                                compilation_id=spec.compilation_identity,
                                spec=spec,
                                logical_task_id=task.logical_task_id,
                                logical_task_ids=(task.logical_task_id,),
                                baseline_thread_local_bytes=int(
                                    task.thread_local_bytes
                                ),
                                physical_equivalence_key=(
                                    _gpu_physical_equivalence_key(
                                        dimensions,
                                        selections,
                                        _GpuPhysicalEffect.ARTIFACT,
                                    )
                                ),
                                selections=tuple(selections.items()),
                                resource_envelope=resource_envelope,
                                tiling_recipe_id=tiling_recipe.recipe_id,
                            )
                        )

        if len(variants) > _MAX_VARIANTS:
            raise RuntimeError("kernel variant search exceeded its bounded budget")
        self._variants = MappingProxyType(
            {variant.variant_id: variant for variant in variants}
        )
        self._rejections = tuple(rejections)
        self._dimensions = () if dimension_contract is None else dimension_contract
        self._tiling_recipes = MappingProxyType(tiling_recipes)
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

    def _initialize_kernel_wide(
        self,
        report,
        snapshot,
        *,
        compile_tiers,
        cuda_min_blocks_per_sm,
        cuda_max_registers,
        structural_mode,
    ):
        """Build artifact-only variants for kernels with multiple offloads."""

        if (
            report is None
            or report.backend != "cuda"
            or report.status != "applied"
        ):
            raise RuntimeError("kernel-wide variant baseline did not apply on CUDA")
        from taichi_forge.lang._gpu_semantics import _GpuPhysicalEffect
        from taichi_forge.lang._gpu_semantics_tuning import (
            _COMPILE_TIER_DIMENSION,
            _CUDA_MAX_REGISTERS_DIMENSION,
            _CUDA_MIN_BLOCKS_DIMENSION,
            _RESIDENCY_DIMENSION,
            _RANGE_WORK_PER_THREAD_DIMENSION,
            _TLS_DIMENSION,
            _WORKGROUP_DIMENSION,
            _derive_gpu_tuning_dimensions,
            _dimension_by_name,
            _gpu_physical_equivalence_key,
        )

        dimensions = _derive_gpu_tuning_dimensions(
            snapshot,
            max_threads=impl.current_cfg().max_block_dim,
            canonical_workgroup_sizes=_BLOCK_DIMS,
            residency_values=_GRID_RESIDENCY_WAVES,
            range_work_per_thread_values=_RANGE_WORK_PER_THREAD_TARGETS,
            compile_tier_values=compile_tiers,
            cuda_min_blocks_values=cuda_min_blocks_per_sm,
            cuda_max_register_values=cuda_max_registers,
        )
        tls_values = _dimension_by_name(dimensions, _TLS_DIMENSION).legal_values
        tier_values = _dimension_by_name(
            dimensions, _COMPILE_TIER_DIMENSION
        ).legal_values
        min_blocks_values = _dimension_by_name(
            dimensions, _CUDA_MIN_BLOCKS_DIMENSION
        ).legal_values
        max_register_values = _dimension_by_name(
            dimensions, _CUDA_MAX_REGISTERS_DIMENSION
        ).legal_values
        baseline = {
            _WORKGROUP_DIMENSION: None,
            _TLS_DIMENSION: "auto",
            _COMPILE_TIER_DIMENSION: "inherit",
            _CUDA_MIN_BLOCKS_DIMENSION: 2,
            _CUDA_MAX_REGISTERS_DIMENSION: None,
            _RANGE_WORK_PER_THREAD_DIMENSION: 1,
            _RESIDENCY_DIMENSION: None,
        }
        if structural_mode == "cartesian":
            selections = [
                {
                    **baseline,
                    _TLS_DIMENSION: thread_local,
                    _COMPILE_TIER_DIMENSION: compile_tier,
                    _CUDA_MIN_BLOCKS_DIMENSION: min_blocks,
                    _CUDA_MAX_REGISTERS_DIMENSION: max_registers,
                }
                for (
                    thread_local,
                    compile_tier,
                    min_blocks,
                    max_registers,
                ) in product(
                    tls_values,
                    tier_values,
                    min_blocks_values,
                    max_register_values,
                )
            ]
        else:
            selections = [baseline]
            for dimension_name, values in (
                (_TLS_DIMENSION, tls_values),
                (_COMPILE_TIER_DIMENSION, tier_values),
                (_CUDA_MIN_BLOCKS_DIMENSION, min_blocks_values),
                (_CUDA_MAX_REGISTERS_DIMENSION, max_register_values),
            ):
                for value in values:
                    if value == baseline[dimension_name]:
                        continue
                    selection = dict(baseline)
                    selection[dimension_name] = value
                    selections.append(selection)

        task_ids = tuple(
            dispatch.logical_task_id for dispatch in snapshot.dispatches
        )
        variants = []
        for selection in selections:
            spec = self._kernel_wide_spec(
                selection[_TLS_DIMENSION],
                selection[_COMPILE_TIER_DIMENSION],
                selection[_CUDA_MIN_BLOCKS_DIMENSION],
                selection[_CUDA_MAX_REGISTERS_DIMENSION],
            )
            variant_id = spec.identity or "kos1:baseline"
            compilation_id = spec.compilation_identity or "kos1:baseline"
            variants.append(
                _KernelVariant(
                    variant_id=variant_id,
                    compilation_id=compilation_id,
                    spec=spec,
                    logical_task_id=snapshot.program.specialization_id,
                    logical_task_ids=task_ids,
                    baseline_thread_local_bytes=sum(
                        int(task.thread_local_bytes or 0) for task in report.tasks
                    ),
                    physical_equivalence_key=_gpu_physical_equivalence_key(
                        dimensions,
                        selection,
                        _GpuPhysicalEffect.ARTIFACT,
                    ),
                    selections=tuple(selection.items()),
                    resource_envelope=None,
                    tiling_recipe_id=None,
                )
            )
        if len(variants) > _MAX_VARIANTS:
            raise RuntimeError("kernel variant search exceeded its bounded budget")
        self._kernel_wide = True
        self._variants = MappingProxyType(
            {variant.variant_id: variant for variant in variants}
        )
        self._rejections = ()
        self._dimensions = dimensions
        self._tiling_recipes = MappingProxyType({})
        self._compilation_groups = tuple(
            _KernelCompilationGroup(
                compilation_id=variant.compilation_id,
                representative_variant_id=variant.variant_id,
                variant_ids=(variant.variant_id,),
            )
            for variant in variants
        )

    @staticmethod
    def _spec(
        block_dim,
        thread_local,
        work_per_thread,
        waves,
        compile_tier,
        cuda_min_blocks_per_sm,
        cuda_max_registers,
    ):
        return _KernelOptimizationSpec(
            ir=_IrOptimizationOptions(
                thread_local=thread_local,
                compile_tier=compile_tier,
            ),
            backend=_BackendCodegenOptions(
                workgroup_size=block_dim,
                cuda_min_blocks_per_sm=cuda_min_blocks_per_sm,
            ),
            artifact=_ArtifactOptions(cuda_max_registers=cuda_max_registers),
            launch=_LaunchOptions(
                block_mode="require",
                grid_residency_waves=waves,
                range_work_per_thread_target=work_per_thread,
            ),
        )

    @staticmethod
    def _kernel_wide_spec(
        thread_local,
        compile_tier,
        cuda_min_blocks_per_sm,
        cuda_max_registers,
    ):
        return _KernelOptimizationSpec(
            ir=_IrOptimizationOptions(
                thread_local=thread_local,
                compile_tier=compile_tier,
            ),
            backend=_BackendCodegenOptions(
                cuda_min_blocks_per_sm=cuda_min_blocks_per_sm,
            ),
            artifact=_ArtifactOptions(cuda_max_registers=cuda_max_registers),
        )

    @staticmethod
    def _eligible_range_task(
        report,
        snapshot,
        *,
        compile_tiers,
        cuda_min_blocks_per_sm,
        cuda_max_registers,
    ):
        if report.backend != "cuda" or report.status != "applied":
            raise RuntimeError("variant baseline did not apply on CUDA")
        from taichi_forge.lang._gpu_semantics import _GpuAvailability
        from taichi_forge.lang._gpu_semantics_tuning import (
            _WORKGROUP_DIMENSION,
            _derive_gpu_tuning_dimensions,
            _derive_workgroup_resource_envelope,
            _dimension_by_name,
        )

        dimensions = _derive_gpu_tuning_dimensions(
            snapshot,
            max_threads=impl.current_cfg().max_block_dim,
            canonical_workgroup_sizes=_BLOCK_DIMS,
            residency_values=_GRID_RESIDENCY_WAVES,
            range_work_per_thread_values=_RANGE_WORK_PER_THREAD_TARGETS,
            compile_tier_values=compile_tiers,
            cuda_min_blocks_values=cuda_min_blocks_per_sm,
            cuda_max_register_values=cuda_max_registers,
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
        return (
            task_by_id[range_dispatch.physical_dispatch_id],
            dimensions,
            _derive_workgroup_resource_envelope(
                snapshot, impl.current_cfg().max_block_dim
            ),
        )

    @property
    def rejections(self):
        return self._rejections

    @property
    def compilation_groups(self):
        return self._compilation_groups

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def tiling_recipes(self):
        return tuple(self._tiling_recipes.values())

    @property
    def structural_mode(self):
        return self._structural_mode

    @property
    def scope_kind(self):
        return "kernel_artifact" if self._kernel_wide else "range_task"

    def refinement(self, block_dim):
        if block_dim not in self._block_dims:
            raise ValueError("refinement block_dim must belong to the parent session")
        return _KernelVariantSession(
            self._kernel,
            self._args,
            block_dims=(block_dim,),
            range_work_per_thread_targets=self._range_work_per_thread_targets,
            compile_tiers=self._compile_tiers,
            cuda_min_blocks_per_sm=self._cuda_min_blocks_per_sm,
            cuda_max_registers=self._cuda_max_registers,
            structural_mode="cartesian",
        )

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
