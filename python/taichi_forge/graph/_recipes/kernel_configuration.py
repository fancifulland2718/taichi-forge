"""Lazy kernel configuration recipes over existing compilation contracts.

Repeated occurrences share a configuration and compiled specialization, while
the whole-Graph assembler retains each occurrence's original argument binding.
No alternative is compiled by discovery or by ordinary Graph execution.
"""

from dataclasses import dataclass, replace
import hashlib
import json

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _choice_fragment,
    _recipe_operation_dispatch_count,
    runtime_family_provider_descriptor,
)
from taichi_forge.lang._kernel_optimization import (
    _BackendCodegenOptions,
    _IrOptimizationOptions,
    _KernelOptimizationSpec,
    _LaunchOptions,
)


@dataclass(frozen=True)
class _ConfigurationManifest:
    recipe_id: str
    payload_json: str

    def to_dict(self):
        return json.loads(self.payload_json)


class GraphKernelConfigurationSource:
    def __init__(
        self, kernel_fn, args, kernel_cpp, template_args, facts, tier, max_block
    ):
        self.kernel_fn = kernel_fn
        self.args = tuple(args)
        self.baseline_kernel_cpp = kernel_cpp
        self.template_args = None if template_args is None else dict(template_args)
        self.facts = facts
        self.tier = tier
        self.max_block = max_block
        self._recipe_source_key = ""
        self._choices = None

    @classmethod
    def try_create(cls, kernel_fn, args, kernel_cpp, template_args, facts):
        from taichi_forge._lib import core
        from taichi_forge.lang import impl, kernel_impl

        kernel = (
            kernel_fn
            if isinstance(kernel_fn, kernel_impl.Kernel)
            else getattr(kernel_fn, "_primal", None)
        )
        if (
            facts is None
            or not isinstance(kernel, kernel_impl.Kernel)
            or kernel.autodiff_mode != core.AutodiffMode.NONE
        ):
            return None
        cfg = impl.current_cfg()
        return cls(
            kernel_fn,
            args,
            kernel_cpp,
            template_args,
            facts,
            kernel.opt_level or cfg.compile_tier,
            cfg.max_block_dim,
        )

    def choices(self):
        if self._choices is not None:
            return self._choices
        tasks = self.facts.snapshot.manifests
        if not tasks:
            self._choices = ()
            return self._choices
        from taichi_forge.lang import impl
        from taichi_forge.lang._gpu_semantics import _GpuAvailability
        from taichi_forge.lang._gpu_semantics_snapshot import (
            _build_resident_gpu_semantics,
        )
        from taichi_forge.lang._gpu_semantics_tuning import (
            _COMPILE_TIER_DIMENSION,
            _CUDA_MIN_BLOCKS_DIMENSION,
            _WORKGROUP_DIMENSION,
            _derive_gpu_tuning_dimensions,
            _dimension_by_name,
            _effective_workgroup_limit,
        )

        semantics = _build_resident_gpu_semantics(
            impl.get_runtime().prog._kernel_gpu_semantics_snapshot(
                self.baseline_kernel_cpp
            )
        )
        dimensions = _derive_gpu_tuning_dimensions(
            semantics,
            max_threads=_effective_workgroup_limit(self.max_block),
            canonical_workgroup_sizes=(32, 64, 128, 256, 512, 1024),
        )

        def values(name):
            dimension = _dimension_by_name(dimensions, name)
            return (
                dimension.legal_values
                if dimension.status.availability == _GpuAvailability.PROVEN
                else ()
            )

        # These are representative requests, not an exhaustive domain or a
        # limit on legal block sizes. Native compilation remains the authority
        # for geometry-sensitive operations and device resource constraints.
        specs = []
        if self.tier != "full" and "full" in values(_COMPILE_TIER_DIMENSION):
            specs.append(
                _KernelOptimizationSpec(ir=_IrOptimizationOptions(compile_tier="full"))
            )
        ranges = tuple(task for task in tasks if task.task_type == "range_for")
        if ranges and not any(
            task.source_block_size_explicit
            or task.static_shared_bytes
            or task.dynamic_shared_bytes
            for task in tasks
        ):
            specs.extend(
                _KernelOptimizationSpec(
                    backend=_BackendCodegenOptions(workgroup_size=block),
                    launch=_LaunchOptions(block_mode="require"),
                )
                for block in values(_WORKGROUP_DIMENSION)
                if any(block != task.selected_block_size for task in ranges)
            )
        if tasks[0].backend == "cuda" and ranges:
            specs.extend(
                _KernelOptimizationSpec(
                    backend=_BackendCodegenOptions(cuda_min_blocks_per_sm=value)
                )
                for value in values(_CUDA_MIN_BLOCKS_DIMENSION)
                if value != 2
            )
        self._choices = tuple((self.manifest(spec), spec) for spec in specs)
        return self._choices

    def manifest(self, spec):
        tasks = self.facts.snapshot.manifests
        payload = {
            "schema": "graph-kernel-configuration-v2",
            "baseline_tasks": tuple(task.logical_task_id for task in tasks),
            "backend": tasks[0].backend,
            "optimization_spec": json.loads(spec.stable_payload),
            # Requested configuration; final task/resource facts are observed
            # after compilation rather than copied from baseline measurements.
            "strategy": "compile-kernel-configuration",
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        key = ":".join(
            (
                "kernel-config",
                hashlib.sha256(encoded.encode()).hexdigest(),
                spec.ir.compile_tier,
                str(spec.backend.workgroup_size or 0),
                str(spec.backend.cuda_min_blocks_per_sm),
            )
        )
        return _ConfigurationManifest(key, encoded)

    def resolve(self, choice_id):
        # A stable key carries the supported controls, so restoring a generated
        # neighbor needs neither search history nor a Cartesian product table.
        try:
            prefix, _, tier, block, minimum = choice_id.split(":")
            if prefix != "kernel-config":
                raise ValueError("invalid configuration prefix")
            block, minimum = int(block), int(minimum)
            singles = tuple(spec for _, spec in self.choices())
            if (
                tier not in {"inherit", *(spec.ir.compile_tier for spec in singles)}
                or block
                not in {0, *(spec.backend.workgroup_size or 0 for spec in singles)}
                or minimum
                not in {2, *(spec.backend.cuda_min_blocks_per_sm for spec in singles)}
            ):
                raise ValueError("configuration outside qualified dimensions")
            spec = _KernelOptimizationSpec(
                ir=_IrOptimizationOptions(compile_tier=tier),
                backend=_BackendCodegenOptions(
                    workgroup_size=block or None, cuda_min_blocks_per_sm=minimum
                ),
                launch=_LaunchOptions(block_mode="require" if block else "auto"),
            )
            manifest = self.manifest(spec)
            if spec.is_baseline or manifest.recipe_id != choice_id:
                raise ValueError("configuration identity differs")
            return manifest, spec
        except (TypeError, ValueError) as error:
            raise ValueError("kernel configuration recipe is unavailable") from error

    def neighbors(self, spec):
        for _, delta in self.choices():
            if delta.ir.compile_tier != "inherit":
                neighbor = replace(spec, ir=delta.ir)
            elif delta.backend.workgroup_size is not None:
                neighbor = replace(
                    spec,
                    backend=replace(
                        spec.backend, workgroup_size=delta.backend.workgroup_size
                    ),
                    launch=delta.launch,
                )
            else:
                neighbor = replace(
                    spec,
                    backend=replace(
                        spec.backend,
                        cuda_min_blocks_per_sm=delta.backend.cuda_min_blocks_per_sm,
                    ),
                )
            if neighbor != spec:
                yield self.manifest(neighbor)

    def materialize(self, choice_id):
        from taichi_forge.graph._graph import gen_cpp_kernel, _kernel_task_manifests

        _, spec = self.resolve(choice_id)
        kernel_cpp = gen_cpp_kernel(
            self.kernel_fn,
            self.args,
            template_args=self.template_args,
            kernel_optimization_spec=spec,
        )
        tasks = _kernel_task_manifests(kernel_cpp)
        if not kernel_cpp._same_source_ir_as(self.baseline_kernel_cpp):
            raise ValueError(
                "kernel configuration source changed after Graph freeze; rebuild the definition"
            )
        if not tasks or any(
            task.optimization_spec_id != spec.compilation_identity for task in tasks
        ):
            raise ValueError(
                "compiled kernel configuration differs from the requested artifact"
            )
        return kernel_cpp


def _source_groups(definition):
    """Map exact replay occurrences, including intervening other node kinds.

    Matching semantic identities alone is insufficient when equal kernels occur
    in different regions. Only source objects retained in replay are eligible.
    """
    sources = getattr(
        definition._runtime_spec, "_graph_kernel_configuration_sources", ()
    )
    if not sources:
        return ()
    source_ids = {id(source): source for source in sources}
    regions = {region.path: region for region in definition.regions}
    groups = {}
    for node_index, node in enumerate(definition._runtime_spec.nodes):
        dispatch_index = 0
        for operation in getattr(node, "recipe_operations", ()):
            count = _recipe_operation_dispatch_count(operation)
            if operation[0] == "dispatch":
                matches = tuple(
                    source_ids[id(item)]
                    for item in operation[4:]
                    if id(item) in source_ids
                )
                if matches:
                    (source,) = matches
                    path = f"graph/{node_index}:{node.ir_node.kind}/{dispatch_index}:dispatch"
                    region = regions.get(path)
                    if region is not None:
                        group = groups.setdefault(source.baseline_kernel_cpp, ([], []))
                        group[0].append(source)
                        group[1].append(region.region_id)
            dispatch_index += count
    return tuple(
        (tuple(sources), tuple(coverage)) for sources, coverage in groups.values()
    )


class GraphKernelConfigurationRecipeProvider(GraphRuntimeFragmentProvider):
    descriptor = runtime_family_provider_descriptor(
        "kernel_configuration",
        capabilities=("lazy-kernel-configuration-v1", "typed-runtime-fragment"),
        domain_version="shared-specialization-configuration-v3",
        semantic_fingerprint="in-place-primal-configuration-neighbors-v3",
    )

    def fragments(self, definition):
        return tuple(
            _choice_fragment(
                definition,
                "kernel_configuration",
                sources[0]._recipe_source_key,
                coverage,
                manifest,
                provider_descriptor=self.descriptor,
                application_kind="dispatch_configuration",
            )
            for sources, coverage in _source_groups(definition)
            for manifest, _ in sources[0].choices()
        )

    def _resolve_group(self, definition, fragment_key):
        for sources, coverage in _source_groups(definition):
            prefix = f"kernel_configuration:{sources[0]._recipe_source_key}:"
            if fragment_key.startswith(prefix):
                manifest, spec = sources[0].resolve(fragment_key[len(prefix) :])
                return sources, coverage, manifest, spec
        raise KeyError("kernel configuration source group is unavailable")

    def resolve(self, definition, fragment_key):
        sources, coverage, manifest, _ = self._resolve_group(definition, fragment_key)
        return _choice_fragment(
            definition,
            "kernel_configuration",
            sources[0]._recipe_source_key,
            coverage,
            manifest,
            provider_descriptor=self.descriptor,
            application_kind="dispatch_configuration",
        )

    def expand(self, definition, fragment_key):
        sources, coverage, _, spec = self._resolve_group(definition, fragment_key)
        return tuple(
            _choice_fragment(
                definition,
                "kernel_configuration",
                sources[0]._recipe_source_key,
                coverage,
                manifest,
                provider_descriptor=self.descriptor,
                application_kind="dispatch_configuration",
            )
            for manifest in sources[0].neighbors(spec)
        )

    def contribute_runtime(self, assembly, selection):
        matches = tuple(
            sources
            for sources, coverage in _source_groups(assembly.definition)
            if sources[0]._recipe_source_key == selection.source_key
            and coverage == selection.coverage_region_ids
        )
        if len(matches) != 1:
            raise ValueError("kernel configuration source group is unavailable")
        # Compile/validate the shared code once, then replay the original
        # bindings of every occurrence through the existing assembler.
        compiled = []

        def rewrite():
            if not compiled:
                compiled.append(matches[0][0].materialize(selection.choice_id))
            return compiled[0], (), ()

        for source in matches[0]:
            assembly.select_dispatch(source, rewrite)

    def explain_discovery(self, definition):
        return {
            "family": "kernel_configuration",
            "supported_scope": "CUDA/Vulkan primal compiled specializations; shared configuration per code group",
            "compilation": "deferred until trial materialization; native validation may reject a request",
            "source_group_count": len(_source_groups(definition)),
        }
