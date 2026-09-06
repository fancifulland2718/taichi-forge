"""Prepared sparse-dense physical plans in the whole-Graph composer."""

from taichi_forge.graph._recipes.families import (
    GraphRuntimeFragmentProvider,
    _fragment,
    _recipe_operation_dispatch_count,
    runtime_family_provider_descriptor,
)
from taichi_forge.graph._recipes.fragments import GraphFragmentTask
from taichi_forge.graph._recipes.providers import GraphRecipeProviderError


def _sources(definition):
    regions = {region.path: region for region in definition.regions}
    for node_index, node in enumerate(definition._runtime_spec.nodes):
        ordinal = 0
        for operation in getattr(node, "recipe_operations", ()):
            if operation[0] == "native":
                executable = operation[1]
                recording = getattr(executable, "_recording", None)
                source = getattr(recording, "_graph_spmm_source", None)
                if source is not None:
                    path = f"graph/{node_index}:{node.ir_node.kind}/{ordinal}:native_call"
                    region = regions.get(path)
                    if region is None:
                        raise ValueError("SpMM source is not covered by the frozen semantic Graph")
                    yield path, region.region_id, source, executable
            ordinal += _recipe_operation_dispatch_count(operation)


class SparseSpmmRecipeProvider(GraphRuntimeFragmentProvider):
    """Explicit provider for prepared fixed-pattern sparse-dense operations.

    Pass alongside graph.default_recipe_providers(). Only a complete recipe
    is scheduled by CompileIQ. Preparation requires the application's existing
    dense arrays; discovery never allocates duplicate arrays or runs the math.
    """

    descriptor = runtime_family_provider_descriptor(
        "sparse_spmm",
        capabilities=("fixed-pattern-spmm", "frozen-preprocess-policy"),
        domain_version="spmm-retained-plans-v3",
        semantic_fingerprint="f32-csr-row-major-relaxed-finite-v1",
    )

    def fragments(self, definition):
        fragments = []
        for path, region_id, source, _executable in _sources(definition):
            prepared = source.preparation_report()
            if "row_streamed" not in prepared:
                raise GraphRecipeProviderError(
                    "Prepare Graph SpMM with the actual dense arrays before recipe search",
                    error_key="spmm_plan_preparation_required",
                    provider_namespace=self.descriptor.namespace,
                )
            for strategy, info in prepared.items():
                if strategy == "row_streamed":
                    continue
                task = GraphFragmentTask.create(
                    f"{path}:{strategy}",
                    "retained_sparse_dense_matmul",
                    physical={
                        "semantic_contract": source.semantics,
                        "config": source.physical_config(strategy),
                        "component": source.component,
                        "workspace_bytes": info["workspace_bytes"],
                        "preprocessed": info["preprocessed"],
                        "vendor_internal_kernel_topology": "unknown",
                    },
                )
                fragments.append(
                    _fragment(
                        definition,
                        family="sparse_spmm",
                        source_key=path,
                        choice_id=strategy,
                        coverage=(region_id,),
                        tasks=(task,),
                        exclusive_submission=True,
                        provider_descriptor=self.descriptor,
                        compatible_executor_kinds=("cuda_immutable_argument_frames",),
                    )
                )
        return tuple(fragments)

    def contribute_runtime(self, assembly, selection):
        matches = tuple(item for item in _sources(assembly.definition) if item[0] == selection.source_key)
        if len(matches) != 1:
            raise ValueError("Frozen SpMM semantic source is unavailable")
        _, _region_id, source, executable = matches[0]
        strategy = selection.materialization_choice
        if strategy not in source.preparation_report():
            raise ValueError("Frozen SpMM physical plan was not prepared; no substitution was performed")

        def rewrite(builder, operation):
            recording = source._recording(strategy)
            builder._append_native(recording, admission=operation[2])

        assembly.select_operation(executable, rewrite)

    def describe(self, definition, fragment_key):
        fragment = self.resolve(definition, fragment_key)
        selection = fragment.provider_metadata["family_selection"]
        source = next(item[2] for item in _sources(definition) if item[0] == selection["source_key"])
        strategy = selection["materialization_choice"]
        return {
            **fragment.provider_metadata,
            "semantic_contract": source.semantics,
            "frozen_config": source.physical_config(strategy),
            "component_applicability": source.component,
            "preparation_observation": {
                **source.preparation_report()[strategy],
                "measurement_scope": "host_elapsed_for_spmm_preparation_or_cache_reuse",
                "shared_initialization": "not_separated",
                "selected_only_restore": "not_measured",
                "preparation_origin": source._preparation_origin,
                "plan_ownership": (
                    "independent_search_and_execution_leases" if source._owned_plans else "legacy_matrix_cache"
                ),
                "restoration_boundary": "first_bound_native_preparation" if source._owned_plans else "unavailable",
            },
            "limitations": (
                "CUDA scalar f32 CSR SparsePattern only; compact row-major dense bindings",
                "finite inputs and caller-qualified tolerance; no bitwise cross-version promise",
                "matrix-owned workspace shared by equal RHS/algorithm; runtime-ordered submissions",
                "close releases search leases; live Graph commands and pre-existing matrix caches retain their own plans",
                "frozen definitions hold matrix identity and expected facts, not search plans; per-plan bytes are not process residency",
                "selected plan creation requires real dense bindings; restored preparation timing remains historical, not freshly measured",
                "setup observation is outside trial metrics; production qualification remains downstream-owned",
            ),
        }


__all__ = ["SparseSpmmRecipeProvider"]
