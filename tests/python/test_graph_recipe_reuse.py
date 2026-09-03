import copy
import hashlib
import json
import multiprocessing
from types import SimpleNamespace

import pytest
import taichi_forge as ti
from taichi_forge.graph._ir import (
    DispatchNode,
    GraphAccess,
    ResourceEffect,
    SequentialRegion,
)
from taichi_forge.graph._recipes import (
    GraphDefinition,
    GraphFragmentTask,
    GraphRecipeComposer,
    GraphRecipeFragment,
    GraphRecipeProviderDescriptor,
    GraphRecipeProviderSet,
)


def _portable_definition(backend="cuda", *, extra_node=False):
    nodes = [
        DispatchNode(
            name="load",
            effects=(
                ResourceEffect("values", GraphAccess.READ),
                ResourceEffect("middle", GraphAccess.WRITE),
            ),
            logical_kernel_identity="kernel:portable-load",
        ),
        DispatchNode(
            name="store",
            effects=(
                ResourceEffect("middle", GraphAccess.READ),
                ResourceEffect("output", GraphAccess.WRITE),
            ),
            logical_kernel_identity="kernel:portable-store",
        ),
    ]
    if extra_node:
        nodes.append(
            DispatchNode(
                name="observe",
                effects=(ResourceEffect("output", GraphAccess.READ),),
                logical_kernel_identity="kernel:portable-observe",
            )
        )
    root = SequentialRegion(tuple(nodes), name="portable_graph")
    spec = SimpleNamespace(
        pre_optimization_ir_root=root,
        definition_semantic_root=root,
        definition_semantic_sources=(),
        ir_root=root,
        runtime_arg_names=frozenset(),
        fixed_runtime_args={},
        temporary_runtime_arg_names=frozenset(),
        derived_runtime_arg_names=frozenset(),
        execution_definition={
            "nodes": (),
            "dispatch_count": len(nodes),
            "native_count": 0,
            "observation_count": 0,
            "structured_control_count": 0,
            "max_structured_depth": 0,
            "runtime_arg_count": 0,
            "fixed_runtime_arg_count": 0,
            "internal_storage_bytes": 0,
            "temporary_memory_plan": {},
        },
    )
    return GraphDefinition._from_graph_spec(
        spec,
        backend,
        core_commit="portable-test-core",
    )


class _PortableProvider:
    def __init__(self, *, semantic="portable-provider-v1", route="fused"):
        self.descriptor = GraphRecipeProviderDescriptor(
            namespace="tests.portable_recipe",
            provider_version="1",
            domain_version="portable-domain-v1",
            semantic_fingerprint=semantic,
        )
        self.route = route

    def _fragment(self, definition):
        return GraphRecipeFragment.create(
            definition,
            provider_namespace=self.descriptor.namespace,
            provider_version=self.descriptor.provider_version,
            provider_domain_version=self.descriptor.domain_version,
            fragment_key="stable:fused-load-store",
            coverage_region_ids=tuple(
                source.region_id for source in definition.sources
            ),
            tasks=(
                GraphFragmentTask.create(
                    "fused-load-store",
                    "synthetic_fused_kernel",
                    physical={"route": self.route},
                ),
            ),
        )

    def discover(self, definition):
        return (self._fragment(definition),)

    def resolve(self, definition, fragment_key):
        if fragment_key != "stable:fused-load-store":
            raise KeyError(fragment_key)
        return self._fragment(definition)


def _portable_artifact():
    definition = _portable_definition()
    provider = _PortableProvider()
    provider_set = GraphRecipeProviderSet(definition, (provider,))
    fragment = provider.discover(definition)[0]
    recipe = GraphRecipeComposer(definition).compose((fragment,))
    recipe_manifest = recipe.to_dict()
    manifest_json = json.dumps(
        recipe_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    artifact = ti.graph.GraphRecipeSelectionArtifact.create(
        structure={
            "semantic_graph_id": definition.semantic_graph_id,
            "backend": definition.backend,
            "provider_registry_id": provider_set.provider_registry_id,
            "generation_domain_id": provider_set.generation_domain_id,
            "provider_registry": provider_set.to_dict(),
            "recipe_id": recipe.recipe_id,
            "recipe_manifest_digest": (
                "graph-recipe-manifest-v1:"
                + hashlib.sha256(manifest_json.encode("ascii")).hexdigest()
            ),
            "planned_physical_id": recipe.planned_physical_id,
            "materialized_physical_id": "materialized:test-route",
        },
        recipe_manifest=recipe_manifest,
        evidence={"reuse_scope": "portable"},
    )
    return definition, provider, recipe, artifact


def _resolve_in_fresh_process(artifact, result_queue):
    try:
        definition = _portable_definition()
        resolved = definition.resolve_recipe(
            artifact,
            providers=(_PortableProvider(),),
        )
        result_queue.put(
            {
                "recipe_id": resolved.recipe_id,
                "planned_physical_id": resolved.planned_physical_id,
            }
        )
    except BaseException as error:
        result_queue.put(
            {
                "error": type(error).__name__,
                "message": str(error),
            }
        )


def test_graph_reuse_contexts_are_canonical_and_reject_implicit_objects():
    first = ti.graph.GraphWorkloadContext(
        {
            "shape": (32, 17),
            "distribution": {"kind": "clustered", "seed": 7},
        }
    )
    second = ti.graph.GraphWorkloadContext(
        {
            "distribution": {"seed": 7, "kind": "clustered"},
            "shape": [32, 17],
        }
    )
    assert first == second
    assert first.workload_context_id == second.workload_context_id
    assert ti.graph.GraphWorkloadContext.from_dict(first.to_dict()) == first
    json.dumps(first.to_dict(), sort_keys=True, allow_nan=False)

    evaluation = ti.graph.GraphEvaluationContract(
        {
            "warmup": 5,
            "repeat": 20,
            "synchronization": "after-sample",
            "correctness_oracle": "application-owned-v3",
        }
    )
    backend = ti.graph.GraphBackendEnvironment(
        {
            "backend": "cuda",
            "device": "test-device",
            "driver": "test-driver",
        }
    )
    assert evaluation.evaluation_contract_id.startswith("graph-evaluation-contract-v1:")
    assert backend.backend_environment_id.startswith("graph-backend-environment-v1:")

    with pytest.raises(ValueError, match="non-finite"):
        ti.graph.GraphWorkloadContext({"bad": float("nan")})
    with pytest.raises(ValueError, match="keys"):
        ti.graph.GraphWorkloadContext({1: "implicit-key-coercion"})
    with pytest.raises(TypeError, match="JSON-safe"):
        ti.graph.GraphEvaluationContract({"callback": object()})


def test_graph_reuse_artifacts_are_deterministic_and_tamper_evident():
    checkpoint = ti.graph.GraphRecipeSearchCheckpointV1.create(
        contract={"semantic_graph_id": "semantic:test"},
        generation={"recipes": (), "fragments": ()},
        compileiq_checkpoint={"records": (), "stages": (), "batches": ()},
    )
    restored = ti.graph.GraphRecipeSearchCheckpointV1.from_dict(checkpoint.to_dict())
    assert restored.checkpoint_id == checkpoint.checkpoint_id
    assert restored.compileiq_checkpoint["records"] == []

    artifact = ti.graph.GraphRecipeSelectionArtifact.create(
        structure={
            "semantic_graph_id": "semantic:test",
            "recipe_id": "recipe:test",
        },
        recipe_manifest={"recipe_id": "recipe:test", "fragments": ()},
        evidence={"checkpoint_id": checkpoint.checkpoint_id},
    )
    assert (
        ti.graph.GraphRecipeSelectionArtifact.from_dict(artifact).artifact_id
        == artifact.artifact_id
    )
    json.dumps(artifact.to_dict(), sort_keys=True, allow_nan=False)

    tampered_checkpoint = copy.deepcopy(checkpoint.to_dict())
    tampered_checkpoint["contract"]["semantic_graph_id"] = "semantic:other"
    with pytest.raises(ValueError, match="identity mismatch"):
        ti.graph.GraphRecipeSearchCheckpointV1.from_dict(tampered_checkpoint)

    tampered_artifact = copy.deepcopy(artifact.to_dict())
    tampered_artifact["recipe_manifest"]["recipe_id"] = "recipe:other"
    with pytest.raises(ValueError, match="identity mismatch"):
        ti.graph.GraphRecipeSelectionArtifact.from_dict(tampered_artifact)


def test_portable_selection_resolves_by_stable_key_in_a_fresh_process():
    definition, provider, recipe, artifact = _portable_artifact()
    resolved = definition.resolve_recipe(artifact, providers=(provider,))
    assert resolved.recipe_id == recipe.recipe_id
    assert resolved.planned_physical_id == recipe.planned_physical_id

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_resolve_in_fresh_process,
        args=(artifact.to_dict(), result_queue),
    )
    process.start()
    process.join(timeout=30)
    assert not process.is_alive()
    assert process.exitcode == 0
    assert result_queue.get(timeout=5) == {
        "recipe_id": recipe.recipe_id,
        "planned_physical_id": recipe.planned_physical_id,
    }


def test_portable_selection_rejects_structural_drift_before_materialization():
    definition, provider, _recipe, artifact = _portable_artifact()

    with pytest.raises(ti.graph.GraphRecipeReuseError) as semantic:
        _portable_definition(extra_node=True).resolve_recipe(
            artifact,
            providers=(_PortableProvider(),),
        )
    assert semantic.value.error_key == "semantic_graph_drift"

    with pytest.raises(ti.graph.GraphRecipeReuseError) as backend:
        _portable_definition(backend="cpu").resolve_recipe(
            artifact,
            providers=(_PortableProvider(),),
        )
    assert backend.value.error_key == "backend_unavailable"

    with pytest.raises(ti.graph.GraphRecipeReuseError) as registry:
        definition.resolve_recipe(
            artifact,
            providers=(_PortableProvider(semantic="provider-v2"),),
        )
    assert registry.value.error_key == "provider_registry_drift"

    with pytest.raises(ti.graph.GraphRecipeReuseError) as fragment:
        definition.resolve_recipe(
            artifact,
            providers=(_PortableProvider(route="different-physical-route"),),
        )
    assert fragment.value.error_key == "recipe_fragment_drift"

    report = _portable_definition(backend="cpu").check_recipe_applicability(
        artifact,
        providers=(_PortableProvider(),),
    )
    assert report.status == "backend_unavailable"
    assert not report.structurally_resolvable
    assert not report.evidence_applicable
