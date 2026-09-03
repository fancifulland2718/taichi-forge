import copy
import json

import pytest
import taichi_forge as ti


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
    assert evaluation.evaluation_contract_id.startswith(
        "graph-evaluation-contract-v1:"
    )
    assert backend.backend_environment_id.startswith(
        "graph-backend-environment-v1:"
    )

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
    restored = ti.graph.GraphRecipeSearchCheckpointV1.from_dict(
        checkpoint.to_dict()
    )
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
