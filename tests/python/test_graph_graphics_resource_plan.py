"""Resource-sensitive physical identity, explicit numerical permission and reuse."""

from dataclasses import replace
import json

import numpy as np
import pytest
import taichi_forge as ti

from tests import test_utils
from tests.python.test_hardware_graphics_identity import _pipeline


def _attachment_plan(texture):
    # This fixture owns these two formats; no generic byte-size inference.
    texel_bytes = {ti.Format.rgba32f: 16, ti.Format.rgba16f: 8}[texture.fmt]
    logical_bytes = int(np.prod(texture.shape)) * texel_bytes
    return ti.graph.GraphPhysicalResourceManifest.create(
        resource_id="external:target",
        kind="texture",
        requested_bytes=logical_bytes,
        allocated_bytes=logical_bytes,
        alignment=1,
        ownership="external",
        lifetime="graph",
        scope="public_external",
        binding_name="target",
        properties={"format": texture.fmt.name, "shape": texture.shape, "mip_levels": texture.mip_levels},
    )


class _AttachmentProvider:
    """The test owns the operation and permitted approximation, not Forge."""

    def __init__(self, definition, texture, *, allow_approximation):
        self.definition = definition
        self.plan = _attachment_plan(texture)
        self.allowed = allow_approximation
        self.descriptor = ti.graph.GraphRecipeProviderDescriptor(
            namespace="test.graphics_attachment",
            provider_version="1",
            domain_version="1",
            semantic_fingerprint=json.dumps({"allow_approximation": allow_approximation, "max_abs_error": 0.001}),
            assembly_protocols=(ti.graph.PROVIDER_OWNED_WHOLE_GRAPH_V1,),
        )

    def discover(self, definition):
        if not self.allowed or definition.semantic_graph_id != self.definition.semantic_graph_id:
            return ()
        return (
            ti.graph.GraphRecipeFragment.create(
                definition,
                provider_namespace=self.descriptor.namespace,
                provider_version="1",
                provider_domain_version="1",
                fragment_key="half-attachment",
                coverage_region_ids=tuple(region.region_id for region in definition.regions),
                tasks=(
                    ti.graph.GraphFragmentTask.create("draw-consume", "complete_graph", physical=self.plan.properties),
                ),
                assembly_protocol=ti.graph.PROVIDER_OWNED_WHOLE_GRAPH_V1,
                provider_metadata={
                    "numerical_contract": {"approximate": True, "max_abs_error": 0.001, "rgb_range": [0, 1]}
                },
            ),
        )

    def resolve(self, definition, key):
        fragments = self.discover(definition)
        if not fragments or key != fragments[0].fragment_key:
            raise ValueError("attachment recipe unavailable")
        return fragments[0]

    def expand(self, definition, key):
        self.resolve(definition, key)
        return ()

    def materialize(self, scope, fragment):
        graph = scope.own_executor(scope.definition.compile())
        return ti.graph.GraphMaterializedFragment.create(fragment, graph)

    def assemble(self, scope, definition, recipe, fragments):
        graph = fragments[0].payload
        return ti.graph.GraphMaterializationProduct(
            graph,
            ti.graph.CompiledGraphPhysicalManifest.from_graph(
                definition,
                recipe,
                graph,
                external_resource_plan=(self.plan,),
            ),
        )

    def describe(self, definition, key):
        self.resolve(definition, key)
        return {
            "limitations": "fixture-owned bounded RGB, not general HDR qualification",
            "attachment": self.plan.properties,
        }


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_explicit_attachment_plan_survives_execution_search_and_reuse(monkeypatch):
    @ti.kernel
    def read(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        value = image.fetch(ti.Vector([4, 4]), 0)
        for c in ti.static(range(3)):
            output[c] = value[c]

    exact = ti.Texture(ti.Format.rgba32f, (8, 8))
    approximate = ti.Texture(ti.Format.rgba16f, (8, 8))
    recreated = ti.Texture(ti.Format.rgba16f, (8, 8))
    vertices = ti.ndarray(ti.f32, 15)
    expected = np.array([0.37, 0.51, 0.83], np.float32)
    packed = np.empty((3, 5), np.float32)
    packed[:, :2] = [[-1, -1], [3, -1], [-1, 3]]
    packed[:, 2:] = expected
    vertices.from_numpy(packed.reshape(-1))
    output = ti.ndarray(ti.f32, 3)
    with _pipeline() as pipeline:
        builder = ti.graph.GraphBuilder()
        builder.append_native(
            pipeline.record_pass(
                (
                    pipeline.pass_draw(
                        ti.hardware.graphics.Draw(3),
                        vertex_buffers={0: "vertices"},
                    ),
                ),
                color="target",
            ),
            admission="auto",
        )
        builder.dispatch(
            read,
            ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "target", ndim=2),
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1),
        )
        definition = builder.freeze()
        baseline = definition.recipe_catalog(providers=()).baseline.recipe
        graph = definition.compile()
        try:

            def observe(texture):
                return ti.graph.CompiledGraphPhysicalManifest.from_graph(
                    definition,
                    baseline,
                    graph,
                    external_resource_plan=(_attachment_plan(texture),),
                )

            full, half = observe(exact), observe(approximate)
            assert full.materialized_physical_id != half.materialized_physical_id
            assert observe(recreated).materialized_physical_id == half.materialized_physical_id
            symbolic = ti.graph.CompiledGraphPhysicalManifest.from_graph(definition, baseline, graph)
            assert half.persistent_requested_bytes == symbolic.persistent_requested_bytes
            assert half.to_dict()["resource_plan"][-1]["properties"]["format"] == "rgba16f"
            binding = graph.bind(dict(target=approximate, vertices=vertices, output=output))
            graph.submit(binding).wait()
            np.testing.assert_allclose(output.to_numpy(), expected, atol=0.001, rtol=0)
            assert (
                half.refresh_from_graph(definition, baseline, graph).materialized_physical_id
                == half.materialized_physical_id
            )
            with pytest.raises(ValueError, match="public external"):
                ti.graph.CompiledGraphPhysicalManifest.from_graph(
                    definition,
                    baseline,
                    graph,
                    external_resource_plan=(replace(_attachment_plan(exact), scope="internal", binding_name=""),),
                )
            with pytest.raises(ValueError, match="unique"):
                ti.graph.CompiledGraphPhysicalManifest.from_graph(
                    definition,
                    baseline,
                    graph,
                    external_resource_plan=(_attachment_plan(exact), _attachment_plan(approximate)),
                )
            with monkeypatch.context() as patch:

                def unexpected(*args, **kwargs):
                    raise AssertionError("resource observation must not enter replay")

                patch.setattr(ti.graph.CompiledGraphPhysicalManifest, "from_graph", unexpected)
                for _ in range(3):
                    graph.run(binding)
                graph.submit(binding).wait()
        finally:
            graph.close()

        provider = _AttachmentProvider(definition, approximate, allow_approximation=True)
        disabled = _AttachmentProvider(definition, approximate, allow_approximation=False)
        assert definition.recipe_catalog(providers=(disabled,)).fragments == ()
        workload = ti.graph.GraphWorkloadContext({"shape": [8, 8], "rgb_range": [0, 1], "baseline_format": "rgba32f"})
        evaluation = ti.graph.GraphEvaluationContract(
            {"numerical_policy": {"allow_approximation": True, "max_abs_error": 0.001}}
        )
        environment = ti.graph.GraphBackendEnvironment({"fixture": "current-vulkan-device"})
        contracts = dict(workload_context=workload, evaluation_contract=evaluation, backend_environment=environment)
        target = ti.graph.GraphOptimizationTarget(objectives=(("attachment_bytes", "min"),))
        observed = {}

        def evaluate(executor, recipe):
            texture = exact if recipe.manifest.is_baseline else approximate
            binding = executor.bind(dict(target=texture, vertices=vertices, output=output))
            executor.submit(binding).wait()
            actual = output.to_numpy()
            np.testing.assert_allclose(actual, expected, atol=0.001, rtol=0)
            observed[recipe.recipe_id] = float(np.max(np.abs(actual - expected)))
            # Deterministic contract test, not an application acceleration claim.
            return {"attachment_bytes": float(_attachment_plan(texture).requested_bytes)}

        decision = definition.search_recipes(
            providers=(provider,),
            target=target,
            **contracts,
            budget=ti.graph.GraphSearchBudget(evaluation_limit=4, repeat_count=1),
            strategy=ti.graph.GraphRecipeSearchStrategy(mode="exact_if_bounded"),
        ).run(evaluate)
        assert decision.status == "selected", decision.report.results
        assert len(observed) == 2 and not decision.selection.manifest.is_baseline
        annotations = decision.report.recipe_annotations
        chosen = next(item for item in annotations if item["recipe_id"] == decision.selection.recipe_id)
        assert "rgba16f" in json.dumps(chosen["frozen_fragments"])
        assert "max_abs_error" in decision.report.to_markdown()
        for trial in chosen["trial_boundaries"]:
            assert (
                trial["after_materialization"]["materialized_physical_id"]
                == trial["after_evaluator"]["materialized_physical_id"]
            )
        selection = definition.resolve_recipe(decision.selection_artifact, providers=(provider,))
        with definition.materialize(selection, providers=(provider,)) as restored:
            assert restored.manifest.external_resource_plan == (provider.plan,)
            assert restored.manifest.materialized_physical_id in chosen["measurement"]["materialized_physical_ids"]
            evaluate(restored.executor, selection)
        changed = dict(
            contracts,
            evaluation_contract=ti.graph.GraphEvaluationContract(
                {"numerical_policy": {"allow_approximation": False, "max_abs_error": 3e-5}}
            ),
        )
        applicability = definition.check_recipe_applicability(
            decision.selection_artifact, providers=(provider,), target=target, **changed
        )
        assert not applicability.evidence_applicable
        with pytest.raises(ValueError, match="contract|evaluation"):
            definition.search_recipes(
                providers=(provider,),
                target=target,
                **changed,
                budget=ti.graph.GraphSearchBudget(evaluation_limit=8),
                checkpoint=decision.checkpoint,
            )
        with pytest.raises(ValueError, match="provider"):
            definition.resolve_recipe(decision.selection_artifact, providers=(disabled,))
