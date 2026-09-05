import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge import _compileiq_opaque as _shared_compileiq_opaque
from taichi_forge._lib import core as ti_core
from taichi_forge.graph import (
    CompileIQGraphUnavailableError,
    _compileiq_opaque,
    compileiq_recipe_search,
)
from tests import test_utils

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("failure_phase", (None, "materialization", "objective", "observation", "cleanup", "protocol"))
def test_trial_boundary_evidence_survives_each_failure_without_inventing_memory(monkeypatch, failure_phase):
    from compileiq.forge_support import TrialCleanupV2, TrialFailureV2, TrialOutcomeV2
    from taichi_forge.graph._recipes import physical
    from taichi_forge.graph._trial_observations import _PROVENANCE_KEY, _boundary_markdown, _trial_boundaries

    events = []
    recipe = SimpleNamespace(planned_physical_id="planned", recipe_id="recipe")
    request = SimpleNamespace(
        recipe_id="recipe",
        batch_fingerprint="batch",
        measurement_key="measurement",
        observation_index=0,
        stage_index=1,
        fidelity_name="full",
    )

    def manifest(size):
        return SimpleNamespace(
            materialized_physical_id=f"physical-{size}",
            allocation_topology_exact=True,
            persistent_requested_bytes=size,
            persistent_allocated_bytes=size,
            transient_requested_bytes=0,
            transient_allocated_bytes=0,
        )

    def enter(phase):
        events.append(phase)
        if failure_phase == phase:
            error = ValueError(f"injected {phase}")
            error.cleanup_complete = True
            raise error

    def materialize(*args, **kwargs):
        enter("materialization")
        return SimpleNamespace(manifest=manifest(16), executor=object(), close=lambda: enter("cleanup"))

    def evaluate(*args):
        enter("objective")
        return {"score": float("nan") if failure_phase == "protocol" else 1.0}

    def observe(*args):
        enter("observation")
        return manifest(176)

    session = object.__new__(_compileiq_opaque._CompleteGraphRecipeSearchSessionV2)
    session._plans = SimpleNamespace(
        _catalog=SimpleNamespace(entry=lambda identity: SimpleNamespace(recipe=recipe)),
        _definition=object(),
        backend="cuda",
        python_source_lock="source",
        semantic_plan_id="semantic",
    )
    session._context = object()
    session._materialize_recipe = materialize
    session._objective_function = evaluate
    session._outcome_type = TrialOutcomeV2
    session._cleanup_type = TrialCleanupV2
    session._failure_type = TrialFailureV2
    session._scalar_metric = False
    session._target_contract = SimpleNamespace(metric_names=("score", "materialized_memory_bytes"))
    monkeypatch.setattr(physical, "observe_graph_physical_manifest", observe)
    outcome = session._evaluate(request)
    assert (outcome.failure is None) == (failure_phase is None)
    encoded = outcome.provenance[_PROVENANCE_KEY]
    assert len(encoded.encode("utf-8")) < 4096
    # Projection must include failed/older-stage records, but not fabricate
    # observations for checkpoints produced before this annotation existed.
    record = {"request": vars(request), "outcome": outcome.model_dump()}
    legacy = {"request": vars(request), "outcome": {**record["outcome"], "provenance": {}}}
    (observation,) = _trial_boundaries((legacy, record))["recipe"]
    assert observation["trial_failed"] == (failure_phase is not None)
    assert observation["stage_index"] == 1
    timings = observation["host_wall_seconds"]
    assert timings["materialization"] >= 0
    if failure_phase == "materialization":
        assert events == ["materialization"]
        assert observation["after_materialization"] is None
        assert observation["after_evaluator_status"] == "not_run"
        assert timings["evaluator"] is None
        assert timings["cleanup"] is None
    else:
        assert observation["after_materialization"]["persistent_allocated_bytes"] == 16
        assert events == (
            ["materialization", "objective"] + ([] if failure_phase == "objective" else ["observation"]) + ["cleanup"]
        )
        assert timings["evaluator"] >= 0
        assert timings["cleanup"] >= 0
    if failure_phase in ("materialization", "objective", "observation"):
        assert observation["after_evaluator"] is None
        assert observation["after_evaluator_status"] != "observed"
        markdown = "\n".join(_boundary_markdown(({"recipe_id": "recipe", "trial_boundaries": (observation,)},)))
        assert "unavailable" in markdown
        assert "0 / 1" in markdown
    else:
        assert observation["after_evaluator"]["persistent_allocated_bytes"] == 176
        assert observation["after_evaluator_status"] == "observed"
        assert outcome.materialized_memory_bytes == 176
        assert timings["post_evaluator_observation"] >= 0
    if failure_phase is None:
        assert outcome.metrics["materialized_memory_bytes"] == 176
    elif failure_phase == "cleanup":
        assert observation["cleanup_status"] == "incomplete"


def test_compileiq_public_search_surface_is_graph_owned():
    assert ti.graph.compileiq_recipe_search is compileiq_recipe_search
    assert not hasattr(ti.graph, "CompileIQGraphRecipeSearch")
    assert not hasattr(ti.graph, "GraphExecutableRecipeSelection")
    assert not hasattr(ti, "compileiq_offload_execution_plan_search")
    assert not hasattr(ti.lang, "compileiq_offload_execution_plan_search")
    assert not hasattr(ti, "CompileIQOffloadExecutionPlanSearch")
    assert not hasattr(ti.lang, "CompileIQOffloadExecutionPlanSearch")
    assert not hasattr(ti.algorithms, "compileiq_reduce_provider_search")
    assert not hasattr(ti.algorithms, "compileiq_segmented_scan_search")
    assert not hasattr(ti.algorithms, "CompileIQReduceProviderSearch")
    assert not hasattr(ti.algorithms, "CompileIQSegmentedScanSearch")


def _search_budget(evaluations):
    from compileiq.forge_support import ForgeOpaqueSearchBudgetV2

    return ForgeOpaqueSearchBudgetV2(
        evaluation_limit=evaluations,
        time_limit_seconds=300.0,
        materialized_memory_limit_bytes=1 << 30,
    )


class _Literal:
    def __init__(self, value):
        self.value = value


class _Choice:
    def __init__(self, values):
        self.vals = list(values)


class _OpaqueRecipeDomain:
    SCHEMA = "compileiq.opaque-recipe-domain.v1"
    MAX_RECIPE_IDS = 4096
    MAX_FIELD_UTF8_BYTES = 4096
    MAX_CANONICAL_BYTES = 4 * 1024 * 1024

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)
        self.recipe_ids = tuple(
            sorted(self.recipe_ids, key=lambda value: value.encode())
        )
        payload = json.dumps(
            {
                "provider_namespace": self.provider_namespace,
                "domain_version": self.domain_version,
                "provider_semantic_fingerprint": self.provider_semantic_fingerprint,
                "recipe_ids": self.recipe_ids,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.domain_fingerprint = "ciq-domain-v1:" + hashlib.sha256(payload).hexdigest()

    def to_search_space(self):
        return {
            "domain_fingerprint": _Literal(self.domain_fingerprint),
            "recipe_id": _Choice(
                f"ciq-recipe-v1-{ordinal:04d}"
                for ordinal in range(len(self.recipe_ids))
            ),
        }

    def model_dump(self, *, by_alias):
        assert by_alias is True
        return {
            "schema": self.SCHEMA,
            "provider_namespace": self.provider_namespace,
            "domain_version": self.domain_version,
            "provider_semantic_fingerprint": self.provider_semantic_fingerprint,
            "compileiq_capability_id": self.compileiq_capability_id,
            "compileiq_core_commit": self.compileiq_core_commit,
            "compileiq_core_lock": self.compileiq_core_lock,
            "recipe_ids": self.recipe_ids,
        }


class _Worker:
    PROTOCOL = "forge_main_thread_serial_v1"


class _ExhaustiveSearch:
    PROTOCOL = "bounded_exhaustive_main_thread_v1"


class _SearchSessionV2:
    PROTOCOL = "dynamic_batch_pareto_racing_main_thread_v2"


class _BudgetV2:
    pass


class _OutcomeV2:
    SCHEMA = "compileiq.taichi-forge-trial-outcome.v2"


class _CleanupV2:
    pass


class _BatchV2:
    SCHEMA = "compileiq.opaque-recipe-batch.v2"


class _FidelityV2:
    pass


class _LineageV2:
    pass


class _DynamicDomainV2:
    SCHEMA = "compileiq.opaque-dynamic-recipe-domain.v2"


class _EvaluationContextV1:
    SCHEMA = "compileiq.taichi-forge-evaluation-context.v1"


class _FinalizationV1:
    SCHEMA = "compileiq.taichi-forge-search-finalization.v1"


class _SearchStatusV2:
    SCHEMA = "compileiq.taichi-forge-search-status.v2"


class _OptimizationReportV1:
    SCHEMA = "compileiq.opaque-optimization-report.v1"


class _TargetContract:
    SCHEMA = "compileiq.taichi-forge-opaque-target-contract.v1"


def _capability():
    return MappingProxyType(
        {
            "schema": "compileiq.taichi-forge-recipe-search-capability.v2",
            "protocol_revision": 6,
            "fork_build_id": "compileiq-taichi-forge-complete-recipes.v2",
            "package_version": "1.0.0dev6+taichiforge.report1",
            "opaque_recipe_domain_schema": "compileiq.opaque-recipe-domain.v1",
            "opaque_recipe_batch_schema": "compileiq.opaque-recipe-batch.v2",
            "opaque_dynamic_recipe_domain_schema": (
                "compileiq.opaque-dynamic-recipe-domain.v2"
            ),
            "selection_audit_schema": "compileiq.opaque-recipe-selection.v1",
            "opaque_target_contract_schema": (
                "compileiq.taichi-forge-opaque-target-contract.v1"
            ),
            "opaque_target_selection": (
                "uncertainty_aware_pareto_layers_no_scalarization_v2"
            ),
            "trial_outcome_schema": "compileiq.taichi-forge-trial-outcome.v2",
            "search_checkpoint_schema": (
                "compileiq.taichi-forge-search-checkpoint.v2"
            ),
            "evaluation_context_schema": (
                "compileiq.taichi-forge-evaluation-context.v1"
            ),
            "search_finalization_schema": (
                "compileiq.taichi-forge-search-finalization.v1"
            ),
            "search_status_schema": "compileiq.taichi-forge-search-status.v2",
            "optimization_report_schema": (
                "compileiq.opaque-optimization-report.v1"
            ),
            "optimization_report_renderer": (
                "json_fact_source_markdown_projection_v1"
            ),
            "max_recipe_ids": 4096,
            "max_field_utf8_bytes": 4096,
            "max_canonical_bytes": 4 * 1024 * 1024,
            "provider_recipe_ids_cross_core_boundary": False,
            "core_verification": (
                "bundled_manifest_lock_and_platform_hashes_at_search_start_no_override"
            ),
            "opaque_domain_binding": "capability_id_core_commit_core_lock",
            "objective_worker": "forge_main_thread_serial_v1",
            "opaque_recipe_search": (
                "dynamic_batch_pareto_racing_main_thread_v2"
            ),
            "opaque_recipe_search_v1": "bounded_exhaustive_main_thread_v1",
            "core_manifest_schema_version": 1,
            "core_commit": _compileiq_opaque._EXPECTED_CORE_COMMIT,
            "core_lock": _compileiq_opaque._EXPECTED_CORE_LOCK,
            "capability_id": _compileiq_opaque._EXPECTED_CAPABILITY_ID,
        }
    )


def test_importing_public_graph_api_does_not_import_compileiq():
    script = """
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "python"))
import taichi_forge.graph

loaded = [
    name
    for name in sys.modules
    if name == "compileiq" or name.startswith("compileiq.")
]
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(_ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_public_graph_search_rejects_legacy_executable_spaces():
    with pytest.raises(TypeError, match="frozen GraphDefinition"):
        compileiq_recipe_search(SimpleNamespace(definition=None))


def test_missing_or_different_compileiq_cannot_use_the_public_path(monkeypatch):
    def missing(_):
        raise ImportError("not installed")

    monkeypatch.setattr(_compileiq_opaque, "import_module", missing)
    with pytest.raises(CompileIQGraphUnavailableError, match="compatible modified"):
        _compileiq_opaque._validated_compileiq_capability()

    capability = dict(_capability())
    capability["fork_build_id"] = "upstream"
    support = SimpleNamespace(
        forge_recipe_search_capability=lambda: SimpleNamespace(
            as_dict=lambda: capability
        ),
        ForgeMainThreadWorker=_Worker,
        ForgeOpaqueSearchSessionV2=_SearchSessionV2,
        ForgeOpaqueSearchBudgetV2=_BudgetV2,
        TrialOutcomeV2=_OutcomeV2,
        TrialCleanupV2=_CleanupV2,
        ForgeOpaqueEvaluationContextV1=_EvaluationContextV1,
        ForgeOpaqueSearchFinalizationV1=_FinalizationV1,
        ForgeOpaqueSearchStatusV2=_SearchStatusV2,
        OpaqueOptimizationReportV1=_OptimizationReportV1,
        opaque_optimization_report_json_schema=lambda: {},
        ForgeOpaqueRecipeExhaustiveSearchV1=_ExhaustiveSearch,
        ForgeOpaqueTargetContractV1=_TargetContract,
    )
    recipes = SimpleNamespace(
        OpaqueRecipeDomainV1=_OpaqueRecipeDomain,
        OpaqueRecipeBatchV2=_BatchV2,
        OpaqueRecipeFidelityV2=_FidelityV2,
        OpaqueRecipeLineageV2=_LineageV2,
        OpaqueDynamicRecipeDomainV2=_DynamicDomainV2,
    )
    modules = {
        "compileiq.forge_support": support,
        "compileiq.recipes": recipes,
    }
    monkeypatch.setattr(_compileiq_opaque, "import_module", modules.__getitem__)

    with pytest.raises(CompileIQGraphUnavailableError, match="incompatible"):
        _compileiq_opaque._validated_compileiq_capability()


def test_reviewed_compileiq_distribution_keeps_qualification_out_of_acceptance():
    distribution = _shared_compileiq_opaque._reviewed_distribution_manifest()
    snapshot = distribution["qualified_snapshot"]

    assert distribution["acceptance"] == (
        "compatible_capability_not_commit_or_wheel_hash"
    )
    assert "commit" not in distribution
    assert "wheel_sha256" not in distribution
    assert snapshot["commit"] == _shared_compileiq_opaque._REVIEWED_FORK_COMMIT
    assert snapshot["wheel_platform"] == "win32/amd64"
    assert snapshot["wheel_sha256"] == (
        "fe8c45f71341736609cc9b2374d7c79e6d9e25e984c5fad2fd087714b2608c9c"
    )
    assert snapshot["core_lock"] == (
        "sha256:b4838970b7b913bbb7ce6bd50aaa0d132b0df8b11765bd76284736be8a16040b"
    )


def test_future_compatible_compileiq_build_is_not_commit_or_wheel_locked(tmp_path):
    capability = dict(_capability())
    capability.update(
        {
            "package_version": "1.1.0+taichiforge.report2",
            "core_manifest_schema_version": 2,
            "core_commit": "b" * 40,
            "core_lock": "sha256:" + ("c" * 64),
            "compatible_additive_fact": "accepted",
        }
    )
    identity_payload = {
        name: value for name, value in capability.items() if name != "capability_id"
    }
    capability["capability_id"] = _shared_compileiq_opaque._identity(
        _shared_compileiq_opaque._CAPABILITY_ID_PREFIX,
        identity_payload,
    )

    package_root = tmp_path / "compileiq"
    package_root.mkdir()
    support_path = package_root / "forge_support.py"
    support_path.write_text("# compatible future support\n", encoding="utf-8")
    (package_root / "future_protocol_helper.py").write_text(
        "# additive source\n",
        encoding="utf-8",
    )
    support = SimpleNamespace(
        __file__=str(support_path),
        forge_recipe_search_capability=lambda: SimpleNamespace(
            as_dict=lambda: capability
        ),
        ForgeMainThreadWorker=_Worker,
        ForgeOpaqueSearchSessionV2=_SearchSessionV2,
        ForgeOpaqueSearchBudgetV2=_BudgetV2,
        TrialOutcomeV2=_OutcomeV2,
        TrialCleanupV2=_CleanupV2,
        ForgeOpaqueEvaluationContextV1=_EvaluationContextV1,
        ForgeOpaqueSearchFinalizationV1=_FinalizationV1,
        ForgeOpaqueSearchStatusV2=_SearchStatusV2,
        OpaqueOptimizationReportV1=_OptimizationReportV1,
        opaque_optimization_report_json_schema=lambda: {},
        ForgeOpaqueRecipeExhaustiveSearchV1=_ExhaustiveSearch,
        ForgeOpaqueTargetContractV1=_TargetContract,
    )
    recipes = SimpleNamespace(
        OpaqueRecipeDomainV1=_OpaqueRecipeDomain,
        OpaqueRecipeBatchV2=_BatchV2,
        OpaqueRecipeFidelityV2=_FidelityV2,
        OpaqueRecipeLineageV2=_LineageV2,
        OpaqueDynamicRecipeDomainV2=_DynamicDomainV2,
    )
    modules = {
        "compileiq.forge_support": support,
        "compileiq.recipes": recipes,
    }

    accepted, _, _, source_lock = (
        _shared_compileiq_opaque._validated_compileiq_capability(
            importer=modules.__getitem__
        )
    )
    assert accepted["package_version"] == "1.1.0+taichiforge.report2"
    assert accepted["compatible_additive_fact"] == "accepted"
    assert source_lock.startswith("ciq-python-source-v1:")
    assert source_lock != _shared_compileiq_opaque._EXPECTED_PYTHON_SOURCE_LOCK


@test_utils.test(arch=ti.cpu)
def test_public_v2_search_does_not_build_runtime_fusion_admission_space():
    @ti.kernel
    def copy(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for index in source:
            output[index] = source[index]

    source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder(_map_recipe="baseline")
    builder.dispatch(copy, source, output)
    graph = builder.compile()

    assert graph._spec._executable_optimization_space is None
    search = compileiq_recipe_search(graph)
    assert search.manifest()["baseline_recipe_id"] == search.baseline_recipe_id
    assert graph._spec._executable_optimization_space is None

    assert graph._executable_optimization_space.baseline.fusion_recipe_ids == ()
    assert graph._spec._executable_optimization_space is not None


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_modified_compileiq_exhausts_exact_graph_partitions():
    count = 257

    @ti.kernel
    def first(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            temporary[i] = source[i] * 2

    @ti.kernel
    def second(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
        middle: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            middle[i] = temporary[i] + 3

    @ti.kernel
    def third(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        middle: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = middle[i] * 4

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("source", "temporary", "middle", "output")
    }

    def build():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(first, symbolic["source"], symbolic["temporary"])
        builder.dispatch(
            second,
            symbolic["source"],
            symbolic["temporary"],
            symbolic["middle"],
        )
        builder.dispatch(
            third,
            symbolic["source"],
            symbolic["middle"],
            symbolic["output"],
        )
        return builder.compile(
            workspace_lanes=2,
            workspace_saturation="raise",
        )

    baseline = build()
    plans = compileiq_recipe_search(baseline)
    assert len(plans.recipe_ids) == 4

    source = ti.ndarray(ti.i32, shape=count)
    temporary = ti.ndarray(ti.i32, shape=count)
    middle = ti.ndarray(ti.i32, shape=count)
    output = ti.ndarray(ti.i32, shape=count)
    source_np = np.arange(count, dtype=np.int32)
    source.from_numpy(source_np)
    arguments = {
        "source": source,
        "temporary": temporary,
        "middle": middle,
        "output": output,
    }
    materialized = []

    def objective(graph, request):
        assert graph._workspace_lane_capacity == 2
        assert graph._workspace_saturation == "raise"
        graph.run(arguments)
        ti.sync()
        materialized.append(
            (
                request.recipe_id,
                graph.physical_plan()["physical_dispatch_count"],
            )
        )
        return float(plans.recipe_ids.index(request.recipe_id))

    with plans.compileiq_search(
        objective,
        budget=_search_budget(len(plans.recipe_ids)),
    ) as compileiq_search:
        result = compileiq_search.start()
        coverage = plans.require_complete_search(compileiq_search)
        selected = plans.select_best_result(compileiq_search, result)

    assert coverage["complete"]
    assert coverage["evaluation_count"] == len(plans.recipe_ids)
    assert {recipe_id for recipe_id, _ in materialized} == set(plans.recipe_ids)
    assert {dispatches for _, dispatches in materialized} == {1, 2, 3}
    assert selected.spec_id == plans.recipe_ids[0]
    np.testing.assert_array_equal(output.to_numpy(), (source_np * 2 + 3) * 4)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_complete_recipe_materializes_eight_stage_disjoint_fusion_fragments():
    count = 257

    @ti.kernel
    def stage(
        domain: ti.types.ndarray(dtype=ti.i32, ndim=1),
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        destination: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in domain:
            destination[i] = source[i] + 1

    symbolic_domain = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "domain", ti.i32, ndim=1)
    symbolic = tuple(
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, f"value_{index}", ti.i32, ndim=1)
        for index in range(9)
    )

    def build():
        builder = ti.graph.GraphBuilder()
        for index in range(8):
            builder.dispatch(
                stage,
                symbolic_domain,
                symbolic[index],
                symbolic[index + 1],
            )
        return builder.compile()

    baseline = build()
    public_search = baseline.definition.search_recipes(
        engine="compileiq",
        target=ti.graph.GraphOptimizationTarget(
            objectives=(("device_time_ns", "min"),)
        ),
        budget=ti.graph.GraphSearchBudget(evaluation_limit=108),
    )
    assert len(public_search.recipes) > 1
    assert len({recipe.planned_physical_id for recipe in public_search.recipes}) == len(
        public_search.recipes
    )
    assert public_search.baseline.manifest.is_baseline
    assert {
        family
        for recipe in public_search.recipes
        for family in recipe.manifest.families
    } == {"baseline", "map_fusion", "recording_partition"}
    with baseline.definition.materialize(public_search.baseline) as materialized:
        physical = materialized.executor.physical_plan()
        assert physical["logical_dispatch_count"] == 8
        assert physical["physical_dispatch_count"] == 8

    catalog = baseline.definition.recipe_catalog()
    fragments_by_group = {
        tuple(
            int(value)
            for value in fragment.provider_metadata["family_selection"][
                "source_key"
            ].removeprefix(
                "dispatches:"
            ).split(",")
        ): fragment
        for fragment in catalog.fragments
        if fragment.provider_metadata["family_selection"]["family"]
        == "map_fusion"
    }
    expected_physical = {
        ((0, 1, 2, 3), (4, 5, 6, 7)): 2,
        ((0, 1), (3, 4, 5), (6, 7)): 4,
    }
    domain = ti.ndarray(ti.i32, shape=count)
    arrays = tuple(ti.ndarray(ti.i32, shape=count) for _ in symbolic)
    source_np = np.arange(count, dtype=np.int32)
    arrays[0].from_numpy(source_np)
    arguments = {
        "domain": domain,
        **{f"value_{index}": array for index, array in enumerate(arrays)},
    }

    for source_groups, physical_dispatches in expected_physical.items():
        entry = catalog.compose(
            tuple(fragments_by_group[group].fragment_id for group in source_groups),
            stage="compatible-composition",
            parent_recipe_ids=(catalog.baseline.recipe.recipe_id,),
        )
        assert len(entry.recipe.fragments) == len(source_groups)
        with baseline.definition.materialize(entry.recipe) as materialized:
            graph = materialized.executor
            assert graph.physical_plan()["logical_dispatch_count"] == 8
            assert (
                graph.physical_plan()["physical_dispatch_count"] == physical_dispatches
            )
            graph.run(arguments)
            ti.sync()
            np.testing.assert_array_equal(arrays[-1].to_numpy(), source_np + 8)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_modified_compileiq_exhausts_complete_graph_bounded_recipes():
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])

    capacity = 257
    block_dim = 32

    @ti.kernel
    def publish(
        requested: ti.i32,
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.device_extent_publish(extent, capacity, requested)

    @ti.kernel
    def consume(
        extent: ti.types.ndarray(dtype=ti.i32, ndim=1),
        observed: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        ti.loop_config(block_dim=block_dim)
        for i in range(capacity):
            if i < ti.device_extent_count(extent):
                ti.atomic_add(observed[0], i + 1)

    requested_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "requested", ti.i32)
    extent_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "extent", ti.i32, ndim=1)
    first_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "first", ti.i32, ndim=1)
    second_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "second", ti.i32, ndim=1)

    def build(*, physical_grid="auto", consumer_count=2):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(publish, requested_arg, extent_arg)
        builder.dispatch_bounded(
            consume,
            extent_arg,
            first_arg,
            extent=extent_arg,
            capacity=capacity,
            block_dim=block_dim,
            physical_grid=physical_grid,
        )
        if consumer_count == 2:
            builder.dispatch_bounded(
                consume,
                extent_arg,
                second_arg,
                extent=extent_arg,
                capacity=capacity,
                block_dim=block_dim,
                physical_grid=physical_grid,
            )
        return builder.compile()

    baseline = build()
    plans = compileiq_recipe_search(baseline)
    manifest = plans.manifest()
    assert (
        plans.search_space.provider_namespace == "taichi_forge.graph.bounded_execution"
    )
    assert plans.search_space.domain_version == "graph-bounded-complete-recipe.v2"
    assert manifest["recipe_kind"] == "graph_bounded_execution"
    assert manifest["runtime_admission"] == "explicit_materialization_context_only"

    expected_strategies = (
        "logical_exact",
        "adaptive_per_node",
        "adaptive_grouped",
        "masked_capacity",
    )
    assert {
        recipe["bounded_recipe_manifest"]["strategy"] for recipe in manifest["recipes"]
    } == set(expected_strategies)

    extent = ti.DeviceExtent(capacity)
    first = ti.ndarray(ti.i32, shape=1)
    second = ti.ndarray(ti.i32, shape=1)
    arguments = {
        "requested": 17,
        "extent": extent,
        "first": first,
        "second": second,
    }
    materialized = {}
    semantic_graph_ids = set()

    def objective(graph, request):
        strategy = plans.recipe_manifest(request.recipe_id)["bounded_recipe_manifest"][
            "strategy"
        ]
        semantic_graph_ids.add(graph.definition.semantic_graph_id)
        first.fill(0)
        second.fill(0)
        graph.run(arguments)
        ti.sync()
        expected = 17 * 18 // 2
        assert int(first.to_numpy()[0]) == expected
        assert int(second.to_numpy()[0]) == expected
        materialized[strategy] = (
            graph.execution_stats().memory.persistent_bounded_control_bytes
        )
        return float(expected_strategies.index(strategy))

    with plans.compileiq_search(
        objective,
        budget=_search_budget(len(plans.recipe_ids)),
    ) as compileiq_search:
        result = compileiq_search.start()
        coverage = plans.require_complete_search(compileiq_search)
        selected = plans.select_best_result(compileiq_search, result)

    assert coverage["complete"]
    assert coverage["evaluation_count"] == 4
    assert set(materialized) == set(expected_strategies)
    assert semantic_graph_ids == {baseline.definition.semantic_graph_id}
    assert materialized["logical_exact"] == 0
    assert materialized["masked_capacity"] == 0
    assert materialized["adaptive_per_node"] > 0
    assert materialized["adaptive_grouped"] > 0
    assert selected.bounded_recipe_manifest.strategy == "logical_exact"

    forced = build(physical_grid="capacity")
    forced_search = compileiq_recipe_search(forced)
    assert forced_search.recipe_ids == (forced_search.baseline_recipe_id,)
    assert forced_search.manifest()["families"] == ()

    single = build(consumer_count=1)
    single_plans = compileiq_recipe_search(single)
    assert {
        recipe["bounded_recipe_manifest"]["strategy"]
        for recipe in single_plans.manifest()["recipes"]
    } == {"logical_exact", "adaptive_per_node", "masked_capacity"}
