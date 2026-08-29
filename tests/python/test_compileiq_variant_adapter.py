import sys
from types import ModuleType, SimpleNamespace

import pytest

import taichi_forge as ti
from taichi_forge.lang._compileiq_adapter import (
    _CompileIQSearchStage,
    _CompileIQVariantAdapter,
    _CompileIQVariantBundle,
    _CompileIQWinnerScope,
)
from taichi_forge.lang._kernel_variant_tuning import _KernelVariantSession
from tests import test_utils


class _FakeSession:
    def __init__(self):
        self._variants = {
            "v0-auto": SimpleNamespace(
                variant_id="v0-auto",
                compilation_id="c0",
                spec=SimpleNamespace(stable_payload="spec-v0-auto"),
            ),
            "v0-one": SimpleNamespace(
                variant_id="v0-one",
                compilation_id="c0",
                spec=SimpleNamespace(stable_payload="spec-v0-one"),
            ),
            "v1-auto": SimpleNamespace(
                variant_id="v1-auto",
                compilation_id="c1",
                spec=SimpleNamespace(stable_payload="spec-v1-auto"),
            ),
        }
        self.compilation_groups = (
            SimpleNamespace(
                compilation_id="c0",
                representative_variant_id="v0-auto",
                variant_ids=("v0-auto", "v0-one"),
            ),
            SimpleNamespace(
                compilation_id="c1",
                representative_variant_id="v1-auto",
                variant_ids=("v1-auto",),
            ),
        )

    def variant_ids(self):
        return tuple(self._variants)

    def variant(self, variant_id):
        return self._variants[variant_id]

    def bind(self, variant_id):
        return ("bound", variant_id)


class _FakeCartesianSession(_FakeSession):
    structural_mode = "cartesian"

    def __init__(self):
        self._variants = {}
        groups = []
        for min_blocks in (1, 2):
            for max_registers in (None, 24):
                variant_id = f"v-{min_blocks}-{max_registers}"
                selections = (
                    ("cuda_min_blocks_per_sm", min_blocks),
                    ("cuda_max_registers", max_registers),
                )
                self._variants[variant_id] = SimpleNamespace(
                    variant_id=variant_id,
                    compilation_id=variant_id,
                    spec=SimpleNamespace(stable_payload=variant_id),
                    selections=selections,
                )
                groups.append(
                    SimpleNamespace(
                        compilation_id=variant_id,
                        representative_variant_id=variant_id,
                        variant_ids=(variant_id,),
                    )
                )
        self.compilation_groups = tuple(groups)


def _install_fake_compileiq(monkeypatch):
    package = ModuleType("compileiq")
    search_spaces = ModuleType("compileiq.search_spaces")
    base = ModuleType("compileiq.search_spaces.base")
    compilers = ModuleType("compileiq.search_spaces.compilers")

    base.choice = lambda values: ("choice", tuple(values))

    class PtxasSearchSpace:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    compilers.PtxasSearchSpace = PtxasSearchSpace
    monkeypatch.setitem(sys.modules, "compileiq", package)
    monkeypatch.setitem(sys.modules, "compileiq.search_spaces", search_spaces)
    monkeypatch.setitem(sys.modules, "compileiq.search_spaces.base", base)
    monkeypatch.setitem(sys.modules, "compileiq.search_spaces.compilers", compilers)


def test_compileiq_adapter_keeps_structural_and_launch_stages_separate(monkeypatch):
    _install_fake_compileiq(monkeypatch)
    adapter = _CompileIQVariantAdapter(_FakeSession())

    assert adapter.variant_ids() == ("v0-auto", "v1-auto")
    assert adapter.variant_ids("launch", compilation_id="c0") == (
        "v0-auto",
        "v0-one",
    )
    assert adapter.variant_ids("full") == ("v0-auto", "v0-one", "v1-auto")
    assert adapter.search_space() == {
        "forge_variant": ("choice", ("v0-auto", "v1-auto"))
    }
    assert adapter.bind({"forge_variant": "v0-one"}) == ("bound", "v0-one")

    provider = adapter.ptxas_search_space(version="13.3", variant="sm89")
    assert provider.kwargs == {
        "version": "13.3",
        "variant": "sm89",
        "tag": "latest",
    }


def test_compileiq_adapter_validates_samples_and_emits_plain_manifest():
    adapter = _CompileIQVariantAdapter(_FakeSession())

    with pytest.raises(ValueError, match="launch search requires"):
        adapter.variant_ids("launch")
    with pytest.raises(KeyError, match="unknown Forge compilation group"):
        adapter.variant_ids("launch", compilation_id="missing")
    with pytest.raises(KeyError, match="require 'forge_variant'"):
        adapter.select({})
    with pytest.raises(KeyError, match="unknown Forge kernel variant"):
        adapter.select({"forge_variant": "missing"})

    manifest = adapter.manifest()
    assert manifest["schema_version"] == 3
    assert manifest["structural_variant_ids"] == ("v0-auto", "v1-auto")
    assert manifest["dimensions"] == ()
    assert manifest["tiling_recipes"] == ()
    assert [item["compilation_id"] for item in manifest["variants"]] == [
        "c0",
        "c0",
        "c1",
    ]
    assert all(item["selections"] == () for item in manifest["variants"])
    assert all(item["resource_envelope"] is None for item in manifest["variants"])
    assert all(item["tiling_recipe_id"] is None for item in manifest["variants"])


def test_compileiq_adapter_supports_named_user_space_dimension(monkeypatch):
    _install_fake_compileiq(monkeypatch)
    adapter = _CompileIQVariantAdapter(
        _FakeSession(), parameter="forge_variant_composite"
    )

    assert adapter.search_space("structural") == {
        "forge_variant_composite": (
            "choice",
            ("v0-auto", "v1-auto"),
        )
    }
    with pytest.raises(ValueError, match="Cartesian"):
        adapter.search_space("full")
    assert adapter.select({"forge_variant_composite": "v1-auto"}).variant_id == (
        "v1-auto"
    )
    assert adapter.manifest()["parameter"] == "forge_variant_composite"


def test_compileiq_adapter_exposes_legal_cartesian_refinement_axes(monkeypatch):
    _install_fake_compileiq(monkeypatch)
    adapter = _CompileIQVariantAdapter(_FakeCartesianSession())

    assert adapter.factorized_search_space() == {
        "cuda_min_blocks_per_sm": ("choice", (1, 2)),
        "cuda_max_registers": ("choice", ("auto", 24)),
    }
    selected = adapter.select_factorized(
        {"cuda_min_blocks_per_sm": 1, "cuda_max_registers": 24}
    )
    assert selected.variant_id == "v-1-24"
    assert adapter.bind_factorized(
        {"cuda_min_blocks_per_sm": 2, "cuda_max_registers": "auto"}
    ) == ("bound", "v-2-None")
    assert adapter.manifest()["factorized_refinement_available"]

    with pytest.raises(ValueError, match="exactly match"):
        adapter.select_factorized({"cuda_min_blocks_per_sm": 1})
    with pytest.raises(ValueError, match="cartesian refinement"):
        _CompileIQVariantAdapter(_FakeSession()).factorized_search_space()


@pytest.mark.parametrize("parameter", ["", "1variant", "forge.variant", None])
def test_compileiq_adapter_rejects_invalid_parameter_name(parameter):
    with pytest.raises((TypeError, ValueError)):
        _CompileIQVariantAdapter(_FakeSession(), parameter=parameter)


def test_compileiq_bundle_composes_forge_only_kernel_dimensions(monkeypatch):
    _install_fake_compileiq(monkeypatch)
    bundle = _CompileIQVariantBundle(
        {"composite": _FakeSession(), "combine": _FakeSession()}
    )

    assert bundle.kernel_names == ("composite", "combine")
    assert bundle.search_space("structural") == {
        "forge_variant_composite": ("choice", ("v0-auto", "v1-auto")),
        "forge_variant_combine": ("choice", ("v0-auto", "v1-auto")),
    }
    params = {
        "forge_variant_composite": "v0-one",
        "forge_variant_combine": "v1-auto",
    }
    assert dict(bundle.bind(params)) == {
        "composite": ("bound", "v0-one"),
        "combine": ("bound", "v1-auto"),
    }
    manifest = bundle.manifest()
    assert manifest["provider"] == "compileiq_user_space"
    assert not manifest["uses_ptxas_search_space"]
    assert manifest["kernels"]["combine"]["parameter"] == (
        "forge_variant_combine"
    )


def test_compileiq_bundle_rejects_invalid_shape_and_ambiguous_launch_stage():
    with pytest.raises(TypeError):
        _CompileIQVariantBundle([])
    with pytest.raises(ValueError, match="at least one"):
        _CompileIQVariantBundle({})
    with pytest.raises(ValueError, match="kernel names"):
        _CompileIQVariantBundle({"bad-name": _FakeSession()})

    bundle = _CompileIQVariantBundle({"kernel": _FakeSession()})
    with pytest.raises(ValueError, match="Cartesian search is disabled"):
        bundle.search_space("launch")


def test_compileiq_adapter_builds_balanced_exhaustive_stage_schedule():
    adapter = _CompileIQVariantAdapter(_FakeSession())

    schedule = adapter.paired_schedule(blocks=4)
    assert len(schedule) == 8
    assert [trial.variant_id for trial in schedule] == [
        "v0-auto",
        "v0-auto",
        "v0-auto",
        "v0-auto",
        "v1-auto",
        "v1-auto",
        "v1-auto",
        "v1-auto",
    ]
    assert [trial.order for trial in schedule[:4]] == [
        ("baseline", "candidate"),
        ("candidate", "baseline"),
        ("baseline", "candidate"),
        ("candidate", "baseline"),
    ]

    launch = adapter.paired_schedule("launch", compilation_id="c0", blocks=2)
    assert [(trial.variant_id, trial.block) for trial in launch] == [
        ("v0-auto", 0),
        ("v0-auto", 1),
        ("v0-one", 0),
        ("v0-one", 1),
    ]


def test_compileiq_adapter_ranks_complete_paired_evidence_fail_closed():
    adapter = _CompileIQVariantAdapter(_FakeSession())
    ranked = adapter.rank_paired(
        {
            "v0-auto": (0.93, 1.01, 0.95, 0.97),
            "v1-auto": (0.98, 0.99, 0.97, 0.96),
        },
        blocks=4,
    )

    assert [item.variant_id for item in ranked] == ["v1-auto", "v0-auto"]
    assert ranked[0].worst_ratio == pytest.approx(0.99)
    assert ranked[0].median_ratio == pytest.approx(0.975)
    assert ranked[0].worst_positive
    assert ranked[1].worst_ratio == pytest.approx(1.01)
    assert not ranked[1].worst_positive


@pytest.mark.parametrize("blocks", [0, 1, 3, True])
def test_compileiq_adapter_rejects_unbalanced_pairing(blocks):
    adapter = _CompileIQVariantAdapter(_FakeSession())
    with pytest.raises((TypeError, ValueError)):
        adapter.paired_schedule(blocks=blocks)


def test_compileiq_adapter_rejects_incomplete_or_invalid_evidence():
    adapter = _CompileIQVariantAdapter(_FakeSession())
    with pytest.raises(ValueError, match="missing"):
        adapter.rank_paired({"v0-auto": (0.9, 0.9)}, blocks=2)
    with pytest.raises(ValueError, match="exactly 2"):
        adapter.rank_paired({"v0-auto": (0.9,), "v1-auto": (0.9, 0.9)}, blocks=2)
    with pytest.raises(ValueError, match="finite and positive"):
        adapter.rank_paired(
            {"v0-auto": (0.9, float("nan")), "v1-auto": (0.9, 0.9)},
            blocks=2,
        )


def _winner_scope(final_candidate_id, provider_scope_id="explicit-acf:sha256"):
    return _CompileIQWinnerScope(
        final_candidate_id=final_candidate_id,
        forge_specialization_id="kernel:contact-v3",
        workload_profile_id="tlw1:production-sap",
        shape_scope_id="dofs=32768/contacts=8192",
        replay_scope_id="fresh-process-reset-v2",
        runtime_scope_id="cuda:uuid:driver",
        compiler_scope_id="llvm20:ptxas13.3",
        provider_scope_id=provider_scope_id,
        variant_manifest_id="manifest:sha256",
    )


def test_compileiq_staged_plan_exhausts_bounded_groups_before_qualification():
    adapter = _CompileIQVariantAdapter(_FakeSession())
    plan = adapter.staged_plan(
        structural_blocks=4,
        launch_blocks=4,
        qualification_blocks=10,
        structural_shortlist=2,
    )
    structural = {
        "v0-auto": (0.98, 0.99, 0.97, 0.98),
        "v1-auto": (0.96, 0.97, 0.98, 0.97),
    }

    assert plan.structural_stage.candidate_kind == "forge_structural"
    assert len(plan.structural_stage.schedule) == 8
    assert plan.shortlisted_compilation_ids(structural) == ("c1", "c0")
    launch_stages = plan.launch_stages(structural)
    assert [stage.compilation_id for stage in launch_stages] == ["c1", "c0"]
    assert all(len(stage.candidate_ids) <= 32 for stage in launch_stages)

    launch = {
        "c1": {"v1-auto": (1.0, 1.0, 1.0, 1.0)},
        "c0": {
            "v0-auto": (1.0, 1.0, 1.0, 1.0),
            "v0-one": (0.96, 0.97, 0.95, 0.96),
        },
    }
    finalist_variants = plan.launch_finalists(structural, launch)
    assert finalist_variants == ("v1-auto", "v0-one")
    finalists = tuple(
        plan.final_candidate(variant_id) for variant_id in finalist_variants
    )
    assert len(plan.qualification_stage(finalists).schedule) == 20

    with pytest.raises(RuntimeError, match="disabled by default"):
        plan.ptxas_stage("v0-one", ("acf-baseline", "acf-candidate"))
    assert plan.manifest()["compileiq_stage"] == "forge_variants_only"
    assert not plan.manifest()["ptxas_search_enabled"]

    ptxas_plan = adapter.staged_plan(include_ptxas_search=True)
    ptxas = ptxas_plan.ptxas_stage(
        "v0-one", ("acf-baseline", "acf-candidate")
    )
    assert ptxas.candidate_kind == "ptxas_control"
    assert ptxas.forge_variant_id == "v0-one"
    assert ptxas_plan.manifest()["compileiq_stage"] == "optional_ptxas_control_only"
    assert ptxas_plan.manifest()["ptxas_search_enabled"]


def test_compileiq_staged_plan_requires_boolean_ptxas_opt_in():
    adapter = _CompileIQVariantAdapter(_FakeSession())
    with pytest.raises(TypeError, match="include_ptxas_search"):
        adapter.staged_plan(include_ptxas_search="yes")


def test_compileiq_final_qualification_binds_exact_scope_and_worst_gate():
    adapter = _CompileIQVariantAdapter(_FakeSession())
    plan = adapter.staged_plan(qualification_blocks=10)
    finalists = (
        plan.final_candidate("v0-one", "acf-fast"),
        plan.final_candidate("v1-auto", "baseline"),
    )
    candidate_ids = tuple(candidate.identity for candidate in finalists)
    decision = plan.qualify(
        {
            candidate_ids[0]: (0.96,) * 10,
            candidate_ids[1]: (0.95,) * 9 + (1.01,),
        },
        finalists,
        scopes={
            candidate_ids[0]: _winner_scope(
                candidate_ids[0], "acf-fast:sha256"
            ),
            candidate_ids[1]: _winner_scope(
                candidate_ids[1], "driver-baseline"
            ),
        },
        correctness={candidate_ids[0]: True, candidate_ids[1]: True},
        memory_stable={candidate_ids[0]: True, candidate_ids[1]: True},
    )

    assert decision.admitted
    assert decision.selected_candidate_id == candidate_ids[0]
    assert decision.selected_forge_variant_id == "v0-one"
    assert decision.selected_provider_candidate_id == "acf-fast"
    assert decision.scope_id == _winner_scope(
        candidate_ids[0], "acf-fast:sha256"
    ).identity
    by_id = {item.variant_id: item for item in decision.evidence}
    assert by_id[candidate_ids[0]].ratio_cv == pytest.approx(0.0)
    assert by_id[candidate_ids[1]].worst_ratio == pytest.approx(1.01)

    rejected_candidate = plan.final_candidate("v0-one")
    rejected = plan.qualify(
        {rejected_candidate.identity: (0.99,) * 9 + (1.001,)},
        (rejected_candidate,),
        scopes={
            rejected_candidate.identity: _winner_scope(
                rejected_candidate.identity, "baseline"
            )
        },
        correctness={rejected_candidate.identity: True},
        memory_stable={rejected_candidate.identity: True},
    )
    assert not rejected.admitted
    assert rejected.selected_candidate_id is None
    assert "worst-positive" in rejected.reason


def test_compileiq_staged_contract_rejects_oversized_or_weak_final_stage():
    with pytest.raises(ValueError, match="at most 32"):
        _CompileIQSearchStage(
            stage_id="oversized",
            candidate_kind="forge_structural",
            candidate_ids=tuple(f"v{index}" for index in range(33)),
            blocks=2,
        )
    with pytest.raises(ValueError, match="at least 10"):
        _CompileIQSearchStage(
            stage_id="weak-final",
            candidate_kind="qualification",
            candidate_ids=("v0",),
            blocks=8,
        )
    with pytest.raises(ValueError, match="non-empty single-line"):
        _CompileIQWinnerScope(
            final_candidate_id="candidate",
            forge_specialization_id="",
            workload_profile_id="profile",
            shape_scope_id="shape",
            replay_scope_id="replay",
            runtime_scope_id="runtime",
            compiler_scope_id="compiler",
            provider_scope_id="provider",
            variant_manifest_id="manifest",
        )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_compileiq_staged_plan_covers_real_bounded_cuda_variant_groups():
    count = 4096
    values = ti.ndarray(ti.i32, shape=count)
    result = ti.field(ti.i32, shape=())

    @ti.kernel
    def reduce(inp: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(count):
            result[None] += inp[i]

    adapter = _CompileIQVariantAdapter(_KernelVariantSession(reduce, (values,)))
    plan = adapter.staged_plan(structural_shortlist=2)
    structural_ids = plan.structural_stage.candidate_ids
    measurements = {
        candidate_id: (1.0 + index * 0.001,) * 4
        for index, candidate_id in enumerate(structural_ids)
    }
    launch_stages = plan.launch_stages(measurements)

    assert len(structural_ids) == 24
    assert len(plan.structural_stage.schedule) == 96
    assert len(launch_stages) == 2
    assert all(len(stage.candidate_ids) == 16 for stage in launch_stages)
    assert all(len(stage.schedule) == 64 for stage in launch_stages)
    manifest = adapter.manifest()
    recipes = {
        recipe["recipe_id"]: recipe for recipe in manifest["tiling_recipes"]
    }
    assert recipes
    assert all(
        variant["tiling_recipe_id"] in recipes
        for variant in manifest["variants"]
    )
    selected = {
        recipes[variant["tiling_recipe_id"]]["strategy"]
        for variant in manifest["variants"]
    }
    assert selected == {"baseline", "thread_coarsened"}
    assert {
        recipe["strategy"]
        for recipe in recipes.values()
        if recipe["availability"] == "unsupported"
    } == {"shared_staged", "layout_specialized"}
