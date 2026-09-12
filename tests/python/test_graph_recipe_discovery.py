from types import SimpleNamespace

import pytest

from taichi_forge.graph._recipes.discovery import dispatch_source_explanation


def _definition(backend="cuda", count=2, bindings=()):
    return SimpleNamespace(
        backend=backend,
        semantic_root={
            "kind": "sequential_region",
            "children": [
                {"kind": "dispatch", "bindings": bindings} for _ in range(count)
            ],
        },
        sources=tuple(
            SimpleNamespace(
                kind="dispatch",
                semantic_identity="kernel",
                region_id=f"region:{i}",
                path=f"graph/{i}:dispatch",
            )
            for i in range(count)
        ),
    )


def _source(key, kinds=("serial", "range_for"), failures=None):
    return SimpleNamespace(
        _candidates=(),
        _candidate_failures=failures or {},
        _recipe_source_key=key,
        kernel_fn=_source,
        baseline_plan=SimpleNamespace(
            semantic_kernel_identity="kernel",
            tasks=tuple(SimpleNamespace(task_kind=kind) for kind in kinds),
        ),
        baseline_manifests=(),
        candidates=lambda: pytest.fail("report must not compile"),
    )


def _explain(definition, sources=(), family="offload_phase_fusion"):
    return dispatch_source_explanation(
        definition, sources, family=family, supported_scope="test scope"
    )


def test_discovery_attributes_repeated_kernel_attempts_to_distinct_regions():
    report = _explain(_definition(), (_source("first"), _source("second")))
    first, second = report["sources"]
    assert first["matching_region_ids"] == ("region:0",)
    assert second["matching_region_ids"] == ("region:1",)
    assert first["matching_region_paths"] == ("graph/0:dispatch",)
    assert first["domain_exclusions"][0]["reason_code"] == "no_adjacent_range_tasks"
    assert not report["unregistered_regions"]


def test_discovery_names_bound_data_oriented_kernel_without_calling_it():
    source = _source("bound")
    source.kernel_fn = SimpleNamespace(__name__=None, _primal=SimpleNamespace(func=_source))
    assert _explain(_definition(count=1), (source,))["sources"][0]["kernel_name"] == _source.__qualname__


@pytest.mark.parametrize(
    "backend,code",
    [("cuda", "insufficient_symbolic_ndarrays"), ("vulkan", "unsupported_backend")],
)
def test_discovery_explains_every_unregistered_field_dispatch(backend, code):
    report = _explain(_definition(backend), family="graph_memory")
    assert [item["region_id"] for item in report["unregistered_regions"]] == [
        "region:0",
        "region:1",
    ]
    assert all(item["reason_code"] == code for item in report["unregistered_regions"])
    if backend == "cuda":
        assert "this dispatch has 0" in report["unregistered_regions"][0]["reason"]


def test_discovery_preserves_native_access_reason_and_exact_task_group():
    reason = "offload phase fusion rejected source tasks 1..2: non-pointwise dense field access (rank=0, index=scalar)"
    source = _source(
        "fusion", ("serial", "range_for", "range_for"), {((1, 2),): reason}
    )
    report = _explain(_definition(count=1), (source,))
    rejection = report["sources"][0]["generation_rejections"][0]
    assert rejection["reason"] == reason
    assert rejection["category"] == "fusion_legality_rejection"
    assert rejection["source_task_indices"] == (1, 2)
    assert report["sources"][0]["status"] == "candidate_generation_rejected"


def test_discovery_does_not_classify_unexpected_compiler_failure_as_inapplicable():
    source = _source(
        "fusion", failures={(): "internal offload topology assertion failed"}
    )
    rejection = _explain(_definition(count=1), (source,))["sources"][0][
        "generation_rejections"
    ][0]
    assert "category" not in rejection
    assert "internal offload topology assertion failed" == rejection["reason"]


def test_discovery_does_not_infer_missing_ndarrays_when_symbolic_pair_exists():
    bindings = (
        {"kind": "ndarray", "name": "input"},
        {"kind": "ndarray", "name": "output"},
    )
    report = _explain(_definition(count=1, bindings=bindings), family="graph_memory")
    assert report["unregistered_regions"][0]["reason_code"] == "source_not_registered"
