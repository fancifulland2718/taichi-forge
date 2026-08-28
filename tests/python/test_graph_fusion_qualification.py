import json

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from taichi_forge.graph._graph import (
    _graph_fusion_runtime_scope,
)
from taichi_forge.graph._optimization import (
    _GRAPH_FUSION_QUALIFICATION_SCHEMA,
    _GraphFusionQualificationCache,
)
from tests import test_utils


_CACHE_ENV = "TAICHI_FORGE_INTERNAL_GRAPH_FUSION_QUALIFICATION"
_REPLAY_ENV = "TAICHI_FORGE_INTERNAL_GRAPH_FUSION_EXPECTED_REPLAYS"
_FUSION_ENV = "TAICHI_FORGE_INTERNAL_MAP_FUSION"


def _entry(
    *,
    semantic_plan_id="semantic-plan:" + "1" * 24,
    baseline_execution_identity="baseline-identity",
    selected_spec_id="executable:" + "2" * 24,
    execution_identity="selected-identity",
    source_commit="a" * 40,
    runtime_scope=None,
    minimum_expected_replays=100,
    extent_min=256,
    extent_max=512,
):
    return {
        "semantic_plan_id": semantic_plan_id,
        "backend": "cuda",
        "baseline_execution_identity": baseline_execution_identity,
        "selected_spec_id": selected_spec_id,
        "execution_identity": execution_identity,
        "source_commit": source_commit,
        "runtime_scope": (
            {"core_commit": source_commit} if runtime_scope is None else runtime_scope
        ),
        "binding_scope": [
            {
                "name": "values",
                "kind": "ndarray",
                "dtype": "f32",
                "rank": 2,
                "element_shape": [],
                "shape_min": [extent_min, 3],
                "shape_max": [extent_max, 3],
            },
            {
                "name": "count",
                "kind": "scalar",
                "minimum": extent_min,
                "maximum": extent_max,
            },
        ],
        "minimum_expected_replays": minimum_expected_replays,
        "evidence_id": "fresh-process-abba:test",
        "qualification": {
            "correctness": True,
            "memory_stable": True,
            "worst_positive": True,
        },
    }


def _cache(*entries):
    return {
        "schema": _GRAPH_FUSION_QUALIFICATION_SCHEMA,
        "entries": list(entries),
    }


def test_fusion_qualification_matches_exact_runtime_and_binding_scope():
    source_commit = "a" * 40
    value = _entry(source_commit=source_commit)
    cache = _GraphFusionQualificationCache.from_dict(_cache(value))
    entry, reason = cache.select(
        semantic_plan_id=value["semantic_plan_id"],
        backend="cuda",
        source_commit=source_commit,
        runtime_scope={"core_commit": source_commit},
        bindings={
            "values": {
                "kind": "ndarray",
                "dtype": "f32",
                "rank": 2,
                "shape": (384, 3),
                "element_shape": (),
            },
            "count": {"kind": "scalar", "value": 384},
        },
        expected_replays=100,
    )
    assert reason == "qualified"
    assert entry.selected_spec_id == value["selected_spec_id"]

    entry, reason = cache.select(
        semantic_plan_id=value["semantic_plan_id"],
        backend="cuda",
        source_commit=source_commit,
        runtime_scope={"core_commit": source_commit},
        bindings={
            "values": {
                "kind": "ndarray",
                "dtype": "f32",
                "rank": 2,
                "shape": (384, 3),
                "element_shape": (),
            },
            "count": {"kind": "scalar", "value": 384},
            "unqualified": {"kind": "scalar", "value": 1},
        },
        expected_replays=100,
    )
    assert entry is None
    assert reason == "no_exact_qualification"

    entry, reason = cache.select(
        semantic_plan_id=value["semantic_plan_id"],
        backend="cuda",
        source_commit=source_commit,
        runtime_scope={"core_commit": source_commit},
        bindings={
            "values": {
                "kind": "ndarray",
                "dtype": "f32",
                "rank": 2,
                "shape": (128, 3),
                "element_shape": (),
            },
            "count": {"kind": "scalar", "value": 128},
        },
        expected_replays=99,
    )
    assert entry is None
    assert reason == "no_exact_qualification"


@pytest.mark.parametrize("gate", ["correctness", "memory_stable", "worst_positive"])
def test_fusion_qualification_rejects_any_failed_gate(gate):
    value = _entry()
    value["qualification"][gate] = False
    with pytest.raises(ValueError, match="admission gate"):
        _GraphFusionQualificationCache.from_dict(_cache(value))


def test_fusion_qualification_rejects_ambiguous_exact_entries():
    value = _entry()
    second = _entry()
    second["evidence_id"] = "fresh-process-abba:second"
    cache = _GraphFusionQualificationCache.from_dict(_cache(value, second))
    selected, reason = cache.select(
        semantic_plan_id=value["semantic_plan_id"],
        backend="cuda",
        source_commit=value["source_commit"],
        runtime_scope=value["runtime_scope"],
        bindings={
            "values": {
                "kind": "ndarray",
                "dtype": "f32",
                "rank": 2,
                "shape": (384, 3),
                "element_shape": (),
            },
            "count": {"kind": "scalar", "value": 384},
        },
        expected_replays=100,
    )
    assert selected is None
    assert reason == "ambiguous_qualification"


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_qualified_fusion_materializes_once_and_falls_back_by_binding(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(_FUSION_ENV, "baseline")

    @ti.kernel
    def stage_one(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        first: ti.types.ndarray(dtype=ti.i32, ndim=1),
        count: ti.i32,
    ):
        for i in range(count):
            first[i] = source[i] * 2

    @ti.kernel
    def stage_two(
        first: ti.types.ndarray(dtype=ti.i32, ndim=1),
        second: ti.types.ndarray(dtype=ti.i32, ndim=1),
        count: ti.i32,
    ):
        for i in range(count):
            second[i] = first[i] + 3

    @ti.kernel
    def stage_three(
        second: ti.types.ndarray(dtype=ti.i32, ndim=1),
        third: ti.types.ndarray(dtype=ti.i32, ndim=1),
        count: ti.i32,
    ):
        for i in range(count):
            third[i] = second[i] * 4

    @ti.kernel
    def stage_four(
        third: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
        count: ti.i32,
    ):
        for i in range(count):
            output[i] = third[i] - 5

    array_names = ("source", "first", "second", "third", "output")
    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in array_names
    }
    symbolic["count"] = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)

    def build():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            stage_one,
            symbolic["source"],
            symbolic["first"],
            symbolic["count"],
        )
        builder.dispatch(
            stage_two,
            symbolic["first"],
            symbolic["second"],
            symbolic["count"],
        )
        builder.dispatch(
            stage_three,
            symbolic["second"],
            symbolic["third"],
            symbolic["count"],
        )
        builder.dispatch(
            stage_four,
            symbolic["third"],
            symbolic["output"],
            symbolic["count"],
        )
        return builder.compile()

    probe = build()
    space = probe._executable_optimization_space
    selected = next(
        spec
        for spec in space.candidates
        if any(recipe.startswith("fusion:map4:") for recipe in spec.fusion_recipe_ids)
    )
    runtime_scope = _graph_fusion_runtime_scope("cuda")
    source_commit = _ti_core.get_commit_hash().lower()
    cache_path = tmp_path / "fusion-qualification.json"
    cache_path.write_text(
        json.dumps(
            _cache(
                {
                    "semantic_plan_id": space.semantic_plan_id,
                    "backend": "cuda",
                    "baseline_execution_identity": (space.baseline.execution_identity),
                    "selected_spec_id": selected.spec_id,
                    "execution_identity": selected.execution_identity,
                    "source_commit": source_commit,
                    "runtime_scope": runtime_scope,
                    "binding_scope": [
                        *[
                            {
                                "name": name,
                                "kind": "ndarray",
                                "dtype": "i32",
                                "rank": 1,
                                "element_shape": [],
                                "shape_min": [257],
                                "shape_max": [257],
                            }
                            for name in array_names
                        ],
                        {
                            "name": "count",
                            "kind": "scalar",
                            "minimum": 257,
                            "maximum": 257,
                        },
                    ],
                    "minimum_expected_replays": 10,
                    "evidence_id": "fresh-process-abba:integration",
                    "qualification": {
                        "correctness": True,
                        "memory_stable": True,
                        "worst_positive": True,
                    },
                }
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(_CACHE_ENV, str(cache_path))
    monkeypatch.setenv(_REPLAY_ENV, "10")

    graph = build()
    assert graph.physical_plan()["physical_dispatch_count"] == 4
    assert graph._qualified_fusion_stats["materializations"] == 0

    def allocate(size):
        arrays = {name: ti.ndarray(ti.i32, shape=size) for name in array_names}
        arrays["source"].from_numpy(np.arange(size, dtype=np.int32))
        for name in array_names[1:]:
            arrays[name].fill(0)
        return arrays

    arrays = allocate(257)
    runtime_args = {**arrays, "count": 257}
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    with ti.ad.Tape(loss=loss):
        with pytest.raises(RuntimeError, match="primal-only"):
            graph.run(runtime_args)
    assert graph._qualified_fusion_stats["materializations"] == 0

    graph.run(runtime_args)
    graph.run(runtime_args)
    graph.submit(runtime_args).wait()
    np.testing.assert_array_equal(
        arrays["output"].to_numpy(), np.arange(257, dtype=np.int32) * 8 + 7
    )
    stats = graph._qualified_fusion_stats
    assert stats["qualified_selections"] == 3
    assert stats["materializations"] == 1
    assert stats["retained_variants"] == 1
    assert stats["last_reason"] == "qualified"

    first_async = allocate(257)
    second_async = allocate(257)
    first_async["source"].fill(3)
    second_async["source"].fill(9)
    first_ticket = graph.submit({**first_async, "count": 257})
    second_ticket = graph.submit({**second_async, "count": 257})
    first_ticket.wait()
    second_ticket.wait()
    np.testing.assert_array_equal(
        first_async["output"].to_numpy(), np.full(257, 31, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        second_async["output"].to_numpy(), np.full(257, 79, dtype=np.int32)
    )
    for _ in range(32):
        graph.run(runtime_args)
    stats = graph._qualified_fusion_stats
    assert stats["qualified_selections"] == 37
    assert stats["materializations"] == 1
    assert stats["retained_variants"] == 1

    smaller = allocate(128)
    graph.run({**smaller, "count": 128})
    np.testing.assert_array_equal(
        smaller["output"].to_numpy(), np.arange(128, dtype=np.int32) * 8 + 7
    )
    stats = graph._qualified_fusion_stats
    assert stats["baseline_fallbacks"] == 1
    assert stats["materializations"] == 1
    assert stats["last_reason"] == "no_exact_qualification"

    ti.reset()
    with pytest.raises(RuntimeError, match="reset|valid|reinitialization"):
        graph.run(runtime_args)
