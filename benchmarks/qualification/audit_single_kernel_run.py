from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Sequence


SCHEMA = "taichi_forge.single_kernel_microbench.v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-9, abs_tol=1.0e-12)


def _percentile(values: Sequence[float], percent: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_median(speedups: Sequence[float], seed: int) -> dict[str, float]:
    logs = [math.log(float(value)) for value in speedups]
    median = statistics.median(logs)
    if len(logs) == 1:
        low = high = logs[0]
    else:
        rng = random.Random(seed)
        bootstrapped = [statistics.median(rng.choice(logs) for _ in logs) for _ in range(10_000)]
        low = _percentile(bootstrapped, 2.5)
        high = _percentile(bootstrapped, 97.5)
    return {
        "median_speedup_x": math.exp(median),
        "bootstrap_95_low_x": math.exp(low),
        "bootstrap_95_high_x": math.exp(high),
    }


def _check(condition: bool, name: str, failures: list[str]) -> None:
    if not condition:
        failures.append(name)


def _reported_exact_i32_vector_valid(vector: dict[str, Any]) -> bool:
    actual = vector.get("actual_values_i32")
    expected = vector.get("expected_values_i32")
    if (
        not isinstance(actual, list)
        or not isinstance(expected, list)
        or not actual
        or len(actual) != len(expected)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < -(2**31) or value >= 2**31
            for value in actual + expected
        )
    ):
        return False

    def evidence(values: list[int]) -> dict[str, Any]:
        payload = bytearray()
        for value in values:
            payload.extend(int(value).to_bytes(4, "little", signed=True))
        count = len(values)
        sample_indices = sorted(
            set(
                (
                    0,
                    count // 4,
                    count // 2,
                    (3 * count) // 4,
                    count - 1,
                )
            )
        )
        return {
            "count": count,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "sum": sum(values),
            "minimum": min(values),
            "maximum": max(values),
            "sample_indices": sample_indices,
            "samples": [values[index] for index in sample_indices],
        }

    actual_evidence = evidence(actual)
    expected_evidence = evidence(expected)
    mismatches = [index for index, (left, right) in enumerate(zip(actual, expected)) if left != right]
    return bool(
        vector.get("count") == actual_evidence["count"]
        and vector.get("expected_count") == expected_evidence["count"]
        and vector.get("actual_sha256") == actual_evidence["sha256"]
        and vector.get("expected_sha256") == expected_evidence["sha256"]
        and vector.get("actual_sum") == actual_evidence["sum"]
        and vector.get("expected_sum") == expected_evidence["sum"]
        and vector.get("actual_minimum") == actual_evidence["minimum"]
        and vector.get("expected_minimum") == expected_evidence["minimum"]
        and vector.get("actual_maximum") == actual_evidence["maximum"]
        and vector.get("expected_maximum") == expected_evidence["maximum"]
        and vector.get("sample_indices") == actual_evidence["sample_indices"]
        and vector.get("expected_sample_indices") == expected_evidence["sample_indices"]
        and vector.get("actual_samples") == actual_evidence["samples"]
        and vector.get("expected_samples") == expected_evidence["samples"]
        and vector.get("mismatch_count") == len(mismatches)
        and vector.get("first_mismatch") == (None if not mismatches else mismatches[0])
        and not mismatches
    )


def _endpoint_equivalent(left_result: dict[str, Any], right_result: dict[str, Any]) -> bool:
    if left_result["operation"] in ("fill", "copy", "saxpy", "stencil2d", "reduce_chunks"):
        for validation_name in ("validation_before", "validation_after"):
            left_validation = left_result[validation_name]
            right_validation = right_result[validation_name]
            if not left_validation.get("passed") or not right_validation.get("passed"):
                return False
            left = left_validation.get("endpoint_fingerprint") or {}
            right = right_validation.get("endpoint_fingerprint") or {}
            if not left.get("finite") or not right.get("finite"):
                return False
            if left.get("count") != right.get("count") or left.get("sample_indices") != right.get("sample_indices"):
                return False
            if len(left.get("sample_values", [])) != len(left.get("sample_indices", [])) or len(
                right.get("sample_values", [])
            ) != len(right.get("sample_indices", [])):
                return False
            count = int(left["count"])
            element_tolerance = 2.0 * max(
                float(left_validation.get("effective_tolerance", 0.0)),
                float(right_validation.get("effective_tolerance", 0.0)),
            )
            for key in ("minimum", "maximum"):
                if not math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=element_tolerance):
                    return False
            if not math.isclose(
                float(left["sum"]), float(right["sum"]), rel_tol=0.0, abs_tol=element_tolerance * max(1, count)
            ):
                return False
            if any(
                not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=element_tolerance)
                for a, b in zip(left["sample_values"], right["sample_values"])
            ):
                return False
        return True
    if left_result["operation"] == "native_reduce":
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            if not left.get("passed") or not right.get("passed"):
                return False
            for value in (left, right):
                if not all(isinstance(value.get(key), int) for key in ("actual", "expected", "absolute_error")):
                    return False
                if value["actual"] != value["expected"] or value["absolute_error"] != 0:
                    return False
            if left["actual"] != right["actual"] or left["expected"] != right["expected"]:
                return False
        return True
    if left_result["operation"] == "native_transform":
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            for value in (left, right):
                if not value.get("passed") or value.get("comparison") != "exact_i32_affine_transform":
                    return False
                if (
                    not isinstance(value.get("count"), int)
                    or value["count"] <= 0
                    or value.get("mismatch_count") != 0
                    or value.get("first_mismatch") is not None
                ):
                    return False
                if (
                    not isinstance(value.get("actual_sha256"), str)
                    or len(value["actual_sha256"]) != 64
                    or value["actual_sha256"] != value.get("expected_sha256")
                ):
                    return False
                for suffix in ("sum", "minimum", "maximum", "samples"):
                    if value.get(f"actual_{suffix}") != value.get(f"expected_{suffix}"):
                        return False
                if len(value.get("sample_indices", [])) != len(value.get("actual_samples", [])):
                    return False
            for key in (
                "count",
                "actual_sha256",
                "expected_sha256",
                "actual_sum",
                "expected_sum",
                "actual_minimum",
                "expected_minimum",
                "actual_maximum",
                "expected_maximum",
                "sample_indices",
                "actual_samples",
                "expected_samples",
            ):
                if left.get(key) != right.get(key):
                    return False
        return True
    if left_result["operation"] in ("native_gather", "native_scatter"):
        operation = left_result["operation"].removeprefix("native_")
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            for value in (left, right):
                if not value.get("passed") or value.get("comparison") != f"exact_i32_{operation}":
                    return False
                if (
                    not isinstance(value.get("count"), int)
                    or value["count"] <= 0
                    or value.get("mismatch_count") != 0
                    or value.get("first_mismatch") is not None
                ):
                    return False
                if (
                    not isinstance(value.get("actual_sha256"), str)
                    or len(value["actual_sha256"]) != 64
                    or value["actual_sha256"] != value.get("expected_sha256")
                ):
                    return False
                for suffix in ("sum", "minimum", "maximum", "samples"):
                    if value.get(f"actual_{suffix}") != value.get(f"expected_{suffix}"):
                        return False
                if len(value.get("sample_indices", [])) != len(value.get("actual_samples", [])):
                    return False
            for key in (
                "count",
                "actual_sha256",
                "expected_sha256",
                "actual_sum",
                "expected_sum",
                "actual_minimum",
                "expected_minimum",
                "actual_maximum",
                "expected_maximum",
                "sample_indices",
                "actual_samples",
                "expected_samples",
            ):
                if left.get(key) != right.get(key):
                    return False
        return True
    if left_result["operation"] == "native_compact":
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            for value in (left, right):
                if not value.get("passed") or value.get("comparison") != "exact_stable_i32_compact":
                    return False
                if (
                    not isinstance(value.get("actual_count"), int)
                    or value["actual_count"] <= 0
                    or value["actual_count"] != value.get("expected_count")
                    or value.get("mismatch_count") != 0
                    or value.get("first_mismatch") is not None
                ):
                    return False
                if (
                    not isinstance(value.get("actual_sha256"), str)
                    or len(value["actual_sha256"]) != 64
                    or value["actual_sha256"] != value.get("expected_sha256")
                ):
                    return False
                for suffix in ("sum", "minimum", "maximum", "samples"):
                    if value.get(f"actual_{suffix}") != value.get(f"expected_{suffix}"):
                        return False
                if len(value.get("sample_indices", [])) != len(value.get("actual_samples", [])):
                    return False
            for key in (
                "actual_count",
                "expected_count",
                "actual_sha256",
                "expected_sha256",
                "actual_sum",
                "expected_sum",
                "actual_minimum",
                "expected_minimum",
                "actual_maximum",
                "expected_maximum",
                "sample_indices",
                "actual_samples",
                "expected_samples",
            ):
                if left.get(key) != right.get(key):
                    return False
        return True
    if left_result["operation"] == "device_prefix_chain":
        exact_keys = (
            "actual_sha256",
            "expected_sha256",
            "actual_sum",
            "expected_sum",
            "actual_minimum",
            "expected_minimum",
            "actual_maximum",
            "expected_maximum",
            "sample_indices",
            "actual_samples",
            "expected_samples",
            "mismatch_count",
            "first_mismatch",
        )
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            for value in (left, right):
                if (
                    not value.get("passed")
                    or value.get("comparison") != "exact_device_count_stable_compact_then_scan"
                    or not isinstance(value.get("actual_count"), int)
                    or value["actual_count"] <= 0
                    or value["actual_count"] != value.get("expected_count")
                ):
                    return False
                for vector_name in ("compacted", "scanned"):
                    vector = value.get(vector_name, {})
                    if (
                        not isinstance(vector.get("actual_sha256"), str)
                        or len(vector["actual_sha256"]) != 64
                        or vector["actual_sha256"] != vector.get("expected_sha256")
                        or vector.get("mismatch_count") != 0
                        or vector.get("first_mismatch") is not None
                    ):
                        return False
                    for suffix in ("sum", "minimum", "maximum", "samples"):
                        if vector.get(f"actual_{suffix}") != vector.get(f"expected_{suffix}"):
                            return False
                    if len(vector.get("sample_indices", [])) != len(vector.get("actual_samples", [])):
                        return False
            if left["actual_count"] != right["actual_count"] or left["expected_count"] != right["expected_count"]:
                return False
            for vector_name in ("compacted", "scanned"):
                for key in exact_keys:
                    if left[vector_name].get(key) != right[vector_name].get(key):
                        return False
        return True
    if left_result["operation"] == "particle_spatial_hash":
        vector_names = ("keys", "offsets", "canonical_output", "neighbors")
        exact_keys = (
            "count",
            "actual_sha256",
            "expected_sha256",
            "actual_sum",
            "expected_sum",
            "actual_minimum",
            "expected_minimum",
            "actual_maximum",
            "expected_maximum",
            "sample_indices",
            "actual_samples",
            "expected_samples",
            "mismatch_count",
            "first_mismatch",
        )
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            for value in (left, right):
                if (
                    not value.get("passed")
                    or value.get("comparison") != "exact_2d_cell_hash_buckets_and_neighbor_counts"
                    or set(value.get("endpoint_vectors", {})) != set(vector_names)
                ):
                    return False
                for vector_name in vector_names:
                    vector = value["endpoint_vectors"][vector_name]
                    if (
                        not isinstance(vector.get("count"), int)
                        or vector["count"] <= 0
                        or not isinstance(vector.get("actual_sha256"), str)
                        or len(vector["actual_sha256"]) != 64
                        or vector["actual_sha256"] != vector.get("expected_sha256")
                        or vector.get("mismatch_count") != 0
                        or vector.get("first_mismatch") is not None
                    ):
                        return False
                    for suffix in ("sum", "minimum", "maximum", "samples"):
                        if vector.get(f"actual_{suffix}") != vector.get(f"expected_{suffix}"):
                            return False
                    if len(vector.get("sample_indices", [])) != len(vector.get("actual_samples", [])):
                        return False
            for vector_name in vector_names:
                for key in exact_keys:
                    if left["endpoint_vectors"][vector_name].get(key) != right["endpoint_vectors"][vector_name].get(
                        key
                    ):
                        return False
        return True
    if left_result["operation"] == "marching_squares":
        vector_names = ("selected_cells_i32", "case_codes_i32")
        exact_keys = (
            "count",
            "expected_count",
            "actual_sha256",
            "expected_sha256",
            "actual_sum",
            "expected_sum",
            "actual_minimum",
            "expected_minimum",
            "actual_maximum",
            "expected_maximum",
            "sample_indices",
            "expected_sample_indices",
            "actual_samples",
            "expected_samples",
            "mismatch_count",
            "first_mismatch",
        )
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            for value in (left, right):
                if (
                    not value.get("passed")
                    or value.get("comparison") != "exact_stable_marching_squares_full_cell_and_case_vectors"
                    or not isinstance(value.get("actual_count"), int)
                    or value["actual_count"] <= 0
                    or value["actual_count"] != value.get("expected_count")
                    or set(value.get("endpoint_vectors", {})) != set(vector_names)
                ):
                    return False
                for vector_name in vector_names:
                    vector = value["endpoint_vectors"][vector_name]
                    if (
                        not isinstance(vector.get("count"), int)
                        or vector["count"] <= 0
                        or vector["count"] != vector.get("expected_count")
                        or vector["count"] != value["actual_count"]
                        or not isinstance(vector.get("actual_sha256"), str)
                        or len(vector["actual_sha256"]) != 64
                        or vector["actual_sha256"] != vector.get("expected_sha256")
                        or vector.get("mismatch_count") != 0
                        or vector.get("first_mismatch") is not None
                    ):
                        return False
                    for suffix in ("sum", "minimum", "maximum", "samples"):
                        if vector.get(f"actual_{suffix}") != vector.get(f"expected_{suffix}"):
                            return False
                    if vector.get("sample_indices") != vector.get("expected_sample_indices") or len(
                        vector.get("sample_indices", [])
                    ) != len(vector.get("actual_samples", [])):
                        return False
                fingerprint = value.get("endpoint_fingerprint", {})
                if (
                    fingerprint.get("finite") is not True
                    or fingerprint.get("selected_count") != value["actual_count"]
                    or fingerprint.get("selected_cells_sha256")
                    != value["endpoint_vectors"]["selected_cells_i32"]["actual_sha256"]
                    or fingerprint.get("case_codes_sha256")
                    != value["endpoint_vectors"]["case_codes_i32"]["actual_sha256"]
                ):
                    return False
            if left["actual_count"] != right["actual_count"] or left["expected_count"] != right["expected_count"]:
                return False
            for vector_name in vector_names:
                for key in exact_keys:
                    if left["endpoint_vectors"][vector_name].get(key) != right["endpoint_vectors"][vector_name].get(
                        key
                    ):
                        return False
        for result in (left_result, right_result):
            before = result["validation_before"]
            after = result["validation_after"]
            if before.get("actual_count") != after.get("actual_count") or before.get("expected_count") != after.get(
                "expected_count"
            ):
                return False
            for vector_name in vector_names:
                if any(
                    before["endpoint_vectors"][vector_name].get(key) != after["endpoint_vectors"][vector_name].get(key)
                    for key in exact_keys
                ):
                    return False
        return True
    if left_result["operation"] == "adaptive_pbd":
        observed_names = ("positions_f32", "residuals_f32")
        exact_names = ("active_history_i32", "final_active_ids_i32")
        observed_keys = (
            "count",
            "dtype",
            "sha256",
            "sum",
            "minimum",
            "maximum",
            "sample_indices",
            "samples",
        )
        exact_keys = (
            "count",
            "expected_count",
            "actual_sha256",
            "expected_sha256",
            "actual_sum",
            "expected_sum",
            "actual_minimum",
            "expected_minimum",
            "actual_maximum",
            "expected_maximum",
            "sample_indices",
            "expected_sample_indices",
            "actual_samples",
            "expected_samples",
            "mismatch_count",
            "first_mismatch",
        )
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            for value in (left, right):
                if (
                    not value.get("passed")
                    or value.get("comparison") != "analytic_full_state_and_exact_cross_route_adaptive_pbd"
                    or set(value.get("endpoint_vectors", {})) != set((*observed_names, *exact_names))
                    or not value.get("endpoint_fingerprint", {}).get("finite")
                    or float(value.get("max_left_position_error", math.inf)) > 2.0e-5
                    or float(value.get("max_right_position_error", math.inf)) > 2.0e-5
                    or float(value.get("max_residual_error", math.inf)) > 3.0e-5
                    or float(value.get("max_y_error", math.inf)) != 0.0
                    or value.get("history_mismatch_count") != 0
                    or value.get("active_id_mismatch_count") != 0
                    or value.get("active_id_first_mismatch") is not None
                    or value.get("actual_active_extent") != value.get("expected_active_extent")
                ):
                    return False
                for vector_name in observed_names:
                    vector = value["endpoint_vectors"][vector_name]
                    if (
                        not isinstance(vector.get("count"), int)
                        or vector["count"] <= 0
                        or vector.get("dtype") != "float32"
                        or not isinstance(vector.get("sha256"), str)
                        or len(vector["sha256"]) != 64
                        or not all(math.isfinite(float(vector.get(key))) for key in ("sum", "minimum", "maximum"))
                        or len(vector.get("sample_indices", [])) != len(vector.get("samples", []))
                    ):
                        return False
                for vector_name in exact_names:
                    vector = value["endpoint_vectors"][vector_name]
                    if (
                        not isinstance(vector.get("count"), int)
                        or vector["count"] <= 0
                        or vector["count"] != vector.get("expected_count")
                        or not isinstance(vector.get("actual_sha256"), str)
                        or len(vector["actual_sha256"]) != 64
                        or vector["actual_sha256"] != vector.get("expected_sha256")
                        or vector.get("mismatch_count") != 0
                        or vector.get("first_mismatch") is not None
                    ):
                        return False
                    for suffix in ("sum", "minimum", "maximum", "samples"):
                        if vector.get(f"actual_{suffix}") != vector.get(f"expected_{suffix}"):
                            return False
                    if vector.get("sample_indices") != vector.get("expected_sample_indices") or len(
                        vector.get("sample_indices", [])
                    ) != len(vector.get("actual_samples", [])):
                        return False
                fingerprint = value["endpoint_fingerprint"]
                if (
                    fingerprint.get("positions_sha256") != value["endpoint_vectors"]["positions_f32"]["sha256"]
                    or fingerprint.get("residuals_sha256") != value["endpoint_vectors"]["residuals_f32"]["sha256"]
                    or fingerprint.get("active_history_sha256")
                    != value["endpoint_vectors"]["active_history_i32"]["actual_sha256"]
                    or fingerprint.get("final_active_ids_sha256")
                    != value["endpoint_vectors"]["final_active_ids_i32"]["actual_sha256"]
                ):
                    return False
            for vector_name in observed_names:
                for key in observed_keys:
                    if left["endpoint_vectors"][vector_name].get(key) != right["endpoint_vectors"][vector_name].get(
                        key
                    ):
                        return False
            for vector_name in exact_names:
                for key in exact_keys:
                    if left["endpoint_vectors"][vector_name].get(key) != right["endpoint_vectors"][vector_name].get(
                        key
                    ):
                        return False
        for result in (left_result, right_result):
            before = result["validation_before"]["endpoint_vectors"]
            after = result["validation_after"]["endpoint_vectors"]
            for vector_name in observed_names:
                if any(before[vector_name].get(key) != after[vector_name].get(key) for key in observed_keys):
                    return False
            for vector_name in exact_names:
                if any(before[vector_name].get(key) != after[vector_name].get(key) for key in exact_keys):
                    return False
        return True
    if left_result["operation"] == "bfs_worklist":
        vector_names = ("distance_i32", "frontier_history_i32")
        exact_keys = (
            "count",
            "expected_count",
            "actual_sha256",
            "expected_sha256",
            "actual_sum",
            "expected_sum",
            "actual_minimum",
            "expected_minimum",
            "actual_maximum",
            "expected_maximum",
            "sample_indices",
            "expected_sample_indices",
            "actual_samples",
            "expected_samples",
            "actual_values_i32",
            "expected_values_i32",
            "mismatch_count",
            "first_mismatch",
        )
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            for value in (left, right):
                if (
                    not value.get("passed")
                    or value.get("comparison") != "exact_fixed_depth_grid_bfs_full_distance_and_frontier_vectors"
                    or value.get("distance_mismatch_count") != 0
                    or value.get("history_mismatch_count") != 0
                    or value.get("visited_count") != value.get("expected_visited_count")
                    or set(value.get("endpoint_vectors", {})) != set(vector_names)
                ):
                    return False
                vectors = value["endpoint_vectors"]
                if any(not _reported_exact_i32_vector_valid(vectors[name]) for name in vector_names):
                    return False
                actual_distance = vectors["distance_i32"]["actual_values_i32"]
                expected_distance = vectors["distance_i32"]["expected_values_i32"]
                actual_history = vectors["frontier_history_i32"]["actual_values_i32"]
                expected_history = vectors["frontier_history_i32"]["expected_values_i32"]
                visited = sum(distance >= 0 for distance in actual_distance)
                if (
                    visited != value["visited_count"]
                    or sum(distance >= 0 for distance in expected_distance) != value["expected_visited_count"]
                    or value.get("frontier_history") != actual_history
                    or value.get("expected_frontier_history") != expected_history
                ):
                    return False
                fingerprint = value.get("endpoint_fingerprint", {})
                if (
                    fingerprint.get("finite") is not True
                    or fingerprint.get("visited_count") != visited
                    or fingerprint.get("distance_sha256") != vectors["distance_i32"]["actual_sha256"]
                    or fingerprint.get("frontier_history_sha256") != vectors["frontier_history_i32"]["actual_sha256"]
                ):
                    return False
            for vector_name in vector_names:
                for key in exact_keys:
                    if left["endpoint_vectors"][vector_name].get(key) != right["endpoint_vectors"][vector_name].get(
                        key
                    ):
                        return False
        for result in (left_result, right_result):
            before = result["validation_before"]["endpoint_vectors"]
            after = result["validation_after"]["endpoint_vectors"]
            for vector_name in vector_names:
                if any(before[vector_name].get(key) != after[vector_name].get(key) for key in exact_keys):
                    return False
        return True
    if left_result["operation"] == "falling_sand":
        vector_names = (
            "grid_i32",
            "winner_source_by_destination_i32",
            "destinations_i32",
            "priorities_i32",
        )
        exact_keys = (
            "count",
            "expected_count",
            "actual_sha256",
            "expected_sha256",
            "actual_sum",
            "expected_sum",
            "actual_minimum",
            "expected_minimum",
            "actual_maximum",
            "expected_maximum",
            "sample_indices",
            "expected_sample_indices",
            "actual_samples",
            "expected_samples",
            "actual_values_i32",
            "expected_values_i32",
            "mismatch_count",
            "first_mismatch",
        )
        count_keys = (
            "candidate_count",
            "expected_candidate_count",
            "winner_count",
            "expected_winner_count",
            "conflict_count",
            "expected_conflict_count",
        )
        fingerprint_map = {
            "grid_sha256": "grid_i32",
            "winner_sources_sha256": "winner_source_by_destination_i32",
            "destinations_sha256": "destinations_i32",
            "priorities_sha256": "priorities_i32",
        }
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            for value in (left, right):
                counts = [value.get(key) for key in count_keys]
                if (
                    not value.get("passed")
                    or value.get("comparison") != "exact_falling_sand_grid_candidates_and_keyed_winners"
                    or any(not isinstance(item, int) or isinstance(item, bool) for item in counts)
                    or value.get("candidate_count") != value.get("expected_candidate_count")
                    or value.get("winner_count") != value.get("expected_winner_count")
                    or value.get("conflict_count") != value.get("expected_conflict_count")
                    or value.get("conflict_count") != value.get("candidate_count") - value.get("winner_count")
                    or value.get("conflict_count", 0) <= 0
                    or set(value.get("endpoint_vectors", {})) != set(vector_names)
                ):
                    return False
                vectors = value["endpoint_vectors"]
                if any(not _reported_exact_i32_vector_valid(vectors[name]) for name in vector_names):
                    return False
                fingerprint = value.get("endpoint_fingerprint", {})
                if fingerprint.get("finite") is not True:
                    return False
                for fingerprint_key, vector_name in fingerprint_map.items():
                    if fingerprint.get(fingerprint_key) != vectors[vector_name]["actual_sha256"]:
                        return False
            for key in count_keys:
                if left.get(key) != right.get(key):
                    return False
            for vector_name in vector_names:
                for key in exact_keys:
                    if left["endpoint_vectors"][vector_name].get(key) != right["endpoint_vectors"][vector_name].get(
                        key
                    ):
                        return False
        for result in (left_result, right_result):
            before = result["validation_before"]
            after = result["validation_after"]
            for key in count_keys:
                if before.get(key) != after.get(key):
                    return False
            for vector_name in vector_names:
                if any(
                    before["endpoint_vectors"][vector_name].get(key) != after["endpoint_vectors"][vector_name].get(key)
                    for key in exact_keys
                ):
                    return False
        return True
    if left_result["operation"] == "sparse_block_stencil":
        comparison = "coordinate_dense_oracle_for_rebuilt_sparse_five_point_" "weighted_jacobi"
        fingerprint_keys = (
            "count",
            "sha256",
            "sum",
            "minimum",
            "maximum",
            "sample_indices",
            "sample_values",
        )
        for validation_name in ("validation_before", "validation_after"):
            left = left_result[validation_name]
            right = right_result[validation_name]
            for value in (left, right):
                tolerance = float(value.get("effective_tolerance", -1.0))
                actual = value.get("endpoint_fingerprint", {})
                expected = value.get("expected_endpoint_fingerprint", {})
                if (
                    not value.get("passed")
                    or value.get("comparison") != comparison
                    or value.get("active_blocks") != value.get("expected_active_blocks")
                    or tolerance < 0.0
                    or float(value.get("max_abs_error", math.inf)) > tolerance
                    or not math.isfinite(float(value.get("rmse", math.inf)))
                    or actual.get("finite") is not True
                    or expected.get("finite") is not True
                    or actual.get("count") != expected.get("count")
                    or actual.get("sample_indices") != expected.get("sample_indices")
                    or not isinstance(actual.get("sha256"), str)
                    or len(actual["sha256"]) != 64
                    or not isinstance(expected.get("sha256"), str)
                    or len(expected["sha256"]) != 64
                ):
                    return False
                count = int(actual["count"])
                if (
                    count <= 0
                    or len(actual.get("sample_values", [])) != len(actual.get("sample_indices", []))
                    or len(expected.get("sample_values", [])) != len(expected.get("sample_indices", []))
                ):
                    return False
                for key in ("minimum", "maximum"):
                    if not math.isclose(float(actual[key]), float(expected[key]), rel_tol=0.0, abs_tol=tolerance):
                        return False
                if not math.isclose(
                    float(actual["sum"]), float(expected["sum"]), rel_tol=0.0, abs_tol=tolerance * count
                ):
                    return False
                if any(
                    not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
                    for a, b in zip(actual["sample_values"], expected["sample_values"])
                ):
                    return False
            left_fp = left["endpoint_fingerprint"]
            right_fp = right["endpoint_fingerprint"]
            if any(left_fp.get(key) != right_fp.get(key) for key in fingerprint_keys):
                return False
        for result in (left_result, right_result):
            before = result["validation_before"]["endpoint_fingerprint"]
            after = result["validation_after"]["endpoint_fingerprint"]
            if any(before.get(key) != after.get(key) for key in fingerprint_keys):
                return False
        return True
    if left_result["operation"] not in ("mpm_graph", "mpm_direct", "active_grid_mpm"):
        return True
    for validation_name in ("validation_before", "validation_after"):
        left_validation = left_result[validation_name]
        right_validation = right_result[validation_name]
        if not left_validation.get("passed") or not right_validation.get("passed"):
            return False
        if left_result["operation"] == "active_grid_mpm":
            for key in (
                "active_count",
                "mass_active_count",
                "published_count",
                "active_flags_sha256",
                "mass_mask_sha256",
            ):
                if left_validation.get(key) != right_validation.get(key):
                    return False
            for value in (left_validation, right_validation):
                if (
                    not isinstance(value.get("active_flags_sha256"), str)
                    or len(value["active_flags_sha256"]) != 64
                    or value["active_flags_sha256"] != value.get("mass_mask_sha256")
                ):
                    return False
        left = left_validation["endpoint_fingerprint"]
        right = right_validation["endpoint_fingerprint"]
        if not left.get("finite") or not right.get("finite"):
            return False
        for key in ("x_mean", "v_mean", "C_mean", "sample_x", "sample_v"):
            if len(left[key]) != len(right[key]):
                return False
            if any(
                not math.isclose(float(a), float(b), rel_tol=5.0e-5, abs_tol=5.0e-5)
                for a, b in zip(left[key], right[key])
            ):
                return False
        if not math.isclose(float(left["J_mean"]), float(right["J_mean"]), rel_tol=5.0e-5, abs_tol=5.0e-5):
            return False
        if float(left["image_sum"]) != float(right["image_sum"]) or float(left["image_max"]) != float(
            right["image_max"]
        ):
            return False
    return True


def _adaptive_pbd_kernel_control_route_isolated(child: dict[str, Any]) -> bool:
    if child.get("operation") != "adaptive_pbd" or child.get("runtime") not in ("forge_kernel", "vanilla_kernel"):
        return True
    route = child.get("route", {})
    contract = child.get("workload_contract", {})
    return bool(
        route.get("passed") is True
        and route.get("classification") == f"{child['runtime']}_equivalent_adaptive_pbd_kernel_pipeline"
        and route.get("adapter") == "benchmark_defined_ti_kernel_pipeline"
        and "native" not in json.dumps(route, sort_keys=True).lower()
        and route.get("kernel_source_owner") == "benchmark"
        and route.get("kernel_source_sha256") == contract.get("kernel_source_sha256")
        and route.get("helper_api_used") is False
        and route.get("specialized_api_used") is False
        and route.get("benchmark_workspace_kind") == contract.get("kernel_benchmark_workspace_kind")
        and route.get("benchmark_workspace_field_count") == 6
        and route.get("benchmark_workspace_field_count") == contract.get("kernel_benchmark_workspace_field_count")
        and route.get("scan_algorithm") == "inclusive_hillis_steele_ping_pong"
        and route.get("scan_algorithm") == contract.get("kernel_scan_algorithm")
        and route.get("scan_elements") == contract.get("kernel_scan_elements")
        and route.get("scan_pipelines_per_replay") == contract.get("kernel_scan_pipelines_per_replay")
        and route.get("scan_steps_per_pipeline") == contract.get("kernel_scan_steps_per_pipeline")
        and route.get("final_scan_copy_kernel_invocations_per_pipeline")
        == contract.get("kernel_final_scan_copy_kernel_invocations_per_pipeline")
        and route.get("non_scan_ti_kernel_invocations_per_replay")
        == contract.get("kernel_non_scan_ti_invocations_per_replay")
        and route.get("stage_kernel_names") == contract.get("kernel_stage_names")
        and route.get("ti_kernel_invocations_per_replay") == contract.get("kernel_ti_invocations_per_replay")
        and route.get("physical_backend_launches_assumed") is False
        and route.get("expected_backend") == child.get("backend")
        and route.get("observed_backend") == child.get("backend")
        and route.get("constraints") == contract.get("constraints")
        and route.get("iterations") == contract.get("iterations")
    )


def _marching_squares_kernel_control_route_isolated(child: dict[str, Any]) -> bool:
    if child.get("operation") != "marching_squares" or child.get("runtime") not in ("forge_kernel", "vanilla_kernel"):
        return True
    route = child.get("route", {})
    contract = child.get("workload_contract", {})
    return bool(
        route.get("passed") is True
        and route.get("classification") == f"{child['runtime']}_equivalent_marching_squares_kernel_pipeline"
        and route.get("adapter") == "benchmark_defined_ti_kernel_pipeline"
        and "native" not in json.dumps(route, sort_keys=True).lower()
        and route.get("kernel_source_owner") == "benchmark"
        and route.get("kernel_source_sha256") == contract.get("kernel_source_sha256")
        and route.get("helper_api_used") is False
        and route.get("specialized_api_used") is False
        and route.get("benchmark_workspace_kind") == contract.get("kernel_benchmark_workspace_kind")
        and route.get("benchmark_workspace_field_count") == 2
        and route.get("benchmark_workspace_field_count") == contract.get("kernel_benchmark_workspace_field_count")
        and route.get("scan_algorithm") == "inclusive_hillis_steele_ping_pong"
        and route.get("scan_algorithm") == contract.get("kernel_scan_algorithm")
        and route.get("scan_elements") == contract.get("kernel_scan_elements")
        and route.get("scan_steps") == contract.get("kernel_scan_steps")
        and route.get("final_scan_copy_kernel_invocations") == contract.get("kernel_final_scan_copy_kernel_invocations")
        and route.get("non_scan_ti_kernel_invocations_per_replay")
        == contract.get("kernel_non_scan_ti_invocations_per_replay")
        and route.get("stage_kernel_names") == contract.get("kernel_stage_names")
        and route.get("ti_kernel_invocations_per_replay") == contract.get("kernel_ti_invocations_per_replay")
        and route.get("physical_backend_launches_assumed") is False
        and route.get("expected_backend") == child.get("backend")
        and route.get("observed_backend") == child.get("backend")
        and route.get("cells") == contract.get("cells")
    )


def _bfs_worklist_kernel_control_route_isolated(child: dict[str, Any]) -> bool:
    if child.get("operation") != "bfs_worklist" or child.get("runtime") not in ("forge_kernel", "vanilla_kernel"):
        return True
    route = child.get("route", {})
    contract = child.get("workload_contract", {})
    return bool(
        route.get("passed") is True
        and route.get("classification") == f"{child['runtime']}_equivalent_bfs_kernel_pipeline"
        and route.get("adapter") == "benchmark_defined_ti_kernel_pipeline"
        and "native" not in json.dumps(route, sort_keys=True).lower()
        and route.get("kernel_source_owner") == "benchmark"
        and route.get("kernel_source_sha256") == contract.get("kernel_source_sha256")
        and route.get("helper_api_used") is False
        and route.get("specialized_api_used") is False
        and route.get("benchmark_workspace_kind") == contract.get("kernel_benchmark_workspace_kind")
        and route.get("benchmark_workspace_field_count") == 4
        and route.get("benchmark_workspace_field_count") == contract.get("kernel_benchmark_workspace_field_count")
        and route.get("stage_kernel_names") == contract.get("kernel_stage_names")
        and route.get("initialize_ti_kernel_invocations_per_replay") == 1
        and route.get("reset_extent_ti_kernel_invocations_per_replay")
        == contract.get("kernel_reset_extent_ti_invocations_per_replay")
        and route.get("expand_ti_kernel_invocations_per_replay")
        == contract.get("kernel_expand_ti_invocations_per_replay")
        and route.get("record_ti_kernel_invocations_per_replay")
        == contract.get("kernel_record_ti_invocations_per_replay")
        and route.get("finalize_ti_kernel_invocations_per_replay") == 1
        and route.get("ti_kernel_invocations_per_replay") == contract.get("kernel_ti_invocations_per_replay")
        and route.get("physical_backend_launches_assumed") is False
        and route.get("expected_backend") == child.get("backend")
        and route.get("observed_backend") == child.get("backend")
        and route.get("nodes") == contract.get("nodes")
        and route.get("levels") == contract.get("levels")
    )


def _bfs_worklist_native_route_admitted(child: dict[str, Any]) -> bool:
    if child.get("operation") != "bfs_worklist" or child.get("runtime") != "forge":
        return True
    route = child.get("route", {})
    contract = child.get("workload_contract", {})
    memory = route.get("memory_report", {})
    transition = route.get("last_transition_statistics", {})
    contract_profile = contract.get("contract_profile", "legacy")
    current_contract = contract_profile == "current"
    return bool(
        route.get("passed") is True
        and route.get("classification") == "forge_native_device_worklist_bfs_pipeline"
        and route.get("adapter") == "forge_native_device_worklist_frontier_pipeline"
        and route.get("kernel_source_owner") == "benchmark"
        and route.get("kernel_source_sha256") == contract.get("kernel_source_sha256")
        and route.get("capacity") == contract.get("nodes")
        and route.get("nodes") == contract.get("nodes")
        and route.get("levels") == contract.get("levels")
        and route.get("benchmark_ti_kernel_invocations_per_replay") == 2 + 2 * contract.get("levels", -1)
        and route.get("device_worklist_transitions_per_replay") == (0 if current_contract else contract.get("levels"))
        and route.get("fused_recycle_boundaries_per_replay") == (contract.get("levels") if current_contract else 0)
        and route.get("contract_profile") == contract_profile
        and route.get("transition_mode") == ("direct" if current_contract else "staged")
        and route.get("telemetry_enabled") is (not current_contract)
        and route.get("physical_backend_launches_assumed") is False
        and route.get("expected_backend") == child.get("backend")
        and route.get("observed_backend") == child.get("backend")
        and memory.get("fixed_capacity") is True
        and memory.get("replay_allocation_count") == 0
        and (
            (
                transition.get("generated") is None
                and transition.get("accepted") is None
                and transition.get("rejected") is None
            )
            if current_contract
            else (
                transition.get("generated") == contract.get("expected_last_frontier")
                and transition.get("accepted") == contract.get("expected_last_frontier")
                and transition.get("rejected") == 0
            )
        )
        and transition.get("overflow") is False
    )


def _falling_sand_kernel_control_route_isolated(child: dict[str, Any]) -> bool:
    if child.get("operation") != "falling_sand" or child.get("runtime") not in ("forge_kernel", "vanilla_kernel"):
        return True
    route = child.get("route", {})
    contract = child.get("workload_contract", {})
    return bool(
        route.get("passed") is True
        and route.get("classification") == f"{child['runtime']}_equivalent_falling_sand_kernel_pipeline"
        and route.get("adapter") == "benchmark_defined_ti_kernel_pipeline"
        and "native" not in json.dumps(route, sort_keys=True).lower()
        and route.get("kernel_source_owner") == "benchmark"
        and route.get("kernel_source_sha256") == contract.get("kernel_source_sha256")
        and route.get("helper_api_used") is False
        and route.get("specialized_api_used") is False
        and route.get("benchmark_workspace_kind") == contract.get("kernel_benchmark_workspace_kind")
        and route.get("benchmark_workspace_field_count") == 1
        and route.get("benchmark_workspace_field_count") == contract.get("kernel_benchmark_workspace_field_count")
        and route.get("stage_kernel_names") == contract.get("kernel_stage_names")
        and route.get("ti_kernel_invocations_per_replay") == contract.get("kernel_ti_invocations_per_replay")
        and route.get("claim_policy") == "atomic_min_priority_then_source_ordinal"
        and route.get("control_only_claim_workspace_reset") is True
        and route.get("control_only_claim_workspace_reset") == contract.get("kernel_control_claim_workspace_reset")
        and route.get("physical_backend_launches_assumed") is False
        and route.get("expected_candidates") == contract.get("expected_candidate_count")
        and route.get("expected_winners") == contract.get("expected_winner_count")
        and route.get("observed_candidates") == contract.get("expected_candidate_count")
        and route.get("observed_winners") == contract.get("expected_winner_count")
        and route.get("elements") == contract.get("cells")
        and route.get("expected_backend") == child.get("backend")
        and route.get("observed_backend") == child.get("backend")
    )


def _falling_sand_native_route_admitted(child: dict[str, Any]) -> bool:
    if child.get("operation") != "falling_sand" or child.get("runtime") != "forge":
        return True
    route = child.get("route", {})
    contract = child.get("workload_contract", {})
    memory = route.get("memory_report", {})
    transition = route.get("last_transition_statistics", {})
    candidates = contract.get("expected_candidate_count")
    winners = contract.get("expected_winner_count")
    contract_profile = contract.get("contract_profile", "legacy")
    current_contract = contract_profile == "current"
    return bool(
        isinstance(candidates, int)
        and isinstance(winners, int)
        and route.get("passed") is True
        and route.get("classification") == "forge_native_falling_sand_keyed_claim_pipeline"
        and route.get("adapter") == "forge_native_device_worklist_keyed_claim"
        and route.get("kernel_source_owner") == "benchmark"
        and route.get("kernel_source_sha256") == contract.get("kernel_source_sha256")
        and route.get("benchmark_stage_kernel_names") == contract.get("native_benchmark_stage_kernel_names")
        and route.get("benchmark_ti_kernel_invocations_per_replay") == (3 if current_contract else 5)
        and route.get("device_worklist_transitions_per_replay") == (1 if current_contract else 2)
        and route.get("contract_profile") == contract_profile
        and route.get("conflict_output_shape") == ("dense_winner_table" if current_contract else "compact_winner_list")
        and route.get("telemetry_enabled") is (not current_contract)
        and route.get("worklist_transition_names")
        == (
            ["fixed_domain_masked_dense_claim"]
            if current_contract
            else [
                "stable_select_candidates",
                "deterministic_keyed_claim",
            ]
        )
        and route.get("claim_policy") == "min_priority_then_source_ordinal"
        and route.get("control_only_claim_workspace_reset") is False
        and route.get("control_only_claim_workspace_reset") == contract.get("native_control_claim_workspace_reset")
        and route.get("physical_backend_launches_assumed") is False
        and route.get("capacity") == contract.get("cells")
        and route.get("elements") == contract.get("cells")
        and route.get("expected_backend") == child.get("backend")
        and route.get("observed_backend") == child.get("backend")
        and memory.get("fixed_capacity") is True
        and memory.get("replay_allocation_count") == 0
        and transition.get("generated") == (None if current_contract else candidates)
        and (
            (
                transition.get("accepted") is None
                and transition.get("rejected") is None
                and transition.get("conflicts") is None
                and transition.get("winners") is None
            )
            if current_contract
            else (
                transition.get("accepted") == winners
                and transition.get("rejected") == candidates - winners
                and transition.get("conflicts") == candidates - winners
                and transition.get("winners") == winners
            )
        )
        and transition.get("overflow") is False
        and route.get("observed_candidates") == candidates
        and route.get("observed_winners") == winners
    )


def _sparse_block_stencil_route_isolated(child: dict[str, Any]) -> bool:
    if child.get("operation") != "sparse_block_stencil":
        return True
    route = child.get("route", {})
    contract = child.get("workload_contract", {})
    runtime = child.get("runtime")
    return bool(
        runtime in ("forge", "vanilla")
        and route.get("classification") == f"{runtime}_shared_sparse_block_stencil"
        and route.get("adapter") == "shared_vanilla_compatible_sparse_taichi_pipeline"
        and route.get("kernel_source_owner") == "benchmark"
        and route.get("kernel_source_sha256") == contract.get("kernel_source_sha256")
        and route.get("native_or_helper_api_used") is False
        and route.get("capacity_hint_used") is False
        and route.get("timed_host_deactivate_calls_per_replay") == 1
        and route.get("ti_kernel_invocations_per_replay") == contract.get("ti_kernel_invocations_per_replay")
        and route.get("physical_backend_launches_assumed") is False
        and route.get("expected_active_blocks") == contract.get("active_blocks")
        and route.get("observed_active_blocks") == contract.get("active_blocks")
        and route.get("passed") is True
    )


def _audit_failed_run(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    failure_path = run_dir / "failure.json"
    failures: list[str] = []
    _check(failure_path.is_file(), "failure.json", failures)
    failure = _read_json(failure_path) if failure_path.is_file() else {}
    _check(manifest.get("schema") == SCHEMA, "manifest schema", failures)
    _check(failure.get("schema") == SCHEMA, "failure schema", failures)
    _check(manifest.get("run_id") == failure.get("run_id"), "run id", failures)
    _check(failure.get("ready_for_performance_claim") is False, "failed run claim eligibility", failures)
    _check(manifest.get("failure", {}).get("reason") == failure.get("reason"), "failure reason identity", failures)
    for filename in ("failure.zh-CN.md", "failure.en.md"):
        path = run_dir / filename
        _check(path.is_file() and path.stat().st_size > 0, f"bilingual failure artifact {filename}", failures)
    child_paths = list((run_dir / "children").glob("pair-*.json"))
    config = manifest.get("config", {})
    for path in child_paths:
        child = _read_json(path)
        _check(child.get("schema") == SCHEMA, f"child schema {path.name}", failures)
        for name in ("operation", "backend", "preset"):
            _check(child.get(name) == config.get(name), f"child {name} {path.name}", failures)
        _check(
            child.get("status") in ("passed", "failed", "error", "rejected"),
            f"child terminal status {path.name}",
            failures,
        )
        stability_failure = child.get("stability_failure")
        if stability_failure is not None:
            requested = int(stability_failure.get("requested_replays", -1))
            completed = int(stability_failure.get("completed_replays", -1))
            _check(
                requested == int(config.get("stability_replays", -2)),
                f"stability failure requested replays {path.name}",
                failures,
            )
            _check(0 <= completed < requested, f"stability failure completed replays {path.name}", failures)
            _check(
                stability_failure.get("failed_replay_one_based") == completed + 1,
                f"stability failure replay index {path.name}",
                failures,
            )
            _check(bool(stability_failure.get("cause")), f"stability failure cause {path.name}", failures)
            _check(
                "rss_before_bytes" in stability_failure and "rss_at_failure_bytes" in stability_failure,
                f"stability failure RSS evidence {path.name}",
                failures,
            )
            if config.get("backend") != "cpu":
                _check(
                    "gpu_before_mib" in stability_failure and "gpu_at_failure_mib" in stability_failure,
                    f"stability failure GPU evidence {path.name}",
                    failures,
                )
            _check(
                len(child.get("samples", [])) == int(config.get("samples", -1)),
                f"pre-failure scored samples {path.name}",
                failures,
            )
            _check(
                child.get("failure_route", {}).get("cycles_completed", -1)
                >= child.get("route_before_scoring", {}).get("cycles_completed", 0),
                f"failure route progress {path.name}",
                failures,
            )
    scored_children = len(child_paths)
    return {
        "schema": "taichi_forge.single_kernel_microbench.audit.v1",
        "run_id": manifest.get("run_id"),
        "run_status": "failed",
        "audit_passed": not failures,
        "audit_failures": failures,
        "scored_child_count": scored_children,
        "pair_count": int(manifest.get("config", {}).get("pairs", 0)),
        "ready_for_performance_claim": False,
        "recomputed_paired": None,
    }


def _audit(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    if not (run_dir / "summary.json").is_file():
        return _audit_failed_run(run_dir, manifest)
    summary = _read_json(run_dir / "summary.json")
    failures: list[str] = []
    definition = summary.get(
        "comparison_definition",
        {
            "name": "forge-vs-vanilla",
            "subject": "forge",
            "baseline": "vanilla",
        },
    )
    participants = (definition["subject"], definition["baseline"])
    extended_contract = "physical_device_binding" in summary.get("method_checks", {})
    _check(manifest.get("schema") == SCHEMA, "manifest schema", failures)
    _check(summary.get("schema") == SCHEMA, "summary schema", failures)
    _check(manifest.get("run_id") == summary.get("run_id"), "run id", failures)
    _check(manifest.get("config") == summary.get("config"), "config identity", failures)
    if "comparison_definition" in summary:
        _check(
            summary["comparison_definition"] == summary["config"].get("comparison_definition"),
            "comparison definition identity",
            failures,
        )
    _check(manifest.get("exclusive_driver_lock", {}).get("acquired") is True, "exclusive benchmark lock", failures)

    for wheel in manifest.get("forge_wheels", {}).values():
        path = Path(wheel["path"])
        _check(path.is_file() and _sha256(path) == wheel["sha256"], f"wheel hash {path.name}", failures)

    children = []
    for path in sorted((run_dir / "children").glob("pair-*.json")):
        child = _read_json(path)
        child["_artifact_path"] = str(path)
        children.append(child)
    pair_count = int(summary["config"]["pairs"])
    _check(len(children) == pair_count * 2, "scored child count", failures)

    groups: dict[int, list[dict[str, Any]]] = {}
    for child in children:
        groups.setdefault(int(child["pair_index"]), []).append(child)
        _check(child.get("status") == "passed", "child status", failures)
        _check(child.get("arch_match") is True, "backend match", failures)
        _check(child.get("environment_isolated") is True, "environment isolation", failures)
        if extended_contract:
            _check(
                child.get("device_identity", {}).get("binding_verified") is True, "physical device binding", failures
            )
            _check(child.get("route", {}).get("passed") is True, "execution route", failures)
        _check(child["validation_before"]["passed"] is True, "correctness before", failures)
        _check(child["validation_after"]["passed"] is True, "correctness after", failures)
        _check(child["teardown"]["sync_error"] is None, "teardown sync", failures)
        _check(child["teardown"]["reset_error"] is None, "teardown reset", failures)
        raw = [float(value) for value in child["raw_batch_ms"]]
        samples = [float(value) for value in child["samples"]]
        batch = int(child["batch_size"])
        if "batched_score_warmup" in summary.get("method_checks", {}):
            warmup_batch = int(child.get("warmup_batch_size", 0))
            warmup_raw = [float(value) for value in child.get("warmup_raw_batch_ms", [])]
            warmup_per_replay = [float(value) for value in child.get("warmup_ms", [])]
            _check(warmup_batch == batch, "score warmup common batch", failures)
            _check(len(warmup_raw) == int(summary["config"]["warmups"]), "score warmup batch count", failures)
            _check(len(warmup_raw) == len(warmup_per_replay), "score warmup sample length", failures)
            _check(
                all(math.isfinite(value) and value > 0.0 for value in warmup_raw),
                "score warmup finite positive batches",
                failures,
            )
            _check(
                all(
                    _close(per_replay, raw_value / warmup_batch)
                    for per_replay, raw_value in zip(warmup_per_replay, warmup_raw)
                ),
                "score warmup sample derivation",
                failures,
            )
        _check(len(raw) == int(summary["config"]["samples"]), "raw sample count", failures)
        _check(len(raw) == len(samples), "sample length", failures)
        _check(
            all(_close(sample, raw_value / batch) for sample, raw_value in zip(samples, raw)),
            "sample derivation",
            failures,
        )
        recomputed_median = statistics.median(samples)
        recomputed_p95 = _percentile(samples, 95.0)
        recomputed_mean = statistics.fmean(samples)
        recomputed_cv = 0.0 if recomputed_mean == 0.0 else statistics.pstdev(samples) / recomputed_mean * 100.0
        _check(_close(recomputed_median, child["summary"]["median_ms"]), "child median", failures)
        _check(_close(recomputed_p95, child["summary"]["p95_ms"]), "child p95", failures)
        _check(_close(recomputed_cv, child["summary"]["cv_percent"]), "child CV", failures)
        if "latency_samples" in summary["config"]:
            latency_values = [float(value) for value in child.get("warm_single_call_latency_ms", [])]
            _check(
                len(latency_values) == int(summary["config"]["latency_samples"]),
                "warm single-call latency sample count",
                failures,
            )
            if latency_values:
                latency_summary = child.get("warm_single_call_latency_summary", {})
                latency_mean = statistics.fmean(latency_values)
                latency_cv = 0.0 if latency_mean == 0.0 else statistics.pstdev(latency_values) / latency_mean * 100.0
                _check(
                    _close(statistics.median(latency_values), latency_summary.get("median_ms")),
                    "warm single-call latency median",
                    failures,
                )
                _check(
                    _close(_percentile(latency_values, 95.0), latency_summary.get("p95_ms")),
                    "warm single-call latency p95",
                    failures,
                )
                _check(_close(latency_cv, latency_summary.get("cv_percent")), "warm single-call latency CV", failures)

    neutral_signatures = set()
    workload_signatures = set()
    comparison_classes = set()
    for child in children:
        environment = child["environment"]
        dependency_versions = tuple(sorted((name, row["version"]) for name, row in environment["dependencies"].items()))
        neutral_signatures.add((environment["python_version"], dependency_versions))
        workload_signatures.add(
            (
                child["operation"],
                child["backend"],
                child["preset"],
                child["logical_bytes"],
                child["traffic_model"],
                child["batch_size"],
                child.get("measurement_scope"),
                tuple(sorted(child["measurement_config"].items())),
                json.dumps(child.get("workload_contract", {}), sort_keys=True),
            )
        )
        comparison_class = child.get("workload_contract", {}).get("comparison_class")
        if comparison_class is not None:
            comparison_classes.add(comparison_class)
    _check(len(neutral_signatures) == 1, "neutral dependency parity", failures)
    _check(len(workload_signatures) == 1, "workload parity", failures)
    if "comparison_class" in summary:
        _check(len(comparison_classes) == 1, "comparison class consistency", failures)
        _check(
            len(comparison_classes) == 1 and summary["comparison_class"] == next(iter(comparison_classes)),
            "comparison class summary",
            failures,
        )

    recomputed_rows = []
    all_intervals = []
    orders = manifest["pair_orders"]
    for pair_index in range(1, pair_count + 1):
        pair = groups.get(pair_index, [])
        _check(len(pair) == 2, f"pair {pair_index} cardinality", failures)
        if len(pair) != 2:
            continue
        by_runtime = {child["runtime"]: child for child in pair}
        _check(set(by_runtime) == set(participants), f"pair {pair_index} runtimes", failures)
        order = tuple(orders[pair_index - 1])
        ordered = sorted(pair, key=lambda child: child["position_in_pair"])
        _check(tuple(child["runtime"] for child in ordered) == order, f"pair {pair_index} order", failures)
        first, second = ordered
        _check(
            first["parent_launch_finished_ns"] <= second["parent_launch_started_ns"],
            f"pair {pair_index} non-overlap",
            failures,
        )
        all_intervals.extend(
            (
                (first["parent_launch_started_ns"], first["parent_launch_finished_ns"]),
                (second["parent_launch_started_ns"], second["parent_launch_finished_ns"]),
            )
        )
        subject = by_runtime[definition["subject"]]
        baseline = by_runtime[definition["baseline"]]
        recomputed_rows.append(
            {
                "median_speedup_x": (baseline["summary"]["median_ms"] / subject["summary"]["median_ms"]),
                "p95_speedup_x": (baseline["summary"]["p95_ms"] / subject["summary"]["p95_ms"]),
                "endpoint_equivalent": _endpoint_equivalent(subject, baseline),
                "warm_latency_speedup_x": (
                    (
                        baseline["warm_single_call_latency_summary"]["median_ms"]
                        / subject["warm_single_call_latency_summary"]["median_ms"]
                    )
                    if "latency_samples" in summary["config"]
                    else None
                ),
            }
        )
    ordered_intervals = sorted(all_intervals)
    _check(
        all(left[1] <= right[0] for left, right in zip(ordered_intervals, ordered_intervals[1:])),
        "global child non-overlap",
        failures,
    )

    stored_rows = summary["pair_rows"]
    _check(len(stored_rows) == len(recomputed_rows), "pair row count", failures)
    for index, (stored, recomputed) in enumerate(zip(stored_rows, recomputed_rows), start=1):
        if "subject" in stored:
            _check(
                stored["subject"] == definition["subject"] and stored["baseline"] == definition["baseline"],
                f"pair {index} comparison roles",
                failures,
            )
        _check(
            _close(stored["median_speedup_x"], recomputed["median_speedup_x"]), f"pair {index} median speedup", failures
        )
        _check(_close(stored["p95_speedup_x"], recomputed["p95_speedup_x"]), f"pair {index} p95 speedup", failures)
        if extended_contract:
            _check(
                stored.get("endpoint_equivalent", stored.get("cross_runtime_endpoint_equivalent"))
                is recomputed["endpoint_equivalent"],
                f"pair {index} comparison endpoint",
                failures,
            )
        if recomputed["warm_latency_speedup_x"] is not None:
            _check(
                _close(stored.get("warm_latency_speedup_x"), recomputed["warm_latency_speedup_x"]),
                f"pair {index} warm latency speedup",
                failures,
            )

    median_speedups = [row["median_speedup_x"] for row in recomputed_rows]
    p95_speedups = [row["p95_speedup_x"] for row in recomputed_rows]
    latency_speedups = [
        row["warm_latency_speedup_x"] for row in recomputed_rows if row["warm_latency_speedup_x"] is not None
    ]
    if median_speedups:
        for name, expected in _bootstrap_median(median_speedups, int(summary["config"]["seed"])).items():
            _check(_close(summary["paired_summary"][name], expected), f"paired {name}", failures)
        for name, expected in _bootstrap_median(p95_speedups, int(summary["config"]["seed"]) + 1).items():
            _check(_close(summary["p95_paired_summary"][name], expected), f"paired p95 {name}", failures)
        if latency_speedups:
            for name, expected in _bootstrap_median(latency_speedups, int(summary["config"]["seed"]) + 2).items():
                _check(
                    _close(summary["warm_single_call_latency_paired_summary"][name], expected),
                    f"paired warm latency {name}",
                    failures,
                )

    expected_noise_count = 1 + pair_count * 3
    observations = manifest["noise_observations"]
    _check(len(observations) == expected_noise_count, "noise observation count", failures)
    _check(all(item["passed"] for item in observations), "noise admission", failures)
    for filename in (
        "report.zh-CN.md",
        "report.en.md",
        "validation.zh-CN.md",
        "validation.en.md",
    ):
        path = run_dir / filename
        _check(path.is_file() and path.stat().st_size > 0, f"bilingual artifact {filename}", failures)

    config = summary["config"]
    qualification_policy = bool(
        config["intent"] == "qualification"
        and pair_count >= 10
        and pair_count % 2 == 0
        and int(config["samples"]) >= 30
        and int(config["warmups"]) >= 5
        and float(config["target_sample_ms"]) >= 100.0
        and int(config["stability_replays"]) >= 1_000
        and config["cpu_affinity"] != "none"
        and float(config.get("max_cpu_util", 20.0)) <= 20.0
        and (
            config["backend"] == "cpu"
            or (float(config.get("max_gpu_util", 15.0)) <= 15.0 and float(config.get("max_gpu_temp", 65.0)) <= 65.0)
        )
    )
    forward_order = "->".join(participants)
    reverse_order = "->".join(reversed(participants))
    order_counts = {
        forward_order: sum(tuple(order) == participants for order in orders),
        reverse_order: sum(tuple(order) == tuple(reversed(participants)) for order in orders),
    }
    stability_complete = all(
        child.get("stability") is not None
        and child["stability"]["replays"] >= int(config["stability_replays"])
        and child["stability"]["memory_guard_passed"]
        and (
            not extended_contract
            or child.get("runtime_package", child["runtime"]) != "forge"
            or child["stability"].get("enhanced_plateau", {}).get("passed") is True
        )
        for child in children
    )
    timing_window_complete = all(
        statistics.median(child["raw_batch_ms"]) >= float(config["target_sample_ms"]) for child in children
    )
    expected_axes = {
        "forge": (
            "forge",
            (
                "native"
                if config["operation"].startswith("native_")
                or config["operation"]
                in (
                    "device_prefix_chain",
                    "active_grid_mpm",
                    "particle_spatial_hash",
                    "adaptive_pbd",
                    "marching_squares",
                    "bfs_worklist",
                    "falling_sand",
                )
                else "kernel"
            ),
        ),
        "forge_kernel": ("forge", "kernel"),
        "vanilla": ("vanilla", "kernel"),
        "vanilla_kernel": ("vanilla", "kernel"),
    }
    comparison_axis_verified = all(
        (child.get("runtime_package"), child.get("adapter_kind")) == expected_axes[child["runtime"]]
        for child in children
    )
    kernel_control_route_isolated = all(
        (
            (
                child["route"]["classification"].startswith(f"{child['runtime']}_")
                and "native" not in json.dumps(child["route"], sort_keys=True).lower()
                and (
                    child["operation"]
                    not in (
                        "native_reduce",
                        "native_transform",
                        "native_gather",
                        "native_scatter",
                        "native_compact",
                        "device_prefix_chain",
                        "active_grid_mpm",
                        "particle_spatial_hash",
                    )
                    or (
                        child["route"].get("adapter")
                        == (
                            "benchmark_defined_ti_kernel_pipeline"
                            if child["operation"] in ("native_compact", "device_prefix_chain", "particle_spatial_hash")
                            else (
                                "benchmark_defined_ti_kernel_graph_pipeline"
                                if child["operation"] == "active_grid_mpm"
                                else "benchmark_defined_ti_kernel"
                            )
                        )
                        and child["route"].get("kernel_source_owner") == "benchmark"
                        and child["route"].get("kernel_source_sha256")
                        == child["workload_contract"].get("kernel_source_sha256")
                        and child["route"].get("helper_api_used") is False
                        and (
                            (
                                child["route"].get("specialized_api_used") is False
                                and child["route"].get("benchmark_workspace_field_count") == 0
                                and child["route"].get("graph_kernel_names")
                                == child["workload_contract"].get("kernel_graph_kernel_names")
                                and child["route"].get("graph_dispatches_per_replay")
                                == child["workload_contract"].get("kernel_graph_dispatches_per_replay")
                            )
                            if child["operation"] == "active_grid_mpm"
                            else (
                                child["route"].get("specialized_api_used") is False
                                and child["route"].get("benchmark_workspace_field_count") == 2
                                and child["route"].get("scan_algorithm") == "inclusive_hillis_steele_ping_pong"
                                and child["route"].get("scan_steps")
                                == child["workload_contract"].get("kernel_scan_steps")
                                and child["route"].get("final_scan_copy_kernel_invocations")
                                == child["workload_contract"].get("kernel_final_scan_copy_kernel_invocations")
                                if child["operation"] == "native_compact"
                                else (
                                    child["route"].get("specialized_api_used") is False
                                    and child["route"].get("benchmark_workspace_field_count") == 4
                                    and child["route"].get("scan_algorithm") == "inclusive_hillis_steele_ping_pong"
                                    and child["route"].get("scan_pipelines_per_replay")
                                    == child["workload_contract"].get("kernel_scan_pipelines_per_replay")
                                    and child["route"].get("scan_steps_per_pipeline")
                                    == child["workload_contract"].get("kernel_scan_steps_per_pipeline")
                                    and child["route"].get("final_scan_copy_kernel_invocations_" "per_pipeline")
                                    == child["workload_contract"].get(
                                        "kernel_final_scan_copy_kernel_" "invocations_per_pipeline"
                                    )
                                    and child["route"].get("stage_ti_kernel_invocations_per_replay")
                                    == child["workload_contract"].get("kernel_stage_ti_invocations_per_replay")
                                    if child["operation"] == "device_prefix_chain"
                                    else (
                                        child["route"].get("specialized_api_used") is False
                                        and child["route"].get("benchmark_workspace_field_count") == 2
                                        and child["route"].get("scan_algorithm") == "inclusive_hillis_steele_ping_pong"
                                        and child["route"].get("scan_elements")
                                        == child["workload_contract"].get("kernel_scan_elements")
                                        and child["route"].get("scan_steps")
                                        == child["workload_contract"].get("kernel_scan_steps")
                                        and child["route"].get("final_scan_copy_kernel_" "invocations")
                                        == child["workload_contract"].get(
                                            "kernel_final_scan_copy_kernel_" "invocations"
                                        )
                                        and child["route"].get("non_scan_ti_kernel_" "invocations_per_replay")
                                        == child["workload_contract"].get(
                                            "kernel_non_scan_ti_" "invocations_per_replay"
                                        )
                                        and child["route"].get("stage_kernel_names")
                                        == child["workload_contract"].get("kernel_stage_names")
                                        if child["operation"] == "particle_spatial_hash"
                                        else child["route"].get("workspace_present") is False
                                    )
                                )
                            )
                        )
                        and child["route"].get("ti_kernel_invocations_per_replay")
                        == child["workload_contract"].get("kernel_ti_invocations_per_replay")
                        and child["route"].get("physical_backend_launches_assumed") is False
                    )
                )
            )
            if child["runtime"] in ("forge_kernel", "vanilla_kernel")
            else True
        )
        for child in children
    )
    adaptive_pbd_kernel_control_route_isolated = all(
        _adaptive_pbd_kernel_control_route_isolated(child) for child in children
    )
    marching_squares_kernel_control_route_isolated = all(
        _marching_squares_kernel_control_route_isolated(child) for child in children
    )
    bfs_worklist_kernel_control_route_isolated = all(
        _bfs_worklist_kernel_control_route_isolated(child) for child in children
    )
    bfs_worklist_native_route_admitted = all(_bfs_worklist_native_route_admitted(child) for child in children)
    falling_sand_kernel_control_route_isolated = all(
        _falling_sand_kernel_control_route_isolated(child) for child in children
    )
    falling_sand_native_route_admitted = all(_falling_sand_native_route_admitted(child) for child in children)
    sparse_block_stencil_route_isolated = all(_sparse_block_stencil_route_isolated(child) for child in children)
    ordinary_control_route_isolated = all(
        (
            child["route"].get("classification") == f"{child['runtime']}_ordinary_taichi_kernel"
            and child["route"].get("adapter") == "direct_ti_kernel"
            and child["route"].get("kernel_source_owner") == "benchmark"
            and child["route"].get("kernel_source_sha256") == child["workload_contract"].get("kernel_source_sha256")
            and child["route"].get("native_or_helper_api_used") is False
            and child["route"].get("ti_kernel_invocations_per_replay") == 1
            and child["route"].get("physical_backend_launches_assumed") is False
            if child["operation"] in ("fill", "copy", "saxpy", "stencil2d", "reduce_chunks")
            else True
        )
        for child in children
    )
    forge_binary_signatures = {
        (
            child["environment"]["package_distribution"],
            child["environment"]["package_version"],
            child["environment"]["package_path"],
            child["environment"]["core_path"],
            child["environment"]["core_sha256"],
            child["native_commit"],
        )
        for child in children
        if child.get("runtime_package", child["runtime"]) == "forge"
    }
    same_forge_binary_identity = bool(
        definition["name"] != "forge-native-vs-forge-kernel" or len(forge_binary_signatures) == 1
    )
    stable_replay_input = all(
        (
            child.get("measurement_scope") == "device_reset_plus_operation"
            if child["operation"] in ("prefix_sum", "parallel_sort", "sparse_block_stencil", "falling_sand")
            else True
        )
        for child in children
    )
    snode_lifecycle_plateau = all(
        child["operation"] not in ("snode_churn", "snode_concurrent", "sparse_block_stencil")
        or (
            child.get("stability") is not None
            and child["stability"].get("snode_lifecycle_plateau", {}).get("passed") is True
        )
        for child in children
    )
    batched_score_warmup = all(
        child.get("warmup_batch_size") == child.get("batch_size")
        and len(child.get("warmup_raw_batch_ms", [])) == int(config["warmups"])
        and all(math.isfinite(float(value)) and float(value) > 0.0 for value in child.get("warmup_raw_batch_ms", []))
        for child in children
    )
    if extended_contract:
        if "comparison_class_consistent" in summary.get("method_checks", {}):
            _check(
                summary["method_checks"]["comparison_class_consistent"] is (len(comparison_classes) == 1),
                "comparison class method check",
                failures,
            )
        _check(
            summary.get("method_checks", {}).get("physical_device_binding")
            is all(child["device_identity"]["binding_verified"] for child in children),
            "physical device method check",
            failures,
        )
        _check(
            summary.get("method_checks", {}).get("route_verified")
            is all(
                child["route"]["passed"] and child.get("route_before_scoring", child["route"])["passed"]
                for child in children
            ),
            "route method check",
            failures,
        )
        endpoint_key = (
            "endpoint_equivalence"
            if "endpoint_equivalence" in summary.get("method_checks", {})
            else "cross_runtime_endpoint_equivalence"
        )
        _check(
            summary.get("method_checks", {}).get(endpoint_key)
            is all(row["endpoint_equivalent"] for row in recomputed_rows),
            "comparison endpoint method check",
            failures,
        )
        if "comparison_axis_verified" in summary.get("method_checks", {}):
            _check(
                summary["method_checks"]["comparison_axis_verified"] is comparison_axis_verified,
                "comparison axis method check",
                failures,
            )
            if "kernel_control_route_isolated" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["kernel_control_route_isolated"] is kernel_control_route_isolated,
                    "kernel control route method check",
                    failures,
                )
            if "adaptive_pbd_kernel_control_route_isolated" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["adaptive_pbd_kernel_control_route_isolated"]
                    is adaptive_pbd_kernel_control_route_isolated,
                    "adaptive PBD kernel control route method check",
                    failures,
                )
            if "marching_squares_kernel_control_route_isolated" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["marching_squares_kernel_control_route_isolated"]
                    is marching_squares_kernel_control_route_isolated,
                    "Marching Squares kernel control route method check",
                    failures,
                )
            if "bfs_worklist_kernel_control_route_isolated" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["bfs_worklist_kernel_control_route_isolated"]
                    is bfs_worklist_kernel_control_route_isolated,
                    "BFS worklist kernel control route method check",
                    failures,
                )
            if "bfs_worklist_native_route_admitted" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["bfs_worklist_native_route_admitted"]
                    is bfs_worklist_native_route_admitted,
                    "BFS worklist native route method check",
                    failures,
                )
            if "falling_sand_kernel_control_route_isolated" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["falling_sand_kernel_control_route_isolated"]
                    is falling_sand_kernel_control_route_isolated,
                    "falling sand kernel control route method check",
                    failures,
                )
            if "falling_sand_native_route_admitted" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["falling_sand_native_route_admitted"]
                    is falling_sand_native_route_admitted,
                    "falling sand native route method check",
                    failures,
                )
            if "sparse_block_stencil_route_isolated" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["sparse_block_stencil_route_isolated"]
                    is sparse_block_stencil_route_isolated,
                    "sparse block stencil route method check",
                    failures,
                )
            if "ordinary_control_route_isolated" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["ordinary_control_route_isolated"] is ordinary_control_route_isolated,
                    "ordinary control route method check",
                    failures,
                )
            _check(
                summary["method_checks"]["same_forge_binary_identity"] is same_forge_binary_identity,
                "Forge binary identity method check",
                failures,
            )
            _check(
                summary["method_checks"]["stable_replay_input"] is stable_replay_input,
                "stable replay input method check",
                failures,
            )
            if "batched_score_warmup" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["batched_score_warmup"] is batched_score_warmup,
                    "batched score warmup method check",
                    failures,
                )
            if "snode_lifecycle_plateau" in summary["method_checks"]:
                _check(
                    summary["method_checks"]["snode_lifecycle_plateau"] is snode_lifecycle_plateau,
                    "SNode lifecycle plateau method check",
                    failures,
                )
    _check(
        summary.get("method_checks", {}).get("stability_complete") is stability_complete,
        "stability method check",
        failures,
    )
    all_method_checks = bool(
        not failures
        and order_counts[forward_order] == order_counts[reverse_order]
        and stability_complete
        and snode_lifecycle_plateau
        and ordinary_control_route_isolated
        and timing_window_complete
        and all(bool(value) for value in summary.get("method_checks", {}).values())
    )
    favorable_fraction = (
        sum(value > 1.0 for value in median_speedups) / len(median_speedups) if median_speedups else 0.0
    )
    maximum_cv = max((float(child["summary"]["cv_percent"]) for child in children), default=math.inf)
    expected_gates = {
        "qualification_policy": qualification_policy,
        "all_method_checks": all_method_checks,
        "paired_median_above_1_03": (
            bool(median_speedups) and statistics.median(math.log(value) for value in median_speedups) > math.log(1.03)
        ),
        "paired_bootstrap_low_above_1": (
            bool(median_speedups)
            and _bootstrap_median(median_speedups, int(config["seed"]))["bootstrap_95_low_x"] > 1.0
        ),
        "paired_p95_median_above_1": (
            bool(p95_speedups) and statistics.median(math.log(value) for value in p95_speedups) > 0.0
        ),
        "favorable_pair_fraction_at_least_0_8": favorable_fraction >= 0.8,
        "no_pair_below_0_97": bool(median_speedups) and min(median_speedups) >= 0.97,
        "max_child_cv_at_most_5_percent": maximum_cv <= 5.0,
    }
    _check(summary.get("claim_gate_results") == expected_gates, "claim gate recomputation", failures)
    _check(
        summary.get("ready_for_performance_claim") is all(expected_gates.values()),
        "claim eligibility recomputation",
        failures,
    )

    return {
        "schema": "taichi_forge.single_kernel_microbench.audit.v1",
        "run_id": summary["run_id"],
        "run_status": "completed",
        "audit_passed": not failures,
        "audit_failures": failures,
        "scored_child_count": len(children),
        "pair_count": pair_count,
        "ready_for_performance_claim": summary["ready_for_performance_claim"],
        "recomputed_paired": (
            None if not median_speedups else _bootstrap_median(median_speedups, int(summary["config"]["seed"]))
        ),
    }


def audit_artifact(run_dir: Path) -> dict[str, Any]:
    """Recompute one stored microbenchmark artifact without rewriting it."""
    return _audit(Path(run_dir).resolve())


def _write_reports(run_dir: Path, result: dict[str, Any]) -> None:
    status_zh = "通过" if result["audit_passed"] else "失败"
    status_en = "pass" if result["audit_passed"] else "fail"
    failures = result["audit_failures"]
    failure_zh = "无" if not failures else "、".join(failures)
    failure_en = "none" if not failures else ", ".join(failures)
    run_status_zh = "运行失败" if result["run_status"] == "failed" else "运行完成"
    (run_dir / "audit.zh-CN.md").write_text(
        "# 独立 artifact 审计\n\n"
        f"- Run ID：`{result['run_id']}`\n"
        f"- Run 状态：{run_status_zh}\n"
        f"- 审计结果：{status_zh}\n"
        f"- 计分子进程：{result['scored_child_count']}\n"
        f"- A/B 对：{result['pair_count']}\n"
        f"- 性能宣称资格：{'通过' if result['ready_for_performance_claim'] else '未通过'}\n"
        f"- 审计失败项：{failure_zh}\n",
        encoding="utf-8",
    )
    (run_dir / "audit.en.md").write_text(
        "# Independent artifact audit\n\n"
        f"- Run ID: `{result['run_id']}`\n"
        f"- Run status: {result['run_status']}\n"
        f"- Audit result: {status_en}\n"
        f"- Scored child processes: {result['scored_child_count']}\n"
        f"- A/B pairs: {result['pair_count']}\n"
        f"- Performance-claim eligibility: "
        f"{'pass' if result['ready_for_performance_claim'] else 'fail'}\n"
        f"- Audit failures: {failure_en}\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently recompute and validate one microbench run")
    parser.add_argument("run_directory")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_directory).resolve()
    result = audit_artifact(run_dir)
    (run_dir / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_reports(run_dir, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
