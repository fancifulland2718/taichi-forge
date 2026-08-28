import gc
import os

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


_FUSION_ENV = "TAICHI_FORGE_INTERNAL_MAP_FUSION"


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_augmented_pointwise_update_stays_an_atomic_fusion_blocker(monkeypatch):
    monkeypatch.setenv(_FUSION_ENV, "map4")

    @ti.kernel
    def atomic_increment(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] += 1

    values_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    for _ in range(4):
        builder.dispatch(atomic_increment, values_arg)
    graph = builder.compile()

    fusion = graph._ir_debug_info["fusion_plan"]
    assert graph.physical_plan()["physical_dispatch_count"] == 4
    assert fusion["candidate_groups"] == 0
    assert fusion["blockers"] == {"atomic_effect": 4}
    assert fusion["applied_groups"] == 0

    values = ti.ndarray(ti.i32, shape=257)
    values.fill(0)
    graph.run({"values": values})
    np.testing.assert_array_equal(
        values.to_numpy(), np.full(257, 4, dtype=np.int32)
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_map4_fusion_preserves_runtime_alias_order(monkeypatch):
    monkeypatch.setenv(_FUSION_ENV, "map4")

    @ti.kernel
    def stage_one(
        left: ti.types.ndarray(dtype=ti.i32, ndim=1),
        right: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in left:
            right[i] = left[i] * 2

    @ti.kernel
    def stage_two(
        left: ti.types.ndarray(dtype=ti.i32, ndim=1),
        right: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in left:
            left[i] = right[i] + 3

    @ti.kernel
    def stage_three(
        left: ti.types.ndarray(dtype=ti.i32, ndim=1),
        right: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in left:
            right[i] = left[i] * 4

    @ti.kernel
    def stage_four(
        left: ti.types.ndarray(dtype=ti.i32, ndim=1),
        right: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in left:
            left[i] = right[i] - 5

    left = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "left", ti.i32, ndim=1)
    right = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "right", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(stage_one, left, right)
    builder.dispatch(stage_two, left, right)
    builder.dispatch(stage_three, left, right)
    builder.dispatch(stage_four, left, right)
    graph = builder.compile()

    assert graph.physical_plan()["physical_dispatch_count"] == 1
    assert graph._ir_debug_info["fusion_plan"]["unmatched_applied_groups"] == 0

    rng = np.random.default_rng(20260828)
    source = rng.integers(-1000, 1000, size=1021, dtype=np.int32)
    shared = ti.ndarray(ti.i32, shape=source.size)
    shared.from_numpy(source)
    graph.run({"left": shared, "right": shared})
    np.testing.assert_array_equal(
        shared.to_numpy(), (source * 2 + 3) * 4 - 5
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_map4_physics_vector_chain_matches_unfused(monkeypatch):
    @ti.kernel
    def integrate_velocity(
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=1),
        force: ti.types.ndarray(dtype=ti.f32, ndim=1),
        inverse_mass: ti.types.ndarray(dtype=ti.f32, ndim=1),
        dt: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            velocity[i] = velocity[i] + dt * force[i] * inverse_mass[i]

    @ti.kernel
    def integrate_position(
        position: ti.types.ndarray(dtype=ti.f32, ndim=1),
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=1),
        dt: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            position[i] = position[i] + dt * velocity[i]

    @ti.kernel
    def project_ground(
        position: ti.types.ndarray(dtype=ti.f32, ndim=1),
        correction: ti.types.ndarray(dtype=ti.f32, ndim=1),
        count: ti.i32,
    ):
        for i in range(count):
            correction[i] = ti.min(position[i], 0.0)

    @ti.kernel
    def apply_correction(
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=1),
        correction: ti.types.ndarray(dtype=ti.f32, ndim=1),
        dt: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            velocity[i] = velocity[i] - correction[i] / dt

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("position", "velocity", "force", "inverse_mass", "correction")
    }
    symbolic["dt"] = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "dt", ti.f32)
    symbolic["count"] = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "count", ti.i32
    )

    def build(recipe):
        monkeypatch.setenv(_FUSION_ENV, recipe)
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            integrate_velocity,
            symbolic["velocity"],
            symbolic["force"],
            symbolic["inverse_mass"],
            symbolic["dt"],
            symbolic["count"],
        )
        builder.dispatch(
            integrate_position,
            symbolic["position"],
            symbolic["velocity"],
            symbolic["dt"],
            symbolic["count"],
        )
        builder.dispatch(
            project_ground,
            symbolic["position"],
            symbolic["correction"],
            symbolic["count"],
        )
        builder.dispatch(
            apply_correction,
            symbolic["velocity"],
            symbolic["correction"],
            symbolic["dt"],
            symbolic["count"],
        )
        return builder.compile()

    baseline = build("baseline")
    fused = build("map4")
    assert baseline.physical_plan()["physical_dispatch_count"] == 4
    assert fused.physical_plan()["physical_dispatch_count"] == 1
    assert (
        baseline._executable_optimization_space.semantic_plan_id
        == fused._executable_optimization_space.semantic_plan_id
    )

    rng = np.random.default_rng(42)
    count = 4099
    inputs = {
        "position": rng.normal(0.0, 2.0, count).astype(np.float32),
        "velocity": rng.normal(0.0, 3.0, count).astype(np.float32),
        "force": rng.normal(0.0, 5.0, count).astype(np.float32),
        "inverse_mass": rng.uniform(0.1, 2.0, count).astype(np.float32),
    }

    def execute(graph):
        arrays = {
            name: ti.ndarray(ti.f32, shape=count)
            for name in (
                "position",
                "velocity",
                "force",
                "inverse_mass",
                "correction",
            )
        }
        for name, values in inputs.items():
            arrays[name].from_numpy(values)
        arrays["correction"].fill(0.0)
        graph.run({**arrays, "dt": 0.01, "count": count})
        return arrays["position"].to_numpy(), arrays["velocity"].to_numpy()

    baseline_position, baseline_velocity = execute(baseline)
    fused_position, fused_velocity = execute(fused)
    np.testing.assert_allclose(fused_position, baseline_position, rtol=0, atol=0)
    np.testing.assert_allclose(fused_velocity, baseline_velocity, rtol=0, atol=0)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_map4_replay_memory_identity_and_reset_lifecycle(monkeypatch):
    monkeypatch.setenv(_FUSION_ENV, "map4")

    @ti.kernel
    def add_one(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] = values[i] + 1

    values_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.i32, ndim=1
    )

    def build():
        builder = ti.graph.GraphBuilder()
        for _ in range(4):
            builder.dispatch(add_one, values_arg)
        return builder.compile()

    graphs = tuple(build() for _ in range(3))
    identities = tuple(
        graph._executable_optimization_space.selected.compilation_identity
        for graph in graphs
    )
    assert len(set(identities)) == 1
    assert all(
        graph.physical_plan()["physical_dispatch_count"] == 1
        for graph in graphs
    )

    graph = graphs[0]
    values = ti.ndarray(ti.i32, shape=257)
    values.fill(0)
    graph.run({"values": values})
    ti.sync()
    program = impl.get_runtime().prog
    gc.collect()
    memory_before = program._runtime_statistics_snapshot()["memory"]
    iterations = int(os.environ.get("TI_MAP_FUSION_STRESS_ITERATIONS", "256"))
    assert 1 <= iterations <= 10_000
    for _ in range(iterations):
        graph.run({"values": values})
    ti.sync()
    gc.collect()
    memory_after = program._runtime_statistics_snapshot()["memory"]
    assert memory_after == memory_before
    np.testing.assert_array_equal(
        values.to_numpy(), np.full(257, 4 * (iterations + 1), dtype=np.int32)
    )
    if impl.current_cfg().arch == ti.vulkan:
        segment = graph.execution_stats().segments[0]
        assert segment.last_path == "vulkan_replay", (
            segment.last_path,
            segment.fallback_reason,
        )
        assert segment.fallback_reason == "none"

        ordinary_builder = ti.graph.GraphBuilder()
        ordinary_builder.dispatch(add_one, values_arg)
        ordinary = ordinary_builder.compile()
        ordinary_values = ti.ndarray(ti.i32, shape=16)
        ordinary_values.fill(0)
        ordinary.run({"values": ordinary_values})
        ordinary.run({"values": ordinary_values})
        ti.sync()
        ordinary_segment = ordinary.execution_stats().segments[0]
        assert ordinary_segment.last_path == "ordinary_fallback"
        assert ordinary_segment.fallback_reason == "insufficient_dispatches"

    ti.reset()
    with pytest.raises(RuntimeError, match="reset|stale|valid"):
        graph.run({"values": values})
    with pytest.raises(RuntimeError, match="reset|stale|valid"):
        _ = graph._executable_optimization_space
