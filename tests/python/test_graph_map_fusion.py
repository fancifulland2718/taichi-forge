import gc
import os

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.lang import impl
from tests import test_utils


_FUSION_ENV = "TAICHI_FORGE_INTERNAL_MAP_FUSION"


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_map_recipe_identity_tracks_source_kernel_code(monkeypatch):
    monkeypatch.setenv(_FUSION_ENV, "map2")

    @ti.kernel
    def producer_add(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            temporary[i] = source[i] + 1

    @ti.kernel
    def producer_multiply(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            temporary[i] = source[i] * 2

    @ti.kernel
    def consumer(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        temporary: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = temporary[i] - source[i]

    symbolic = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
        for name in ("source", "temporary", "output")
    }

    def build(producer):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            producer,
            symbolic["source"],
            symbolic["temporary"],
        )
        builder.dispatch(
            consumer,
            symbolic["source"],
            symbolic["temporary"],
            symbolic["output"],
        )
        return builder.compile()

    add_graph = build(producer_add)
    multiply_graph = build(producer_multiply)
    add_recipe = add_graph._ir_debug_info["fusion_plan"]["recipes"][0]
    multiply_recipe = multiply_graph._ir_debug_info["fusion_plan"]["recipes"][0]

    assert add_recipe["schema_version"] == 2
    assert multiply_recipe["schema_version"] == 2
    assert all(add_recipe["source_kernel_identities"])
    assert all(multiply_recipe["source_kernel_identities"])
    assert (
        add_recipe["source_kernel_identities"][1]
        == multiply_recipe["source_kernel_identities"][1]
    )
    assert (
        add_recipe["source_kernel_identities"][0]
        != multiply_recipe["source_kernel_identities"][0]
    )
    assert add_recipe["recipe_id"] != multiply_recipe["recipe_id"]
    assert (
        add_graph._executable_optimization_space.semantic_plan_id
        != multiply_graph._executable_optimization_space.semantic_plan_id
    )


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


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_map4_particle_contact_chain_matches_unfused(monkeypatch):
    @ti.kernel
    def integrate_velocity(
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        force: ti.types.ndarray(dtype=ti.f32, ndim=2),
        inverse_mass: ti.types.ndarray(dtype=ti.f32, ndim=1),
        dt: ti.f32,
        gravity: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            velocity[i, 0] = (
                velocity[i, 0] + dt * force[i, 0] * inverse_mass[i]
            )
            velocity[i, 1] = velocity[i, 1] + dt * (
                force[i, 1] * inverse_mass[i] + gravity
            )
            velocity[i, 2] = (
                velocity[i, 2] + dt * force[i, 2] * inverse_mass[i]
            )

    @ti.kernel
    def integrate_position(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        dt: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            for axis in ti.static(range(3)):
                position[i, axis] = (
                    position[i, axis] + dt * velocity[i, axis]
                )

    @ti.kernel
    def project_plane_contact(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        correction: ti.types.ndarray(dtype=ti.f32, ndim=2),
        friction: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            correction[i, 0] = 0.0
            correction[i, 1] = 0.0
            correction[i, 2] = 0.0
            if position[i, 1] < 0.0:
                normal_delta = ti.max(-velocity[i, 1], 0.0)
                normal_delta = normal_delta + ti.min(
                    -position[i, 1] * 32.0, 8.0
                )
                tangent_speed = ti.sqrt(
                    velocity[i, 0] * velocity[i, 0]
                    + velocity[i, 2] * velocity[i, 2]
                    + 1e-12
                )
                tangent_scale = ti.max(
                    0.0,
                    1.0 - friction * normal_delta / tangent_speed,
                )
                correction[i, 0] = (
                    tangent_scale - 1.0
                ) * velocity[i, 0]
                correction[i, 1] = normal_delta
                correction[i, 2] = (
                    tangent_scale - 1.0
                ) * velocity[i, 2]

    @ti.kernel
    def apply_contact(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        correction: ti.types.ndarray(dtype=ti.f32, ndim=2),
        count: ti.i32,
    ):
        for i in range(count):
            for axis in ti.static(range(3)):
                velocity[i, axis] = (
                    velocity[i, axis] + correction[i, axis]
                )
            position[i, 1] = ti.max(position[i, 1], 0.0)

    symbolic = {
        name: ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY,
            name,
            ti.f32,
            ndim=2 if name != "inverse_mass" else 1,
        )
        for name in (
            "position",
            "velocity",
            "force",
            "inverse_mass",
            "correction",
        )
    }
    for name in ("dt", "gravity", "friction"):
        symbolic[name] = ti.graph.Arg(
            ti.graph.ArgKind.SCALAR, name, ti.f32
        )
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
            symbolic["gravity"],
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
            project_plane_contact,
            symbolic["position"],
            symbolic["velocity"],
            symbolic["correction"],
            symbolic["friction"],
            symbolic["count"],
        )
        builder.dispatch(
            apply_contact,
            symbolic["position"],
            symbolic["velocity"],
            symbolic["correction"],
            symbolic["count"],
        )
        return builder.compile()

    baseline = build("baseline")
    fused = build("map4")
    assert baseline.physical_plan()["physical_dispatch_count"] == 4
    assert fused.physical_plan()["physical_dispatch_count"] == 1, (
        fused._ir_debug_info["fusion_plan"]
    )

    rng = np.random.default_rng(20260828)
    count = 4099
    inputs = {
        "position": rng.normal(0.0, 1.5, (count, 3)).astype(np.float32),
        "velocity": rng.normal(0.0, 4.0, (count, 3)).astype(np.float32),
        "force": rng.normal(0.0, 10.0, (count, 3)).astype(np.float32),
        "inverse_mass": rng.uniform(0.0, 2.0, count).astype(np.float32),
    }

    def execute(graph):
        arrays = {
            name: ti.ndarray(ti.f32, shape=values.shape)
            for name, values in inputs.items()
        }
        arrays["correction"] = ti.ndarray(ti.f32, shape=(count, 3))
        for name, values in inputs.items():
            arrays[name].from_numpy(values)
        arrays["correction"].fill(0.0)
        graph.run(
            {
                **arrays,
                "dt": 1.0 / 120.0,
                "gravity": -9.81,
                "friction": 0.45,
                "count": count,
            }
        )
        return (
            arrays["position"].to_numpy(),
            arrays["velocity"].to_numpy(),
            arrays["correction"].to_numpy(),
        )

    baseline_results = execute(baseline)
    fused_results = execute(fused)
    for baseline_value, fused_value in zip(baseline_results, fused_results):
        np.testing.assert_allclose(fused_value, baseline_value, rtol=0, atol=0)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_multicomponent_dynamic_lane_stays_a_fusion_blocker(monkeypatch):
    monkeypatch.setenv(_FUSION_ENV, "map2")

    @ti.kernel
    def dynamic_lane(
        source: ti.types.ndarray(dtype=ti.f32, ndim=2),
        output: ti.types.ndarray(dtype=ti.f32, ndim=1),
        count: ti.i32,
    ):
        for i in range(count):
            output[i] = source[i, i % 3]

    source = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=2
    )
    output = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    count = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(dynamic_lane, source, output, count)
    builder.dispatch(dynamic_lane, source, output, count)
    graph = builder.compile()

    fusion = graph._ir_debug_info["fusion_plan"]
    assert graph.physical_plan()["physical_dispatch_count"] == 2
    assert fusion["candidate_groups"] == 0
    assert fusion["blockers"] == {"non_pointwise_access": 2}


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_map4_graph_rejects_active_ad_without_poisoning_replay(monkeypatch):
    monkeypatch.setenv(_FUSION_ENV, "map4")

    @ti.kernel
    def add_one(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in values:
            values[i] = values[i] + 1.0

    values_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "values", ti.f32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    for _ in range(4):
        builder.dispatch(add_one, values_arg)
    graph = builder.compile()
    assert graph.physical_plan()["physical_dispatch_count"] == 1

    values = ti.ndarray(ti.f32, shape=257)
    values.fill(0.0)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    with ti.ad.Tape(loss=loss):
        with pytest.raises(RuntimeError, match="primal-only"):
            graph.run({"values": values})

    graph.run({"values": values})
    np.testing.assert_array_equal(
        values.to_numpy(), np.full(257, 4.0, dtype=np.float32)
    )


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
