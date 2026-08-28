import numpy as np

import taichi_forge as ti
from tests import test_utils


_FUSION_ENV = "TAICHI_FORGE_INTERNAL_MAP_FUSION"


def _ndarray_args(names, dtype, ndim):
    return {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=ndim)
        for name in names
    }


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_rigid_integrate_orientation_and_aabb_map3(monkeypatch):
    @ti.kernel
    def integrate_velocity(
        linear_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        angular_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        force: ti.types.ndarray(dtype=ti.f32, ndim=2),
        torque: ti.types.ndarray(dtype=ti.f32, ndim=2),
        inverse_mass: ti.types.ndarray(dtype=ti.f32, ndim=1),
        inverse_inertia: ti.types.ndarray(dtype=ti.f32, ndim=2),
        dt: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            for axis in ti.static(range(3)):
                linear_velocity[i, axis] = linear_velocity[i, axis] + (
                    dt * force[i, axis] * inverse_mass[i]
                )
                angular_velocity[i, axis] = angular_velocity[i, axis] + (
                    dt * torque[i, axis] * inverse_inertia[i, axis]
                )

    @ti.kernel
    def integrate_pose(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        orientation: ti.types.ndarray(dtype=ti.f32, ndim=2),
        linear_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        angular_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        dt: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            for axis in ti.static(range(3)):
                position[i, axis] = position[i, axis] + dt * linear_velocity[i, axis]
            w = orientation[i, 0]
            x = orientation[i, 1]
            y = orientation[i, 2]
            z = orientation[i, 3]
            wx = angular_velocity[i, 0]
            wy = angular_velocity[i, 1]
            wz = angular_velocity[i, 2]
            nw = w + 0.5 * dt * (-wx * x - wy * y - wz * z)
            nx = x + 0.5 * dt * (wx * w + wz * y - wy * z)
            ny = y + 0.5 * dt * (wy * w - wz * x + wx * z)
            nz = z + 0.5 * dt * (wz * w + wy * x - wx * y)
            inv_norm = ti.rsqrt(nw * nw + nx * nx + ny * ny + nz * nz)
            orientation[i, 0] = nw * inv_norm
            orientation[i, 1] = nx * inv_norm
            orientation[i, 2] = ny * inv_norm
            orientation[i, 3] = nz * inv_norm

    @ti.kernel
    def update_aabb(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        orientation: ti.types.ndarray(dtype=ti.f32, ndim=2),
        half_extent: ti.types.ndarray(dtype=ti.f32, ndim=2),
        aabb_min: ti.types.ndarray(dtype=ti.f32, ndim=2),
        aabb_max: ti.types.ndarray(dtype=ti.f32, ndim=2),
        count: ti.i32,
    ):
        for i in range(count):
            w = orientation[i, 0]
            x = orientation[i, 1]
            y = orientation[i, 2]
            z = orientation[i, 3]
            hx = half_extent[i, 0]
            hy = half_extent[i, 1]
            hz = half_extent[i, 2]
            ex = (
                ti.abs(1.0 - 2.0 * (y * y + z * z)) * hx
                + ti.abs(2.0 * (x * y - z * w)) * hy
                + ti.abs(2.0 * (x * z + y * w)) * hz
            )
            ey = (
                ti.abs(2.0 * (x * y + z * w)) * hx
                + ti.abs(1.0 - 2.0 * (x * x + z * z)) * hy
                + ti.abs(2.0 * (y * z - x * w)) * hz
            )
            ez = (
                ti.abs(2.0 * (x * z - y * w)) * hx
                + ti.abs(2.0 * (y * z + x * w)) * hy
                + ti.abs(1.0 - 2.0 * (x * x + y * y)) * hz
            )
            aabb_min[i, 0] = position[i, 0] - ex
            aabb_min[i, 1] = position[i, 1] - ey
            aabb_min[i, 2] = position[i, 2] - ez
            aabb_max[i, 0] = position[i, 0] + ex
            aabb_max[i, 1] = position[i, 1] + ey
            aabb_max[i, 2] = position[i, 2] + ez

    vec3_names = (
        "position",
        "linear_velocity",
        "angular_velocity",
        "force",
        "torque",
        "inverse_inertia",
        "half_extent",
        "aabb_min",
        "aabb_max",
    )
    symbolic = _ndarray_args(vec3_names, ti.f32, 2)
    symbolic["orientation"] = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "orientation", ti.f32, ndim=2
    )
    symbolic["inverse_mass"] = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "inverse_mass", ti.f32, ndim=1
    )
    symbolic["dt"] = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "dt", ti.f32)
    symbolic["count"] = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)

    def build(recipe):
        monkeypatch.setenv(_FUSION_ENV, recipe)
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            integrate_velocity,
            symbolic["linear_velocity"],
            symbolic["angular_velocity"],
            symbolic["force"],
            symbolic["torque"],
            symbolic["inverse_mass"],
            symbolic["inverse_inertia"],
            symbolic["dt"],
            symbolic["count"],
        )
        builder.dispatch(
            integrate_pose,
            symbolic["position"],
            symbolic["orientation"],
            symbolic["linear_velocity"],
            symbolic["angular_velocity"],
            symbolic["dt"],
            symbolic["count"],
        )
        builder.dispatch(
            update_aabb,
            symbolic["position"],
            symbolic["orientation"],
            symbolic["half_extent"],
            symbolic["aabb_min"],
            symbolic["aabb_max"],
            symbolic["count"],
        )
        return builder.compile()

    baseline = build("baseline")
    fused = build("map3")
    assert baseline.physical_plan()["physical_dispatch_count"] == 3
    assert fused.physical_plan()["physical_dispatch_count"] == 1

    rng = np.random.default_rng(101)
    count = 2053
    host = {
        name: rng.normal(0.0, 1.0, (count, 3)).astype(np.float32)
        for name in vec3_names[:-2]
    }
    host["inverse_inertia"] = rng.uniform(0.1, 2.0, (count, 3)).astype(np.float32)
    host["half_extent"] = rng.uniform(0.01, 1.5, (count, 3)).astype(np.float32)
    orientation = rng.normal(0.0, 1.0, (count, 4)).astype(np.float32)
    orientation /= np.linalg.norm(orientation, axis=1, keepdims=True)
    inverse_mass = rng.uniform(0.0, 2.0, count).astype(np.float32)

    def execute(graph):
        arrays = {name: ti.ndarray(ti.f32, shape=(count, 3)) for name in vec3_names}
        for name, value in host.items():
            arrays[name].from_numpy(value)
        arrays["aabb_min"].fill(0.0)
        arrays["aabb_max"].fill(0.0)
        arrays["orientation"] = ti.ndarray(ti.f32, shape=(count, 4))
        arrays["orientation"].from_numpy(orientation)
        arrays["inverse_mass"] = ti.ndarray(ti.f32, shape=count)
        arrays["inverse_mass"].from_numpy(inverse_mass)
        graph.run({**arrays, "dt": 1.0 / 240.0, "count": count})
        return {
            name: arrays[name].to_numpy()
            for name in ("position", "orientation", "aabb_min", "aabb_max")
        }

    baseline_result = execute(baseline)
    fused_result = execute(fused)
    for name in baseline_result:
        np.testing.assert_allclose(
            fused_result[name], baseline_result[name], rtol=2e-6, atol=2e-6
        )
    np.testing.assert_allclose(
        np.linalg.norm(fused_result["orientation"], axis=1),
        1.0,
        rtol=2e-6,
        atol=2e-6,
    )
    assert np.all(fused_result["aabb_min"] <= fused_result["aabb_max"])


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_mpm_particle_local_state_and_boundary_map4(monkeypatch):
    @ti.kernel
    def update_deformation(
        deformation: ti.types.ndarray(dtype=ti.f32, ndim=2),
        affine: ti.types.ndarray(dtype=ti.f32, ndim=2),
        dt: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            f00 = deformation[i, 0]
            f01 = deformation[i, 1]
            f10 = deformation[i, 2]
            f11 = deformation[i, 3]
            c00 = affine[i, 0]
            c01 = affine[i, 1]
            c10 = affine[i, 2]
            c11 = affine[i, 3]
            deformation[i, 0] = (1.0 + dt * c00) * f00 + dt * c01 * f10
            deformation[i, 1] = (1.0 + dt * c00) * f01 + dt * c01 * f11
            deformation[i, 2] = dt * c10 * f00 + (1.0 + dt * c11) * f10
            deformation[i, 3] = dt * c10 * f01 + (1.0 + dt * c11) * f11

    @ti.kernel
    def update_constitutive(
        deformation: ti.types.ndarray(dtype=ti.f32, ndim=2),
        volume_ratio: ti.types.ndarray(dtype=ti.f32, ndim=1),
        pressure: ti.types.ndarray(dtype=ti.f32, ndim=1),
        bulk: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            determinant = (
                deformation[i, 0] * deformation[i, 3]
                - deformation[i, 1] * deformation[i, 2]
            )
            volume_ratio[i] = determinant
            pressure[i] = bulk * (determinant - 1.0)

    @ti.kernel
    def advect_particles(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        dt: ti.f32,
        gravity: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            velocity[i, 1] = velocity[i, 1] + dt * gravity
            position[i, 0] = position[i, 0] + dt * velocity[i, 0]
            position[i, 1] = position[i, 1] + dt * velocity[i, 1]

    @ti.kernel
    def project_boundary(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        boundary: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            for axis in ti.static(range(2)):
                if position[i, axis] < boundary:
                    position[i, axis] = boundary
                    velocity[i, axis] = ti.max(velocity[i, axis], 0.0)
                if position[i, axis] > 1.0 - boundary:
                    position[i, axis] = 1.0 - boundary
                    velocity[i, axis] = ti.min(velocity[i, axis], 0.0)

    symbolic = {
        "position": ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "position", ti.f32, ndim=2),
        "velocity": ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "velocity", ti.f32, ndim=2),
        "deformation": ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "deformation", ti.f32, ndim=2
        ),
        "affine": ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "affine", ti.f32, ndim=2),
        "volume_ratio": ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, "volume_ratio", ti.f32, ndim=1
        ),
        "pressure": ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "pressure", ti.f32, ndim=1),
        "dt": ti.graph.Arg(ti.graph.ArgKind.SCALAR, "dt", ti.f32),
        "gravity": ti.graph.Arg(ti.graph.ArgKind.SCALAR, "gravity", ti.f32),
        "bulk": ti.graph.Arg(ti.graph.ArgKind.SCALAR, "bulk", ti.f32),
        "boundary": ti.graph.Arg(ti.graph.ArgKind.SCALAR, "boundary", ti.f32),
        "count": ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32),
    }

    def build(recipe):
        monkeypatch.setenv(_FUSION_ENV, recipe)
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            update_deformation,
            symbolic["deformation"],
            symbolic["affine"],
            symbolic["dt"],
            symbolic["count"],
        )
        builder.dispatch(
            update_constitutive,
            symbolic["deformation"],
            symbolic["volume_ratio"],
            symbolic["pressure"],
            symbolic["bulk"],
            symbolic["count"],
        )
        builder.dispatch(
            advect_particles,
            symbolic["position"],
            symbolic["velocity"],
            symbolic["dt"],
            symbolic["gravity"],
            symbolic["count"],
        )
        builder.dispatch(
            project_boundary,
            symbolic["position"],
            symbolic["velocity"],
            symbolic["boundary"],
            symbolic["count"],
        )
        return builder.compile()

    baseline = build("baseline")
    fused = build("map4")
    assert baseline.physical_plan()["physical_dispatch_count"] == 4
    assert fused.physical_plan()["physical_dispatch_count"] == 1

    rng = np.random.default_rng(202)
    count = 2053
    host = {
        "position": rng.uniform(-0.02, 1.02, (count, 2)).astype(np.float32),
        "velocity": rng.normal(0.0, 2.0, (count, 2)).astype(np.float32),
        "deformation": (
            np.tile(np.eye(2, dtype=np.float32).reshape(1, 4), (count, 1))
            + rng.normal(0.0, 0.02, (count, 4)).astype(np.float32)
        ),
        "affine": rng.normal(0.0, 0.4, (count, 4)).astype(np.float32),
    }

    def execute(graph):
        arrays = {
            name: ti.ndarray(ti.f32, shape=value.shape) for name, value in host.items()
        }
        for name, value in host.items():
            arrays[name].from_numpy(value)
        arrays["volume_ratio"] = ti.ndarray(ti.f32, shape=count)
        arrays["pressure"] = ti.ndarray(ti.f32, shape=count)
        graph.run(
            {
                **arrays,
                "dt": 1.0 / 120.0,
                "gravity": -9.81,
                "bulk": 24.0,
                "boundary": 0.025,
                "count": count,
            }
        )
        return {
            name: arrays[name].to_numpy()
            for name in (
                "position",
                "velocity",
                "deformation",
                "volume_ratio",
                "pressure",
            )
        }

    baseline_result = execute(baseline)
    fused_result = execute(fused)
    for name in baseline_result:
        np.testing.assert_array_equal(fused_result[name], baseline_result[name])
    assert np.all(fused_result["position"] >= np.float32(0.025))
    assert np.all(fused_result["position"] <= np.float32(0.975))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_fem_node_integrate_and_fixed_boundary_map3(monkeypatch):
    @ti.kernel
    def integrate_velocity(
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        force: ti.types.ndarray(dtype=ti.f32, ndim=2),
        inverse_mass: ti.types.ndarray(dtype=ti.f32, ndim=1),
        dt: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            for axis in ti.static(range(3)):
                velocity[i, axis] = (
                    velocity[i, axis] + dt * force[i, axis] * inverse_mass[i]
                )

    @ti.kernel
    def predict_position(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        dt: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            for axis in ti.static(range(3)):
                position[i, axis] = position[i, axis] + dt * velocity[i, axis]

    @ti.kernel
    def apply_fixed_boundary(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        rest_position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        fixed: ti.types.ndarray(dtype=ti.i32, ndim=1),
        count: ti.i32,
    ):
        for i in range(count):
            if fixed[i] != 0:
                for axis in ti.static(range(3)):
                    position[i, axis] = rest_position[i, axis]
                    velocity[i, axis] = 0.0

    symbolic = _ndarray_args(
        ("position", "rest_position", "velocity", "force"), ti.f32, 2
    )
    symbolic["inverse_mass"] = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "inverse_mass", ti.f32, ndim=1
    )
    symbolic["fixed"] = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "fixed", ti.i32, ndim=1)
    symbolic["dt"] = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "dt", ti.f32)
    symbolic["count"] = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)

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
            predict_position,
            symbolic["position"],
            symbolic["velocity"],
            symbolic["dt"],
            symbolic["count"],
        )
        builder.dispatch(
            apply_fixed_boundary,
            symbolic["position"],
            symbolic["rest_position"],
            symbolic["velocity"],
            symbolic["fixed"],
            symbolic["count"],
        )
        return builder.compile()

    baseline = build("baseline")
    fused = build("map3")
    assert baseline.physical_plan()["physical_dispatch_count"] == 3
    assert fused.physical_plan()["physical_dispatch_count"] == 1

    rng = np.random.default_rng(303)
    count = 2053
    rest = rng.normal(0.0, 1.0, (count, 3)).astype(np.float32)
    host = {
        "position": rest + rng.normal(0.0, 0.1, (count, 3)).astype(np.float32),
        "rest_position": rest,
        "velocity": rng.normal(0.0, 2.0, (count, 3)).astype(np.float32),
        "force": rng.normal(0.0, 5.0, (count, 3)).astype(np.float32),
        "inverse_mass": rng.uniform(0.0, 2.0, count).astype(np.float32),
        "fixed": (np.arange(count) % 17 == 0).astype(np.int32),
    }

    def execute(graph):
        arrays = {
            name: ti.ndarray(
                ti.i32 if value.dtype == np.int32 else ti.f32,
                shape=value.shape,
            )
            for name, value in host.items()
        }
        for name, value in host.items():
            arrays[name].from_numpy(value)
        graph.run({**arrays, "dt": 1.0 / 240.0, "count": count})
        return arrays["position"].to_numpy(), arrays["velocity"].to_numpy()

    baseline_position, baseline_velocity = execute(baseline)
    fused_position, fused_velocity = execute(fused)
    np.testing.assert_array_equal(fused_position, baseline_position)
    np.testing.assert_array_equal(fused_velocity, baseline_velocity)
    fixed = host["fixed"].astype(bool)
    np.testing.assert_array_equal(fused_position[fixed], rest[fixed])
    np.testing.assert_array_equal(fused_velocity[fixed], 0.0)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_pbd_predict_and_precomputed_correction_apply_map4(monkeypatch):
    @ti.kernel
    def predict(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        predicted: ti.types.ndarray(dtype=ti.f32, ndim=2),
        dt: ti.f32,
        gravity: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            predicted[i, 0] = position[i, 0] + dt * velocity[i, 0]
            predicted[i, 1] = position[i, 1] + dt * velocity[i, 1] + dt * dt * gravity
            predicted[i, 2] = position[i, 2] + dt * velocity[i, 2]

    @ti.kernel
    def apply_precomputed_correction(
        predicted: ti.types.ndarray(dtype=ti.f32, ndim=2),
        correction: ti.types.ndarray(dtype=ti.f32, ndim=2),
        inverse_mass: ti.types.ndarray(dtype=ti.f32, ndim=1),
        count: ti.i32,
    ):
        for i in range(count):
            for axis in ti.static(range(3)):
                predicted[i, axis] = (
                    predicted[i, axis] + inverse_mass[i] * correction[i, axis]
                )

    @ti.kernel
    def update_velocity(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        predicted: ti.types.ndarray(dtype=ti.f32, ndim=2),
        inverse_dt: ti.f32,
        count: ti.i32,
    ):
        for i in range(count):
            for axis in ti.static(range(3)):
                velocity[i, axis] = (
                    predicted[i, axis] - position[i, axis]
                ) * inverse_dt

    @ti.kernel
    def finalize(
        position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        predicted: ti.types.ndarray(dtype=ti.f32, ndim=2),
        count: ti.i32,
    ):
        for i in range(count):
            for axis in ti.static(range(3)):
                position[i, axis] = predicted[i, axis]

    symbolic = _ndarray_args(
        ("position", "velocity", "predicted", "correction"), ti.f32, 2
    )
    symbolic["inverse_mass"] = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "inverse_mass", ti.f32, ndim=1
    )
    for name in ("dt", "gravity", "inverse_dt"):
        symbolic[name] = ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, ti.f32)
    symbolic["count"] = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)

    def build(recipe):
        monkeypatch.setenv(_FUSION_ENV, recipe)
        builder = ti.graph.GraphBuilder()
        builder.dispatch(
            predict,
            symbolic["position"],
            symbolic["velocity"],
            symbolic["predicted"],
            symbolic["dt"],
            symbolic["gravity"],
            symbolic["count"],
        )
        builder.dispatch(
            apply_precomputed_correction,
            symbolic["predicted"],
            symbolic["correction"],
            symbolic["inverse_mass"],
            symbolic["count"],
        )
        builder.dispatch(
            update_velocity,
            symbolic["position"],
            symbolic["velocity"],
            symbolic["predicted"],
            symbolic["inverse_dt"],
            symbolic["count"],
        )
        builder.dispatch(
            finalize,
            symbolic["position"],
            symbolic["predicted"],
            symbolic["count"],
        )
        return builder.compile()

    baseline = build("baseline")
    fused = build("map4")
    assert baseline.physical_plan()["physical_dispatch_count"] == 4
    assert fused.physical_plan()["physical_dispatch_count"] == 1

    rng = np.random.default_rng(404)
    count = 2053
    host = {
        "position": rng.normal(0.0, 1.0, (count, 3)).astype(np.float32),
        "velocity": rng.normal(0.0, 2.0, (count, 3)).astype(np.float32),
        "correction": rng.normal(0.0, 0.01, (count, 3)).astype(np.float32),
        "inverse_mass": rng.uniform(0.0, 2.0, count).astype(np.float32),
    }
    dt = np.float32(1.0 / 120.0)

    def execute(graph):
        arrays = {
            name: ti.ndarray(ti.f32, shape=value.shape) for name, value in host.items()
        }
        for name, value in host.items():
            arrays[name].from_numpy(value)
        arrays["predicted"] = ti.ndarray(ti.f32, shape=(count, 3))
        graph.run(
            {
                **arrays,
                "dt": float(dt),
                "gravity": -9.81,
                "inverse_dt": float(np.float32(1.0) / dt),
                "count": count,
            }
        )
        return arrays["position"].to_numpy(), arrays["velocity"].to_numpy()

    baseline_position, baseline_velocity = execute(baseline)
    fused_position, fused_velocity = execute(fused)
    np.testing.assert_allclose(fused_position, baseline_position, rtol=2e-6, atol=2e-6)
    # Fusion may change one final f32 rounding in the predicted position. The
    # velocity reconstruction amplifies that difference by inverse_dt, so keep
    # a separate, mechanism-derived absolute budget instead of weakening every
    # physics result in this suite.
    position_delta = np.max(np.abs(fused_position - baseline_position))
    velocity_atol = max(2e-6, float(position_delta) / float(dt) + 2e-6)
    np.testing.assert_allclose(
        fused_velocity, baseline_velocity, rtol=2e-6, atol=velocity_atol
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan])
def test_neighbor_constraint_projection_is_not_a_fusion_candidate(monkeypatch):
    @ti.kernel
    def project_edges(
        predicted: ti.types.ndarray(dtype=ti.f32, ndim=2),
        neighbor: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.f32, ndim=2),
        count: ti.i32,
    ):
        for i in range(count):
            j = neighbor[i]
            for axis in ti.static(range(3)):
                output[i, axis] = predicted[i, axis] - predicted[j, axis]

    predicted = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicted", ti.f32, ndim=2)
    neighbor = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "neighbor", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2)
    count = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32)
    monkeypatch.setenv(_FUSION_ENV, "map2")
    builder = ti.graph.GraphBuilder()
    builder.dispatch(project_edges, predicted, neighbor, output, count)
    builder.dispatch(project_edges, predicted, neighbor, output, count)
    graph = builder.compile()

    assert graph.physical_plan()["physical_dispatch_count"] == 2
    assert graph._ir_debug_info["fusion_plan"]["blockers"] == {
        "non_pointwise_access": 2
    }
