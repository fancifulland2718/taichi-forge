import pytest

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from tests import test_utils


def _make_cursor_kernels():
    # GeoPhys's asset-free cursor reproducer. Preserve the loop-carried struct
    # member and nested vector branches that exposed skipped SPIR-V passes.
    group_count = 1
    points_per_group = 2
    count = ti.field(ti.i32, shape=())
    active = ti.field(ti.i32, shape=group_count)
    ncon = ti.field(ti.i32, shape=group_count)
    starts = ti.field(ti.i32, shape=group_count)
    condim = ti.field(ti.i32, shape=group_count)
    normal = ti.Vector.field(3, ti.f32, shape=group_count)
    friction = ti.field(ti.f32, shape=group_count)
    spin_friction = ti.field(ti.f32, shape=group_count)
    roll_friction = ti.field(ti.f32, shape=group_count)
    gap = ti.field(ti.i32, shape=group_count * points_per_group)
    distance = ti.field(ti.f32, shape=group_count * points_per_group)
    tangent = ti.Vector.field(3, ti.f32, shape=group_count * points_per_group)
    spin = ti.field(ti.f32, shape=group_count * points_per_group)
    rolling = ti.Vector.field(3, ti.f32, shape=group_count * points_per_group)
    out_dim = ti.field(ti.i32, shape=group_count * points_per_group)
    out_tangent = ti.Vector.field(3, ti.f32, shape=group_count * points_per_group)
    out_spin = ti.field(ti.f32, shape=group_count * points_per_group)
    out_rolling = ti.Vector.field(3, ti.f32, shape=group_count * points_per_group)

    @ti.kernel
    def setup():
        count[None] = group_count
        for m in range(group_count):
            active[m] = 1
            ncon[m] = points_per_group
            starts[m] = points_per_group * m
            condim[m] = 6
            normal[m] = ti.Vector([0.0, 1.0, 0.0])
            friction[m] = 0.5
            spin_friction[m] = 0.1
            roll_friction[m] = 0.1
        for i in range(group_count * points_per_group):
            gap[i] = 0
            distance[i] = -0.001
            tangent[i] = ti.Vector([1.0, 2.0, 3.0])
            spin[i] = 0.25
            rolling[i] = ti.Vector([4.0, 5.0, 6.0])

    @ti.func
    def emit_one(dim, mu, mu_spin, mu_roll, n, d, t, s, r, cursor: ti.template()):
        index = cursor.fact
        cursor.fact += 1
        if index < group_count * points_per_group:
            norm = n.norm()
            if norm > 1e-8:
                n = n / norm
            effective = ti.cast(1, ti.i32)
            if dim >= 3 and ti.max(mu, 0.0) > 1e-8:
                effective = 3
            if dim >= 4 and ti.max(mu_spin, 0.0) > 1e-8:
                effective = 4
            if dim >= 6 and ti.max(mu_roll, 0.0) > 1e-8:
                effective = 6
            if effective < 4:
                s = 0.0
            if effective < 6:
                r = ti.Vector.zero(ti.f32, 3)
            if d > 1e-7:
                t = ti.Vector.zero(ti.f32, 3)
                s = 0.0
                r = ti.Vector.zero(ti.f32, 3)
            out_dim[index] = effective
            out_tangent[index] = t
            out_spin[index] = s
            out_rolling[index] = r - n * r.dot(n)

    @ti.kernel
    def run():
        ti.loop_config(block_dim=32)
        for m in range(count[None]):
            cursor = ti.Struct(fact=starts[m])
            if active[m] != 0:
                n = normal[m]
                for k in range(points_per_group):
                    if k < ncon[m]:
                        i = points_per_group * m + k
                        t = tangent[i]
                        t = t - n * t.dot(n)
                        if gap[i] >= 0:
                            emit_one(
                                condim[m],
                                friction[m],
                                spin_friction[m],
                                roll_friction[m],
                                n,
                                distance[i],
                                t,
                                spin[i],
                                rolling[i],
                                cursor,
                            )

    return setup, run, (out_dim, out_tangent, out_spin, out_rolling)


def _compile_without_launch(kernel):
    key = kernel._primal.ensure_compiled()
    program = ti.lang.impl.get_runtime().prog
    program.compile_kernel(
        program.config(),
        program.get_device_caps(),
        kernel._primal.compiled_kernels[key],
    )
    return _ti_core.get_last_vulkan_spv_stats()


def _assert_cursor_output(outputs):
    dimensions, tangents, spins, rolling = outputs
    assert dimensions.to_numpy().tolist() == [6, 6]
    assert tangents.to_numpy().tolist() == [[1.0, 0.0, 3.0]] * 2
    assert spins.to_numpy().tolist() == [0.25, 0.25]
    assert rolling.to_numpy().tolist() == [[4.0, 0.0, 6.0]] * 2


@pytest.mark.parametrize(
    "parallel,skip_unroll,disabled",
    [
        (False, False, []),
        (True, False, []),
        (False, True, []),
        (False, False, ["LoopUnroll"]),
    ],
)
@test_utils.test(
    arch=ti.vulkan,
    offline_cache=False,
    vulkan_spv_stats=True,
    vulkan_spv_stats_filter="all",
    num_compile_threads=2,
)
def test_optimizer_runs_for_later_tasks_and_kernels(parallel, skip_unroll, disabled):
    ti.cfg.spirv_parallel_codegen = parallel
    ti.cfg.spirv_skip_loop_unroll = skip_unroll
    ti.cfg.spirv_disabled_passes = disabled
    # setup and the serial task in run consume the old thread-local pass list.
    # Check the generated code before submission so a regression fails cleanly
    # instead of crashing the driver when the cursor kernel is launched.
    for _ in range(3):
        setup, run, outputs = _make_cursor_kernels()
        setup()
        stats = _compile_without_launch(run)
        ranges = [item for item in stats if item["type"] == "range_for"]
        assert len(stats) == 2
        assert len(ranges) == 1
        for task in ranges:
            assert task["opt_run"] and task["opt_ok"]
            assert task["word_after"] < task["word_before"]
            assert ("LoopUnroll" in task["skipped_passes"]) == bool(skip_unroll or disabled)
        run()
        ti.sync()
        _assert_cursor_output(outputs)


@pytest.mark.parametrize("tier,level", [("balanced", 0), ("fast", 3)])
@test_utils.test(
    arch=ti.vulkan,
    offline_cache=False,
    vulkan_spv_stats=True,
    vulkan_spv_stats_filter="all",
)
def test_optimizer_disabled_modes_preserve_unoptimized_output(tier, level):
    ti.cfg.compile_tier = tier
    ti.cfg.external_optimization_level = level
    for _ in range(2):
        _, run, _ = _make_cursor_kernels()
        stats = _compile_without_launch(run)
        assert stats
        for task in stats:
            assert not task["opt_run"]
            assert task["word_after"] == task["word_before"]
    # Compile-only: this repair preserves opt-out semantics. The unoptimized
    # cursor shader still triggers a compiler fault on affected NVIDIA drivers.


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_cursor_nested_vector_outputs_other_backends():
    setup, run, outputs = _make_cursor_kernels()
    setup()
    run()
    ti.sync()
    _assert_cursor_output(outputs)
