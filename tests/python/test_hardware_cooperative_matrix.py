import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core as _ti_core
from tests import test_utils


def _executable_spec():
    specs = ti.hardware.matrix.cooperative_matrix_specs(executable_only=True)
    if not specs:
        pytest.skip("no executable Vulkan cooperative-matrix tuple")
    return next(
        (spec for spec in specs if (spec.m, spec.n, spec.k) == (16, 16, 16)),
        specs[0],
    )


def _matrix_types(spec):
    return (
        ti.types.matrix(spec.m, spec.k, ti.f16),
        ti.types.matrix(spec.k, spec.n, ti.f16),
        ti.types.matrix(spec.m, spec.n, ti.f32),
    )


@test_utils.test(arch=ti.cpu)
def test_vulkan_cooperative_matrix_is_explicit_and_backend_gated():
    assert not ti.hardware.matrix.cooperative_matrix_is_available()
    assert ti.hardware.matrix.cooperative_matrix_specs() == ()


@test_utils.test(
    arch=ti.vulkan,
    offline_cache=False,
    vulkan_spv_stats=True,
    vulkan_spv_stats_filter="all",
)
def test_vulkan_cooperative_matrix_matches_oracle_and_retains_hardware_ops():
    specs = ti.hardware.matrix.cooperative_matrix_specs(executable_only=True)
    if not specs:
        pytest.skip("no executable Vulkan cooperative-matrix tuple")

    rng = np.random.default_rng(20260826)
    for spec in specs:
        a_type, b_type, accumulator_type = _matrix_types(spec)
        batch = 3

        @ti.kernel
        def cooperative_mma(
            a: ti.types.ndarray(dtype=a_type, ndim=1),
            b: ti.types.ndarray(dtype=b_type, ndim=1),
            c: ti.types.ndarray(dtype=accumulator_type, ndim=1),
            output: ti.types.ndarray(dtype=accumulator_type, ndim=1),
        ):
            ti.loop_config(block_dim=spec.subgroup_size * 4)
            for lane in range(batch * spec.subgroup_size):
                ti.hardware.matrix.cooperative_mma_f16_f32(a, b, c, output, lane, spec)

        a_values = rng.uniform(-0.5, 0.5, (batch, spec.m, spec.k)).astype(np.float16)
        b_values = rng.uniform(-0.5, 0.5, (batch, spec.k, spec.n)).astype(np.float16)
        c_values = rng.uniform(-0.25, 0.25, (batch, spec.m, spec.n)).astype(np.float32)
        a = ti.ndarray(dtype=a_type, shape=batch)
        b = ti.ndarray(dtype=b_type, shape=batch)
        c = ti.ndarray(dtype=accumulator_type, shape=batch)
        output = ti.ndarray(dtype=accumulator_type, shape=batch)
        a.from_numpy(a_values)
        b.from_numpy(b_values)
        c.from_numpy(c_values)

        cooperative_mma(a, b, c, output)
        ti.sync()
        expected = a_values.astype(np.float32) @ b_values.astype(np.float32)
        expected += c_values
        np.testing.assert_allclose(output.to_numpy(), expected, rtol=2e-3, atol=2e-3)

        stats = _ti_core.get_last_vulkan_spv_stats()
        assert sum(item["cooperative_matrix_load_after"] for item in stats) == 3
        assert sum(item["cooperative_matrix_mul_add_after"] for item in stats) == 1
        assert sum(item["cooperative_matrix_store_after"] for item in stats) == 1


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_cooperative_matrix_rejects_invalid_tuple_shape_and_control():
    spec = _executable_spec()
    a_type, b_type, accumulator_type = _matrix_types(spec)
    bad_a_type = ti.types.matrix(spec.m, spec.k + 1, ti.f16)
    a = ti.ndarray(dtype=a_type, shape=1)
    b = ti.ndarray(dtype=b_type, shape=1)
    c = ti.ndarray(dtype=accumulator_type, shape=1)
    output = ti.ndarray(dtype=accumulator_type, shape=1)
    bad_a = ti.ndarray(dtype=bad_a_type, shape=1)

    unsupported = ti.hardware.matrix.CooperativeMatrixSpec(
        m=spec.m + 1,
        n=spec.n,
        k=spec.k,
        a_type=spec.a_type,
        b_type=spec.b_type,
        c_type=spec.c_type,
        result_type=spec.result_type,
        scope=spec.scope,
        saturating_accumulation=spec.saturating_accumulation,
        subgroup_size=spec.subgroup_size,
    )

    @ti.kernel
    def bad_shape(
        bad: ti.types.ndarray(dtype=bad_a_type, ndim=1),
        rhs: ti.types.ndarray(dtype=b_type, ndim=1),
        addend: ti.types.ndarray(dtype=accumulator_type, ndim=1),
        result: ti.types.ndarray(dtype=accumulator_type, ndim=1),
    ):
        ti.loop_config(block_dim=spec.subgroup_size)
        for lane in range(spec.subgroup_size):
            ti.hardware.matrix.cooperative_mma_f16_f32(
                bad, rhs, addend, result, lane, spec
            )

    with pytest.raises(
        (ti.TaichiCompilationError, RuntimeError),
        match="operand 'a'.*compact matrix",
    ):
        bad_shape(bad_a, b, c, output)

    @ti.kernel
    def bad_tuple(
        lhs: ti.types.ndarray(dtype=a_type, ndim=1),
        rhs: ti.types.ndarray(dtype=b_type, ndim=1),
        addend: ti.types.ndarray(dtype=accumulator_type, ndim=1),
        result: ti.types.ndarray(dtype=accumulator_type, ndim=1),
    ):
        ti.loop_config(block_dim=spec.subgroup_size)
        for lane in range(spec.subgroup_size):
            ti.hardware.matrix.cooperative_mma_f16_f32(
                lhs, rhs, addend, result, lane, unsupported
            )

    with pytest.raises(
        (ti.TaichiCompilationError, RuntimeError),
        match="not supported by the active Vulkan",
    ):
        bad_tuple(a, b, c, output)

    @ti.kernel
    def nested_control(
        lhs: ti.types.ndarray(dtype=a_type, ndim=1),
        rhs: ti.types.ndarray(dtype=b_type, ndim=1),
        addend: ti.types.ndarray(dtype=accumulator_type, ndim=1),
        result: ti.types.ndarray(dtype=accumulator_type, ndim=1),
    ):
        ti.loop_config(block_dim=spec.subgroup_size)
        for lane in range(spec.subgroup_size):
            if lane >= 0:
                ti.hardware.matrix.cooperative_mma_f16_f32(
                    lhs, rhs, addend, result, lane, spec
                )

    with pytest.raises(
        (ti.TaichiCompilationError, RuntimeError),
        match="top-level dense range-loop",
    ):
        nested_control(a, b, c, output)

    @ti.kernel
    def transformed_lane(
        lhs: ti.types.ndarray(dtype=a_type, ndim=1),
        rhs: ti.types.ndarray(dtype=b_type, ndim=1),
        addend: ti.types.ndarray(dtype=accumulator_type, ndim=1),
        result: ti.types.ndarray(dtype=accumulator_type, ndim=1),
    ):
        ti.loop_config(block_dim=spec.subgroup_size)
        for lane in range(spec.subgroup_size):
            ti.hardware.matrix.cooperative_mma_f16_f32(
                lhs,
                rhs,
                addend,
                result,
                lane + ti.cast(addend[0][0, 0], ti.i32),
                spec,
            )

    with pytest.raises(
        (ti.TaichiCompilationError, RuntimeError),
        match="direct range loop index",
    ):
        transformed_lane(a, b, c, output)

    @ti.kernel
    def bad_block_dim(
        lhs: ti.types.ndarray(dtype=a_type, ndim=1),
        rhs: ti.types.ndarray(dtype=b_type, ndim=1),
        addend: ti.types.ndarray(dtype=accumulator_type, ndim=1),
        result: ti.types.ndarray(dtype=accumulator_type, ndim=1),
    ):
        ti.loop_config(block_dim=spec.subgroup_size // 2)
        for lane in range(spec.subgroup_size):
            ti.hardware.matrix.cooperative_mma_f16_f32(
                lhs, rhs, addend, result, lane, spec
            )

    with pytest.raises(
        (ti.TaichiCompilationError, RuntimeError),
        match="block_dim divisible",
    ):
        bad_block_dim(a, b, c, output)


@pytest.mark.run_in_serial
def test_vulkan_cooperative_matrix_runtime_generation_recreation():
    observed = False
    for _ in range(3):
        ti.init(arch=ti.vulkan, offline_cache=False)
        try:
            specs = ti.hardware.matrix.cooperative_matrix_specs(executable_only=True)
            if specs:
                observed = True
                assert ti.hardware.matrix.cooperative_matrix_is_available()
                assert all(spec.subgroup_size > 0 for spec in specs)
        finally:
            ti.reset()
    if not observed:
        pytest.skip("no executable Vulkan cooperative-matrix tuple")
