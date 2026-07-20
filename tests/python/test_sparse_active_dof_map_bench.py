import gc
import importlib.util
import weakref
from pathlib import Path

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


_BENCH_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "sparse_active_dof_map_bench.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "sparse_active_dof_map_bench", _BENCH_PATH
)
sparse_active_dof_map_bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sparse_active_dof_map_bench)


def test_generic_csr_galerkin_accepts_arbitrary_aggregate_map():
    size = 7
    rows = [{row: 1.0} for row in range(size)]
    for row, column, weight in (
        (0, 1, 1.0),
        (1, 2, 2.0),
        (2, 3, 0.5),
        (0, 4, 0.75),
        (4, 5, 1.25),
        (5, 6, 0.9),
        (2, 6, 0.4),
    ):
        rows[row][row] += weight
        rows[column][column] += weight
        rows[row][column] = -weight
        rows[column][row] = -weight
    csr = sparse_active_dof_map_bench._csr_from_rows_reference(rows)
    aggregate_map = np.asarray([0, 0, 1, 1, 2, 2, 2], dtype=np.int32)
    projected = sparse_active_dof_map_bench._galerkin_csr_reference(
        *csr, aggregate_map, 3
    )
    dense = sparse_active_dof_map_bench._csr_to_dense_reference(*csr)
    prolongation = np.zeros((size, 3), dtype=np.float64)
    prolongation[np.arange(size), aggregate_map] = 1.0
    expected = prolongation.T @ dense @ prolongation
    np.testing.assert_allclose(
        projected["reconstructed"], expected, rtol=0.0, atol=0.0
    )
    assert np.allclose(
        projected["reconstructed"],
        projected["reconstructed"].T,
        rtol=0.0,
        atol=0.0,
    )
    assert np.linalg.eigvalsh(projected["reconstructed"])[0] > 0.0


def _recursive_hierarchy_case(dimensions):
    if dimensions == 2:
        coordinates = [
            (i, j)
            for i in range(8)
            for j in range(8)
            if (i, j) not in ((3, 3), (3, 4))
        ]
        coordinates += [(12, 12), (12, 13), (13, 12), (13, 13)]
        bottom_component_cap = 4
    else:
        coordinates = [
            (i, j, k)
            for i in range(4)
            for j in range(4)
            for k in range(4)
            if (i, j, k) != (1, 1, 1)
        ]
        coordinates += [
            (i, j, k)
            for i in range(8, 10)
            for j in range(8, 10)
            for k in range(8, 10)
        ]
        bottom_component_cap = 2
    coordinates = np.asarray(coordinates, dtype=np.int32)
    csr = sparse_active_dof_map_bench._coordinate_poisson_csr_reference(
        coordinates, dimensions
    )
    return coordinates, (
        sparse_active_dof_map_bench._build_recursive_csr_hierarchy_reference(
            dimensions=dimensions,
            coordinates=coordinates,
            row_offsets=csr[0],
            column_indices=csr[1],
            values=csr[2],
            bottom_component_cap=bottom_component_cap,
        )
    )


@pytest.mark.parametrize("dimensions", [2, 3])
def test_recursive_csr_hierarchy_is_spd_and_linearly_bounded(dimensions):
    coordinates, hierarchy = _recursive_hierarchy_case(dimensions)
    levels = hierarchy["levels"]
    assert hierarchy["level_count"] >= 3
    assert hierarchy["nonbottom_level_count"] == len(levels) - 1
    assert hierarchy["sum_level_dofs"] < 2 * len(coordinates)
    assert hierarchy["operator_projection"] == "generic_csr_galerkin"
    assert hierarchy["bottom_dense_scope"] == (
        "independent_connected_components"
    )
    assert not hierarchy["smoother_steps_selected"]
    assert all(level["symmetric"] for level in levels)
    assert all(level["min_eigenvalue"] > 0.0 for level in levels)
    assert all(level["galerkin_matches_dense_oracle"] for level in levels)
    assert all(level["directional_stencil_eligible"] for level in levels)
    assert all(
        levels[index + 1]["size"] < levels[index]["size"]
        for index in range(len(levels) - 1)
    )
    assert max(levels[-1]["component_sizes"]) <= hierarchy[
        "bottom_component_cap"
    ]
    assert hierarchy["bottom_component_inverse_bytes"] == (
        sum(size**2 for size in levels[-1]["component_sizes"])
        * np.dtype(np.float32).itemsize
    )
    assert hierarchy["operator_pattern_bytes"] == sum(
        level["operator_pattern_bytes"] for level in levels
    )
    assert hierarchy["operator_value_bytes"] == sum(
        level["operator_value_bytes"] for level in levels
    )
    assert hierarchy["aggregate_map_bytes"] == sum(
        level["map_to_coarse_bytes"] for level in levels
    )
    assert hierarchy["workspace_upper_bytes"] == sum(
        level["workspace_upper_bytes"] for level in levels
    )
    assert hierarchy["steady_reserved_bytes_upper"] == sum(
        (
            hierarchy["operator_pattern_bytes"],
            hierarchy["operator_value_bytes"],
            hierarchy["aggregate_map_bytes"],
            hierarchy["workspace_upper_bytes"],
            hierarchy["bottom_component_inverse_bytes"],
        )
    )


@pytest.mark.parametrize("dimensions", [2, 3])
def test_symmetric_vcycle_reference_is_linear_and_spd(dimensions):
    _, hierarchy = _recursive_hierarchy_case(dimensions)
    vcycle = sparse_active_dof_map_bench._assemble_symmetric_vcycle_reference(
        hierarchy
    )
    assert vcycle["pre_smoother_steps"] == 1
    assert vcycle["post_smoother_steps"] == 1
    assert vcycle["host_algebra_correctness_only"]
    assert vcycle["fixed_linear_operator"]
    assert vcycle["symmetric_pre_post_composition"]
    assert vcycle["bottom_inverse_spd"]
    assert vcycle["smoother_numeric_bytes"] == (
        hierarchy["nonbottom_level_count"] * np.dtype(np.float32).itemsize
    )
    assert vcycle["steady_reserved_bytes_upper"] == (
        hierarchy["steady_reserved_bytes_upper"]
        + vcycle["smoother_numeric_bytes"]
    )
    assert vcycle["linearity_difference_linf"] <= 1e-13
    assert vcycle["symmetry_difference_linf"] <= 1e-13
    assert vcycle["minimum_eigenvalue"] > 0.0
    assert vcycle["logical_dispatch_upper_bound"] == (
        1 + hierarchy["nonbottom_level_count"] * 5
    )
    for smoother in vcycle["damping"]:
        assert smoother["normalized_absolute_row_sum_bound"] >= (
            smoother["normalized_maximum_eigenvalue"] - 1e-14
        )
        assert smoother["damping"] == pytest.approx(
            1.0 / smoother["normalized_absolute_row_sum_bound"],
            rel=0.0,
            abs=0.0,
        )
        assert smoother["strict_spd_smoother_bound"]


def _nonstencil_spd_hierarchy_case():
    size = 8
    rows = [{row: 1.0} for row in range(size)]
    for row, column, weight in (
        (0, 2, 0.7),
        (2, 5, 1.1),
        (5, 7, 0.6),
        (7, 1, 1.3),
        (1, 4, 0.8),
        (4, 6, 1.2),
        (6, 3, 0.9),
        (3, 0, 0.5),
    ):
        rows[row][row] += weight
        rows[column][column] += weight
        rows[row][column] = -weight
        rows[column][row] = -weight
    csr = sparse_active_dof_map_bench._csr_from_rows_reference(rows)
    coordinates = np.asarray([(row, 0) for row in range(size)], dtype=np.int32)
    hierarchy = (
        sparse_active_dof_map_bench._build_recursive_csr_hierarchy_reference(
            dimensions=2,
            coordinates=coordinates,
            row_offsets=csr[0],
            column_indices=csr[1],
            values=csr[2],
            bottom_component_cap=2,
        )
    )
    return csr, coordinates, hierarchy


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
)
def test_galerkin_csr_can_reuse_device_resident_primitives():
    csr, _, hierarchy = _nonstencil_spd_hierarchy_case()
    fine = hierarchy["_reference_levels"][0]
    coarse = hierarchy["_reference_levels"][1]
    fine_size = len(fine["row_offsets"]) - 1
    coarse_size = len(coarse["row_offsets"]) - 1
    capacity = len(fine["column_indices"])
    assert (fine_size, coarse_size, capacity) == (8, 4, 24)
    assert capacity > len(coarse["column_indices"])

    source_row_offsets = ti.ndarray(ti.i32, shape=fine_size + 1)
    source_column_indices = ti.ndarray(ti.i32, shape=capacity)
    source_values = ti.ndarray(ti.f32, shape=capacity)
    fine_to_coarse = ti.ndarray(ti.i32, shape=fine_size)
    source_row_offsets.from_numpy(fine["row_offsets"])
    source_column_indices.from_numpy(fine["column_indices"])
    source_values.from_numpy(fine["values"].astype(np.float32))
    fine_to_coarse.from_numpy(fine["fine_to_coarse"])

    sorted_keys = ti.ndarray(ti.u64, shape=capacity)
    sorted_values = ti.ndarray(ti.f32, shape=capacity)
    unique_keys = ti.ndarray(ti.u64, shape=capacity)
    run_ends = ti.ndarray(ti.i32, shape=capacity)
    run_count = ti.ndarray(ti.i32, shape=1)
    row_offsets = ti.ndarray(ti.i32, shape=coarse_size + 1)
    column_indices = ti.ndarray(ti.i32, shape=capacity)
    unique_values = ti.ndarray(ti.f32, shape=capacity)
    control = ti.ndarray(ti.i32, shape=2)

    @ti.kernel
    def emit_galerkin_triplets(
        input_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        input_column_indices: ti.types.ndarray(dtype=ti.i32, ndim=1),
        input_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        aggregate: ti.types.ndarray(dtype=ti.i32, ndim=1),
        keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        status: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for fine_row in range(fine_size):
            coarse_row = aggregate[fine_row]
            for offset in range(
                input_row_offsets[fine_row],
                input_row_offsets[fine_row + 1],
            ):
                fine_column = input_column_indices[offset]
                coarse_column = 0
                value = input_values[offset]
                valid = 0 <= coarse_row < coarse_size
                valid = valid and 0 <= fine_column < fine_size
                if valid:
                    coarse_column = aggregate[fine_column]
                    valid = 0 <= coarse_column < coarse_size
                if ti.math.isnan(value) or ti.math.isinf(value):
                    ti.atomic_max(status[0], 2)
                    valid = False
                if not valid:
                    ti.atomic_max(status[0], 1)
                    coarse_row = 0
                    coarse_column = 0
                    value = 0.0
                keys[offset] = (
                    ti.cast(coarse_row, ti.u64) << 32
                ) | ti.cast(coarse_column, ti.u64)
                values[offset] = value

    @ti.kernel
    def reduce_runs_and_count_rows(
        keys: ti.types.ndarray(dtype=ti.u64, ndim=1),
        values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ends: ti.types.ndarray(dtype=ti.i32, ndim=1),
        count: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output_columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output_row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        status: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for segment in range(capacity):
            if segment < count[0]:
                begin = 0
                if segment > 0:
                    begin = ends[segment - 1]
                end = ends[segment]
                total = ti.cast(0.0, ti.f32)
                for offset in range(begin, end):
                    total += values[offset]
                    if ti.math.isnan(total) or ti.math.isinf(total):
                        ti.atomic_max(status[0], 3)
                key = keys[segment]
                row = ti.cast(key >> 32, ti.i32)
                column = ti.cast(
                    key & ti.u64(0xFFFFFFFF), ti.i32
                )
                output_columns[segment] = column
                output_values[segment] = total
                ti.atomic_add(output_row_offsets[row + 1], 1)
            if segment == 0:
                status[1] = count[0]

    arch = ti.lang.impl.current_cfg().arch
    sort_method = {
        ti.cpu: "cpu_native",
        ti.cuda: "cuda_device",
        ti.vulkan: "vulkan_native_radix_u32",
    }[arch]
    compact_method = {
        ti.cpu: "cpu_native",
        ti.cuda: "cuda_device",
        ti.vulkan: "vulkan_native",
    }[arch]
    sort_workspace = ti.algorithms.SortWorkspace(max_items=capacity)
    rle_workspace = ti.algorithms.RunLengthWorkspace(max_items=capacity)
    run_scan = ti.algorithms.PrefixSumExecutor(capacity)
    row_scan = ti.algorithms.PrefixSumExecutor(coarse_size + 1)

    def assemble_once():
        control.fill(0)
        run_count.fill(0)
        run_ends.fill(0)
        row_offsets.fill(0)
        emit_galerkin_triplets(
            source_row_offsets,
            source_column_indices,
            source_values,
            fine_to_coarse,
            sorted_keys,
            sorted_values,
            control,
        )
        ti.algorithms.sort(
            sorted_keys,
            sorted_values,
            method=sort_method,
            workspace=sort_workspace,
        )
        ti.algorithms.experimental_run_length_encode(
            sorted_keys,
            unique_keys,
            run_ends,
            run_count,
            method=compact_method,
            workspace=rle_workspace,
        )
        run_scan.run(run_ends)
        reduce_runs_and_count_rows(
            unique_keys,
            sorted_values,
            run_ends,
            run_count,
            column_indices,
            unique_values,
            row_offsets,
            control,
        )
        row_scan.run(row_offsets)

        control_host = control.to_numpy()
        unique_nnz = int(control_host[1])
        return {
            "control": control_host,
            "row_offsets": row_offsets.to_numpy(),
            "column_indices": column_indices.to_numpy()[:unique_nnz],
            "values": unique_values.to_numpy()[:unique_nnz],
        }

    first = assemble_once()
    second = assemble_once()
    expected_nnz = len(coarse["column_indices"])
    assert expected_nnz == 16
    assert first["control"].tolist() == [0, expected_nnz]
    assert second["control"].tolist() == [0, expected_nnz]
    np.testing.assert_array_equal(first["row_offsets"], coarse["row_offsets"])
    np.testing.assert_array_equal(
        first["column_indices"], coarse["column_indices"]
    )
    np.testing.assert_allclose(
        first["values"],
        coarse["values"].astype(np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        second["row_offsets"], first["row_offsets"]
    )
    np.testing.assert_array_equal(
        second["column_indices"], first["column_indices"]
    )
    np.testing.assert_array_equal(second["values"], first["values"])

    expected_backend = {
        ti.cpu: "cpu_native",
        ti.cuda: "cuda_device",
        ti.vulkan: "vulkan_native",
    }[arch]
    if arch == ti.cuda:
        assert sort_workspace._cuda_device_active
    elif arch == ti.vulkan:
        assert sort_workspace._vulkan_native_active
    assert (
        rle_workspace.compact_workspace._native_compact_plan.backend
        == expected_backend
    )
    assert run_scan._native_scan_plan.backend == expected_backend
    assert row_scan._native_scan_plan.backend == expected_backend

    persistent_staging_bytes = 32 * capacity + 4 * coarse_size + 16
    exact_output_bytes = 4 * (coarse_size + 1) + 8 * expected_nnz
    capacity_output_bytes = 4 * (coarse_size + 1) + 8 * capacity
    assert persistent_staging_bytes == 800
    assert exact_output_bytes == 148
    assert capacity_output_bytes == 212
    assert capacity_output_bytes - exact_output_bytes == 64
    assert control.shape[0] * np.dtype(np.int32).itemsize == 8
    assert persistent_staging_bytes > capacity_output_bytes
    assert exact_output_bytes < capacity_output_bytes
    assert capacity - expected_nnz == (
        len(fine["column_indices"]) - len(coarse["column_indices"])
    )
    assert sort_workspace.workspace_bytes_peak >= 0
    assert rle_workspace.workspace_bytes_peak >= capacity * 4


def test_symmetric_vcycle_accepts_nonstencil_spd_csr():
    _, _, hierarchy = _nonstencil_spd_hierarchy_case()
    assert hierarchy["operator_projection"] == "generic_csr_galerkin"
    assert not hierarchy["levels"][0]["directional_stencil_eligible"]
    assert all(level["symmetric"] for level in hierarchy["levels"])
    assert all(
        level["min_eigenvalue"] > 0.0 for level in hierarchy["levels"]
    )
    vcycle = sparse_active_dof_map_bench._assemble_symmetric_vcycle_reference(
        hierarchy
    )
    assert vcycle["linearity_difference_linf"] <= 1e-13
    assert vcycle["symmetry_difference_linf"] <= 1e-13
    assert vcycle["minimum_eigenvalue"] > 0.0
    assert all(
        smoother["strict_spd_smoother_bound"]
        for smoother in vcycle["damping"]
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_recursive_vcycle_graph_drives_private_compiled_pcg():
    csr, coordinates, hierarchy = _nonstencil_spd_hierarchy_case()
    size = hierarchy["levels"][0]["size"]
    topology_version = 83
    numeric_version = 89
    pattern_numpy = np.concatenate((csr[0], csr[1])).astype(np.int32)
    values_numpy = csr[2].astype(np.float32)
    pattern = ti.ndarray(ti.i32, shape=len(pattern_numpy))
    pattern.from_numpy(pattern_numpy)
    values = ti.ndarray(ti.f32, shape=len(values_numpy))
    values.from_numpy(values_numpy)
    exact_numpy = np.asarray(
        [1.0, -0.5, 2.0, -1.5, 0.75, -2.5, 1.25, -0.25],
        dtype=np.float32,
    )
    exact = ti.ndarray(ti.f32, shape=size)
    exact.from_numpy(exact_numpy)
    rhs = ti.ndarray(ti.f32, shape=size)
    rhs.fill(0.0)
    probe = ti.ndarray(ti.f32, shape=size)
    probe.fill(0.0)

    @ti.kernel
    def apply_csr(
        active_size: ti.i32,
        packed_pattern: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            total = 0.0
            for offset in range(packed_pattern[row], packed_pattern[row + 1]):
                column = packed_pattern[active_size + 1 + offset]
                total += numeric_values[offset] * x[column]
            y[row] = total

    @ti.kernel
    def vcycle_pre(
        active_size: ti.i32,
        diagonal: ti.types.ndarray(dtype=ti.f32, ndim=1),
        damping: ti.types.ndarray(dtype=ti.f32, ndim=1),
        level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
        pre_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            pre_solution[row] = damping[0] * level_rhs[row] / diagonal[row]

    @ti.kernel
    def vcycle_restrict(
        active_size: ti.i32,
        coarse_size: ti.i32,
        row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
        fine_to_coarse: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
        pre_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
        coarse_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(coarse_size):
            coarse_rhs[row] = 0.0
        for row in range(active_size):
            total = 0.0
            for offset in range(row_offsets[row], row_offsets[row + 1]):
                total += numeric_values[offset] * pre_solution[columns[offset]]
            ti.atomic_add(
                coarse_rhs[fine_to_coarse[row]], level_rhs[row] - total
            )

    @ti.kernel
    def vcycle_bottom(
        active_size: ti.i32,
        component_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        component_rows: ti.types.ndarray(dtype=ti.i32, ndim=1),
        inverse_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        row_component: ti.types.ndarray(dtype=ti.i32, ndim=1),
        row_local: ti.types.ndarray(dtype=ti.i32, ndim=1),
        inverse_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
        level_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            component = row_component[row]
            start = component_offsets[component]
            block_size = component_offsets[component + 1] - start
            inverse_start = inverse_offsets[component]
            total = 0.0
            for local_column in range(block_size):
                column = component_rows[start + local_column]
                total += (
                    inverse_values[
                        inverse_start
                        + row_local[row] * block_size
                        + local_column
                    ]
                    * level_rhs[column]
                )
            level_solution[row] = total

    @ti.kernel
    def vcycle_post(
        active_size: ti.i32,
        row_offsets: ti.types.ndarray(dtype=ti.i32, ndim=1),
        columns: ti.types.ndarray(dtype=ti.i32, ndim=1),
        fine_to_coarse: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
        diagonal: ti.types.ndarray(dtype=ti.f32, ndim=1),
        damping: ti.types.ndarray(dtype=ti.f32, ndim=1),
        level_rhs: ti.types.ndarray(dtype=ti.f32, ndim=1),
        pre_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
        coarse_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
        level_solution: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            corrected = (
                pre_solution[row] + coarse_solution[fine_to_coarse[row]]
            )
            total = 0.0
            for offset in range(row_offsets[row], row_offsets[row + 1]):
                column = columns[offset]
                neighbor_corrected = (
                    pre_solution[column]
                    + coarse_solution[fine_to_coarse[column]]
                )
                total += numeric_values[offset] * neighbor_corrected
            level_solution[row] = corrected + (
                damping[0] * (level_rhs[row] - total) / diagonal[row]
            )

    program = ti.lang.impl.get_runtime().prog
    target_primal = apply_csr._primal
    target_key = target_primal.ensure_compiled(
        size, pattern, values, exact, rhs
    )
    target = (
        program._create_compiled_kernel_linear_operator_with_numeric_data(
            target_primal.compiled_kernels[target_key],
            size,
            topology_version,
            numeric_version,
            pattern.arr,
            values.arr,
        )
    )
    plan = sparse_active_dof_map_bench._RecursiveVcycleGraphPlan(
        ti,
        program=program,
        hierarchy=hierarchy,
        topology_version=topology_version,
        numeric_version=numeric_version,
        pre_kernel=vcycle_pre,
        restrict_kernel=vcycle_restrict,
        bottom_kernel=vcycle_bottom,
        post_kernel=vcycle_post,
    )
    target.spmv(program, exact.arr, rhs.arr)
    plan.apply(rhs, probe)
    ti.sync()
    host_vcycle = (
        sparse_active_dof_map_bench._assemble_symmetric_vcycle_reference(
            hierarchy
        )["inverse_operator"]
    )
    expected_probe = host_vcycle @ rhs.to_numpy().astype(np.float64)
    np.testing.assert_allclose(
        probe.to_numpy(), expected_probe, rtol=0.0, atol=2e-5
    )
    plan_stats = plan.debug_runtime_stats()
    assert plan_stats["identity"]["level_count"] == 3
    assert plan_stats["operations"]["graph_node_count"] == 1
    assert plan_stats["operations"]["graph_dispatch_count"] == 7
    assert plan_stats["operations"]["kernel_dispatches_per_apply"] == 7
    assert plan_stats["operations"]["host_graph_submissions_per_apply"] == 1
    assert plan_stats["operations"]["explicit_apply_host_synchronizations"] == 0
    assert plan_stats["contract"]["no_host_fallback"]
    assert plan_stats["resources"]["topology_reserved_bytes"] == 304
    assert plan_stats["resources"]["numeric_reserved_bytes"] == 232
    assert plan_stats["resources"]["workspace_reserved_bytes"] == 96
    assert plan_stats["resources"]["plan_owned_reserved_bytes"] == 632
    numeric_publisher = plan.create_numeric_publisher()
    publisher_stats = numeric_publisher.debug_runtime_stats()
    assert publisher_stats["host_topology_metadata_bytes"] == 292
    assert publisher_stats["device_reserved_bytes"] == 0
    assert publisher_stats["numeric_role_count"] == 7
    assert publisher_stats["numeric_payload_bytes"] == 232

    changed_dense = sparse_active_dof_map_bench._csr_to_dense_reference(*csr)
    changed_dense[0, 0] += 0.25
    changed_dense[1, 1] += 0.25
    changed_dense[0, 1] = -0.25
    changed_dense[1, 0] = -0.25
    changed_rows = [
        {
            column: float(changed_dense[row, column])
            for column in range(size)
            if changed_dense[row, column] != 0.0
        }
        for row in range(size)
    ]
    changed_csr = sparse_active_dof_map_bench._csr_from_rows_reference(
        changed_rows
    )
    changed_hierarchy = (
        sparse_active_dof_map_bench._build_recursive_csr_hierarchy_reference(
            dimensions=2,
            coordinates=coordinates,
            row_offsets=changed_csr[0],
            column_indices=changed_csr[1],
            values=changed_csr[2],
            bottom_component_cap=2,
        )
    )
    assert not sparse_active_dof_map_bench._recursive_vcycle_topology_matches(
        hierarchy, changed_hierarchy
    )
    with pytest.raises(ValueError, match="identical level CSR patterns"):
        numeric_publisher.create_sources(changed_hierarchy)

    replacement_hierarchy = (
        sparse_active_dof_map_bench._build_recursive_csr_hierarchy_reference(
            dimensions=2,
            coordinates=coordinates,
            row_offsets=csr[0],
            column_indices=csr[1],
            values=2.0 * csr[2],
            bottom_component_cap=2,
        )
    )
    assert sparse_active_dof_map_bench._recursive_vcycle_topology_matches(
        hierarchy, replacement_hierarchy
    )

    arch = ti.lang.impl.current_cfg().arch

    def assert_graph_cache_execution(stats, *, complete_counters):
        assert stats["known_compiled_dispatches"] == 7
        assert stats["known_persistent_argument_bytes"] == (
            stats["known_persistent_device_argument_bytes"]
            + stats["known_deferred_host_argument_bytes"]
        )
        assert stats["retained_allocation_leases_are_borrowed"]
        if arch == ti.cpu:
            assert stats["backend"] == "none"
            assert stats["last_path"] == "none"
            assert stats["known_persistent_device_argument_bytes"] == 0
            assert stats["known_deferred_host_argument_bytes"] == 0
            assert stats["retained_allocation_lease_count"] == 0
            assert stats["deferred_replay_batch_count"] == 0
            assert not stats["opaque_driver_runtime_state_present"]
            assert stats["total_owned_device_bytes_reported"]
        elif arch == ti.cuda:
            assert stats["backend"] == "cuda"
            assert stats["last_path"].startswith("cuda_")
            assert stats["known_persistent_device_argument_bytes"] > 0
            assert stats["retained_allocation_lease_count"] > 0
            assert stats["deferred_replay_batch_count"] <= 2
            assert stats["opaque_driver_runtime_state_present"]
            assert not stats["total_owned_device_bytes_reported"]
        else:
            assert stats["backend"] == "vulkan"
            assert stats["last_path"] in ("vulkan_record", "vulkan_replay")
            assert stats["known_persistent_device_argument_bytes"] > 0
            assert stats["known_deferred_host_argument_bytes"] == 0
            assert stats["retained_allocation_lease_count"] == 0
            assert stats["deferred_replay_batch_count"] == 0
            assert stats["opaque_driver_runtime_state_present"]
            assert not stats["total_owned_device_bytes_reported"]
        if complete_counters:
            assert stats["diagnostics_counters_complete"]

    inverse = plan.create_native_operator()
    empty_graph_cache = inverse._debug_graph_cache_stats()
    assert empty_graph_cache["backend"] == "none"
    assert empty_graph_cache["known_compiled_dispatches"] == 0
    assert empty_graph_cache["known_persistent_argument_bytes"] == 0
    assert not empty_graph_cache["opaque_driver_runtime_state_present"]
    assert empty_graph_cache["total_owned_device_bytes_reported"]
    plan_ref = weakref.ref(plan)
    plan = None
    gc.collect()
    assert plan_ref() is None
    numeric_update_sources = numeric_publisher.create_sources(
        replacement_hierarchy
    )
    assert sum(
        int(np.prod(value.shape)) * np.dtype(np.float32).itemsize
        for value in numeric_update_sources.values()
    ) == 232
    numeric_update_source_refs = [
        weakref.ref(value) for value in numeric_update_sources.values()
    ]
    probe.fill(0.0)
    inverse.spmv(program, rhs.arr, probe.arr)
    ti.sync()
    np.testing.assert_allclose(
        probe.to_numpy(), expected_probe, rtol=0.0, atol=2e-5
    )
    assert_graph_cache_execution(
        inverse._debug_graph_cache_stats(), complete_counters=True
    )
    inverse_stats = inverse._debug_runtime_stats()
    assert inverse_stats["resources"]["pattern_reserved_bytes"] == (
        plan_stats["resources"]["topology_reserved_bytes"]
    )
    assert inverse_stats["resources"]["values_reserved_bytes"] == (
        plan_stats["resources"]["numeric_reserved_bytes"]
    )
    assert inverse_stats["resources"]["spmv_workspace_reserved_bytes"] == (
        plan_stats["resources"]["workspace_reserved_bytes"]
    )
    assert inverse_stats["resources"]["operator_owned_reserved_bytes"] == 632
    assert inverse_stats["transfers"]["device_to_device_bytes"] == 632

    preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program, target, inverse, True
        )
    )
    def make_solver(matrix, binding):
        if arch == ti.cpu:
            return ti._lib.core._make_cpu_compiled_kernel_pcg_solver(
                program, matrix, binding, 32, 1e-5
            )
        if arch == ti.cuda:
            return ti._lib.core._make_cuda_compiled_kernel_pcg_solver(
                program, matrix, binding, 32, 1e-5, False
            )
        return ti._lib.core._make_vulkan_compiled_kernel_pcg_convergence_plan(
            program, matrix, binding, 32, 1e-5
        )

    solver = make_solver(target, preconditioner)
    solution = ti.ndarray(ti.f32, shape=size)
    solution.fill(0.0)
    solver.solve(program, solution.arr, rhs.arr)
    ti.sync()
    assert solver.is_success()
    np.testing.assert_allclose(
        solution.to_numpy(), exact_numpy, rtol=0.0, atol=2e-4
    )
    solver_stats = solver._debug_runtime_stats()
    assert solver_stats["identity"]["preconditioner_method"] == (
        "compiled_graph_inverse_apply"
    )
    assert solver_stats["resources"]["external_preconditioner"]

    incomplete_sources = {
        name: value.arr
        for name, value in numeric_update_sources.items()
        if name != "bottom_inverse_values"
    }
    with pytest.raises(RuntimeError, match="complete numeric role set"):
        inverse.update_numeric_data(
            program,
            incomplete_sources,
            topology_version,
            numeric_version,
        )
    rejected_update_stats = inverse._debug_runtime_stats()
    assert rejected_update_stats["identity"]["numeric_version"] == (
        numeric_version
    )
    assert rejected_update_stats["operations"]["numeric_updates"] == 0
    assert rejected_update_stats["resources"][
        "numeric_update_peak_temporary_bytes"
    ] == 0
    probe.fill(0.0)
    inverse.spmv(program, rhs.arr, probe.arr)
    ti.sync()
    np.testing.assert_allclose(
        probe.to_numpy(), expected_probe, rtol=0.0, atol=2e-5
    )

    inverse.update_numeric_data(
        program,
        {name: value.arr for name, value in numeric_update_sources.items()},
        topology_version,
        numeric_version,
    )
    cleared_graph_cache = inverse._debug_graph_cache_stats()
    assert cleared_graph_cache["backend"] == "none"
    assert cleared_graph_cache["known_compiled_dispatches"] == 0
    assert cleared_graph_cache["known_persistent_argument_bytes"] == 0
    assert not cleared_graph_cache["opaque_driver_runtime_state_present"]
    assert cleared_graph_cache["total_owned_device_bytes_reported"]
    replacement_values = ti.ndarray(ti.f32, shape=len(values_numpy))
    replacement_values.from_numpy(2.0 * values_numpy)
    target.update_numeric_data(
        program,
        replacement_values.arr,
        topology_version,
        numeric_version,
    )
    numeric_update_sources = None
    incomplete_sources = None
    gc.collect()
    assert all(reference() is None for reference in numeric_update_source_refs)

    probe.fill(13.0)
    with pytest.raises(RuntimeError, match="preconditioner is stale"):
        preconditioner.apply(program, target, rhs.arr, probe.arr)
    np.testing.assert_array_equal(
        probe.to_numpy(), np.full(size, 13.0, dtype=np.float32)
    )
    solution.fill(17.0)
    with pytest.raises(RuntimeError, match="preconditioner is stale"):
        solver.solve(program, solution.arr, rhs.arr)
    np.testing.assert_array_equal(
        solution.to_numpy(), np.full(size, 17.0, dtype=np.float32)
    )

    target.spmv(program, exact.arr, rhs.arr)
    probe.fill(0.0)
    inverse.spmv(program, rhs.arr, probe.arr)
    ti.sync()
    replacement_vcycle = (
        sparse_active_dof_map_bench._assemble_symmetric_vcycle_reference(
            replacement_hierarchy
        )["inverse_operator"]
    )
    replacement_expected_probe = (
        replacement_vcycle @ rhs.to_numpy().astype(np.float64)
    )
    np.testing.assert_allclose(
        probe.to_numpy(), replacement_expected_probe, rtol=0.0, atol=2e-5
    )
    assert_graph_cache_execution(
        inverse._debug_graph_cache_stats(), complete_counters=True
    )
    refreshed_inverse_stats = inverse._debug_runtime_stats()
    assert refreshed_inverse_stats["identity"]["pattern_version"] == (
        topology_version
    )
    assert refreshed_inverse_stats["identity"]["numeric_version"] == (
        numeric_version + 1
    )
    assert refreshed_inverse_stats["operations"]["numeric_updates"] == 1
    assert refreshed_inverse_stats["resources"][
        "numeric_update_peak_temporary_bytes"
    ] == 232
    assert refreshed_inverse_stats["resources"][
        "operator_owned_reserved_bytes"
    ] == 632
    assert refreshed_inverse_stats["transfers"][
        "device_to_device_bytes"
    ] == 864

    refreshed_preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program, target, inverse, True
        )
    )
    refreshed_solver = make_solver(target, refreshed_preconditioner)
    solution.fill(0.0)
    refreshed_solver.solve(program, solution.arr, rhs.arr)
    ti.sync()
    assert refreshed_solver.is_success()
    np.testing.assert_allclose(
        solution.to_numpy(), exact_numpy, rtol=0.0, atol=2e-4
    )

    old_target_stats = target._debug_runtime_stats()
    old_inverse_stats = inverse._debug_runtime_stats()
    old_solver_stats = refreshed_solver._debug_runtime_stats()
    old_target_bytes = old_target_stats["resources"][
        "operator_owned_reserved_bytes"
    ]
    old_inverse_bytes = old_inverse_stats["resources"][
        "operator_owned_reserved_bytes"
    ]
    old_solver_workspace_bytes = old_solver_stats["resources"][
        "persistent_vector_reserved_bytes"
    ]
    assert (old_target_bytes, old_inverse_bytes, old_solver_workspace_bytes) == (
        228,
        632,
        128,
    )
    initial_publication = sparse_active_dof_map_bench._HierarchyPublication(
        program=program,
        topology_version=topology_version,
        numeric_version=numeric_version + 1,
        size=size,
        target=target,
        inverse=inverse,
        preconditioner=refreshed_preconditioner,
        solver=refreshed_solver,
        numeric_publisher=numeric_publisher,
        target_operator_bytes=old_target_bytes,
        inverse_operator_bytes=old_inverse_bytes,
        solver_workspace_bytes=old_solver_workspace_bytes,
        solver_workspace_materialized_bytes=old_solver_workspace_bytes,
        build_peak_device_bytes=(
            old_target_bytes + old_inverse_bytes + old_solver_workspace_bytes
        ),
    )
    registry = sparse_active_dof_map_bench._HierarchyPublicationRegistry(
        program, capacity_bytes=4096
    )
    initial_publish = registry.publish(
        expected_generation=0,
        topology_version=topology_version,
        estimated_steady_device_bytes=initial_publication.steady_device_bytes,
        estimated_build_peak_device_bytes=(
            initial_publication.build_peak_device_bytes
        ),
        builder=lambda: initial_publication,
    )
    assert initial_publish["published"]
    assert initial_publish["generation"] == 1
    assert initial_publish["steady_device_bytes"] == 988
    old_lease = registry.acquire()
    initial_publication_ref = weakref.ref(initial_publication)
    initial_publication = None
    target = None
    inverse = None
    preconditioner = None
    solver = None
    refreshed_preconditioner = None
    refreshed_solver = None
    numeric_publisher = None
    gc.collect()
    assert initial_publication_ref() is not None

    builder_calls = 0

    def must_not_build():
        nonlocal builder_calls
        builder_calls += 1
        raise AssertionError("capacity rejection invoked the builder")

    generation_attempt = registry.publish(
        expected_generation=0,
        topology_version=topology_version + 1,
        estimated_steady_device_bytes=1020,
        estimated_build_peak_device_bytes=1912,
        builder=must_not_build,
    )
    assert generation_attempt["status"] == "generation_mismatch"
    assert not generation_attempt["builder_invoked"]
    capacity_attempt = registry.publish(
        expected_generation=1,
        topology_version=topology_version + 1,
        estimated_steady_device_bytes=1020,
        estimated_build_peak_device_bytes=3109,
        builder=must_not_build,
    )
    assert capacity_attempt["status"] == "capacity_overflow"
    assert not capacity_attempt["builder_invoked"]
    assert builder_calls == 0

    def fail_build():
        raise RuntimeError("injected hierarchy build failure")

    failed_attempt = registry.publish(
        expected_generation=1,
        topology_version=topology_version + 1,
        estimated_steady_device_bytes=1020,
        estimated_build_peak_device_bytes=1912,
        builder=fail_build,
    )
    assert failed_attempt["status"] == "build_failed"
    assert failed_attempt["builder_invoked"]
    assert failed_attempt["error"] == (
        "RuntimeError: injected hierarchy build failure"
    )
    assert registry.debug_runtime_stats()["identity"]["generation"] == 1

    changed_pattern_numpy = np.concatenate(
        (changed_csr[0], changed_csr[1])
    ).astype(np.int32)
    changed_values_numpy = changed_csr[2].astype(np.float32)
    changed_dense_f32 = (
        sparse_active_dof_map_bench._csr_to_dense_reference(
            changed_csr[0], changed_csr[1], changed_values_numpy
        ).astype(np.float32)
    )
    changed_rhs_numpy = changed_dense_f32 @ exact_numpy
    changed_rhs = ti.ndarray(ti.f32, shape=size)
    changed_rhs.from_numpy(changed_rhs_numpy)
    candidate_publication_ref = None
    candidate_solver_workspace_materialized_bytes = None

    def build_changed_publication():
        nonlocal candidate_publication_ref
        nonlocal candidate_solver_workspace_materialized_bytes
        changed_pattern = ti.ndarray(ti.i32, shape=len(changed_pattern_numpy))
        changed_pattern.from_numpy(changed_pattern_numpy)
        changed_values = ti.ndarray(ti.f32, shape=len(changed_values_numpy))
        changed_values.from_numpy(changed_values_numpy)
        changed_target_key = target_primal.ensure_compiled(
            size, changed_pattern, changed_values, exact, changed_rhs
        )
        changed_target = (
            program._create_compiled_kernel_linear_operator_with_numeric_data(
                target_primal.compiled_kernels[changed_target_key],
                size,
                topology_version + 1,
                1,
                changed_pattern.arr,
                changed_values.arr,
            )
        )
        changed_plan = sparse_active_dof_map_bench._RecursiveVcycleGraphPlan(
            ti,
            program=program,
            hierarchy=changed_hierarchy,
            topology_version=topology_version + 1,
            numeric_version=1,
            pre_kernel=vcycle_pre,
            restrict_kernel=vcycle_restrict,
            bottom_kernel=vcycle_bottom,
            post_kernel=vcycle_post,
        )
        changed_plan_stats = changed_plan.debug_runtime_stats()
        changed_inverse = changed_plan.create_native_operator()
        changed_publisher = changed_plan.create_numeric_publisher()
        changed_preconditioner = (
            ti._lib.core._make_compiled_kernel_preconditioner_plan(
                program, changed_target, changed_inverse, True
            )
        )
        changed_solver = make_solver(changed_target, changed_preconditioner)
        target_stats = changed_target._debug_runtime_stats()
        inverse_stats = changed_inverse._debug_runtime_stats()
        solver_stats = changed_solver._debug_runtime_stats()
        target_bytes = target_stats["resources"][
            "operator_owned_reserved_bytes"
        ]
        inverse_bytes = inverse_stats["resources"][
            "operator_owned_reserved_bytes"
        ]
        solver_workspace_bytes = solver_stats["resources"][
            "persistent_vector_reserved_bytes"
        ]
        solver_workspace_reservation_bytes = 4 * size * np.dtype(np.float32).itemsize
        candidate_solver_workspace_materialized_bytes = solver_workspace_bytes
        source_target_bytes = (
            changed_pattern_numpy.nbytes + changed_values_numpy.nbytes
        )
        publication = sparse_active_dof_map_bench._HierarchyPublication(
            program=program,
            topology_version=topology_version + 1,
            numeric_version=1,
            size=size,
            target=changed_target,
            inverse=changed_inverse,
            preconditioner=changed_preconditioner,
            solver=changed_solver,
            numeric_publisher=changed_publisher,
            target_operator_bytes=target_bytes,
            inverse_operator_bytes=inverse_bytes,
            solver_workspace_bytes=solver_workspace_reservation_bytes,
            solver_workspace_materialized_bytes=solver_workspace_bytes,
            build_peak_device_bytes=(
                source_target_bytes
                + changed_plan_stats["resources"]["plan_owned_reserved_bytes"]
                + target_bytes
                + inverse_bytes
                + solver_workspace_reservation_bytes
            ),
        )
        assert (target_bytes, inverse_bytes) == (244, 648)
        assert solver_workspace_bytes in (0, solver_workspace_reservation_bytes)
        assert publication.build_peak_device_bytes == 1912
        candidate_publication_ref = weakref.ref(publication)
        return publication

    migrated_publish = registry.publish(
        expected_generation=1,
        topology_version=topology_version + 1,
        estimated_steady_device_bytes=1020,
        estimated_build_peak_device_bytes=1912,
        builder=build_changed_publication,
    )
    assert migrated_publish["published"], migrated_publish
    assert migrated_publish["generation"] == 2
    assert migrated_publish["steady_device_bytes"] == 1020
    assert migrated_publish["build_peak_device_bytes"] == 1912
    assert migrated_publish["old_plus_new_steady_device_bytes"] == 2008
    assert candidate_publication_ref() is not None
    new_lease = registry.acquire()
    assert new_lease.generation == 2
    assert new_lease.topology_version == topology_version + 1
    assert old_lease.generation == 1
    assert old_lease.topology_version == topology_version

    retained_capacity_attempt = registry.publish(
        expected_generation=2,
        topology_version=topology_version + 2,
        estimated_steady_device_bytes=1020,
        estimated_build_peak_device_bytes=2100,
        builder=must_not_build,
    )
    assert retained_capacity_attempt["status"] == "capacity_overflow"
    assert not retained_capacity_attempt["builder_invoked"]
    underestimated_candidate = sparse_active_dof_map_bench._HierarchyPublication(
        program=program,
        topology_version=topology_version + 2,
        numeric_version=1,
        size=size,
        target=new_lease._publication.target,
        inverse=new_lease._publication.inverse,
        preconditioner=new_lease._publication.preconditioner,
        solver=new_lease._publication.solver,
        numeric_publisher=new_lease._publication.numeric_publisher,
        target_operator_bytes=244,
        inverse_operator_bytes=648,
        solver_workspace_bytes=128,
        solver_workspace_materialized_bytes=(
            new_lease._publication.solver_workspace_materialized_bytes
        ),
        build_peak_device_bytes=1912,
    )
    underestimated_attempt = registry.publish(
        expected_generation=2,
        topology_version=topology_version + 2,
        estimated_steady_device_bytes=1020,
        estimated_build_peak_device_bytes=1900,
        builder=lambda: underestimated_candidate,
    )
    assert underestimated_attempt["status"] == "candidate_contract_rejected"
    assert underestimated_attempt["builder_invoked"]
    underestimated_candidate = None
    assert registry.debug_runtime_stats()["identity"]["generation"] == 2

    old_solution = ti.ndarray(ti.f32, shape=size)
    old_solution.fill(0.0)
    old_lease.solve(old_solution, rhs)
    new_solution = ti.ndarray(ti.f32, shape=size)
    new_solution.fill(0.0)
    new_lease.solve(new_solution, changed_rhs)
    ti.sync()
    np.testing.assert_allclose(
        old_solution.to_numpy(), exact_numpy, rtol=0.0, atol=2e-4
    )
    np.testing.assert_allclose(
        new_solution.to_numpy(), exact_numpy, rtol=0.0, atol=2e-4
    )
    assert_graph_cache_execution(
        old_lease._publication.inverse._debug_graph_cache_stats(),
        complete_counters=False,
    )
    assert_graph_cache_execution(
        new_lease._publication.inverse._debug_graph_cache_stats(),
        complete_counters=False,
    )
    registry_stats = registry.debug_runtime_stats()
    assert registry_stats["identity"]["generation"] == 2
    assert registry_stats["identity"]["topology_version"] == (
        topology_version + 1
    )
    assert registry_stats["operations"] == {
        "publish_attempts": 7,
        "successful_publishes": 2,
        "rejected_publishes": 5,
        "build_failures": 1,
        "active_leases": 2,
    }
    assert registry_stats["resources"]["current_steady_device_bytes"] == 1020
    assert registry_stats["resources"]["current_target_operator_bytes"] == 244
    assert registry_stats["resources"]["current_inverse_operator_bytes"] == 648
    assert registry_stats["resources"]["current_solver_workspace_bytes"] == 128
    assert registry_stats["resources"][
        "current_solver_workspace_materialized_bytes"
    ] == candidate_solver_workspace_materialized_bytes
    assert registry_stats["resources"][
        "current_publisher_device_reserved_bytes"
    ] == 0
    assert registry_stats["resources"][
        "live_generation_steady_device_bytes"
    ] == 2008
    assert registry_stats["resources"][
        "retired_lease_steady_device_bytes"
    ] == 988
    assert registry_stats["resources"][
        "publish_overlap_peak_device_bytes"
    ] == 2900
    assert registry_stats["resources"][
        "current_publisher_host_metadata_bytes"
    ] == 300
    assert registry_stats["resources"][
        "live_publisher_host_metadata_bytes"
    ] == 592
    assert registry_stats["resources"][
        "retired_lease_publisher_host_metadata_bytes"
    ] == 292
    assert registry_stats["resources"][
        "publish_overlap_peak_publisher_host_metadata_bytes"
    ] == 592
    if arch == ti.cpu:
        assert registry_stats["resources"][
            "graph_runtime_cache_known_device_argument_bytes"
        ] == 0
        assert registry_stats["resources"][
            "graph_runtime_cache_opaque_generation_count"
        ] == 0
        assert registry_stats["resources"]["graph_runtime_cache_device_bytes"] == 0
        assert registry_stats["contract"][
            "graph_runtime_cache_bytes_reported"
        ]
    else:
        assert registry_stats["resources"][
            "graph_runtime_cache_known_device_argument_bytes"
        ] > 0
        assert registry_stats["resources"][
            "graph_runtime_cache_opaque_generation_count"
        ] == 2
        assert (
            registry_stats["resources"]["graph_runtime_cache_device_bytes"]
            is None
        )
        assert not registry_stats["contract"][
            "graph_runtime_cache_bytes_reported"
        ]
    assert registry_stats["resources"]["registry_device_reserved_bytes"] == 0
    assert not registry_stats["contract"][
        "graph_runtime_cache_in_explicit_capacity"
    ]

    old_lease.release()
    old_solution.fill(23.0)
    with pytest.raises(RuntimeError, match="lease was released"):
        old_lease.solve(old_solution, rhs)
    np.testing.assert_array_equal(
        old_solution.to_numpy(), np.full(size, 23.0, dtype=np.float32)
    )
    gc.collect()
    assert initial_publication_ref() is None
    released_stats = registry.debug_runtime_stats()
    assert released_stats["operations"]["active_leases"] == 1
    assert released_stats["resources"][
        "live_generation_steady_device_bytes"
    ] == 1020
    assert released_stats["resources"][
        "retired_lease_steady_device_bytes"
    ] == 0
    assert released_stats["resources"][
        "live_publisher_host_metadata_bytes"
    ] == 300
    assert released_stats["resources"][
        "graph_runtime_cache_opaque_generation_count"
    ] == (0 if arch == ti.cpu else 1)


@pytest.mark.parametrize("dimensions", [2, 3])
@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
    cuda_sparse_pool_auto_size=True,
    cuda_sparse_per_snode_pool=True,
)
def test_active_block_dof_map_matches_snode_oracle_and_migrates(dimensions):
    report = sparse_active_dof_map_bench.run_initialized(
        ti,
        dimensions=dimensions,
    )

    assert report["schema"] == "taichi_forge.sparse_active_dof_map.v1"
    assert report["schema_version"] == 1
    assert report["correct"]
    assert report["topology"]["version"] == 2
    assert report["topology"]["sort_method"] != "auto"
    assert report["topology"]["unique_method"] != "auto"
    assert not report["topology"]["host_active_prefix_read_required_to_build"]
    assert report["topology"]["host_staging_status_read_required_to_publish"]
    assert report["topology"]["publish_model"] == (
        "host_mediated_two_phase_before_snode_mutation"
    )
    assert not report["topology"]["mutation_fault_rollback"]
    assert report["operator_contract"]["snode_accesses_per_compact_apply"] == 0
    assert report["operator_contract"][
        "snode_struct_for_is_correctness_oracle_only"
    ]
    assert (
        report["operator_contract"]["plan_owned_apply_allocation_count"] == 0
    )
    assert (
        report["operator_contract"][
            "explicit_apply_host_synchronization_count"
        ]
        == 0
    )
    assert report["operator_contract"][
        "profile_private_not_solver_provider"
    ]
    assert not report["operator_contract"][
        "profile_plan_reuses_native_sparse_matrix_hook"
    ]
    assert report["operator_contract"][
        "compiled_kernel_provider_reuses_native_sparse_matrix_hook"
    ]
    assert not report["operator_contract"][
        "compiled_kernel_provider_solver_integrated"
    ]

    expected_blocks = 4
    expected_dofs = expected_blocks * 2**dimensions
    for phase_name in ("initial", "after_overflow", "migrated"):
        phase = report["checks"][phase_name]
        assert phase["active_blocks"] == expected_blocks
        assert phase["active_dofs"] == expected_dofs
        assert phase["ordering_matches"]
        assert phase["ordered_keys"] == sorted(phase["ordered_keys"])
        assert phase["gather_difference_l1"] == 0
        assert phase["operator_difference_l1"] == 0
    attempts = report["attempts"]
    assert attempts["initial"]["published"]
    assert attempts["initial"]["version_after"] == 1
    assert attempts["overflow"]["status"] == "capacity_overflow"
    assert not attempts["overflow"]["published"]
    assert attempts["overflow"]["old_topology_preserved"]
    assert attempts["overflow"]["version_before"] == 1
    assert attempts["overflow"]["version_after"] == 1
    assert attempts["migrated"]["published"]
    assert attempts["migrated"]["version_after"] == 2

    memory = report["resources"]["memory_attribution"]
    scalar_bytes = 4
    max_dofs = (report["config"]["candidate_capacity"] - 1) * 2**dimensions
    assert memory["scalar_and_index_bytes"] == scalar_bytes
    assert memory["candidate_staging_reserved_bytes"] == (
        report["config"]["producer_capacity"] * dimensions * scalar_bytes
        + report["config"]["candidate_capacity"] * scalar_bytes
        + 2 * scalar_bytes
    )
    assert memory["dof_map_reserved_bytes"] == (
        report["config"]["candidate_capacity"] * scalar_bytes
        + 2 * scalar_bytes
        + max_dofs * dimensions * scalar_bytes
    )
    assert memory["structured_operator_neighbor_reserved_bytes"] == (
        max_dofs * 2 * dimensions * scalar_bytes
    )
    assert memory["structured_operator_neighbor_active_bytes"] == (
        expected_dofs * 2 * dimensions * scalar_bytes
    )
    assert memory["diagnostic_vector_reserved_bytes"] == (
        memory["diagnostic_vector_capacity_entries"] * scalar_bytes
    )
    assert memory["profile_owned_reserved_bytes"] == sum(
        memory[name]
        for name in (
            "candidate_staging_reserved_bytes",
            "dof_map_reserved_bytes",
            "structured_operator_neighbor_reserved_bytes",
            "diagnostic_vector_reserved_bytes",
            "two_level_residency_peak_reserved_bytes",
        )
    )
    assert memory["native_provider_operator_data_reserved_bytes"] == memory[
        "structured_operator_neighbor_reserved_bytes"
    ]
    assert memory["profile_and_native_provider_peak_reserved_bytes"] == (
        memory["profile_owned_reserved_bytes"]
        + memory["native_provider_operator_data_reserved_bytes"]
    )
    expected_coarse_dofs = expected_blocks
    expected_two_level_steady_bytes = (
        expected_dofs * scalar_bytes
        + expected_coarse_dofs**2 * scalar_bytes
        + 2 * expected_coarse_dofs * scalar_bytes
    )
    assert memory["two_level_preconditioner_steady_reserved_bytes"] == (
        expected_two_level_steady_bytes
    )
    assert memory[
        "two_level_preconditioner_migration_overlap_reserved_bytes"
    ] == (2 * expected_two_level_steady_bytes)
    assert memory[
        "two_level_preconditioner_numeric_refresh_overlap_reserved_bytes"
    ] == (2 * expected_two_level_steady_bytes)
    assert memory[
        "two_level_preconditioner_publish_overlap_peak_reserved_bytes"
    ] == (2 * expected_two_level_steady_bytes)
    assert memory["native_graph_provider_reserved_bytes"] == (
        expected_two_level_steady_bytes
    )
    assert memory["two_level_native_bridge_overlap_reserved_bytes"] == (
        2 * expected_two_level_steady_bytes
    )
    assert memory["two_level_residency_peak_reserved_bytes"] == (
        2 * expected_two_level_steady_bytes
    )
    actual_nnz = expected_dofs + memory["active_neighbor_entries"]
    assert memory["csr_reference_actual_nnz"] == actual_nnz
    assert memory["csr_reference_actual_pattern_bytes"] == (
        expected_dofs + 1 + actual_nnz
    ) * scalar_bytes
    assert memory["csr_reference_actual_value_bytes"] == (
        actual_nnz * scalar_bytes
    )
    assert memory["csr_reference_actual_total_bytes"] <= memory[
        "csr_reference_upper_total_bytes"
    ]
    assert memory["csr_actual_minus_structured_active_operator_bytes"] == (
        memory["csr_reference_actual_total_bytes"]
        - memory["structured_operator_neighbor_active_bytes"]
    )
    assert memory["allocation_scopes"]["coordinate_to_dof_field"] == (
        "included_in_snode_tree_memory"
    )
    assert report["checks"]["stale_initial_probe_l1"] == 0
    plans = report["operator_plans"]
    assert plans["overflow_preserved_initial_plan"]
    assert plans["stale_initial_plan_rejected_before_mutation"]
    assert plans["alias_rejected_before_mutation"]
    assert plans["numeric_version_mismatch_rejected_before_mutation"]
    assert plans["initial"]["identity"]["topology_version"] == 1
    assert plans["initial"]["operations"]["apply_calls"] == 2
    assert plans["initial"]["operations"]["rejected_apply_calls"] == 1
    assert plans["migrated"]["identity"]["topology_version"] == 2
    assert plans["migrated"]["identity"]["numeric_version"] == 1
    assert plans["migrated"]["operations"]["apply_calls"] == 2
    assert plans["migrated"]["operations"]["rejected_apply_calls"] == 2
    assert (
        plans["migrated"]["operations"]["plan_owned_apply_allocations"] == 0
    )
    assert (
        plans["migrated"]["operations"][
            "explicit_apply_host_synchronizations"
        ]
        == 0
    )
    assert not plans["migrated"]["resources"]["owns_snode_tree"]
    assert not plans["migrated"]["resources"]["owns_input_or_output"]
    native = report["native_operator"]
    assert not native["snode_dependencies_allowed"]
    assert native["operator_data_snapshot_owned"]
    assert native["alias_rejected_before_mutation"]
    assert native["operator_data_source_released_before_second_apply"]
    assert native["before_destroy_difference_l1"] == 0
    assert native["post_destroy_difference_l1"] == 0
    native_first = native["before_destroy_stats"]
    native_final = native["post_destroy_stats"]
    assert native_final["identity"]["backend_family"] == report["arch"]
    assert native_final["identity"]["storage_format"] == "matrix_free_kernel"
    assert native_final["identity"]["dtype"] == "f32"
    assert native_final["identity"]["rows"] == expected_dofs
    assert native_final["identity"]["cols"] == expected_dofs
    assert native_final["identity"]["nnz"] == 0
    assert native_final["identity"]["pattern_version"] == 2
    assert native_final["identity"]["numeric_version"] == 1
    assert native_first["operations"]["pattern_builds"] == 1
    assert native_first["operations"]["spmv_calls"] == 1
    assert native_first["operations"]["spmv_plan_builds"] == 1
    assert native_first["operations"]["spmv_plan_reuses"] == 1
    assert native_final["operations"]["spmv_calls"] == 2
    assert native_final["operations"]["spmv_plan_builds"] == 1
    assert native_final["operations"]["spmv_plan_reuses"] == 2
    assert native_final["resources"]["pattern_reserved_bytes"] == memory[
        "native_provider_operator_data_reserved_bytes"
    ]
    assert native_final["resources"]["operator_owned_reserved_bytes"] == (
        memory["native_provider_operator_data_reserved_bytes"]
    )
    assert native_final["transfers"]["device_to_device_bytes"] == memory[
        "native_provider_operator_data_reserved_bytes"
    ]
    assert native_final["provider"]["name"] == (
        "forge_compiled_taichi_kernel"
    )
    native_graph = report["native_graph_operator"]
    assert not native_graph["snode_dependencies_allowed"]
    assert native_graph["resource_roles_explicit"]
    assert native_graph["source_snapshots_owned"]
    assert native_graph["alias_rejected_before_mutation"]
    assert native_graph["source_python_graph_plan_released"]
    assert native_graph["before_destroy_difference_linf"] <= 5e-5
    assert native_graph["post_destroy_difference_linf"] <= 5e-5
    assert not native_graph["solver_integrated"]
    native_graph_first = native_graph["before_destroy_stats"]
    native_graph_final = native_graph["post_destroy_stats"]
    assert native_graph_final["identity"]["backend_family"] == report["arch"]
    assert native_graph_final["identity"]["storage_format"] == (
        "matrix_free_graph"
    )
    assert native_graph_final["identity"]["dtype"] == "f32"
    assert native_graph_final["identity"]["rows"] == expected_dofs
    assert native_graph_final["identity"]["cols"] == expected_dofs
    assert native_graph_final["identity"]["nnz"] == 0
    assert native_graph_final["identity"]["pattern_version"] == 2
    assert native_graph_final["identity"]["numeric_version"] == 2
    assert native_graph_first["operations"]["pattern_builds"] == 1
    assert native_graph_first["operations"]["spmv_calls"] == 1
    assert native_graph_first["operations"]["spmv_plan_builds"] == 1
    assert native_graph_first["operations"]["spmv_plan_reuses"] == 1
    assert native_graph_first["operations"][
        "spmv_workspace_allocations"
    ] == 2
    assert native_graph_final["operations"]["spmv_calls"] == 2
    assert native_graph_final["operations"]["spmv_plan_builds"] == 1
    assert native_graph_final["operations"]["spmv_plan_reuses"] == 2
    assert native_graph_final["resources"]["pattern_reserved_bytes"] == (
        expected_dofs * scalar_bytes
    )
    assert native_graph_final["resources"]["values_reserved_bytes"] == (
        expected_coarse_dofs**2 * scalar_bytes
    )
    assert native_graph_final["resources"][
        "spmv_workspace_reserved_bytes"
    ] == (2 * expected_coarse_dofs * scalar_bytes)
    assert native_graph_final["resources"][
        "operator_owned_reserved_bytes"
    ] == expected_two_level_steady_bytes
    assert native_graph_final["transfers"]["device_to_device_bytes"] == (
        expected_two_level_steady_bytes
    )
    assert native_graph_final["provider"]["name"] == "forge_compiled_graph"
    two_level = report["two_level_preconditioner"]
    assert two_level["method"] == "two_level_additive_galerkin"
    assert two_level["fine_level_storage"] == "compact_active_dof_ndarray"
    assert two_level["coarse_level_storage"] == (
        "one_dof_per_sorted_active_brick"
    )
    for phase_name in (
        "initial",
        "migrated",
        "numeric_refreshed",
        "after_transient_rebind",
        "post_tree_destroy",
    ):
        check = two_level["checks"][phase_name]
        assert check["fine_size"] == expected_dofs
        assert check["coarse_size"] == expected_blocks
        assert check["coarse_ordered_keys"] == sorted(
            check["coarse_ordered_keys"]
        )
        assert check["coarse_ordering_matches_fine_bricks"]
        assert check["fine_to_coarse_complete"]
        assert check["fine_operator_symmetric"]
        assert check["fine_operator_min_eigenvalue"] > 0
        assert check["coarse_operator_symmetric"]
        assert check["coarse_operator_min_eigenvalue"] > 0
        coarse_storage = check["coarse_storage"]
        assert coarse_storage["directional_reconstructs_dense_galerkin"]
        assert coarse_storage["positive_diagonal"]
        assert coarse_storage["nonpositive_offdiagonal"]
        assert coarse_storage["reciprocal_weighted_adjacency"]
        assert coarse_storage["actual_nnz"] <= (
            expected_coarse_dofs * (2 * dimensions + 1)
        )
        assert coarse_storage["max_row_nnz"] <= (
            coarse_storage["row_nnz_upper_bound"]
        )
        assert coarse_storage["row_nnz_upper_bound"] == 2 * dimensions + 1
        assert coarse_storage["directional_topology_bytes"] == (
            expected_coarse_dofs * 2 * dimensions * scalar_bytes
        )
        assert coarse_storage["directional_numeric_bytes"] == (
            expected_coarse_dofs * (2 * dimensions + 1) * scalar_bytes
        )
        assert coarse_storage["directional_total_bytes"] == (
            coarse_storage["directional_topology_bytes"]
            + coarse_storage["directional_numeric_bytes"]
        )
        assert coarse_storage["csr_pattern_bytes"] == (
            expected_coarse_dofs + 1 + coarse_storage["actual_nnz"]
        ) * scalar_bytes
        assert coarse_storage["csr_value_bytes"] == (
            coarse_storage["actual_nnz"] * scalar_bytes
        )
        assert coarse_storage["csr_total_bytes"] == (
            coarse_storage["csr_pattern_bytes"]
            + coarse_storage["csr_value_bytes"]
        )
        assert coarse_storage["dense_inverse_bytes"] == (
            expected_coarse_dofs**2 * scalar_bytes
        )
        assert sum(coarse_storage["component_sizes"]) == expected_coarse_dofs
        assert coarse_storage["component_count"] == len(
            coarse_storage["component_sizes"]
        )
        assert coarse_storage["component_dense_inverse_bytes"] == (
            sum(size**2 for size in coarse_storage["component_sizes"])
            * scalar_bytes
        )
        assert coarse_storage["component_dense_inverse_bytes"] <= (
            coarse_storage["dense_inverse_bytes"]
        )
        assert coarse_storage["directional_bound_scope"] == (
            "axis_aligned_complete_brick_scalar_poisson_only"
        )
        assert coarse_storage["generic_sparse_system_storage"] == "csr_or_bsr"
        assert coarse_storage["directional_stencil_is_optional_specialization"]
        assert not coarse_storage["solve_workspace_included"]
        assert check["preconditioner_symmetric"]
        assert check["preconditioner_min_eigenvalue"] > 0
        assert check["output_difference_linf"] <= 5e-5
    coarse_probe = two_level["coarse_operator_storage_probe"]
    assert coarse_probe["representation"] == "directional_stencil_or_csr"
    assert coarse_probe["public_solver_baseline"] == "generic_csr_or_bsr"
    assert coarse_probe["directional_stencil_scope"] == (
        "profile_private_axis_aligned_brick_specialization"
    )
    assert coarse_probe["storage_complexity"] == "linear_in_coarse_dofs"
    assert coarse_probe["dense_inverse_complexity"] == (
        "quadratic_in_coarse_dofs"
    )
    assert coarse_probe["solve_not_selected_by_storage_probe"]
    initial_level_plan = two_level["plans"]["initial"]
    migrated_level_plan = two_level["plans"]["migrated"]
    refreshed_level_plan = two_level["plans"]["numeric_refreshed"]
    assert initial_level_plan["identity"]["fine_topology_version"] == 1
    assert initial_level_plan["identity"]["coarse_topology_version"] == 1
    assert initial_level_plan["operations"]["apply_calls"] == 1
    assert initial_level_plan["operations"]["rejected_apply_calls"] == 1
    assert initial_level_plan["operations"]["kernel_launches"] == 3
    assert migrated_level_plan["identity"]["fine_topology_version"] == 2
    assert migrated_level_plan["identity"]["coarse_topology_version"] == 2
    assert migrated_level_plan["operations"]["apply_calls"] == 1
    assert migrated_level_plan["operations"]["rejected_apply_calls"] == 2
    assert migrated_level_plan["operations"]["kernel_launches"] == 3
    assert refreshed_level_plan["identity"]["fine_topology_version"] == 2
    assert refreshed_level_plan["identity"]["coarse_topology_version"] == 2
    assert refreshed_level_plan["identity"]["numeric_version"] == 2
    assert refreshed_level_plan["operations"]["apply_calls"] == 4
    assert refreshed_level_plan["operations"]["rejected_apply_calls"] == 0
    assert refreshed_level_plan["operations"]["kernel_launches"] == 12
    assert refreshed_level_plan["operations"]["host_graph_submissions"] == 4
    assert refreshed_level_plan["operations"]["graph_execution_path"] != (
        "not_run"
    )
    if report["arch"] == "cpu":
        assert refreshed_level_plan["operations"]["graph_backend_segments"] == 0
    else:
        assert refreshed_level_plan["operations"]["graph_backend_segments"] == 1
        assert refreshed_level_plan["operations"]["graph_execution_path"].startswith(
            report["arch"]
        )
    assert refreshed_level_plan["operations"][
        "plan_owned_apply_allocations"
    ] == 0
    assert refreshed_level_plan["operations"][
        "explicit_apply_host_synchronizations"
    ] == 0
    assert refreshed_level_plan["resources"]["plan_owned_reserved_bytes"] == (
        expected_two_level_steady_bytes
    )
    assert refreshed_level_plan["resources"]["owns_level_topology"]
    assert refreshed_level_plan["resources"]["owns_level_numeric_data"]
    assert refreshed_level_plan["resources"]["owns_level_workspace"]
    assert not refreshed_level_plan["resources"]["owns_snode_tree"]
    assert refreshed_level_plan["resources"]["owned_allocation_identity_stable"]
    assert not refreshed_level_plan["resources"][
        "plan_argument_dict_retains_last_input_or_output"
    ]
    assert refreshed_level_plan["resources"][
        "graph_fast_arg_cache_retains_last_native_binding"
    ]
    assert refreshed_level_plan["resources"][
        "last_bound_caller_vector_bytes_not_plan_owned"
    ] == (2 * expected_dofs * scalar_bytes)
    assert refreshed_level_plan["resources"][
        "graph_persistent_argument_bytes_outside_plan_payload"
    ] >= 0
    assert refreshed_level_plan["contract"]["additive_spd_contract"]
    assert not refreshed_level_plan["contract"][
        "compiled_single_kernel_eligible"
    ]
    assert refreshed_level_plan["contract"]["graph_sequence_candidate"]
    assert refreshed_level_plan["contract"][
        "cached_graph_execution_integrated"
    ]
    assert not refreshed_level_plan["contract"][
        "backend_replay_required_for_correctness"
    ]
    assert refreshed_level_plan["contract"]["one_host_graph_run_per_apply"]
    assert refreshed_level_plan["contract"][
        "graph_runtime_resource_lease_managed_by_program"
    ]
    assert refreshed_level_plan["contract"]["graph_node_count"] == 1
    assert refreshed_level_plan["contract"]["graph_dispatch_count"] == 3
    assert refreshed_level_plan["contract"]["graph_runtime_arg_count"] == 8
    assert not refreshed_level_plan["contract"]["solver_integrated"]
    assert two_level["lifecycle"]["built_after_topology_publish"]
    assert two_level["lifecycle"]["overflow_preserved_initial_plan"]
    assert two_level["lifecycle"][
        "stale_initial_plan_rejected_before_mutation"
    ]
    assert two_level["lifecycle"]["alias_rejected_before_mutation"]
    assert two_level["lifecycle"][
        "stale_numeric_plan_rejected_before_mutation"
    ]
    assert two_level["lifecycle"]["numeric_refresh_rebuilt_current_plan"]
    assert two_level["lifecycle"]["plan_source_snapshots_released"]
    assert two_level["lifecycle"][
        "transient_input_output_wrappers_released"
    ]
    assert two_level["lifecycle"][
        "graph_cache_pinned_last_transient_native_views"
    ]
    assert two_level["lifecycle"][
        "transient_native_views_released_after_rebind"
    ]
    assert two_level["lifecycle"][
        "migration_kept_stale_and_current_plans_alive"
    ]
    assert two_level["lifecycle"][
        "numeric_refresh_kept_stale_and_current_plans_alive"
    ]
    assert two_level["lifecycle"]["current_plan_survives_snode_tree_destroy"]
    assert two_level["execution"]["kernel_launches_per_apply"] == 3
    assert two_level["execution"]["host_graph_submissions_per_apply"] == 1
    assert not two_level["execution"][
        "single_compiled_kernel_provider_sufficient"
    ]
    assert two_level["execution"]["compiled_graph_sequence_candidate"]
    assert two_level["execution"]["compiled_graph_integrated"]
    assert two_level["execution"][
        "graph_runtime_resource_lease_managed_by_program"
    ]
    assert two_level["execution"][
        "graph_fast_arg_cache_retains_last_native_binding"
    ]
    assert not two_level["execution"]["solver_integrated"]
    assert two_level["memory"]["steady_plan_reserved_bytes"] == (
        expected_two_level_steady_bytes
    )
    assert two_level["memory"]["migration_overlap_peak_reserved_bytes"] == (
        2 * expected_two_level_steady_bytes
    )
    assert two_level["memory"][
        "numeric_refresh_overlap_peak_reserved_bytes"
    ] == (2 * expected_two_level_steady_bytes)
    assert two_level["memory"]["publish_overlap_peak_reserved_bytes"] == (
        2 * expected_two_level_steady_bytes
    )
    assert two_level["memory"]["native_graph_provider_reserved_bytes"] == (
        expected_two_level_steady_bytes
    )
    assert two_level["memory"][
        "native_bridge_handoff_overlap_reserved_bytes"
    ] == (2 * expected_two_level_steady_bytes)
    assert two_level["memory"]["level_residency_peak_reserved_bytes"] == (
        2 * expected_two_level_steady_bytes
    )
    assert two_level["memory"][
        "last_bound_caller_vector_bytes_not_plan_owned"
    ] == (2 * expected_dofs * scalar_bytes)
    assert two_level["memory"][
        "last_bound_caller_vectors_may_be_pinned_until_rebind"
    ]
    assert two_level["memory"][
        "graph_persistent_argument_bytes_outside_plan_payload"
    ] >= 0
    assert two_level["memory"]["graph_persistent_argument_memory_domain"] == (
        "backend_host_runtime_argument_buffers_not_level_payload"
    )
    assert report["lifecycle"]["tree_present_before_destroy"]
    assert report["lifecycle"]["tree_recovered_after_destroy"]
    assert report["lifecycle"]["operator_survives_tree_destroy"]
    assert report["lifecycle"]["post_destroy_operator_difference_l1"] == 0
    assert report["lifecycle"][
        "native_operator_survives_tree_and_source_destroy"
    ]
    assert report["lifecycle"][
        "native_graph_operator_survives_tree_and_source_destroy"
    ]
    assert report["lifecycle"][
        "two_level_preconditioner_survives_tree_destroy"
    ]


def test_active_block_dof_map_rejects_unsupported_dimensions():
    with pytest.raises(ValueError, match="dimensions must be 2 or 3"):
        sparse_active_dof_map_bench._topologies(4)


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_compiled_kernel_operator_rejects_invalid_abi_and_snode_dependency():
    operator_data = ti.ndarray(ti.i32, shape=(1, 2))
    input_array = ti.ndarray(ti.f32, shape=1)
    output_array = ti.ndarray(ti.f32, shape=1)
    field_state = ti.field(ti.f32, shape=1)

    @ti.kernel
    def invalid_abi(
        data: ti.types.ndarray(dtype=ti.i32, ndim=2),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in x:
            y[index] = x[index] + ti.cast(data[index, 0], ti.f32)

    @ti.kernel
    def snode_dependent(
        active_size: ti.i32,
        data: ti.types.ndarray(dtype=ti.i32, ndim=2),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = (
                x[index]
                + field_state[0]
                + ti.cast(data[index, 0], ti.f32)
            )

    @ti.kernel
    def graph_identity(
        active_size: ti.i32,
        data: ti.types.ndarray(dtype=ti.i32, ndim=2),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = x[index] + ti.cast(data[index, 0], ti.f32)

    def compiled_kernel(function, *args):
        primal = function._primal
        key = primal.ensure_compiled(*args)
        return primal.compiled_kernels[key]

    invalid_kernel = compiled_kernel(
        invalid_abi,
        operator_data,
        input_array,
        output_array,
    )
    program = ti.lang.impl.get_runtime().prog
    with pytest.raises(RuntimeError, match="ABI must be exactly"):
        program._create_compiled_kernel_linear_operator(
            invalid_kernel,
            1,
            1,
            1,
            operator_data.arr,
        )

    dependent_kernel = compiled_kernel(
        snode_dependent,
        1,
        operator_data,
        input_array,
        output_array,
    )
    with pytest.raises(RuntimeError, match="must not depend on any SNodeTree"):
        program._create_compiled_kernel_linear_operator(
            dependent_kernel,
            1,
            1,
            1,
            operator_data.arr,
        )

    sym_active_size = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "active_size", ti.i32
    )
    sym_data = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "data", ti.i32, ndim=2
    )
    sym_input = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    sym_output = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(
        graph_identity,
        sym_active_size,
        sym_data,
        sym_input,
        sym_output,
    )
    graph = graph_builder.compile()
    with pytest.raises(RuntimeError, match="graph has 4 arguments but 3 roles"):
        program._create_compiled_graph_linear_operator(
            graph._compiled_graph,
            1,
            1,
            1,
            {},
            {"data": operator_data.arr},
            {},
            {},
        )

    dependent_graph_builder = ti.graph.GraphBuilder()
    dependent_graph_builder.dispatch(
        snode_dependent,
        sym_active_size,
        sym_data,
        sym_input,
        sym_output,
    )
    dependent_graph = dependent_graph_builder.compile()
    with pytest.raises(RuntimeError, match="must not depend on any SNodeTree"):
        program._create_compiled_graph_linear_operator(
            dependent_graph._compiled_graph,
            1,
            1,
            1,
            {"active_size": 1},
            {"data": operator_data.arr},
            {},
            {},
        )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_compiled_kernel_operator_rebinds_persistent_launch_context():
    size = 4
    operator_data = ti.ndarray(ti.i32, shape=1)
    operator_data.fill(3)
    first_input = ti.ndarray(ti.f32, shape=size)
    first_output = ti.ndarray(ti.f32, shape=size)
    first_input.fill(2.0)
    first_output.fill(0.0)

    @ti.kernel
    def apply_scaled_identity(
        active_size: ti.i32,
        data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = ti.cast(data[0], ti.f32) * x[index]

    primal = apply_scaled_identity._primal
    key = primal.ensure_compiled(
        size,
        operator_data,
        first_input,
        first_output,
    )
    program = ti.lang.impl.get_runtime().prog
    operator = program._create_compiled_kernel_linear_operator(
        primal.compiled_kernels[key],
        size,
        1,
        1,
        operator_data.arr,
    )
    operator.spmv(program, first_input.arr, first_output.arr)
    ti.sync()
    assert first_output.to_numpy().tolist() == [6.0] * size

    first_input_ref = weakref.ref(first_input)
    first_output_ref = weakref.ref(first_output)
    first_input = None
    first_output = None
    gc.collect()
    assert first_input_ref() is None
    assert first_output_ref() is None

    second_input = ti.ndarray(ti.f32, shape=size)
    second_output = ti.ndarray(ti.f32, shape=size)
    second_input.fill(4.0)
    second_output.fill(0.0)
    operator.spmv(program, second_input.arr, second_output.arr)
    ti.sync()
    assert second_output.to_numpy().tolist() == [12.0] * size
    stats = operator._debug_runtime_stats()
    assert stats["operations"]["spmv_calls"] == 2
    assert stats["operations"]["spmv_plan_builds"] == 1
    assert stats["operations"]["spmv_plan_reuses"] == 2


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_compiled_kernel_operator_publishes_numeric_snapshot_update():
    size = 4
    topology_version = 7
    numeric_version = 11
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    numeric.from_numpy(np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    input_array = ti.ndarray(ti.f32, shape=size)
    output_array = ti.ndarray(ti.f32, shape=size)
    input_array.from_numpy(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )
    output_array.fill(0.0)

    @ti.kernel
    def apply_weighted_map(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    primal = apply_weighted_map._primal
    key = primal.ensure_compiled(
        size,
        topology,
        numeric,
        input_array,
        output_array,
    )
    program = ti.lang.impl.get_runtime().prog
    operator = (
        program._create_compiled_kernel_linear_operator_with_numeric_data(
            primal.compiled_kernels[key],
            size,
            topology_version,
            numeric_version,
            topology.arr,
            numeric.arr,
        )
    )
    topology_ref = weakref.ref(topology)
    numeric_ref = weakref.ref(numeric)
    topology = None
    numeric = None
    gc.collect()
    assert topology_ref() is None
    assert numeric_ref() is None

    operator.spmv(program, input_array.arr, output_array.arr)
    ti.sync()
    np.testing.assert_array_equal(
        output_array.to_numpy(),
        np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float32),
    )
    initial = operator._debug_runtime_stats()
    topology_bytes = size * np.dtype(np.int32).itemsize
    numeric_bytes = size * np.dtype(np.float32).itemsize
    assert initial["identity"]["pattern_version"] == topology_version
    assert initial["identity"]["numeric_version"] == numeric_version
    assert initial["operations"]["numeric_updates"] == 0
    assert initial["resources"]["pattern_reserved_bytes"] == topology_bytes
    assert initial["resources"]["values_reserved_bytes"] == numeric_bytes
    assert initial["resources"]["operator_owned_reserved_bytes"] == (
        topology_bytes + numeric_bytes
    )
    assert initial["transfers"]["device_to_device_bytes"] == (
        topology_bytes + numeric_bytes
    )

    replacement = ti.ndarray(ti.f32, shape=size)
    replacement.from_numpy(
        np.array([5.0, 4.0, 3.0, 2.0], dtype=np.float32)
    )
    with pytest.raises(RuntimeError, match="version mismatch"):
        operator.update_numeric_data(
            program,
            replacement.arr,
            topology_version - 1,
            numeric_version,
        )
    rejected = operator._debug_runtime_stats()
    assert rejected["identity"]["numeric_version"] == numeric_version
    assert rejected["operations"]["numeric_updates"] == 0
    output_array.fill(0.0)
    operator.spmv(program, input_array.arr, output_array.arr)
    ti.sync()
    np.testing.assert_array_equal(
        output_array.to_numpy(),
        np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float32),
    )

    operator.update_numeric_data(
        program,
        replacement.arr,
        topology_version,
        numeric_version,
    )
    replacement_ref = weakref.ref(replacement)
    replacement = None
    gc.collect()
    assert replacement_ref() is None
    output_array.fill(0.0)
    operator.spmv(program, input_array.arr, output_array.arr)
    ti.sync()
    np.testing.assert_array_equal(
        output_array.to_numpy(),
        np.array([5.0, 8.0, 9.0, 8.0], dtype=np.float32),
    )

    final = operator._debug_runtime_stats()
    assert final["identity"]["pattern_version"] == topology_version
    assert final["identity"]["numeric_version"] == numeric_version + 1
    assert final["operations"]["pattern_builds"] == 1
    assert final["operations"]["numeric_updates"] == 1
    assert final["operations"]["numeric_update_bytes"] == numeric_bytes
    assert final["operations"]["spmv_calls"] == 3
    assert final["operations"]["spmv_plan_builds"] == 1
    assert final["operations"]["spmv_plan_reuses"] == 3
    assert final["transfers"]["device_to_device_bytes"] == (
        topology_bytes + 2 * numeric_bytes
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_compiled_kernel_preconditioner_binds_both_operator_versions():
    size = 4
    target_topology_version = 3
    target_numeric_version = 5
    inverse_topology_version = 7
    inverse_numeric_version = 11
    identity = ti.ndarray(ti.i32, shape=size)
    identity.from_numpy(np.arange(size, dtype=np.int32))
    target_numeric = ti.ndarray(ti.f32, shape=size)
    target_numeric.from_numpy(
        np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    )
    inverse_numeric = ti.ndarray(ti.f32, shape=size)
    inverse_numeric.from_numpy(
        np.array([0.5, 1.0 / 3.0, 0.25, 0.2], dtype=np.float32)
    )
    input_array = ti.ndarray(ti.f32, shape=size)
    output_array = ti.ndarray(ti.f32, shape=size)
    input_array.from_numpy(
        np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float32)
    )
    output_array.fill(0.0)

    @ti.kernel
    def apply_diagonal(
        active_size: ti.i32,
        topology: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric[index] * x[topology[index]]

    def make_operator(topology_version, numeric_version, numeric):
        primal = apply_diagonal._primal
        key = primal.ensure_compiled(
            size,
            identity,
            numeric,
            input_array,
            output_array,
        )
        return program._create_compiled_kernel_linear_operator_with_numeric_data(
            primal.compiled_kernels[key],
            size,
            topology_version,
            numeric_version,
            identity.arr,
            numeric.arr,
        )

    program = ti.lang.impl.get_runtime().prog
    target = make_operator(
        target_topology_version,
        target_numeric_version,
        target_numeric,
    )
    inverse = make_operator(
        inverse_topology_version,
        inverse_numeric_version,
        inverse_numeric,
    )
    with pytest.raises(RuntimeError, match="explicit symmetric-positive"):
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program,
            target,
            inverse,
            False,
        )
    preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program,
            target,
            inverse,
            True,
        )
    )
    preconditioner.apply(
        program,
        target,
        input_array.arr,
        output_array.arr,
    )
    ti.sync()
    np.testing.assert_allclose(
        output_array.to_numpy(),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )

    initial = preconditioner._debug_runtime_stats()
    assert initial["identity"]["method"] == (
        "compiled_kernel_inverse_apply"
    )
    assert initial["identity"]["operator_pattern_version_at_build"] == (
        target_topology_version
    )
    assert initial["identity"]["operator_numeric_version_at_build"] == (
        target_numeric_version
    )
    assert initial["identity"][
        "preconditioner_pattern_version_at_build"
    ] == inverse_topology_version
    assert initial["identity"][
        "preconditioner_numeric_version_at_build"
    ] == inverse_numeric_version
    assert not initial["identity"]["operator_stale"]
    assert not initial["identity"]["preconditioner_stale"]
    assert initial["operations"]["apply_calls"] == 1
    assert initial["resources"]["persistent_inverse_count"] == 0
    assert initial["resources"]["persistent_inverse_reserved_bytes"] == 0
    assert initial["resources"]["ownership_scope"] == (
        "external_inverse_operator"
    )
    assert not initial["contract"]["numeric_refresh_supported"]
    assert initial["contract"]["numeric_update_requires_rebuild"]
    assert not initial["contract"]["in_place_apply_supported"]

    replacement_target = ti.ndarray(ti.f32, shape=size)
    replacement_target.from_numpy(
        np.array([4.0, 5.0, 8.0, 10.0], dtype=np.float32)
    )
    target.update_numeric_data(
        program,
        replacement_target.arr,
        target_topology_version,
        target_numeric_version,
    )
    output_array.fill(17.0)
    with pytest.raises(RuntimeError, match="preconditioner is stale"):
        preconditioner.apply(
            program,
            target,
            input_array.arr,
            output_array.arr,
        )
    ti.sync()
    np.testing.assert_array_equal(
        output_array.to_numpy(),
        np.full(size, 17.0, dtype=np.float32),
    )
    stale_target = preconditioner._debug_runtime_stats()
    assert stale_target["identity"]["operator_stale"]
    assert not stale_target["identity"]["preconditioner_stale"]
    assert stale_target["operations"]["apply_calls"] == 1

    replacement_inverse = ti.ndarray(ti.f32, shape=size)
    replacement_inverse.from_numpy(
        np.array([0.25, 0.2, 0.125, 0.1], dtype=np.float32)
    )
    inverse.update_numeric_data(
        program,
        replacement_inverse.arr,
        inverse_topology_version,
        inverse_numeric_version,
    )
    stale_both = preconditioner._debug_runtime_stats()
    assert stale_both["identity"]["operator_stale"]
    assert stale_both["identity"]["preconditioner_stale"]

    rebuilt = ti._lib.core._make_compiled_kernel_preconditioner_plan(
        program,
        target,
        inverse,
        True,
    )
    output_array.fill(0.0)
    rebuilt.apply(
        program,
        target,
        input_array.arr,
        output_array.arr,
    )
    ti.sync()
    np.testing.assert_allclose(
        output_array.to_numpy(),
        np.array([0.5, 1.2, 1.5, 2.0], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan],
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_compiled_kernel_pcg_accepts_compiled_graph_inverse():
    size = 8
    max_iterations = 16
    tolerance = 1e-4
    target_topology_version = 61
    target_numeric_version = 67
    inverse_topology_version = 71
    inverse_numeric_version = 73
    identity = ti.ndarray(ti.i32, shape=size)
    identity.from_numpy(np.arange(size, dtype=np.int32))
    diagonal = ti.ndarray(ti.f32, shape=size)
    diagonal_numpy = np.array(
        [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        dtype=np.float32,
    )
    diagonal.from_numpy(diagonal_numpy)
    inverse_primary = ti.ndarray(ti.f32, shape=size)
    inverse_secondary = ti.ndarray(ti.f32, shape=size)
    inverse_primary.from_numpy(0.5 / diagonal_numpy)
    inverse_secondary.from_numpy(0.5 / diagonal_numpy)
    inverse_workspace = ti.ndarray(ti.f32, shape=size)
    inverse_workspace.fill(0.0)
    exact = ti.ndarray(ti.f32, shape=size)
    exact_numpy = np.array(
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0],
        dtype=np.float32,
    )
    exact.from_numpy(exact_numpy)
    rhs = ti.ndarray(ti.f32, shape=size)
    rhs.fill(0.0)

    @ti.kernel
    def apply_diagonal(
        active_size: ti.i32,
        topology: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric[index] * x[topology[index]]

    @ti.kernel
    def stage_inverse(
        active_size: ti.i32,
        topology: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_primary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        numeric_secondary: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        workspace: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            workspace[index] = (
                numeric_primary[index] + numeric_secondary[index]
            ) * x[topology[index]]

    @ti.kernel
    def finish_inverse(
        active_size: ti.i32,
        workspace: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = workspace[index]

    program = ti.lang.impl.get_runtime().prog
    target_primal = apply_diagonal._primal
    target_key = target_primal.ensure_compiled(
        size,
        identity,
        diagonal,
        exact,
        rhs,
    )
    target = (
        program._create_compiled_kernel_linear_operator_with_numeric_data(
            target_primal.compiled_kernels[target_key],
            size,
            target_topology_version,
            target_numeric_version,
            identity.arr,
            diagonal.arr,
        )
    )

    sym_active_size = ti.graph.Arg(
        ti.graph.ArgKind.SCALAR, "active_size", ti.i32
    )
    sym_topology = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "topology", ti.i32, ndim=1
    )
    sym_numeric_primary = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "numeric_primary", ti.f32, ndim=1
    )
    sym_numeric_secondary = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "numeric_secondary", ti.f32, ndim=1
    )
    sym_workspace = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "workspace", ti.f32, ndim=1
    )
    sym_input = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "input", ti.f32, ndim=1
    )
    sym_output = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1
    )
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(
        stage_inverse,
        sym_active_size,
        sym_topology,
        sym_numeric_primary,
        sym_numeric_secondary,
        sym_input,
        sym_workspace,
    )
    graph_builder.dispatch(
        finish_inverse,
        sym_active_size,
        sym_workspace,
        sym_output,
    )
    inverse_graph = graph_builder.compile()
    inverse = program._create_compiled_graph_linear_operator(
        inverse_graph._compiled_graph,
        size,
        inverse_topology_version,
        inverse_numeric_version,
        {"active_size": size},
        {"topology": identity.arr},
        {
            "numeric_primary": inverse_primary.arr,
            "numeric_secondary": inverse_secondary.arr,
        },
        {"workspace": inverse_workspace.arr},
    )

    target.spmv(program, exact.arr, rhs.arr)
    ti.sync()
    preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program,
            target,
            inverse,
            True,
        )
    )
    probe = ti.ndarray(ti.f32, shape=size)
    probe.fill(0.0)
    preconditioner.apply(program, target, rhs.arr, probe.arr)
    ti.sync()
    np.testing.assert_allclose(
        probe.to_numpy(), exact_numpy, rtol=0.0, atol=tolerance
    )
    preconditioner_stats = preconditioner._debug_runtime_stats()
    assert preconditioner_stats["identity"]["method"] == (
        "compiled_graph_inverse_apply"
    )
    assert preconditioner_stats["resources"]["ownership_scope"] == (
        "external_inverse_operator"
    )
    inverse_stats = inverse._debug_runtime_stats()
    assert inverse_stats["identity"]["storage_format"] == (
        "matrix_free_graph"
    )
    assert inverse_stats["provider"]["name"] == "forge_compiled_graph"
    assert inverse_stats["resources"]["spmv_workspace_reserved_bytes"] == (
        size * np.dtype(np.float32).itemsize
    )
    numeric_bytes = 2 * size * np.dtype(np.float32).itemsize
    assert inverse_stats["resources"]["values_reserved_bytes"] == (
        numeric_bytes
    )
    assert inverse_stats["resources"][
        "numeric_update_peak_temporary_bytes"
    ] == 0

    arch = ti.lang.impl.current_cfg().arch
    expected_method = (
        "pcg_compiled_kernel_bounded_masked_probe"
        if arch == ti.vulkan
        else "pcg_compiled_kernel"
    )

    def make_solver(binding):
        if arch == ti.cpu:
            return ti._lib.core._make_cpu_compiled_kernel_pcg_solver(
                program,
                target,
                binding,
                max_iterations,
                tolerance,
            )
        if arch == ti.cuda:
            return ti._lib.core._make_cuda_compiled_kernel_pcg_solver(
                program,
                target,
                binding,
                max_iterations,
                tolerance,
                False,
            )
        return (
            ti._lib.core._make_vulkan_compiled_kernel_pcg_convergence_plan(
                program,
                target,
                binding,
                max_iterations,
                tolerance,
            )
        )

    solver = make_solver(preconditioner)
    solution = ti.ndarray(ti.f32, shape=size)
    solution.fill(0.0)
    solver.solve(program, solution.arr, rhs.arr)
    ti.sync()
    assert solver.is_success()
    np.testing.assert_allclose(
        solution.to_numpy(), exact_numpy, rtol=0.0, atol=tolerance
    )
    solve_stats = solver._debug_runtime_stats()
    assert solve_stats["identity"]["method"] == expected_method
    assert solve_stats["identity"]["preconditioner_method"] == (
        "compiled_graph_inverse_apply"
    )
    assert solve_stats["resources"]["external_preconditioner"]
    initial_binding_stats = preconditioner._debug_runtime_stats()
    assert initial_binding_stats["operations"]["apply_calls"] > 1

    replacement_primary = ti.ndarray(ti.f32, shape=size)
    replacement_secondary = ti.ndarray(ti.f32, shape=size)
    replacement_primary.from_numpy(0.25 / diagonal_numpy)
    replacement_secondary.from_numpy(0.25 / diagonal_numpy)
    probe.fill(13.0)
    with pytest.raises(RuntimeError, match="complete numeric role set"):
        inverse.update_numeric_data(
            program,
            {"numeric_primary": replacement_primary.arr},
            inverse_topology_version,
            inverse_numeric_version,
        )
    rejected_stats = inverse._debug_runtime_stats()
    assert rejected_stats["identity"]["numeric_version"] == (
        inverse_numeric_version
    )
    assert rejected_stats["operations"]["numeric_updates"] == 0
    preconditioner.apply(program, target, rhs.arr, probe.arr)
    ti.sync()
    np.testing.assert_allclose(
        probe.to_numpy(), exact_numpy, rtol=0.0, atol=tolerance
    )

    inverse.update_numeric_data(
        program,
        {
            "numeric_primary": replacement_primary.arr,
            "numeric_secondary": replacement_secondary.arr,
        },
        inverse_topology_version,
        inverse_numeric_version,
    )
    replacement_primary_ref = weakref.ref(replacement_primary)
    replacement_secondary_ref = weakref.ref(replacement_secondary)
    replacement_primary = None
    replacement_secondary = None
    gc.collect()
    assert replacement_primary_ref() is None
    assert replacement_secondary_ref() is None

    probe.fill(17.0)
    with pytest.raises(RuntimeError, match="preconditioner is stale"):
        preconditioner.apply(program, target, rhs.arr, probe.arr)
    ti.sync()
    np.testing.assert_array_equal(
        probe.to_numpy(), np.full(size, 17.0, dtype=np.float32)
    )
    stale_binding_stats = preconditioner._debug_runtime_stats()
    assert stale_binding_stats["identity"]["preconditioner_stale"]

    refreshed_preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program,
            target,
            inverse,
            True,
        )
    )
    probe.fill(0.0)
    refreshed_preconditioner.apply(program, target, rhs.arr, probe.arr)
    ti.sync()
    np.testing.assert_allclose(
        probe.to_numpy(), 0.5 * exact_numpy, rtol=0.0, atol=tolerance
    )
    refreshed_stats = inverse._debug_runtime_stats()
    assert refreshed_stats["identity"]["pattern_version"] == (
        inverse_topology_version
    )
    assert refreshed_stats["identity"]["numeric_version"] == (
        inverse_numeric_version + 1
    )
    assert refreshed_stats["operations"]["numeric_updates"] == 1
    assert refreshed_stats["operations"]["numeric_update_bytes"] == (
        numeric_bytes
    )
    assert refreshed_stats["operations"]["spmv_plan_builds"] == 2
    assert refreshed_stats["resources"][
        "numeric_update_peak_temporary_bytes"
    ] == numeric_bytes
    steady_provider_bytes = 4 * size * np.dtype(np.float32).itemsize
    assert refreshed_stats["resources"]["operator_owned_reserved_bytes"] == (
        steady_provider_bytes
    )
    assert refreshed_stats["transfers"]["device_to_device_bytes"] == (
        steady_provider_bytes + numeric_bytes
    )

    refreshed_solver = make_solver(refreshed_preconditioner)
    solution.fill(0.0)
    refreshed_solver.solve(program, solution.arr, rhs.arr)
    ti.sync()
    assert refreshed_solver.is_success()
    np.testing.assert_allclose(
        solution.to_numpy(), exact_numpy, rtol=0.0, atol=tolerance
    )
    refreshed_solve_stats = refreshed_solver._debug_runtime_stats()
    assert refreshed_solve_stats["identity"]["method"] == expected_method
    assert refreshed_solve_stats["identity"][
        "preconditioner_method"
    ] == "compiled_graph_inverse_apply"
    refreshed_binding_stats = (
        refreshed_preconditioner._debug_runtime_stats()
    )
    assert refreshed_binding_stats["operations"]["apply_calls"] > 1
    assert inverse._debug_runtime_stats()["operations"]["spmv_calls"] == (
        stale_binding_stats["operations"]["apply_calls"]
        + refreshed_binding_stats["operations"]["apply_calls"]
    )


@test_utils.test(
    arch=ti.vulkan,
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_vulkan_cg_consumes_compiled_kernel_operator():
    size = 8
    max_iterations = 16
    tolerance = 1e-4
    neighbors_numpy = np.full((size, 2), -1, dtype=np.int32)
    neighbors_numpy[1:, 0] = np.arange(size - 1, dtype=np.int32)
    neighbors_numpy[:-1, 1] = np.arange(1, size, dtype=np.int32)
    operator_data = ti.ndarray(ti.i32, shape=(size, 2))
    operator_data.from_numpy(neighbors_numpy)
    exact = ti.ndarray(ti.f32, shape=size)
    exact_numpy = np.array(
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0],
        dtype=np.float32,
    )
    exact.from_numpy(exact_numpy)
    rhs = ti.ndarray(ti.f32, shape=size)
    rhs.fill(0.0)

    @ti.kernel
    def apply_tridiagonal(
        active_size: ti.i32,
        data: ti.types.ndarray(dtype=ti.i32, ndim=2),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            value = 2.0 * x[index]
            for slot in ti.static(range(2)):
                neighbor = data[index, slot]
                if neighbor >= 0:
                    value -= x[neighbor]
            y[index] = value

    primal = apply_tridiagonal._primal
    key = primal.ensure_compiled(size, operator_data, exact, rhs)
    program = ti.lang.impl.get_runtime().prog
    operator = program._create_compiled_kernel_linear_operator(
        primal.compiled_kernels[key],
        size,
        1,
        1,
        operator_data.arr,
    )
    operator.spmv(program, exact.arr, rhs.arr)
    ti.sync()

    operator_data_ref = weakref.ref(operator_data)
    operator_data = None
    gc.collect()
    assert operator_data_ref() is None
    with pytest.raises(RuntimeError, match="internal Vulkan CSR matrix"):
        ti._lib.core._make_vulkan_cg_convergence_plan(
            program,
            operator,
            max_iterations,
            tolerance,
        )

    plan = ti._lib.core._make_vulkan_compiled_kernel_cg_convergence_plan(
        program,
        operator,
        max_iterations,
        tolerance,
    )
    operator_ref = weakref.ref(operator)
    operator = None
    gc.collect()
    assert operator_ref() is not None

    solution = ti.ndarray(ti.f32, shape=size)
    solution.fill(0.0)
    plan.solve(program, solution.arr, rhs.arr)
    ti.sync()
    assert plan.is_success()
    assert plan.get_status() == 2
    assert plan.get_residual_norm() <= tolerance
    np.testing.assert_allclose(
        solution.to_numpy(), exact_numpy, rtol=0.0, atol=1e-3
    )

    stats = plan._debug_runtime_stats()
    assert stats["identity"]["method"] == (
        "cg_compiled_kernel_bounded_masked_probe"
    )
    assert stats["identity"]["preconditioner_method"] == "identity"
    assert stats["identity"]["operator_pattern_version"] == 1
    assert stats["identity"]["operator_numeric_version"] == 1
    assert stats["operations"]["solve_calls"] == 1
    assert stats["operations"]["operator_apply_calls"] == (
        max_iterations + 1
    )
    assert stats["operations"]["host_synchronizations"] == 1
    assert stats["operations"]["host_scalar_readbacks"] == 4
    assert stats["operations"]["bounded_masked_execution"]
    assert not stats["operations"]["fixed_iteration_only"]
    assert stats["resources"]["persistent_vector_count"] == 3
    assert stats["resources"]["persistent_vector_reserved_bytes"] == (
        3 * size * np.dtype(np.float32).itemsize
    )
    assert not stats["resources"]["external_preconditioner"]

    operator_stats = operator_ref()._debug_runtime_stats()
    assert operator_stats["operations"]["spmv_calls"] == (
        max_iterations + 2
    )
    assert operator_stats["operations"]["spmv_plan_builds"] == 1
    assert operator_stats["operations"]["spmv_plan_reuses"] == (
        max_iterations + 2
    )


@test_utils.test(
    arch=ti.vulkan,
    offline_cache=False,
    vulkan_sparse_experimental=True,
)
def test_private_vulkan_pcg_consumes_compiled_kernel_preconditioner():
    size = 8
    max_iterations = 16
    tolerance = 1e-4
    target_topology_version = 13
    target_numeric_version = 17
    inverse_topology_version = 19
    inverse_numeric_version = 23
    neighbors_numpy = np.full((size, 2), -1, dtype=np.int32)
    neighbors_numpy[1:, 0] = np.arange(size - 1, dtype=np.int32)
    neighbors_numpy[:-1, 1] = np.arange(1, size, dtype=np.int32)
    neighbors = ti.ndarray(ti.i32, shape=(size, 2))
    neighbors.from_numpy(neighbors_numpy)
    identity = ti.ndarray(ti.i32, shape=size)
    identity.from_numpy(np.arange(size, dtype=np.int32))
    diagonal = ti.ndarray(ti.f32, shape=size)
    diagonal.fill(3.0)
    inverse_diagonal = ti.ndarray(ti.f32, shape=size)
    inverse_diagonal.fill(1.0 / 3.0)
    exact = ti.ndarray(ti.f32, shape=size)
    exact_numpy = np.array(
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0],
        dtype=np.float32,
    )
    exact.from_numpy(exact_numpy)
    rhs = ti.ndarray(ti.f32, shape=size)
    rhs.fill(0.0)

    @ti.kernel
    def apply_variable_tridiagonal(
        active_size: ti.i32,
        topology: ti.types.ndarray(dtype=ti.i32, ndim=2),
        numeric: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            value = numeric[index] * x[index]
            for slot in ti.static(range(2)):
                neighbor = topology[index, slot]
                if neighbor >= 0:
                    value -= x[neighbor]
            y[index] = value

    @ti.kernel
    def apply_inverse_diagonal(
        active_size: ti.i32,
        topology: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric[index] * x[topology[index]]

    program = ti.lang.impl.get_runtime().prog
    target_primal = apply_variable_tridiagonal._primal
    target_key = target_primal.ensure_compiled(
        size,
        neighbors,
        diagonal,
        exact,
        rhs,
    )
    target = (
        program._create_compiled_kernel_linear_operator_with_numeric_data(
            target_primal.compiled_kernels[target_key],
            size,
            target_topology_version,
            target_numeric_version,
            neighbors.arr,
            diagonal.arr,
        )
    )
    inverse_primal = apply_inverse_diagonal._primal
    inverse_key = inverse_primal.ensure_compiled(
        size,
        identity,
        inverse_diagonal,
        exact,
        rhs,
    )
    inverse = (
        program._create_compiled_kernel_linear_operator_with_numeric_data(
            inverse_primal.compiled_kernels[inverse_key],
            size,
            inverse_topology_version,
            inverse_numeric_version,
            identity.arr,
            inverse_diagonal.arr,
        )
    )
    target.spmv(program, exact.arr, rhs.arr)
    ti.sync()
    preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program,
            target,
            inverse,
            True,
        )
    )
    plan = (
        ti._lib.core._make_vulkan_compiled_kernel_pcg_convergence_plan(
            program,
            target,
            preconditioner,
            max_iterations,
            tolerance,
        )
    )
    solution = ti.ndarray(ti.f32, shape=size)
    solution.fill(0.0)
    plan.solve(program, solution.arr, rhs.arr)
    ti.sync()
    assert plan.is_success()
    assert plan.get_status() == 2
    assert plan.get_residual_norm() <= tolerance
    np.testing.assert_allclose(
        solution.to_numpy(), exact_numpy, rtol=0.0, atol=1e-3
    )

    solve_stats = plan._debug_runtime_stats()
    assert solve_stats["identity"]["method"] == (
        "pcg_compiled_kernel_bounded_masked_probe"
    )
    assert solve_stats["identity"]["preconditioner_method"] == (
        "compiled_kernel_inverse_apply"
    )
    assert solve_stats["operations"]["operator_apply_calls"] == (
        max_iterations + 1
    )
    assert solve_stats["operations"]["preconditioner_apply_calls"] == (
        max_iterations + 1
    )
    assert solve_stats["resources"]["persistent_vector_count"] == 4
    assert solve_stats["resources"]["external_preconditioner"]
    preconditioner_stats = preconditioner._debug_runtime_stats()
    assert preconditioner_stats["operations"]["apply_calls"] == (
        max_iterations + 1
    )
    assert target._debug_runtime_stats()["operations"]["spmv_calls"] == (
        max_iterations + 2
    )
    assert inverse._debug_runtime_stats()["operations"]["spmv_calls"] == (
        max_iterations + 1
    )

    replacement_diagonal = ti.ndarray(ti.f32, shape=size)
    replacement_diagonal.fill(4.0)
    replacement_inverse = ti.ndarray(ti.f32, shape=size)
    replacement_inverse.fill(0.25)
    target.update_numeric_data(
        program,
        replacement_diagonal.arr,
        target_topology_version,
        target_numeric_version,
    )
    inverse.update_numeric_data(
        program,
        replacement_inverse.arr,
        inverse_topology_version,
        inverse_numeric_version,
    )
    solution.fill(7.0)
    with pytest.raises(RuntimeError, match="preconditioner is stale"):
        plan.solve(program, solution.arr, rhs.arr)
    ti.sync()
    np.testing.assert_array_equal(
        solution.to_numpy(), np.full(size, 7.0, dtype=np.float32)
    )

    target.spmv(program, exact.arr, rhs.arr)
    rebuilt_preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program,
            target,
            inverse,
            True,
        )
    )
    rebuilt_plan = (
        ti._lib.core._make_vulkan_compiled_kernel_pcg_convergence_plan(
            program,
            target,
            rebuilt_preconditioner,
            max_iterations,
            tolerance,
        )
    )
    solution.fill(0.0)
    rebuilt_plan.solve(program, solution.arr, rhs.arr)
    ti.sync()
    assert rebuilt_plan.is_success()
    np.testing.assert_allclose(
        solution.to_numpy(), exact_numpy, rtol=0.0, atol=1e-3
    )


@test_utils.test(
    arch=ti.cuda,
    offline_cache=False,
)
def test_private_cuda_cg_reuses_registered_compiled_kernel_workspace():
    size = 8
    max_iterations = 16
    tolerance = 1e-5
    target_topology_version = 29
    target_numeric_version = 31
    inverse_topology_version = 37
    inverse_numeric_version = 41
    identity = ti.ndarray(ti.i32, shape=size)
    identity.from_numpy(np.arange(size, dtype=np.int32))
    diagonal = ti.ndarray(ti.f32, shape=size)
    diagonal.from_numpy(
        np.array(
            [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            dtype=np.float32,
        )
    )
    inverse_diagonal = ti.ndarray(ti.f32, shape=size)
    inverse_diagonal.from_numpy(1.0 / diagonal.to_numpy())
    exact = ti.ndarray(ti.f32, shape=size)
    exact_numpy = np.array(
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0],
        dtype=np.float32,
    )
    exact.from_numpy(exact_numpy)
    rhs = ti.ndarray(ti.f32, shape=size)
    rhs.fill(0.0)

    @ti.kernel
    def apply_diagonal(
        active_size: ti.i32,
        topology: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric[index] * x[topology[index]]

    def make_operator(topology_version, numeric_version, numeric):
        primal = apply_diagonal._primal
        key = primal.ensure_compiled(
            size,
            identity,
            numeric,
            exact,
            rhs,
        )
        return program._create_compiled_kernel_linear_operator_with_numeric_data(
            primal.compiled_kernels[key],
            size,
            topology_version,
            numeric_version,
            identity.arr,
            numeric.arr,
        )

    program = ti.lang.impl.get_runtime().prog
    target = make_operator(
        target_topology_version,
        target_numeric_version,
        diagonal,
    )
    inverse = make_operator(
        inverse_topology_version,
        inverse_numeric_version,
        inverse_diagonal,
    )
    target.spmv(program, exact.arr, rhs.arr)
    ti.sync()

    identity_solver = ti._lib.core._make_cuda_compiled_kernel_cg_solver(
        program,
        target,
        max_iterations,
        tolerance,
        False,
    )
    identity_solution = ti.ndarray(ti.f32, shape=size)
    identity_solution.fill(0.0)
    identity_solver.solve(program, identity_solution.arr, rhs.arr)
    ti.sync()
    assert identity_solver.is_success()
    np.testing.assert_allclose(
        identity_solution.to_numpy(), exact_numpy, rtol=0.0, atol=1e-4
    )
    identity_stats = identity_solver._debug_runtime_stats()
    assert identity_stats["identity"]["method"] == "cg_compiled_kernel"
    assert identity_stats["identity"]["preconditioner_method"] == (
        "identity"
    )
    assert identity_stats["resources"]["persistent_vector_count"] == 3
    assert identity_stats["resources"]["cublas_handle_count"] == 1
    assert identity_stats["operations"]["host_scalar_reductions"] > 0

    preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program,
            target,
            inverse,
            True,
        )
    )
    pcg = ti._lib.core._make_cuda_compiled_kernel_pcg_solver(
        program,
        target,
        preconditioner,
        max_iterations,
        tolerance,
        False,
    )
    solution = ti.ndarray(ti.f32, shape=size)
    for solve_index in range(2):
        solution.fill(0.0)
        pcg.solve(program, solution.arr, rhs.arr)
        ti.sync()
        assert pcg.is_success()
        np.testing.assert_allclose(
            solution.to_numpy(), exact_numpy, rtol=0.0, atol=1e-5
        )
    pcg_stats = pcg._debug_runtime_stats()
    assert pcg_stats["identity"]["method"] == "pcg_compiled_kernel"
    assert pcg_stats["identity"]["preconditioner_method"] == (
        "compiled_kernel_inverse_apply"
    )
    assert pcg_stats["operations"]["solve_calls"] == 2
    assert pcg_stats["operations"]["workspace_builds"] == 1
    assert pcg_stats["operations"]["workspace_reuses"] == 1
    assert pcg_stats["resources"]["persistent_vector_count"] == 4
    assert pcg_stats["resources"]["external_preconditioner"]
    assert preconditioner._debug_runtime_stats()["operations"][
        "apply_calls"
    ] == 2

    replacement_diagonal = ti.ndarray(ti.f32, shape=size)
    replacement_diagonal.from_numpy(
        np.array(
            [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
            dtype=np.float32,
        )
    )
    replacement_inverse = ti.ndarray(ti.f32, shape=size)
    replacement_inverse.from_numpy(1.0 / replacement_diagonal.to_numpy())
    target.update_numeric_data(
        program,
        replacement_diagonal.arr,
        target_topology_version,
        target_numeric_version,
    )
    inverse.update_numeric_data(
        program,
        replacement_inverse.arr,
        inverse_topology_version,
        inverse_numeric_version,
    )
    solution.fill(7.0)
    with pytest.raises(RuntimeError, match="preconditioner is stale"):
        pcg.solve(program, solution.arr, rhs.arr)
    ti.sync()
    np.testing.assert_array_equal(
        solution.to_numpy(), np.full(size, 7.0, dtype=np.float32)
    )

    target.spmv(program, exact.arr, rhs.arr)
    rebuilt_preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program,
            target,
            inverse,
            True,
        )
    )
    rebuilt_pcg = ti._lib.core._make_cuda_compiled_kernel_pcg_solver(
        program,
        target,
        rebuilt_preconditioner,
        max_iterations,
        tolerance,
        False,
    )
    solution.fill(0.0)
    rebuilt_pcg.solve(program, solution.arr, rhs.arr)
    ti.sync()
    assert rebuilt_pcg.is_success()
    np.testing.assert_allclose(
        solution.to_numpy(), exact_numpy, rtol=0.0, atol=1e-5
    )


@test_utils.test(
    arch=ti.cpu,
    offline_cache=False,
)
def test_private_cpu_pcg_reuses_registered_compiled_kernel_workspace():
    size = 8
    max_iterations = 16
    tolerance = 1e-6
    target_topology_version = 43
    target_numeric_version = 47
    inverse_topology_version = 53
    inverse_numeric_version = 59
    identity = ti.ndarray(ti.i32, shape=size)
    identity.from_numpy(np.arange(size, dtype=np.int32))
    diagonal = ti.ndarray(ti.f32, shape=size)
    diagonal_numpy = np.array(
        [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        dtype=np.float32,
    )
    diagonal.from_numpy(diagonal_numpy)
    inverse_diagonal = ti.ndarray(ti.f32, shape=size)
    inverse_diagonal.from_numpy(1.0 / diagonal_numpy)
    exact = ti.ndarray(ti.f32, shape=size)
    exact_numpy = np.array(
        [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0],
        dtype=np.float32,
    )
    exact.from_numpy(exact_numpy)
    rhs = ti.ndarray(ti.f32, shape=size)
    rhs.fill(0.0)

    @ti.kernel
    def apply_diagonal(
        active_size: ti.i32,
        topology: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric[index] * x[topology[index]]

    def make_operator(topology_version, numeric_version, numeric):
        primal = apply_diagonal._primal
        key = primal.ensure_compiled(
            size,
            identity,
            numeric,
            exact,
            rhs,
        )
        return program._create_compiled_kernel_linear_operator_with_numeric_data(
            primal.compiled_kernels[key],
            size,
            topology_version,
            numeric_version,
            identity.arr,
            numeric.arr,
        )

    program = ti.lang.impl.get_runtime().prog
    target = make_operator(
        target_topology_version,
        target_numeric_version,
        diagonal,
    )
    inverse = make_operator(
        inverse_topology_version,
        inverse_numeric_version,
        inverse_diagonal,
    )
    target.spmv(program, exact.arr, rhs.arr)
    preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program,
            target,
            inverse,
            True,
        )
    )
    pcg = ti._lib.core._make_cpu_compiled_kernel_pcg_solver(
        program,
        target,
        preconditioner,
        max_iterations,
        tolerance,
    )
    solution = ti.ndarray(ti.f32, shape=size)
    for solve_index in range(2):
        solution.fill(0.0)
        pcg.solve(program, solution.arr, rhs.arr)
        assert pcg.is_success()
        np.testing.assert_allclose(
            solution.to_numpy(), exact_numpy, rtol=0.0, atol=1e-6
        )
    stats = pcg._debug_runtime_stats()
    assert stats["identity"]["method"] == "pcg_compiled_kernel"
    assert stats["identity"]["preconditioner_method"] == (
        "compiled_kernel_inverse_apply"
    )
    assert stats["operations"]["solve_calls"] == 2
    assert stats["operations"]["workspace_builds"] == 1
    assert stats["operations"]["workspace_reuses"] == 1
    assert stats["operations"]["host_scalar_reductions"] > 0
    assert stats["resources"]["persistent_vector_count"] == 4
    assert stats["resources"]["persistent_vector_reserved_bytes"] == (
        4 * size * np.dtype(np.float32).itemsize
    )
    assert stats["resources"]["external_preconditioner"]
    assert preconditioner._debug_runtime_stats()["operations"][
        "apply_calls"
    ] == 2

    replacement_diagonal = ti.ndarray(ti.f32, shape=size)
    replacement_diagonal_numpy = np.array(
        [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],
        dtype=np.float32,
    )
    replacement_diagonal.from_numpy(replacement_diagonal_numpy)
    replacement_inverse = ti.ndarray(ti.f32, shape=size)
    replacement_inverse.from_numpy(1.0 / replacement_diagonal_numpy)
    target.update_numeric_data(
        program,
        replacement_diagonal.arr,
        target_topology_version,
        target_numeric_version,
    )
    inverse.update_numeric_data(
        program,
        replacement_inverse.arr,
        inverse_topology_version,
        inverse_numeric_version,
    )
    solution.fill(7.0)
    with pytest.raises(RuntimeError, match="preconditioner is stale"):
        pcg.solve(program, solution.arr, rhs.arr)
    np.testing.assert_array_equal(
        solution.to_numpy(), np.full(size, 7.0, dtype=np.float32)
    )

    target.spmv(program, exact.arr, rhs.arr)
    rebuilt_preconditioner = (
        ti._lib.core._make_compiled_kernel_preconditioner_plan(
            program,
            target,
            inverse,
            True,
        )
    )
    rebuilt_pcg = ti._lib.core._make_cpu_compiled_kernel_pcg_solver(
        program,
        target,
        rebuilt_preconditioner,
        max_iterations,
        tolerance,
    )
    solution.fill(0.0)
    rebuilt_pcg.solve(program, solution.arr, rhs.arr)
    assert rebuilt_pcg.is_success()
    np.testing.assert_allclose(
        solution.to_numpy(), exact_numpy, rtol=0.0, atol=1e-6
    )
