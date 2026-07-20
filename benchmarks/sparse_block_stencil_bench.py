"""Canonical block-sparse 2D Poisson/stencil diagnostic.

This workload is intentionally not tied to MPM. A moving rectangular set of
active bricks stores the vectors of a matrix-free five-point Poisson operator
and runs repeated weighted-Jacobi updates. A dense coordinate reference checks
the same active cells, while sparse lifecycle telemetry attributes listgen,
workspace, memory, migration, and teardown work.

Timings are diagnostic unless the performance flag is requested. GPU
performance runs use the shared idle-admission guard before Taichi creates a
context.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_idle_guard import (  # noqa: E402
    finalize_performance_measurement, prepare_performance_measurement,
)
from sparse_snode_lifecycle_bench import (  # noqa: E402
    _arch_name, _numeric_delta, _phase_sample, _runtime_snapshot, _summary_ns,
)

SCHEMA = "taichi_forge.sparse_block_stencil.v1"
PHASES = (
    "create",
    "materialize",
    "compile_workload",
    "activate_initial",
    "cold_operator_apply",
    "warm_operator_apply",
    "dense_reference_apply",
    "sparse_solver_steps",
    "dense_solver_steps",
    "deactivate_migrate",
    "rebuild_operator_apply",
    "dense_migrated_apply",
    "destroy",
)


def _validate_config(
    *,
    root_blocks,
    block_size,
    active_blocks_per_axis,
    margin_blocks,
    migration_blocks,
    solver_iterations,
):
    positive = {
        "root_blocks": root_blocks,
        "block_size": block_size,
        "active_blocks_per_axis": active_blocks_per_axis,
        "margin_blocks": margin_blocks,
        "migration_blocks": migration_blocks,
        "solver_iterations": solver_iterations,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    migrated_end = (margin_blocks + migration_blocks + active_blocks_per_axis)
    if migrated_end > root_blocks:
        raise ValueError(
            "the migrated active-brick window must fit inside root_blocks")


def run_initialized(
    ti,
    *,
    root_blocks: int = 48,
    block_size: int = 8,
    active_blocks_per_axis: int = 32,
    margin_blocks: int = 4,
    migration_blocks: int = 2,
    solver_iterations: int = 4,
) -> dict:
    """Run the block-sparse Poisson workload in an initialized Program."""
    _validate_config(
        root_blocks=root_blocks,
        block_size=block_size,
        active_blocks_per_axis=active_blocks_per_axis,
        margin_blocks=margin_blocks,
        migration_blocks=migration_blocks,
        solver_iterations=solver_iterations,
    )
    domain_size = root_blocks * block_size
    active_blocks = active_blocks_per_axis**2
    active_cells = active_blocks * block_size**2
    initial_origin = margin_blocks
    migrated_origin = margin_blocks + migration_blocks

    # This is a coordinate-level correctness oracle only. Its kernels visit the
    # same active window, not the full dense domain.
    dense_x = ti.field(dtype=ti.f32, shape=(domain_size, domain_size))
    dense_rhs = ti.field(dtype=ti.f32, shape=(domain_size, domain_size))
    dense_ax = ti.field(dtype=ti.f32, shape=(domain_size, domain_size))
    dense_next = ti.field(dtype=ti.f32, shape=(domain_size, domain_size))
    ti.lang.impl.get_runtime().materialize()
    ti.sync()

    prog = ti.lang.impl.get_runtime().prog
    prog._debug_reset_sparse_listgen_stats()
    baseline = _runtime_snapshot(ti)
    phase_samples = {phase: [] for phase in PHASES}
    phases = {}
    holder = {}

    def measure(name, callback):
        sample = _phase_sample(ti, callback)
        phases[name] = sample
        phase_samples[name].append(sample)

    def create():
        x = ti.field(dtype=ti.f32)
        rhs = ti.field(dtype=ti.f32)
        ax = ti.field(dtype=ti.f32)
        x_next = ti.field(dtype=ti.f32)
        builder = ti.FieldsBuilder()
        pointer_kwargs = ({
            "vk_max_active": active_blocks
        } if _arch_name(ti) in ("cuda", "vulkan") else {})
        pointer = builder.pointer(ti.ij, (root_blocks, root_blocks),
                                  **pointer_kwargs)
        pointer.dense(ti.ij,
                      (block_size, block_size)).place(x, rhs, ax, x_next)
        holder.update(
            x=x,
            rhs=rhs,
            ax=ax,
            x_next=x_next,
            pointer=pointer,
            tree=builder.finalize(),
        )

    measure("create", create)
    x = holder["x"]
    rhs = holder["rhs"]
    ax = holder["ax"]
    x_next = holder["x_next"]
    pointer = holder["pointer"]
    tree = holder["tree"]
    tree_id = str(int(tree.id))
    tree_identity = [int(tree.id), int(tree.generation)]

    @ti.func
    def sparse_neighbor_sum(i, j):
        value = 0.0
        if i > 0:
            value += x[i - 1, j]
        if i + 1 < domain_size:
            value += x[i + 1, j]
        if j > 0:
            value += x[i, j - 1]
        if j + 1 < domain_size:
            value += x[i, j + 1]
        return value

    @ti.func
    def dense_neighbor_sum(i, j):
        value = 0.0
        if i > 0:
            value += dense_x[i - 1, j]
        if i + 1 < domain_size:
            value += dense_x[i + 1, j]
        if j > 0:
            value += dense_x[i, j - 1]
        if j + 1 < domain_size:
            value += dense_x[i, j + 1]
        return value

    @ti.kernel
    def reset_dense():
        for i, j in dense_x:
            dense_x[i, j] = 0.0
            dense_rhs[i, j] = 0.0
            dense_ax[i, j] = 0.0
            dense_next[i, j] = 0.0

    @ti.kernel
    def initialize_sparse(origin_blocks: ti.i32, epoch: ti.i32,
                          enabled: ti.i32):
        for bi, bj, li, lj in ti.ndrange(
                active_blocks_per_axis,
                active_blocks_per_axis,
                block_size,
                block_size,
        ):
            if enabled != 0:
                i = (origin_blocks + bi) * block_size + li
                j = (origin_blocks + bj) * block_size + lj
                x[i, j] = ti.cast(1 + (i * 7 + j * 11 + epoch * 13) % 17,
                                  ti.f32)
                rhs[i, j] = ti.cast(1 + (i * 5 + j * 3 + epoch * 7) % 11,
                                    ti.f32)

    @ti.kernel
    def initialize_dense(origin_blocks: ti.i32, epoch: ti.i32,
                         enabled: ti.i32):
        for bi, bj, li, lj in ti.ndrange(
                active_blocks_per_axis,
                active_blocks_per_axis,
                block_size,
                block_size,
        ):
            if enabled != 0:
                i = (origin_blocks + bi) * block_size + li
                j = (origin_blocks + bj) * block_size + lj
                dense_x[i, j] = ti.cast(1 + (i * 7 + j * 11 + epoch * 13) % 17,
                                        ti.f32)
                dense_rhs[i, j] = ti.cast(1 + (i * 5 + j * 3 + epoch * 7) % 11,
                                          ti.f32)

    @ti.kernel
    def apply_sparse_operator():
        for i, j in x:
            ax[i, j] = 4.0 * x[i, j] - sparse_neighbor_sum(i, j)

    @ti.kernel
    def apply_dense_reference(origin_blocks: ti.i32):
        for bi, bj, li, lj in ti.ndrange(
                active_blocks_per_axis,
                active_blocks_per_axis,
                block_size,
                block_size,
        ):
            i = (origin_blocks + bi) * block_size + li
            j = (origin_blocks + bj) * block_size + lj
            dense_ax[i, j] = (4.0 * dense_x[i, j] - dense_neighbor_sum(i, j))

    @ti.kernel
    def relax_sparse():
        for i, j in x:
            # Weighted Jacobi for A x = rhs, omega=0.5 and diag(A)=4.
            x_next[i, j] = x[i, j] + 0.125 * (rhs[i, j] - ax[i, j])

    @ti.kernel
    def relax_dense(origin_blocks: ti.i32):
        for bi, bj, li, lj in ti.ndrange(
                active_blocks_per_axis,
                active_blocks_per_axis,
                block_size,
                block_size,
        ):
            i = (origin_blocks + bi) * block_size + li
            j = (origin_blocks + bj) * block_size + lj
            dense_next[
                i,
                j] = dense_x[i, j] + 0.125 * (dense_rhs[i, j] - dense_ax[i, j])

    @ti.kernel
    def commit_sparse():
        for i, j in x:
            x[i, j] = x_next[i, j]

    @ti.kernel
    def commit_dense(origin_blocks: ti.i32):
        for bi, bj, li, lj in ti.ndrange(
                active_blocks_per_axis,
                active_blocks_per_axis,
                block_size,
                block_size,
        ):
            i = (origin_blocks + bi) * block_size + li
            j = (origin_blocks + bj) * block_size + lj
            dense_x[i, j] = dense_next[i, j]

    @ti.kernel
    def operator_difference_l1(origin_blocks: ti.i32) -> ti.f32:
        difference = 0.0
        for bi, bj, li, lj in ti.ndrange(
                active_blocks_per_axis,
                active_blocks_per_axis,
                block_size,
                block_size,
        ):
            i = (origin_blocks + bi) * block_size + li
            j = (origin_blocks + bj) * block_size + lj
            difference += ti.abs(ax[i, j] - dense_ax[i, j])
        return difference

    @ti.kernel
    def state_difference_l1(origin_blocks: ti.i32) -> ti.f32:
        difference = 0.0
        for bi, bj, li, lj in ti.ndrange(
                active_blocks_per_axis,
                active_blocks_per_axis,
                block_size,
                block_size,
        ):
            i = (origin_blocks + bi) * block_size + li
            j = (origin_blocks + bj) * block_size + lj
            difference += ti.abs(x[i, j] - dense_x[i, j])
        return difference

    @ti.kernel
    def stale_old_state_l1(old_origin_blocks: ti.i32,
                           new_origin_blocks: ti.i32) -> ti.f32:
        stale = 0.0
        new_begin = new_origin_blocks * block_size
        new_end = (new_origin_blocks + active_blocks_per_axis) * block_size
        for bi, bj, li, lj in ti.ndrange(
                active_blocks_per_axis,
                active_blocks_per_axis,
                block_size,
                block_size,
        ):
            i = (old_origin_blocks + bi) * block_size + li
            j = (old_origin_blocks + bj) * block_size + lj
            if not (i >= new_begin and i < new_end and j >= new_begin
                    and j < new_end):
                stale += ti.abs(x[i, j])
        return stale

    @ti.kernel
    def count_active_blocks() -> ti.i32:
        count = 0
        for bi, bj in ti.ndrange(root_blocks, root_blocks):
            if ti.is_active(pointer, [bi, bj]):
                count += 1
        return count

    measure("materialize", ti.lang.impl.get_runtime().materialize)

    def compile_workload():
        initialize_sparse(initial_origin, 0, 0)
        reset_dense()
        initialize_dense(initial_origin, 0, 0)
        apply_sparse_operator()
        apply_dense_reference(initial_origin)
        relax_sparse()
        relax_dense(initial_origin)
        commit_sparse()
        commit_dense(initial_origin)
        count_active_blocks()

    measure("compile_workload", compile_workload)

    def activate_initial():
        reset_dense()
        initialize_sparse(initial_origin, 0, 1)
        initialize_dense(initial_origin, 0, 1)

    measure("activate_initial", activate_initial)
    initial_active_blocks = int(count_active_blocks())
    measure("cold_operator_apply", apply_sparse_operator)
    measure("warm_operator_apply", apply_sparse_operator)
    measure(
        "dense_reference_apply",
        lambda: apply_dense_reference(initial_origin),
    )
    initial_operator_difference = float(operator_difference_l1(initial_origin))

    def sparse_solver_steps():
        for _ in range(solver_iterations):
            apply_sparse_operator()
            relax_sparse()
            commit_sparse()

    def dense_solver_steps():
        for _ in range(solver_iterations):
            apply_dense_reference(initial_origin)
            relax_dense(initial_origin)
            commit_dense(initial_origin)

    measure("sparse_solver_steps", sparse_solver_steps)
    measure("dense_solver_steps", dense_solver_steps)
    solver_state_difference = float(state_difference_l1(initial_origin))

    def deactivate_migrate():
        pointer.deactivate_all()
        reset_dense()
        initialize_sparse(migrated_origin, 1, 1)
        initialize_dense(migrated_origin, 1, 1)

    measure("deactivate_migrate", deactivate_migrate)
    migrated_active_blocks = int(count_active_blocks())
    stale_state = float(stale_old_state_l1(initial_origin, migrated_origin))
    measure("rebuild_operator_apply", apply_sparse_operator)
    measure(
        "dense_migrated_apply",
        lambda: apply_dense_reference(migrated_origin),
    )
    migrated_operator_difference = float(
        operator_difference_l1(migrated_origin))
    pre_destroy = _runtime_snapshot(ti)
    measure("destroy", tree.destroy)
    final = _runtime_snapshot(ti)

    checks = {
        "expected_active_blocks": active_blocks,
        "initial_active_blocks": initial_active_blocks,
        "migrated_active_blocks": migrated_active_blocks,
        "initial_operator_difference_l1": initial_operator_difference,
        "solver_state_difference_l1": solver_state_difference,
        "stale_old_state_l1": stale_state,
        "migrated_operator_difference_l1": (migrated_operator_difference),
    }
    tolerance = 1e-5
    correct = (initial_active_blocks == active_blocks
               and migrated_active_blocks == active_blocks
               and initial_operator_difference <= tolerance
               and solver_state_difference <= tolerance
               and stale_state <= tolerance
               and migrated_operator_difference <= tolerance
               and tree_id not in final["lifecycle"]["snode_trees"])
    if not correct:
        raise RuntimeError(f"sparse block stencil mismatch: {checks}")

    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "arch": _arch_name(ti),
        "correct": True,
        "config": {
            "dimensions": 2,
            "layout": "pointer_dense_brick",
            "operator": "matrix_free_dirichlet_five_point_poisson",
            "iteration": "weighted_jacobi_omega_0.5",
            "root_blocks_per_axis": root_blocks,
            "block_size_per_axis": block_size,
            "domain_cells_per_axis": domain_size,
            "active_blocks_per_axis": active_blocks_per_axis,
            "active_blocks": active_blocks,
            "active_cells": active_cells,
            "initial_origin_blocks": initial_origin,
            "migrated_origin_blocks": migrated_origin,
            "solver_iterations": solver_iterations,
        },
        "telemetry_contract": {
            "runtime_statistics_schema_version":
            (baseline["runtime_statistics_schema_version"]),
            "timing_scope":
            "callback_plus_backend_sync",
            "dense_reference_scope":
            ("same active coordinates for correctness only; not a "
             "full-domain dense performance baseline"),
            "topology_contract":
            ("one active pointer cell per brick; dense brick cells share "
             "the pointer activity and inactive neighbor reads are zero"),
            "migration_contract":
            ("deactivate all bricks, activate a shifted window, verify "
             "old-only coordinates read zero, then rebuild traversal"),
            "listgen_counter_scope":
            "cumulative_since_debug_reset",
            "performance_policy":
            ("no occupancy or brick-size crossover inference unless the "
             "CLI performance flag is explicitly requested"),
        },
        "tree_identity": tree_identity,
        "phase_order": list(PHASES),
        "phase_summary": {
            phase:
            _summary_ns(
                [sample["duration_ns"] for sample in phase_samples[phase]])
            for phase in PHASES
        },
        "phases": phases,
        "checks": checks,
        "tree_before_destroy":
        pre_destroy["lifecycle"]["snode_trees"][tree_id],
        "lifecycle": {
            "baseline":
            baseline,
            "final":
            final,
            "tree_recovered_after_destroy":
            (tree_id not in final["lifecycle"]["snode_trees"]),
            "program_memory_delta_after_destroy":
            _numeric_delta(baseline["memory"], final["memory"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",
                        choices=("cpu", "cuda", "vulkan"),
                        default="cpu")
    parser.add_argument("--root-blocks", type=int, default=48)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--active-blocks-per-axis", type=int, default=32)
    parser.add_argument("--margin-blocks", type=int, default=4)
    parser.add_argument("--migration-blocks", type=int, default=2)
    parser.add_argument("--solver-iterations", type=int, default=4)
    parser.add_argument(
        "--performance",
        action="store_true",
        help=("Mark timings as performance data. GPU runs then require the "
              "automatic idle admission check."),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    measurement = prepare_performance_measurement(args.arch,
                                                  requested=args.performance)

    import taichi_forge as ti

    arch = {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[args.arch]
    init_started = time.perf_counter_ns()
    ti.init(
        arch=arch,
        enable_fallback=False,
        offline_cache=False,
        vulkan_sparse_experimental=True,
        cuda_sparse_pool_auto_size=True,
        cuda_sparse_per_snode_pool=True,
    )
    init_duration_ns = time.perf_counter_ns() - init_started
    try:
        result = run_initialized(
            ti,
            root_blocks=args.root_blocks,
            block_size=args.block_size,
            active_blocks_per_axis=args.active_blocks_per_axis,
            margin_blocks=args.margin_blocks,
            migration_blocks=args.migration_blocks,
            solver_iterations=args.solver_iterations,
        )
        result["init_duration_ns"] = init_duration_ns
        result.update(
            finalize_performance_measurement(measurement,
                                             correct=result["correct"]))
    finally:
        ti.reset()
    result["program_reset_completed"] = True

    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
