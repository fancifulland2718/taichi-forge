"""Forge-only LinearOperator, SolvePlan, and compiled-Graph qualification.

This reports absolute Forge API completion times.  It cannot establish a
Forge/vanilla or Forge/Warp speedup because those runtimes expose no matching
public LinearOperator/SolvePlan capability.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

try:
    from .runtime_common import (
        git_metadata, host_metadata, process_gpu_memory_mib,
        runtime_device_identity, runtime_memory_observation, summarize_samples,
        working_set_bytes, write_csv, write_json,
    )
    from .single_kernel_microbench import (
        QUALIFICATION_MAX_CV_PERCENT, _ExclusiveBenchmarkLock,
        _apply_affinity, _arch_name, _enhanced_memory_plateau,
        _environment_isolated, _environment_provenance, _load_taichi,
        _noise_observation, _resolve_affinity, select_common_batch,
    )
    from .warp_transform_baseline import (
        _windows_process_ancestors, qualification_policy_errors,
    )
except ImportError:
    from runtime_common import (
        git_metadata, host_metadata, process_gpu_memory_mib,
        runtime_device_identity, runtime_memory_observation, summarize_samples,
        working_set_bytes, write_csv, write_json,
    )
    from single_kernel_microbench import (
        QUALIFICATION_MAX_CV_PERCENT, _ExclusiveBenchmarkLock,
        _apply_affinity, _arch_name, _enhanced_memory_plateau,
        _environment_isolated, _environment_provenance, _load_taichi,
        _noise_observation, _resolve_affinity, select_common_batch,
    )
    from warp_transform_baseline import (
        _windows_process_ancestors, qualification_policy_errors,
    )


SCHEMA = "taichi_forge.linear_operator_solve_plan_qualification.v1"
PRESETS = {"small": 65_536, "medium": 262_144}
MODES = ("eager_device_convergent", "graph_device_convergent")


def balanced_mode_orders(sample_count: int) -> list[tuple[str, str]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return [MODES if index % 2 == 0 else tuple(reversed(MODES))
            for index in range(sample_count)]


def _result_snapshot(result: Any) -> dict[str, Any]:
    return {
        "converged": bool(result.converged),
        "termination_reason": str(result.termination_reason),
        "iterations": int(result.iterations),
        "initial_residual_norm": float(result.initial_residual_norm),
        "residual_norm": float(result.residual_norm),
        "effective_tolerance": float(result.effective_tolerance),
    }


def _timed_batch(ti: Any, launch: Callable[[], Any],
                 batch_size: int) -> tuple[float, Any]:
    ti.sync()
    started = time.perf_counter_ns()
    terminal = None
    for _ in range(batch_size):
        terminal = launch()
    ti.sync()
    return (time.perf_counter_ns() - started) / 1.0e6, terminal


def _calibrate(ti: Any, launch: Callable[[], Any], target_ms: float,
               maximum: int = 4_096) -> tuple[int, list[dict[str, Any]]]:
    batch_size = 1
    attempts = []
    while True:
        elapsed_ms, terminal = _timed_batch(ti, launch, batch_size)
        snapshot = _result_snapshot(terminal)
        attempts.append({"batch_size": batch_size, "elapsed_ms": elapsed_ms,
                         "terminal": snapshot})
        if not snapshot["converged"]:
            raise RuntimeError("solve did not converge during calibration")
        if elapsed_ms >= target_ms or batch_size >= maximum:
            return batch_size, attempts
        estimate = (batch_size * 2 if elapsed_ms <= 0.0 else
                    math.ceil(batch_size * target_ms / elapsed_ms))
        batch_size = min(maximum, max(batch_size * 2, estimate))


def _stability(ti: Any, launch: Callable[[], Any], replays: int,
               checkpoint: int) -> dict[str, Any]:
    enhanced_before = runtime_memory_observation(ti)
    rss_before = working_set_bytes()
    gpu_before = process_gpu_memory_mib(os.getpid())
    windows = []
    completed = 0
    terminal = None
    while completed < replays:
        count = min(checkpoint, replays - completed)
        elapsed, terminal = _timed_batch(ti, launch, count)
        windows.append(elapsed / count)
        completed += count
    enhanced_after = runtime_memory_observation(ti)
    rss_after = working_set_bytes()
    gpu_after = process_gpu_memory_mib(os.getpid())
    enhanced = _enhanced_memory_plateau(enhanced_before, enhanced_after)
    rss_delta = None if rss_before is None or rss_after is None else (
        rss_after - rss_before)
    gpu_delta = None if gpu_before is None or gpu_after is None else (
        gpu_after - gpu_before)
    snapshot = _result_snapshot(terminal)
    return {
        "replays": replays, "checkpoint": checkpoint,
        "window_per_solve_ms": windows,
        "window_summary": summarize_samples(windows), "terminal": snapshot,
        "rss_before_bytes": rss_before, "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_delta, "gpu_before_mib": gpu_before,
        "gpu_after_mib": gpu_after, "gpu_delta_mib": gpu_delta,
        "enhanced_before": enhanced_before, "enhanced_after": enhanced_after,
        "enhanced_plateau": enhanced,
        "memory_guard_passed": bool(
            snapshot["converged"]
            and (rss_delta is None or rss_delta <= 64 * 1024 * 1024)
            and (gpu_delta is None or gpu_delta <= 64.0)
            and enhanced["passed"]),
    }


def _correctness(output: Any, expected: Any, diagonal: Any, rhs_host: Any,
                 terminal: Any) -> dict[str, Any]:
    import numpy as np

    actual = output.to_numpy()
    error = actual - expected
    residual = diagonal * actual - rhs_host
    max_abs_error = float(np.max(np.abs(error), initial=0.0))
    relative_l2 = float(
        np.linalg.norm(error.astype(np.float64)) /
        max(np.linalg.norm(expected.astype(np.float64)),
            np.finfo(np.float64).tiny))
    true_residual = float(np.linalg.norm(residual.astype(np.float64)))
    effective = float(terminal.effective_tolerance)
    rounding = 16.0 * np.finfo(np.float32).eps * max(
        float(np.linalg.norm(rhs_host.astype(np.float64))), 1.0)
    residual_limit = max(effective, rounding)
    return {
        "passed": bool(terminal.converged and np.all(np.isfinite(actual))
                       and max_abs_error <= 5.0e-3
                       and true_residual <= residual_limit),
        "max_abs_error": max_abs_error, "relative_l2_error": relative_l2,
        "true_residual_norm": true_residual,
        "residual_limit": residual_limit,
        "terminal": _result_snapshot(terminal),
    }


def _graph_memory(graph: Any) -> dict[str, Any]:
    memory = graph.execution_stats().memory
    names = (
        "persistent_internal_storage_bytes", "internal_storage_exclusive",
        "workspace_lane_capacity", "workspace_lanes_materialized",
        "workspace_lane_waits", "internal_storage_waits",
    )
    return {name: getattr(memory, name, None) for name in names}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    affinity = _apply_affinity(_resolve_affinity(
        args.cpu_affinity, args.cpu_threads))
    ti, import_ms, core_path = _load_taichi("forge")
    environment = _environment_provenance("forge", ti, core_path)
    init_started = time.perf_counter_ns()
    ti.init(arch=ti.cuda, offline_cache=False, kernel_profiler=False,
            random_seed=0, cpu_max_num_threads=args.cpu_threads)
    init_ms = (time.perf_counter_ns() - init_started) / 1.0e6
    actual_arch = ti.lang.impl.current_cfg().arch
    device = runtime_device_identity(ti, "cuda")
    elements = PRESETS[args.preset]
    diagonal_host = np.linspace(1.0, 4.0, elements, dtype=np.float32)
    exact = np.sin(np.linspace(0.0, 8.0, elements, dtype=np.float32))
    rhs_host = (diagonal_host * exact).astype(np.float32)
    topology = ti.ndarray(ti.i32, shape=elements)
    diagonal = ti.ndarray(ti.f32, shape=elements)
    rhs = ti.ndarray(ti.f32, shape=elements)
    eager_output = ti.ndarray(ti.f32, shape=elements)
    graph_output = ti.ndarray(ti.f32, shape=elements)
    topology.from_numpy(np.arange(elements, dtype=np.int32))
    diagonal.from_numpy(diagonal_host)
    rhs.from_numpy(rhs_host)
    eager_output.fill(0)
    graph_output.fill(0)

    @ti.kernel
    def apply_diagonal(
            active_size: ti.i32,
            topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
            numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
            source: ti.types.ndarray(dtype=ti.f32, ndim=1),
            destination: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for index in range(active_size):
            destination[index] = numeric_data[index] * source[
                topology_data[index]]

    operator = ti.linalg.LinearOperator.from_kernel(
        apply_diagonal, elements, topology, numeric=diagonal,
        traits=ti.linalg.OperatorTraits.spd())
    operator_qualification = ti.linalg.qualify_operator(
        operator, reference=lambda values: diagonal_host * values,
        samples=3, seed=0, warmup=1, repetitions=5,
        atol=5.0e-5, rtol=5.0e-5,
        metadata={"case_id": "FORGEONLY-001-LINEAR-OPERATOR"}).to_dict()

    def new_plan(policy: str | None):
        options = {} if policy is None else {"execution_policy": policy}
        return ti.linalg.experimental.SolvePlan(
            operator, method="cg", max_iterations=32, atol=1.0e-4,
            rtol=1.0e-5, check_interval=4, **options)

    automatic_plan = new_plan(None)
    automatic_stats = automatic_plan.statistics()
    capabilities = automatic_plan.execution_capabilities()
    device_capability = capabilities["device_convergent"]
    if not device_capability["supported"]:
        raise RuntimeError("device_convergent is unavailable: " +
                           str(device_capability["unsupported_reason"]))
    solve_qualification = ti.linalg.experimental.qualify_solve_plan(
        lambda: new_plan("device_convergent"), rhs, reference=exact,
        expected_termination="converged", warmup=1, repetitions=5,
        atol=5.0e-3, rtol=5.0e-4,
        metadata={"case_id": "FORGEONLY-001-SOLVE-PLAN-CG"}).to_dict()

    eager_plan = new_plan("device_convergent")
    graph_plan = new_plan("device_convergent")

    def eager_launch():
        return eager_plan.solve(rhs, out=eager_output)

    rhs_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "rhs", ti.f32, ndim=1)
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    action = graph_plan.graph_action(
        rhs_arg, output_arg, name="qualification_device_cg")
    graph_build_started = time.perf_counter_ns()
    builder = ti.graph.GraphBuilder()
    builder.append_native(action)
    graph = builder.compile()
    graph_build_ms = (time.perf_counter_ns() - graph_build_started) / 1.0e6
    packet = action.allocate_terminal()
    graph_arguments = {"rhs": rhs, "output": graph_output, **packet.arguments}

    def graph_launch():
        graph.submit(graph_arguments).wait()
        return packet.snapshot()

    launches = {MODES[0]: eager_launch, MODES[1]: graph_launch}
    first = {}
    for name in MODES:
        started = time.perf_counter_ns()
        terminal = launches[name]()
        ti.sync()
        first[name] = {
            "completion_ms": (time.perf_counter_ns() - started) / 1.0e6,
            "terminal": _result_snapshot(terminal),
        }
        if not terminal.converged:
            raise RuntimeError(f"{name} first solve did not converge")

    calibration = {}
    suggestions = []
    for name in MODES:
        suggestion, attempts = _calibrate(
            ti, launches[name], args.target_sample_ms)
        calibration[name] = {"suggested_batch_size": suggestion,
                             "attempts": attempts}
        suggestions.append(suggestion)
    common_batch = select_common_batch(suggestions)
    for name in MODES:
        for _ in range(args.warmups):
            _, terminal = _timed_batch(ti, launches[name], common_batch)
            if not terminal.converged:
                raise RuntimeError(f"{name} warmup did not converge")

    raw_batch_ms = {name: [] for name in MODES}
    terminals = {}
    execution_order = balanced_mode_orders(args.samples)
    for order in execution_order:
        for name in order:
            elapsed, terminal = _timed_batch(
                ti, launches[name], common_batch)
            if not terminal.converged:
                raise RuntimeError(f"{name} scored solve did not converge")
            raw_batch_ms[name].append(elapsed)
            terminals[name] = terminal
    samples_ms = {name: [value / common_batch
                         for value in raw_batch_ms[name]] for name in MODES}
    summaries = {name: summarize_samples(samples_ms[name]) for name in MODES}
    correctness = {
        MODES[0]: _correctness(eager_output, exact, diagonal_host, rhs_host,
                               terminals[MODES[0]]),
        MODES[1]: _correctness(graph_output, exact, diagonal_host, rhs_host,
                               terminals[MODES[1]]),
    }
    stability = {name: _stability(
        ti, launches[name], args.stability_replays,
        args.stability_checkpoint) for name in MODES}
    eager_terminal = eager_plan.solve(rhs, out=eager_output)
    graph_terminal = graph_launch()
    ti.sync()
    final_correctness = {
        MODES[0]: _correctness(eager_output, exact, diagonal_host, rhs_host,
                               eager_terminal),
        MODES[1]: _correctness(graph_output, exact, diagonal_host, rhs_host,
                               graph_terminal),
    }
    eager_median = float(summaries[MODES[0]]["median_ms"])
    graph_median = float(summaries[MODES[1]]["median_ms"])
    identity = automatic_stats["identity"]
    result = {
        "schema": SCHEMA, "status": "passed", "case_id": "FORGEONLY-001",
        "comparison_class": "forge-only-api-mode", "intent": args.intent,
        "backend": "cuda", "preset": args.preset, "elements": elements,
        "dtype": "f32", "method": "cg", "import_ms": import_ms,
        "init_ms": init_ms, "requested_arch": "cuda",
        "actual_arch": _arch_name(ti, actual_arch),
        "arch_match": actual_arch == ti.cuda, "environment": environment,
        "affinity": affinity, "device_identity": device,
        "operator_qualification": operator_qualification,
        "solve_qualification": solve_qualification,
        "route": {
            "operator_provider": operator.provider,
            "operator_execution_kind": operator.execution_kind,
            "operator_statistics": operator.statistics(),
            "device_convergent_capability": device_capability,
            "automatic_requested_policy": identity.get(
                "requested_solver_execution_policy"),
            "automatic_selected_policy": identity.get(
                "solver_execution_policy"),
            "automatic_selection_qualified": device_capability.get(
                "automatic_selection_qualified"),
            "graph_memory": _graph_memory(graph),
        },
        "build": {"graph_build_ms": graph_build_ms, "first": first},
        "batch_size": common_batch, "calibration": calibration,
        "sample_execution_order": execution_order,
        "raw_batch_ms": raw_batch_ms, "samples_ms": samples_ms,
        "summaries": summaries,
        "diagnostic_api_mode_ratio": {
            "eager_over_graph_median_x": eager_median / graph_median,
            "scope": ("two Forge completion boundaries for the same explicit "
                      "device-convergent CG; no cross-framework claim"),
        },
        "correctness_before_stability": correctness,
        "stability": stability,
        "correctness_after_stability": final_correctness,
        "plan_statistics": {"eager": eager_plan.statistics(),
                            "graph_plan": graph_plan.statistics()},
        "workload_contract": {
            "operator": "compiled diagonal LinearOperator with SPD traits",
            "system": "A=diag(linspace(1,4)); x=sin(linspace(0,8)); b=A*x",
            "same_operator_rhs_solution_tolerance": True,
            "timed_eager_boundary": "SolvePlan.solve completion",
            "timed_graph_boundary": "compiled Graph submit plus wait completion",
            "output_zero_fill_and_terminal_materialization_included": True,
            "setup_jit_qualification_and_host_solution_readback_excluded": True,
            "external_public_api_equivalent": False,
        },
        "measurement_config": {
            "samples": args.samples, "warmups": args.warmups,
            "target_sample_ms": args.target_sample_ms,
            "stability_replays": args.stability_replays,
            "stability_checkpoint": args.stability_checkpoint,
            "cpu_threads": args.cpu_threads,
            "cpu_affinity": args.cpu_affinity,
        },
    }
    method_checks = {
        "arch": result["arch_match"],
        "isolated_environment": _environment_isolated(environment),
        "device_binding": device["binding_verified"],
        "operator_public_qualification": operator_qualification["passed"],
        "solve_public_qualification": solve_qualification["passed"],
        "explicit_device_route": bool(
            device_capability["supported"]
            and eager_plan.statistics()["identity"][
                "solver_execution_policy"] == "device_convergent"
            and graph_plan.statistics()["identity"][
                "solver_execution_policy"] == "device_convergent"),
        "correctness_before_and_after": all(
            item["passed"] for item in
            (*correctness.values(), *final_correctness.values())),
        "balanced_mode_order": abs(sum(
            order[0] == MODES[0] for order in execution_order) - sum(
                order[0] == MODES[1] for order in execution_order)) <= 1,
        "common_scored_batch": all(
            float(summaries[name]["median_ms"]) * common_batch
            >= args.target_sample_ms for name in MODES),
        "stability_complete": all(
            stability[name]["replays"] >= args.stability_replays
            and stability[name]["memory_guard_passed"] for name in MODES),
    }
    result["method_checks"] = method_checks
    result["ready_for_forge_only_absolute_report"] = bool(
        args.intent == "qualification" and not qualification_policy_errors(args)
        and all(method_checks.values())
        and max(float(summaries[name]["cv_percent"])
                for name in MODES) <= QUALIFICATION_MAX_CV_PERCENT)
    ti.sync()
    reset_error = None
    try:
        ti.reset()
    except Exception as error:  # pragma: no cover
        reset_error = repr(error)
    result["teardown"] = {"reset_error": reset_error}
    if reset_error is not None:
        result["status"] = "failed"
        result["ready_for_forge_only_absolute_report"] = False
    gc.collect()
    return result


def _write_reports(output_dir: Path, result: dict[str, Any]) -> None:
    ready = result.get("ready_for_forge_only_absolute_report", False)
    modes = result.get("summaries", {})
    eager = modes.get(MODES[0], {})
    graph = modes.get(MODES[1], {})
    ratio = result.get("diagnostic_api_mode_ratio", {}).get(
        "eager_over_graph_median_x")
    max_cv = max([row.get("cv_percent", math.inf)
                  for row in modes.values()], default=math.inf)
    (output_dir / "report.zh-CN.md").write_text(
        "# LinearOperator/SolvePlan/Graph solver 本机资格结果\n\n"
        f"- 状态：`{result.get('status')}`\n"
        f"- Forge-only 绝对报告资格：{'通过' if ready else '未通过'}\n"
        f"- Eager device-convergent median：{eager.get('median_ms', 'n/a')} ms\n"
        f"- Graph device-convergent median：{graph.get('median_ms', 'n/a')} ms\n"
        f"- Eager/Graph diagnostic mode ratio：{ratio if ratio is not None else 'n/a'}\n"
        f"- 最大 CV：{max_cv}%\n\n"
        "## 边界\n\n"
        "该案例验证 Forge 新 API 的公开合同、显式 device-convergent route、绝对完成"
        "时间和稳定性。mode ratio 只比较两个 Forge 调用边界；vanilla 与 Warp 没有"
        "同一公开 API，因此没有跨框架 speedup。\n",
        encoding="utf-8")
    (output_dir / "report.en.md").write_text(
        "# Local LinearOperator/SolvePlan/Graph solver qualification\n\n"
        f"- Status: `{result.get('status')}`\n"
        f"- Forge-only absolute-report eligibility: {'pass' if ready else 'fail'}\n"
        f"- Eager device-convergent median: {eager.get('median_ms', 'n/a')} ms\n"
        f"- Graph device-convergent median: {graph.get('median_ms', 'n/a')} ms\n"
        f"- Eager/Graph diagnostic mode ratio: {ratio if ratio is not None else 'n/a'}\n"
        f"- Maximum CV: {max_cv}%\n\n"
        "## Boundary\n\n"
        "This case validates the public contracts, explicit device-convergent "
        "route, absolute completion time, and stability of the new Forge APIs. "
        "The mode ratio compares two Forge call boundaries only. Vanilla and "
        "Warp lack the same public API, so no cross-framework speedup exists.\n",
        encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="small")
    parser.add_argument("--intent", choices=("diagnostic", "qualification"),
                        default="diagnostic")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--target-sample-ms", type=float, default=100.0)
    parser.add_argument("--stability-replays", type=int, default=100)
    parser.add_argument("--stability-checkpoint", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cpu-affinity", default="auto")
    parser.add_argument("--max-cpu-util", type=float, default=20.0)
    parser.add_argument("--max-gpu-util", type=float, default=15.0)
    parser.add_argument("--max-gpu-temp", type=float, default=65.0)
    parser.add_argument("--output-root", default="temp_outputs/qualification/results")
    parser.add_argument("--run-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy_errors = qualification_policy_errors(args)
    if policy_errors:
        raise ValueError("; ".join(policy_errors))
    if args.samples <= 0 or args.warmups < 0 or args.target_sample_ms <= 0:
        raise ValueError("samples/target must be positive and warmups nonnegative")
    if args.stability_replays <= 0 or args.stability_checkpoint <= 0:
        raise ValueError("stability settings must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"linear-solve-{args.preset}-{timestamp}"
    if Path(run_id).name != run_id:
        raise ValueError("run_id must be one path component")
    output_dir = (repo_root / args.output_root / run_id).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_metadata(repo_root), "host": host_metadata(),
        "config": vars(args), "noise_observations": [],
    }
    try:
        with _ExclusiveBenchmarkLock() as lock:
            manifest["exclusive_driver_lock"] = lock
            ignored = [os.getpid(), *_windows_process_ancestors(os.getpid())]
            manifest["noise_ignored_process_lineage"] = ignored
            before = _noise_observation(
                "cuda", ignored, args.max_cpu_util, args.max_gpu_util,
                args.max_gpu_temp)
            manifest["noise_observations"].append({"label": "before", **before})
            write_json(output_dir / "manifest.json", manifest)
            if not before["passed"]:
                raise RuntimeError("noise admission failed before run: " +
                                   "; ".join(before["reasons"]))
            result = _run(args)
            after = _noise_observation(
                "cuda", ignored, args.max_cpu_util, args.max_gpu_util,
                args.max_gpu_temp)
            manifest["noise_observations"].append({"label": "after", **after})
            result["noise_admission"] = {
                "before": before, "after": after,
                "passed": before["passed"] and after["passed"],
            }
            result["method_checks"]["exclusive_driver_lock"] = True
            result["method_checks"]["noise_admission"] = after["passed"]
            result["ready_for_forge_only_absolute_report"] = bool(
                result["ready_for_forge_only_absolute_report"]
                and after["passed"])
            manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(output_dir / "manifest.json", manifest)
            write_json(output_dir / "result.json", result)
            rows = []
            for mode in MODES:
                for index, (batch, per_solve) in enumerate(zip(
                        result["raw_batch_ms"][mode],
                        result["samples_ms"][mode])):
                    rows.append({"mode": mode, "sample": index + 1,
                                 "batch_ms": batch,
                                 "per_solve_ms": per_solve})
            write_csv(output_dir / "samples.csv", rows)
            _write_reports(output_dir, result)
            print(json.dumps({
                "run_id": run_id, "status": result["status"],
                "summaries": result["summaries"],
                "ratio": result["diagnostic_api_mode_ratio"],
                "ready": result["ready_for_forge_only_absolute_report"],
                "output_dir": str(output_dir),
            }, sort_keys=True))
            return 0 if result["status"] == "passed" else 2
    except Exception as error:
        try:
            ti = sys.modules.get("taichi_forge")
            if ti is not None:
                ti.reset()
        except Exception:
            pass
        failure = {
            "schema": SCHEMA, "run_id": run_id, "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "ready_for_forge_only_absolute_report": False,
        }
        manifest["failure"] = failure
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "failure.json", failure)
        (output_dir / "failure.zh-CN.md").write_text(
            "# LinearOperator/SolvePlan 资格运行失败\n\n"
            f"- 原因：`{failure['reason']}`\n- 发布资格：未通过\n",
            encoding="utf-8")
        (output_dir / "failure.en.md").write_text(
            "# LinearOperator/SolvePlan qualification failed\n\n"
            f"- Reason: `{failure['reason']}`\n- Publication eligibility: fail\n",
            encoding="utf-8")
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
