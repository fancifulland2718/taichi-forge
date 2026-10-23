"""Integer-prelude qualification; one process/backend, no implicit GPU fallback.

Run --mode baseline first, then --mode candidate --baseline-summary PATH.
No performance claim follows from task counts. Optional --timing records a
small host-to-completion sample using this same fixture and capacity.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

import numpy as np
import taichi_forge as ti


SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)
TASK_TYPES = {0: "serial", 1: "range_for"}
GEOMETRY_FIELDS = (
    "requested_grid_size", "requested_block_size", "source_block_size_explicit",
    "requested_thread_local_mode", "requested_range_work_per_thread_target",
    "requested_grid_residency_waves", "requested_memory_strategy",
    "selected_grid_size", "selected_block_size", "actual_grid_size",
    "actual_block_size", "actual_geometry_kind", "range_mapping",
    "constant_range_size", "static_shared_bytes", "dynamic_shared_bytes",
    "thread_local_bytes",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_fixture(output, capacity):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("serial_integer_prelude_kernels.py").read_text(encoding="utf8")
    provenance = dict(fixture_sha256=digest(template.encode()), capacity=capacity)
    generated = output / "generated_interval_kernels.py"
    generated.write_text(template, encoding="utf8")
    name = "integer_prelude_fixture_" + digest(str(output).encode())[:12]
    spec = importlib.util.spec_from_file_location(name, generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, module.IntervalFixture(capacity), provenance


def manifest(kernel, *args):
    return [asdict(item) for item in kernel.task_manifest(*args)]


def types(tasks):
    return [item["task_type"] for item in tasks]


def assert_types(tasks, expected):
    assert types(tasks) == expected, (types(tasks), expected)


def geometry(tasks):
    ranges = [item for item in tasks if item["task_type"] == "range_for"]
    assert len(ranges) == 1, types(tasks)
    return {name: ranges[0][name] for name in GEOMETRY_FIELDS}


def graph_for(kernel, *arguments):
    builder = ti.graph.GraphBuilder()
    builder.dispatch(kernel, *arguments, template_args={"self": kernel._kernel_owner})
    return builder.compile()


def expected_classify(previous, maximum, count):
    expected = np.full(len(previous), -91, np.int32)
    active = min(max(count, 0), len(previous))
    expected[:active] = np.where(previous[:active] == SENTINEL, 0,
                                np.where(previous[:active] < maximum[:active], 2, 1))
    return expected


def artifact_tasks(directory, entry, arch):
    # Read the backend's actual exported task metadata, not the producer Graph.
    text = (directory / "metadata.json").read_text(encoding="utf8")
    if arch == "vulkan":
        metadata = json.loads(text)
        kernels = [item for item in metadata["kernels"] if item["name"] == entry]
        assert len(kernels) == 1, (entry, metadata["kernels"])
        tasks = kernels[0]["tasks_attribs"]
    else:
        # LLVM TextSerializer output is diagnostic text, not valid JSON.
        # Restrict every lookup to an exact named container; never count tasks
        # from the seed/report kernels or infer them from a producer manifest.
        kernel = named_container(text, entry, "{", "}")
        compiled = named_container(kernel, "compiled_data", "{", "}")
        body = named_container(compiled, "tasks", "[", "]")
        kinds = re.findall(r'"task_type"\s*:\s*(-?\d+)', body)
        names = re.findall(r'"name"\s*:\s*"([^"\\]*)"', body)
        assert len(kinds) == len(names) and kinds, (entry, body)
        tasks = [dict(task_type=int(kind), name=name) for kind, name in zip(kinds, names)]
    return dict(entry=entry, types=[TASK_TYPES[item["task_type"]] for item in tasks],
                raw_tasks=tasks, metadata_sha256=digest((directory / "metadata.json").read_bytes()))


def named_container(text, key, opening, closing):
    pattern = r'"' + re.escape(key) + r'"\s*:\s*' + re.escape(opening)
    matches = list(re.finditer(pattern, text))
    assert len(matches) == 1, (key, len(matches))
    begin = matches[0].end() - 1
    depth, quoted, escaped = 0, False, False
    for position in range(begin, len(text)):
        character = text[position]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return text[begin:position + 1]
    raise AssertionError("Unterminated metadata container: " + key)


def timing_sample(fixture, graph, previous, maximum, samples=15, batch=8):
    # This routine is opt-in, after qualification, and never changes launch policy.
    capacity = fixture.capacity
    fixture.lower_bound.from_numpy(previous)
    fixture.upper_bound.from_numpy(maximum)
    fixture.active_count[None] = capacity // 2
    builder = ti.graph.GraphBuilder()
    for _ in range(batch):
        builder.dispatch(fixture.classify,
                         template_args={"self": fixture})
    batched = builder.compile()
    result = dict(active=capacity // 2, samples=samples, graph_batch=batch,
                  clock="perf_counter_ns; host submit through ti.sync completion")
    try:
        invocations = {
            "direct": fixture.classify,
            "graph_single": lambda: graph.run({}),
            "graph_batch": lambda: batched.run({}),
        }
        for name, invoke in invocations.items():
            for _ in range(3):
                invoke()
                ti.sync()
            elapsed = []
            for _ in range(samples):
                begin = time.perf_counter_ns()
                invoke()
                ti.sync()
                elapsed.append(time.perf_counter_ns() - begin)
            result[name] = dict(ns=elapsed, median_ns=statistics.median(elapsed))
    finally:
        batched.close()
    return result


def run_classify(output, mode="candidate", capacity=37, timing=False):
    assert capacity >= 37, "The qualification sequence requires capacity >= 37"
    output = Path(output).resolve()
    _, fixture, provenance = make_fixture(output, capacity)
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name
    expected_tasks = ["serial", "range_for"] if mode == "baseline" or arch == "cpu" else ["range_for"]
    previous = np.arange(capacity, dtype=np.uint64)
    previous[:3] = SENTINEL
    maximum = previous.copy()
    maximum[6:9] += np.uint64(1)
    maximum[9:11] -= np.uint64(1)
    fixture.lower_bound.from_numpy(previous)
    fixture.upper_bound.from_numpy(maximum)
    fixture.active_count[None] = 11
    fixture.status.fill(-91)
    # A fresh specialization in a fresh process with offline_cache=False.
    ti.profiler.clear_scoped_profiler_info()
    start = time.perf_counter_ns()
    direct_manifest = manifest(fixture.classify)
    compile_ns = time.perf_counter_ns() - start
    ti.profiler.export_scoped_profiler_csv(str(output / "cold-manifest.csv"))
    assert_types(direct_manifest, expected_tasks)
    start = time.perf_counter_ns()
    fixture.classify()
    ti.sync()
    first_launch_ns = time.perf_counter_ns() - start
    ti.profiler.export_scoped_profiler_csv(str(output / "through-first-launch.csv"))
    result = dict(provenance=provenance, jit_tasks=direct_manifest,
                  cold_manifest_compile_ns=compile_ns,
                  first_launch_after_manifest_ns=first_launch_ns, checks=[])
    graph = graph_for(fixture.classify)
    seed_graph = graph_for(
        fixture.seed,
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "count", ti.i32),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "previous", ti.u64, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "maximum", ti.u64, ndim=1),
    )
    read_graph = graph_for(fixture.report, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1))
    try:
        result["graph_tasks"] = manifest(graph)
        assert_types(result["graph_tasks"], expected_tasks)
        assert geometry(direct_manifest) == geometry(result["graph_tasks"])
        for count in (0, 11, capacity, 3, 0):
            expected = expected_classify(previous, maximum, count)
            for route, invoke in (("jit", fixture.classify),
                                  ("graph", lambda: graph.run({}))):
                fixture.active_count[None] = count
                fixture.status.fill(-91)
                invoke()
                np.testing.assert_array_equal(fixture.status.to_numpy(), expected)
                assert fixture.active_count[None] == count
                result["checks"].append(dict(route=route, count=count, output_sha256=digest(expected.tobytes())))

        module = ti.aot.Module(caps=["spirv_has_int64"] if arch == "vulkan" else [])
        plans = {}
        for name, value in (("seed", seed_graph), ("classify", graph), ("read", read_graph)):
            plans[name] = json.loads(module.add_execution_plan(name, value, snapshot_state=True).to_json())
        operations = plans["classify"]["operations"]
        assert len(operations) == 1 and operations[0]["kind"] == "kernel", operations
        saved = output / "aot"
        module.save(saved)
        result["aot_tasks"] = artifact_tasks(saved, operations[0]["entry"], arch)
        assert result["aot_tasks"]["types"] == expected_tasks
        artifact = output / "classify.tcm"
        module.archive(str(artifact))
        result["artifact_sha256"] = digest(artifact.read_bytes())

        # Loaded-module roots must not accidentally keep the producer's pointers.
        fixture.active_count[None] = -777
        fixture.lower_bound.from_numpy(np.full(capacity, SENTINEL, np.uint64))
        fixture.upper_bound.fill(0)
        fixture.status.fill(-777)
        with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
            arrays = {"previous": runtime.ndarray("u64", capacity),
                      "maximum": runtime.ndarray("u64", capacity),
                      "out": runtime.ndarray("i32", capacity)}
            try:
                arrays["previous"].write(previous.tobytes())
                arrays["maximum"].write(maximum.tobytes())
                for count in (0, 11, capacity, 3, 0):
                    with loaded.prepare_execution("seed", dict(count=count, previous=arrays["previous"],
                                                               maximum=arrays["maximum"])) as call:
                        call()
                    with loaded.prepare_execution("classify", {}) as call:
                        call()
                    arrays["out"].write(np.full(capacity, -888, np.int32).tobytes())
                    with loaded.prepare_execution("read", dict(out=arrays["out"])) as call:
                        call()
                    expected = expected_classify(previous, maximum, count)
                    np.testing.assert_array_equal(np.frombuffer(arrays["out"].read(), np.int32), expected)
                    result["checks"].append(dict(route="aot", count=count, output_sha256=digest(expected.tobytes())))
            finally:
                for array in reversed(list(arrays.values())):
                    array.close()
        assert fixture.active_count[None] == -777
        np.testing.assert_array_equal(fixture.status.to_numpy(), np.full(capacity, -777, np.int32))
        result["aot_producer_isolation"] = True
        if timing:
            result["timing"] = timing_sample(fixture, graph, previous, maximum)
    finally:
        read_graph.close()
        seed_graph.close()
        graph.close()
    return result


def run_guards(output):
    module, _, _ = make_fixture(output, 37)
    guard = module.make_guards(37)
    source, out, x, y = (guard[name] for name in ("source", "out", "x", "y"))
    result = {}
    for name in ("dynamic_range", "source_write", "external_alias"):
        kernel = guard[name]
        args = ()
        if name == "external_alias":
            alias = ti.experimental.ndarray_view(source)
            assert alias.shape == ()
            assert alias.runtime_argument.qualification["zero_copy_qualified"]
            args = (alias,)
        tasks = manifest(kernel, *args)
        assert_types(tasks, ["serial", "range_for"])
        cases = (0, 11) if name == "dynamic_range" else (11,)
        for count in cases:
            source[None] = count
            out.fill(-91)
            kernel(*args)
            expected = np.full(37, count, np.int32)
            if name == "dynamic_range":
                expected.fill(-91)
                expected[:count] = np.arange(count)
            np.testing.assert_array_equal(out.to_numpy(), expected)
            assert source[None] == (count if name == "dynamic_range" else 3)
        result[name] = dict(tasks=tasks, counts=list(cases), passed=True)
    result["external_alias"]["real_zero_copy_scalar_field_alias"] = True

    values = np.arange(37, dtype=np.float32) / 4
    x.from_numpy(values)
    for count in (0, 11):
        source[None] = count
        y.fill(-91)
        with ti.ad.FwdMode(loss=y, param=x, seed=[1.0] * 37):
            guard["forward_prefix"]()
            # FwdMode temporarily sets this kernel's mode until context exit.
            forward_tasks = manifest(guard["forward_prefix"])
        assert_types(forward_tasks, ["serial", "range_for"])
        expected = np.full(37, -91, np.float32)
        expected[:count] = values[:count] ** 2
        derivative = np.zeros(37, np.float32)
        derivative[:count] = 2 * values[:count]
        np.testing.assert_array_equal(y.to_numpy(), expected)
        np.testing.assert_array_equal(y.dual.to_numpy(), derivative)
        y.fill(-91)
        x.grad.fill(0)
        y.grad.fill(1)
        guard["reverse_dynamic"]()
        guard["reverse_dynamic"].grad()
        reverse_tasks = manifest(guard["reverse_dynamic"].grad)
        np.testing.assert_array_equal(y.to_numpy(), expected)
        np.testing.assert_array_equal(x.grad.to_numpy(), derivative)
    result["forward_ad"] = dict(tasks=forward_tasks, counts=[0, 11], passed=True)
    result["reverse_ad"] = dict(tasks=reverse_tasks, counts=[0, 11], passed=True)
    return result


def compare_baseline(result, baseline):
    assert baseline["status"] == "passed" and baseline["mode"] == "baseline"
    assert baseline["arch"] == result["arch"]
    assert baseline["classify"]["provenance"] == result["classify"]["provenance"]
    assert baseline["compile_config"] == result["compile_config"]
    assert baseline["classify"]["checks"] == result["classify"]["checks"]
    assert geometry(baseline["classify"]["jit_tasks"]) == geometry(result["classify"]["jit_tasks"])
    for name, guard in result["guards"].items():
        assert types(guard["tasks"]) == types(baseline["guards"][name]["tasks"]), name
    return dict(classify_tasks_before=2, classify_tasks_after=len(result["classify"]["jit_tasks"]),
                same_outputs=True, same_range_geometry=True,
                same_rejected_case_task_types=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), required=True)
    parser.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--expected-native-commit")
    parser.add_argument("--build-label", help="Human-readable source/overlay identity; independent of the compiled stamp")
    parser.add_argument("--capacity", type=int, default=37)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.mode == "candidate" and args.baseline_summary is None:
        parser.error("candidate mode requires --baseline-summary from the matching baseline backend/capacity")
    output = args.output_dir.resolve()
    baseline = None if args.baseline_summary is None else json.loads(args.baseline_summary.resolve().read_text())
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    native_binary = Path(ti._lib.core.__file__).resolve()
    result = dict(status="running", mode=args.mode, arch=args.arch, executable=sys.executable,
                  package=ti.__file__, native_commit=ti._lib.core.get_commit_hash(),
                  build_label=args.build_label or ("candidate_overlay" if args.mode == "candidate" else "baseline"),
                  native_binary=str(native_binary), native_binary_sha256=digest(native_binary.read_bytes()),
                  timing_requested=args.timing, schema=1)
    try:
        if args.expected_native_commit:
            assert result["native_commit"] == args.expected_native_commit
        ti.init(arch=getattr(ti, args.arch), enable_fallback=False,
                cpu_max_num_threads=2, offline_cache=False,
                advanced_optimization=False, opt_level=1, debug=False)
        assert ti.lang.impl.current_cfg().arch == getattr(ti, args.arch)
        result["compile_config"] = {
            name: getattr(ti.lang.impl.current_cfg(), name)
            for name in ("opt_level", "advanced_optimization", "cfg_optimization", "debug", "offline_cache")
        }
        result["classify"] = run_classify(output / "classify", args.mode, args.capacity, args.timing)
        result["guards"] = run_guards(output / "guards")
        if baseline is not None:
            result["comparison"] = compare_baseline(result, baseline)
        result["status"] = "passed"
    except BaseException:
        result["status"] = "failed"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        from taichi_forge_runtime import _native
        if _native.loaded_path() is not None:
            runtime_binary = Path(_native.loaded_path())
            result["runtime_binary"] = str(runtime_binary)
            result["runtime_binary_sha256"] = digest(runtime_binary.read_bytes())
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print("INTEGER_PRELUDE_QUALIFICATION " + json.dumps({key: result[key] for key in ("status", "arch", "mode", "native_commit")}))


if __name__ == "__main__":
    main()
