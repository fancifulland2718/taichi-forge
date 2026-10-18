"""Histogram reconstruction, immutable algorithm preferences and owned scratch."""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_scan_probe import reject


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4
)
def test_histogram_sequence_runtime_only_and_borrowed(tmp_path, monkeypatch):
    module = ti.aot.Module()
    arch = ti.lang.impl.current_cfg().arch
    for key, value in (
        ("PRIVATE_MIN_N", 65536),
        ("PRIVATE_MAX_BINS", 512),
        ("SINGLE_SHARED_MAX_N", 4096),
    ):
        monkeypatch.setenv("TI_VULKAN_HISTOGRAM_" + key, str(value))
    configurations = [(v, b, 17) for v in (ti.i32, ti.u32) for b in (ti.i32, ti.i64)]
    if arch == ti.cpu:
        configurations.append((ti.i32, ti.i64, 262147))
    cases = []
    for value_type, bin_type, n in configurations:
        if arch == ti.vulkan and not ti.lang.impl.get_runtime().prog.vulkan_histogram_value_type_available(
            2 if value_type == ti.u32 else 0, 4 if bin_type == ti.i64 else 0
        ):
            continue
        bound = 7
        value_numpy, bin_numpy = (np.uint32 if value_type == ti.u32 else np.int32), (
            np.int64 if bin_type == ti.i64 else np.int32
        )
        values, bins = ti.ndarray(value_type, n), ti.ndarray(bin_type, bound)
        initial = np.full(bound, 37, bin_numpy)
        sequence = ti.algorithms.primitive_sequence().histogram(values, bins)
        samples = []
        for shift in (0, 3):
            source = ((np.arange(n, dtype=np.int32) + shift) % (bound + 2) - 1).astype(
                value_numpy
            )
            valid = source[(source >= 0) & (source < bound)]
            expected = np.bincount(valid.astype(np.int64), minlength=bound).astype(
                bin_numpy
            )
            values.from_numpy(source)
            bins.from_numpy(initial)
            if value_type == ti.i32:
                sequence.run()
                np.testing.assert_array_equal(bins.to_numpy(), expected)
            samples.append(
                dict(source=source.tobytes().hex(), expected=expected.tobytes().hex())
            )
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()
        name = f"histogram_{value_type}_{bin_type}_{n}"
        before = values.to_numpy().tobytes(), bins.to_numpy().tobytes()
        plan = module.add_execution_plan(
            name, graph, source_bindings=dict(values=values, bins=bins)
        )
        assert before == (values.to_numpy().tobytes(), bins.to_numpy().tobytes())
        operation = json.loads(plan.to_json())["operations"][0]
        assert operation["operation"] == "histogram"
        if arch == ti.cpu:
            assert operation["parameters"]["cpu_max_threads"] == 4
        elif arch == ti.vulkan:
            assert operation["parameters"]["vulkan_policy"] == dict(
                private_min_n=65536, private_max_bins=512, single_shared_max_n=4096
            )
        cases.append(
            dict(
                name=name,
                size=n,
                num_bins=bound,
                value_dtype=str(value_type),
                counter_dtype=str(bin_type),
                initial=initial.tobytes().hex(),
                samples=samples,
            )
        )
        graph.close()
        sequence.clear()
    artifact = tmp_path / "histogram.tcm"
    module.archive(str(artifact))
    assert not module._exported_kernels
    with zipfile.ZipFile(artifact) as archive:
        assert not any(
            name.endswith((".bc", ".spv", ".ptx", ".cubin"))
            for name in archive.namelist()
        )
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    # Consumer settings deliberately request a private variant. They must not
    # change the exported default recipe or allocate hidden scratch.
    env = dict(
        os.environ,
        TI_VULKAN_HISTOGRAM_PRIVATE_MIN_N="0",
        TI_VULKAN_HISTOGRAM_SINGLE_SHARED_MAX_N="0",
    )
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_histogram_probe.py")),
                mode,
                backend,
                str(artifact),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "Validation Error" not in output and "SYNC-HAZARD" not in output, output


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_histogram_prefix_views_empty_alias_and_private_variants(tmp_path):
    module = ti.aot.Module()
    configurations = [
        ("empty", 0, 7),
        ("single", 17, 7),
        ("direct", 17, 7),
        ("private_shared", 4099, 7),
        ("private", 4099, 4097),
    ]
    for name, n, bound in configurations:
        policy = dict(
            private_min_n=0 if name.startswith("private") else 65536,
            private_max_bins=0,
            single_shared_max_n=4096 if name == "single" else 0,
        )
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(
                [
                    dict(
                        kind="native",
                        provider="builtin",
                        operation="histogram",
                        version=1,
                        parameters=dict(
                            size=n,
                            num_bins=bound,
                            value_dtype="i32",
                            counter_dtype="i32",
                            vulkan_policy=policy,
                        ),
                        bindings=dict(values="source", bins="target"),
                    )
                ],
                resources=[
                    dict(
                        kind="ndarray_view",
                        name="source",
                        array="values",
                        offset=1,
                        shape=[4101],
                    ),
                    dict(
                        kind="ndarray_view",
                        name="target",
                        array="bins",
                        offset=1,
                        shape=[4101],
                    ),
                ],
            ),
        )
    artifact = tmp_path / "prefix.tcm"
    module.archive(str(artifact))
    arch = ti.lang.impl.current_cfg().arch
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        values, bins = runtime.ndarray("i32", 4103), runtime.ndarray("i32", 4103)
        frame = dict(values=values, bins=bins)
        reject(
            lambda: loaded.prepare_execution("single", dict(values=values, bins=values))
        )
        for name, n, bound in configurations:
            bins.write(np.full(4103, 37, np.int32).tobytes())
            with loaded.prepare_execution(name, frame) as call:
                assert bins.read() == np.full(4103, 37, np.int32).tobytes()
                scratch = call.info["workspace_bytes"]
                expected_scratch = (
                    ((n + 2047) // 2048) * bound * 4
                    if arch == ti.vulkan and name.startswith("private")
                    else 0
                )
                assert scratch == expected_scratch
                for shift in (0, 3):
                    source = (np.arange(4103, dtype=np.int32) + shift) % (bound + 2) - 1
                    values.write(source.tobytes())
                    call()
                    valid = source[1 : n + 1]
                    valid = valid[(valid >= 0) & (valid < bound)]
                    expected = np.full(4103, 37, np.int32)
                    expected[1 : bound + 1] = np.bincount(valid, minlength=bound)
                    assert bins.read() == expected.tobytes()
                    assert values.read() == source.tobytes()
        # With no reads, aliasing is safe: only the requested bins are cleared.
        with loaded.prepare_execution(
            "empty", dict(values=values, bins=values)
        ) as call:
            source = np.full(4103, 37, np.int32)
            values.write(source.tobytes())
            call()
            source[1:8] = 0
            assert values.read() == source.tobytes()
        values.close()
        bins.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_histogram_snapshot_composes_with_kernels(tmp_path):
    values, bins = ti.field(ti.i32, 17), ti.field(ti.i32, 7)
    data = np.arange(17, dtype=np.int32) % 9 - 1
    values.from_numpy(data)

    @ti.kernel
    def initialize():
        for i in bins:
            bins[i] = 37

    @ti.kernel
    def publish(output: ti.types.ndarray(ti.i32, 1)):
        for i in range(7):
            output[i] = bins[i] * 3

    sequence = ti.algorithms.primitive_sequence().histogram(values, bins)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize)
    builder.append_native(sequence)
    builder.dispatch(
        publish, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    )
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("step", graph, snapshot_state=True)
    artifact = tmp_path / "snapshot.tcm"
    module.archive(str(artifact))
    graph.close()
    sequence.clear()
    values.fill(0)
    expected = (np.bincount(data[(data >= 0) & (data < 7)], minlength=7) * 3).astype(
        np.int32
    )
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        output = runtime.ndarray("i32", 7)
        with loaded.prepare_execution("step", dict(output=output)) as call:
            for _ in range(2):
                call()
                assert output.read() == expected.tobytes()
        output.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_histogram_export_rejects_changed_strategy():
    values, bins = ti.ndarray(ti.i32, 17), ti.ndarray(ti.i32, 7)
    values.fill(0)
    sequence = ti.algorithms.primitive_sequence().histogram(values, bins)
    sequence.run()
    sequence._calls[0].plan.method_name = "unreconstructible"
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="selected cached strategy"):
        ti.aot.Module().add_execution_plan(
            "bad", graph, source_bindings=dict(values=values, bins=bins)
        )
    graph.close()
    sequence.clear()
