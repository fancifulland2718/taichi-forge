"""Shared native bucket ownership and source-free AOT reconstruction."""

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


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4
)
def test_bucket_native_registry_runtime_only_and_borrowed(tmp_path):
    arch = ti.lang.impl.current_cfg().arch
    module = ti.aot.Module()
    configurations = [
        (dtype, [], 17, 7, False)
        for dtype in ("i32", "u32", "f32", "i64", "u64", "f64")
    ]
    configurations += [
        ("f32", [3], 2051, 7, True),
        ("i32", [], 17, 4097, True),
        ("i32", [], 0, 7, True),
    ]
    if arch == ti.cpu:
        # Exercise both existing parallel policies at their actual activation
        # boundary. A one-lane tensor must retain the scalar copy strategy.
        configurations += [
            ("i32", [1], 262147, 7, False),
            ("f32", [3], 262147, 7, False),
        ]
    cases = []
    for index, (dtype, element_shape, n, bins, private) in enumerate(configurations):
        name = f"bucket_{index}"
        parameters = dict(
            size=n,
            num_bins=bins,
            dtype=dtype,
            element_shape=element_shape,
            cpu_max_threads=4,
            vulkan_policy=dict(
                private_enabled=private,
                private_min_items=0,
                private_max_bins=8192,
                private_max_bytes=1 << 24,
            ),
        )
        bindings = {
            name: name + "_view" for name in ("keys", "values", "offsets", "output")
        }
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(
                [
                    dict(
                        kind="native",
                        provider="builtin",
                        operation="bucket_builder",
                        version=1,
                        parameters=parameters,
                        bindings=bindings,
                    )
                ],
                resources=[
                    dict(
                        kind="ndarray_view",
                        name=target,
                        array=source,
                        offset=1,
                        shape=[bins + 2 if source == "offsets" else n + 1],
                    )
                    for source, target in bindings.items()
                ],
            ),
        )
        cases.append(dict(name=name, **parameters))
    artifact = tmp_path / "bucket.tcm"
    module.archive(str(artifact))
    assert not module._exported_kernels
    with zipfile.ZipFile(artifact) as archive:
        assert not any(
            name.endswith((".bc", ".ptx", ".spv", ".cubin"))
            for name in archive.namelist()
        )
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_bucket_builder_probe.py")),
                mode,
                backend,
                str(artifact),
            ],
            env=dict(os.environ, TI_VULKAN_BUCKET_BUILDER_PRIVATE="0"),
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "Validation Error" not in output and "SYNC-HAZARD" not in output, output


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4
)
def test_bucket_sequence_cold_and_cached_export(tmp_path, monkeypatch):
    from taichi_forge.algorithms._primitive_definitions import BucketBuilderDefinition
    from taichi_forge.graph._native_definition import NativeOperationBinding

    for key, value in (
        ("PRIVATE", 1),
        ("PRIVATE_MIN_N", 0),
        ("PRIVATE_MAX_BINS", 1024),
        ("PRIVATE_MAX_BYTES", 1 << 24),
    ):
        monkeypatch.setenv("TI_VULKAN_BUCKET_BUILDER_" + key, str(value))
    module = ti.aot.Module()
    arch = ti.lang.impl.current_cfg().arch
    cases = []
    for n, lanes, cached in ((17, 1, False), (2051, 3, True)):
        bins = 7
        dtype = ti.i32 if lanes == 1 else ti.f32
        keys, offsets = ti.ndarray(ti.i32, n), ti.ndarray(ti.i32, bins + 1)
        values = (
            ti.ndarray(dtype, n) if lanes == 1 else ti.Vector.ndarray(lanes, dtype, n)
        )
        output = (
            ti.ndarray(dtype, n) if lanes == 1 else ti.Vector.ndarray(lanes, dtype, n)
        )
        keys.from_numpy(np.arange(n, dtype=np.int32) % (bins + 2) - 1)
        values.fill(3)
        offsets.fill(-77)
        output.fill(-77)
        sequence = ti.algorithms.primitive_sequence().bucket_builder(
            keys, values, offsets, output, method="two_level" if cached else "auto"
        )
        if cached:
            sequence.run()
            assert sequence._calls[0].plan is not None
        else:
            assert sequence._calls[0].plan is None
        before = tuple(a.to_numpy().tobytes() for a in (keys, values, offsets, output))
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()
        name = f"sequence_{n}"
        bound = []
        original_bind = BucketBuilderDefinition.bind

        def observe_binding(definition, *args, **kwargs):
            bound.append(definition)
            return original_bind(definition, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(BucketBuilderDefinition, "bind", observe_binding)
            plan = module.add_execution_plan(
                name, graph, source_bindings=dict(keys=keys, values=values, offsets=offsets, output=output)
            )
        assert len(bound) == 1
        if cached:
            assert bound[0] is sequence._calls[0].plan.definition
        assert before == tuple(
            a.to_numpy().tobytes() for a in (keys, values, offsets, output)
        )
        if cached:

            def reject_conversion(*args, **kwargs):
                raise AssertionError("warm bucket replay must not rebuild its native description")

            with monkeypatch.context() as patch:
                patch.setattr(BucketBuilderDefinition, "bind", reject_conversion)
                patch.setattr(NativeOperationBinding, "lower", reject_conversion)
                graph.run({})
                graph.run({})
            assert before == tuple(a.to_numpy().tobytes() for a in (keys, values, offsets, output))
        if not cached:
            workspace = sequence._calls[0].workspace
            assert sequence._calls[0].plan is None
            assert workspace._cursor_ndarray is None
            assert workspace.workspace_bytes_current == 0
        operation = json.loads(plan.to_json())["operations"][0]
        assert operation["operation"] == "bucket_builder"
        parameters = operation["parameters"]
        assert parameters["element_shape"] == ([] if lanes == 1 else [lanes])
        if arch == ti.cpu:
            assert parameters["cpu_max_threads"] == 4
        elif arch == ti.vulkan:
            assert parameters["vulkan_policy"] == dict(
                private_enabled=True,
                private_min_items=0,
                private_max_bins=1024,
                private_max_bytes=1 << 24,
            )
        cases.append(
            dict(
                name=name, prefix=0, capacity=n, offset_capacity=bins + 1, **parameters
            )
        )
        graph.close()
        sequence.clear()
    artifact = tmp_path / "bucket-sequence.tcm"
    module.archive(str(artifact))
    assert not module._exported_kernels
    with zipfile.ZipFile(artifact) as archive:
        assert not any(
            name.endswith((".bc", ".ptx", ".spv", ".cubin"))
            for name in archive.namelist()
        )
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_bucket_builder_probe.py")),
                mode,
                backend,
                str(artifact),
            ],
            env=dict(os.environ, TI_VULKAN_BUCKET_BUILDER_PRIVATE="0"),
            capture_output=True,
            text=True,
            timeout=90,
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "Validation Error" not in text and "SYNC-HAZARD" not in text, text


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_bucket_sequence_rejects_changed_cached_strategy():
    keys, values, output = (ti.ndarray(ti.i32, 17) for _ in range(3))
    offsets = ti.ndarray(ti.i32, 8)
    keys.fill(0)
    sequence = ti.algorithms.primitive_sequence().bucket_builder(
        keys, values, offsets, output
    )
    sequence.run()
    sequence._calls[0].plan.method_name = "unreconstructible"
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="selected cached strategy"):
        ti.aot.Module().add_execution_plan(
            "bad",
            graph,
            source_bindings=dict(
                keys=keys, values=values, offsets=offsets, output=output
            ),
        )
    graph.close()
    sequence.clear()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_bucket_sequence_rejects_tensor_field_strategy():
    keys, offsets = ti.field(ti.i32, 17), ti.field(ti.i32, 8)
    values, output = (ti.Vector.field(3, ti.i32, 17) for _ in range(2))
    sequence = ti.algorithms.primitive_sequence().bucket_builder(
        keys, values, offsets, output
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="scalar Field route"):
        ti.aot.Module().add_execution_plan("bad", graph, snapshot_state=True)
    graph.close()
    sequence.clear()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_bucket_field_snapshot_with_downstream_autodiff(tmp_path):
    n, bins = 17, 7
    keys, values = ti.field(ti.i32, n), ti.field(ti.f32, n)
    offsets = ti.field(ti.i32, bins + 1)
    output = ti.field(ti.f32, n, needs_grad=True)
    loss = ti.field(ti.f32, (), needs_grad=True)
    source_keys = np.arange(n, dtype=np.int32) % (bins + 2) - 1
    source_values = np.arange(n, dtype=np.float32) + 0.25
    keys.from_numpy(source_keys)
    values.from_numpy(source_values)

    @ti.kernel
    def initialize():
        loss[None] = 0
        loss.grad[None] = 1
        for i in output:
            output[i] = -77
            output.grad[i] = 0

    @ti.kernel
    def energy():
        for i in output:
            if i < offsets[bins]:
                loss[None] += output[i] * output[i]

    @ti.kernel
    def publish(
        result: ti.types.ndarray(ti.f32, 1), gradient: ti.types.ndarray(ti.f32, 1)
    ):
        for i in output:
            result[i] = output[i]
            gradient[i] = output.grad[i]

    sequence = ti.algorithms.primitive_sequence().bucket_builder(
        keys, values, offsets, output
    )
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize)
    builder.append_native(sequence)
    builder.dispatch(energy)
    builder.dispatch(energy.grad)
    builder.dispatch(
        publish,
        *(
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
            for name in ("result", "gradient")
        ),
    )
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("snapshot", graph, snapshot_state=True)
    artifact = tmp_path / "bucket-snapshot.tcm"
    module.archive(str(artifact))
    graph.close()
    sequence.clear()
    keys.fill(0)
    values.fill(99)
    accepted = source_values[(source_keys >= 0) & (source_keys < bins)]
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        result, gradient = runtime.ndarray("f32", n), runtime.ndarray("f32", n)
        with loaded.prepare_execution(
            "snapshot", dict(result=result, gradient=gradient)
        ) as call:
            for _ in range(2):
                call()
                actual = np.frombuffer(result.read(), np.float32)
                derivative = np.frombuffer(gradient.read(), np.float32)
                np.testing.assert_array_equal(
                    np.sort(actual[: len(accepted)]), accepted
                )
                np.testing.assert_array_equal(
                    actual[len(accepted) :], np.full(n - len(accepted), -77, np.float32)
                )
                expected = np.zeros(n, np.float32)
                expected[: len(accepted)] = 2 * actual[: len(accepted)]
                np.testing.assert_array_equal(derivative, expected)
        result.close()
        gradient.close()
