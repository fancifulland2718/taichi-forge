"""Cold diagnostic export, device scalar semantics and native reconstruction."""

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
def test_diagnostic_sequence_and_selected_nodes_runtime_only(tmp_path):
    from taichi_forge.algorithms._primitive_definitions import CheckDefinition, MetricDefinition
    from taichi_forge.graph._native_definition import NativeOperationBinding

    arch = ti.lang.impl.current_cfg().arch
    module = ti.aot.Module()
    configurations = [
        ("count_if", "f32", False),
        ("all_if", "i32", True),
        ("nan_count", "f32", False),
        ("inf_count", "f32", False),
        ("all_finite", "f32", True),
        ("index_bounds_check", "u64", False),
        ("max_abs", "f32", False),
        ("max_abs_delta", "f32", True),
    ]
    if arch != ti.vulkan:
        configurations.append(("max_abs_delta", "f64", False))
    cases = []
    for function, dtype, selected in configurations:
        metric = function.startswith("max_abs")
        code = dict(f32=np.float32, f64=np.float64, i32=np.int32, u64=np.uint64)[dtype]
        if dtype.startswith("f"):
            initial = np.array([-0.0, 0, 1, -2, np.nan, np.inf, -np.inf], dtype=code)
        elif dtype == "u64":
            initial = np.array([0, 1, 2, 3, 4, 2, np.iinfo(code).max], dtype=code)
        else:
            initial = np.array([1, 0, 2, 0, 3, 1, -1], dtype=code)
        changed = (
            np.ones(7, dtype=code) if function == "all_if" else np.zeros(7, dtype=code)
        )
        expected_first = {
            "count_if": 5,
            "all_if": 2,
            "nan_count": 1,
            "inf_count": 2,
            "all_finite": 3,
            "index_bounds_check": 2,
            "max_abs": np.inf,
            "max_abs_delta": np.inf,
        }[function]
        values = ti.ndarray(getattr(ti, dtype), 7)
        values.from_numpy(initial)
        other = (
            ti.ndarray(getattr(ti, dtype), 7) if function == "max_abs_delta" else None
        )
        if other is not None:
            other.from_numpy(initial)
        workspace = (
            ti.algorithms.MetricWorkspace()
            if metric
            else ti.algorithms.CheckWorkspace()
        )
        args = (values, other) if other is not None else (values,)
        kwargs = dict(upper=4, lower=-1) if function == "index_bounds_check" else {}
        if selected:
            result = getattr(ti.algorithms, function)(
                *args, workspace=workspace, **kwargs
            )
            output = result.device_scalar
            node = result
        else:
            node = ti.algorithms.primitive_sequence()
            getattr(node, function)(*args, workspace=workspace, **kwargs)
            output = (
                workspace._get_result_ndarray(values.dtype)
                if metric
                else workspace._get_result_i32_ndarray()
            )
        result_type = dtype if metric else "i32"
        result_code = code if metric else np.int32
        output.from_numpy(np.array([37], dtype=result_code))
        builder = ti.graph.GraphBuilder()
        builder.append_native(node)
        graph = builder.compile()
        bindings = dict(values=values, output=output)
        if other is not None:
            bindings["other"] = other
        before = {name: value.to_numpy().tobytes() for name, value in bindings.items()}
        name = function + "_" + dtype
        definition_type = MetricDefinition if metric else CheckDefinition
        bound = []
        original_bind = definition_type.bind

        def observe_binding(definition, *args, **kwargs):
            bound.append(definition)
            return original_bind(definition, *args, **kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(definition_type, "bind", observe_binding)
            plan = module.add_execution_plan(name, graph, source_bindings=bindings)
        assert len(bound) == 1
        if selected:
            assert bound[0] is result._plan.definition
        assert before == {
            name: value.to_numpy().tobytes() for name, value in bindings.items()
        }
        if selected:

            def reject_conversion(*args, **kwargs):
                raise AssertionError("warm diagnostic replay must not rebuild its native description")

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(definition_type, "bind", reject_conversion)
                patch.setattr(NativeOperationBinding, "lower", reject_conversion)
                graph.run({})
                graph.run({})
            assert output.to_numpy()[0] == expected_first
        operation = json.loads(plan.to_json())["operations"][0]
        assert operation["operation"] == ("metric_reduce" if metric else "check_count")
        if function in ("all_if", "all_finite"):
            assert operation["parameters"]["op"] == (
                "zero" if function == "all_if" else "not_finite"
            )
        cases.append(
            dict(
                name=name,
                dtype=dtype,
                result_dtype=result_type,
                other=other is not None,
                initial=np.array([37], dtype=result_code).tobytes().hex(),
                samples=[
                    dict(
                        values=source.tobytes().hex(),
                        expected=np.array([expected], dtype=result_code)
                        .tobytes()
                        .hex(),
                    )
                    for source, expected in ((initial, expected_first), (changed, 0))
                ],
            )
        )
        graph.close()
    # The first scalar is private and feeds another device diagnostic. Export
    # must reconstruct it without requesting an external binding or reading it.
    values = ti.ndarray(ti.i32, 7)
    first_workspace, second_workspace = (
        ti.algorithms.CheckWorkspace(),
        ti.algorithms.CheckWorkspace(),
    )
    private = first_workspace._get_result_i32_ndarray()
    output = second_workspace._get_result_i32_ndarray()
    private.fill(37)
    output.fill(37)
    sequence = ti.algorithms.primitive_sequence().count_if(
        values, workspace=first_workspace
    )
    sequence.count_if(private, workspace=second_workspace)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    plan = module.add_execution_plan(
        "private_chain", graph, source_bindings=dict(values=values, output=output)
    )
    assert private.to_numpy()[0] == output.to_numpy()[0] == 37
    manifest = json.loads(plan.to_json())
    assert len(manifest["operations"]) == 2
    assert any(
        resource["kind"] == "ndarray" and resource["shape"] == [1]
        for resource in manifest["resources"]
    )
    cases.append(
        dict(
            name="private_chain",
            dtype="i32",
            result_dtype="i32",
            other=False,
            initial=np.array([37], np.int32).tobytes().hex(),
            samples=[
                dict(
                    values=np.full(7, value, np.int32).tobytes().hex(),
                    expected=np.array([int(value != 0)], np.int32).tobytes().hex(),
                )
                for value in (2, 0)
            ],
        )
    )
    graph.close()
    artifact = tmp_path / "diagnostic.tcm"
    module.archive(str(artifact))
    assert not module._exported_kernels
    with zipfile.ZipFile(artifact) as archive:
        assert not any(
            name.endswith((".bc", ".spv", ".ptx", ".cubin"))
            for name in archive.namelist()
        )
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_diagnostic_probe.py")),
                mode,
                backend,
                str(artifact),
            ],
            env=dict(
                os.environ,
                TI_VULKAN_CHECK_COUNT_ITEMS_PER_GROUP="256",
                TI_VULKAN_METRIC_REDUCE_ITEMS_PER_GROUP="256",
            ),
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "Validation Error" not in output and "SYNC-HAZARD" not in output, output


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_diagnostic_offset_prefix_empty_and_rebinding(tmp_path):
    module = ti.aot.Module()
    for n in (0, 7):
        module.add_execution_plan(
            str(n),
            ti.aot.ExecutionPlan(
                [
                    dict(
                        kind="native",
                        provider="builtin",
                        operation="check_count",
                        version=1,
                        parameters=dict(size=n, dtype="f32", op="nonzero"),
                        bindings=dict(values="source", output="count"),
                    ),
                    dict(
                        kind="native",
                        provider="builtin",
                        operation="metric_reduce",
                        version=1,
                        parameters=dict(size=n, dtype="f32", op="max_abs"),
                        bindings=dict(values="source", output="magnitude"),
                    ),
                ],
                resources=[
                    dict(
                        kind="ndarray_view",
                        name="source",
                        array="values",
                        offset=1,
                        shape=[7],
                    ),
                    dict(
                        kind="ndarray_view",
                        name="count",
                        array="counts",
                        offset=1,
                        shape=[1],
                    ),
                    dict(
                        kind="ndarray_view",
                        name="magnitude",
                        array="metrics",
                        offset=3,
                        shape=[1],
                    ),
                ],
            ),
        )
    artifact = tmp_path / "prefix.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        values, changed = runtime.ndarray("f32", 9), runtime.ndarray("f32", 9)
        counts, metrics = runtime.ndarray("i32", 5), runtime.ndarray("f32", 5)
        frame = dict(values=values, counts=counts, metrics=metrics)
        reject(lambda: loaded.prepare_execution("7", dict(frame, metrics=values)))
        for n in (0, 7):
            for source, value in ((values, -2), (changed, 0)):
                data = np.full(9, value, np.float32)
                data[[0, -1]] = 99
                source.write(data.tobytes())
                counts.write(np.full(5, 37, np.int32).tobytes())
                metrics.write(np.full(5, 37, np.float32).tobytes())
                with loaded.prepare_execution(
                    str(n), dict(frame, values=source)
                ) as call:
                    assert call.info["workspace_bytes"] == 0
                    call()
                    expected_count = np.full(5, 37, np.int32)
                    expected_count[1] = n if value else 0
                    expected_metric = np.full(5, 37, np.float32)
                    expected_metric[3] = abs(value) if n else 0
                    assert counts.read() == expected_count.tobytes()
                    assert metrics.read() == expected_metric.tobytes()
                    assert source.read() == data.tobytes()
        for array in (values, changed, counts, metrics):
            array.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_diagnostic_cold_export_rejects_changed_strategy():
    values = ti.ndarray(ti.f32, 7)
    sequence = ti.algorithms.primitive_sequence().count_if(values)
    sequence._calls[0].kwargs["method"] = "cuda_device"
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="strategy"):
        module.add_execution_plan("bad", graph, source_bindings=dict(values=values))
    assert not module._content and not module._execution_plans
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_diagnostic_snapshot_results_feed_kernel_conditions(tmp_path):
    values = ti.field(ti.f32, 7)
    values.from_numpy(np.arange(7, dtype=np.float32) - 3)
    count = ti.algorithms.count_if(values)
    magnitude = ti.algorithms.max_abs(values)

    @ti.kernel
    def publish(
        counts: ti.types.ndarray(ti.i32, 1),
        metrics: ti.types.ndarray(ti.f32, 1),
        output: ti.types.ndarray(ti.f32, 1),
    ):
        if counts[0] != 0:
            output[0] = counts[0] + metrics[0]
        else:
            output[0] = -17
        for i in values:
            values[i] = 0

    builder = ti.graph.GraphBuilder()
    builder.append_native(count)
    builder.append_native(magnitude)
    builder.dispatch(
        publish,
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "counts", ti.i32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "metrics", ti.f32, ndim=1),
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1),
    )
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan(
        "step",
        graph,
        snapshot_state=True,
        source_bindings=dict(
            counts=count.device_scalar, metrics=magnitude.device_scalar
        ),
    )
    artifact = tmp_path / "snapshot.tcm"
    module.archive(str(artifact))
    graph.close()
    values.fill(99)
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        frame = dict(
            counts=runtime.ndarray("i32", 1),
            metrics=runtime.ndarray("f32", 1),
            output=runtime.ndarray("f32", 1),
        )
        with loaded.prepare_execution("step", frame) as call:
            for expected in (9, -17):
                call()
                assert (
                    frame["output"].read() == np.array([expected], np.float32).tobytes()
                )
        for value in frame.values():
            value.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_diagnostic_selected_plan_and_layout_rejections():
    values = ti.ndarray(ti.f32, 7)
    result = ti.algorithms.max_abs(values)
    result._plan.n = 8
    builder = ti.graph.GraphBuilder()
    builder.append_native(result)
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="selected cached strategy"):
        module.add_execution_plan("bad", graph, source_bindings=dict(values=values))
    assert not module._content and not module._execution_plans
    graph.close()
    fields = ti.Struct.field({"value": ti.f32, "neighbor": ti.i32}, shape=7)
    fields.value.fill(1)
    sequence = ti.algorithms.primitive_sequence().count_if(fields.value)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="packed"):
        module.add_execution_plan("strided", graph, snapshot_state=True)
    assert not module._content and not module._execution_plans
    graph.close()
