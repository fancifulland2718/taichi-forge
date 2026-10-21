"""Shared native scan export and runtime-only lifetime/gradient contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scan_definition_is_shared_and_replay_stays_direct(monkeypatch):
    from taichi_forge.algorithms import _algorithms as algorithms
    from taichi_forge.algorithms._primitive_definitions import ScanDefinition
    from taichi_forge.graph._native_definition import NativeDefinition

    values = ti.ndarray(ti.i32, 17)
    values.fill(1)
    executor = ti.algorithms.PrefixSumExecutor(17)
    executor.run(values)
    definition = executor._native_scan_plan.definition
    sequence = ti.algorithms.primitive_sequence().scan(values, executor=executor)
    sequence.run()
    assert sequence._calls[0].plan.definition is definition

    def no_conversion(*args, **kwargs):
        raise AssertionError("warm native scan must not construct or serialize definitions")

    with monkeypatch.context() as patch:
        patch.setattr(algorithms, "ScanDefinition", no_conversion)
        patch.setattr(ScanDefinition, "bind", no_conversion)
        patch.setattr(NativeDefinition, "parameter_dict", no_conversion)
        executor.run(values)
        sequence.run()
    expected = np.ones(17, dtype=np.int32)
    for _ in range(4):
        expected = np.cumsum(expected, dtype=np.int32)
    np.testing.assert_array_equal(values.to_numpy(), expected)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    definitions = []
    bind = ScanDefinition.bind

    def record_bind(selected, array):
        definitions.append(selected)
        return bind(selected, array)

    with monkeypatch.context() as patch:
        patch.setattr(ScanDefinition, "bind", record_bind)
        plan = module.add_execution_plan("scan", graph, source_bindings=dict(values=values))
    assert len(definitions) == 1 and definitions[0] is definition
    assert json.loads(plan.to_json())["operations"][0]["parameters"] == dict(size=17, dtype="i32")
    np.testing.assert_array_equal(values.to_numpy(), expected)
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scan_sequence_and_native_prefix_export(tmp_path):
    module = ti.aot.Module()
    cases = []
    for dtype, size in (
        (ti.i32, 17),
        (ti.u32, 4099),
        (ti.i64, 257),
        (ti.u64, 257),
        (ti.f32, 4099),
        (ti.f64, 257),
    ):
        if (
            ti.lang.impl.current_cfg().arch == ti.vulkan
            and not ti.lang.impl.get_runtime().prog.vulkan_scan_value_type_available(
                ti.algorithms._algorithms._SCAN_VALUE_TYPE[dtype]
            )
        ):
            continue
        values = ti.ndarray(dtype, size)
        values.fill(1)
        sequence = ti.algorithms.primitive_sequence().scan(values)
        if dtype == ti.f32:
            sequence.run()  # Qualify a cached native selection as well.
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()
        before = values.to_numpy().tobytes()
        name = f"scan_{dtype}"
        plan = module.add_execution_plan(
            name, graph, source_bindings=dict(values=values)
        )
        assert values.to_numpy().tobytes() == before
        assert [op["operation"] for op in json.loads(plan.to_json())["operations"]] == [
            "scan"
        ]
        cases.append(dict(name=name, dtype=str(dtype), size=size, capacity=size))
        graph.close()
        sequence.clear()
    for size in (0, 1, 31):
        name = f"prefix_{size}"
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(
                [
                    dict(
                        kind="native",
                        provider="builtin",
                        operation="scan",
                        version=1,
                        parameters=dict(size=size, dtype="i32"),
                        bindings=dict(values="values"),
                    )
                ]
            ),
        )
        cases.append(dict(name=name, dtype="i32", size=size, capacity=size + 7))
    assert not module._exported_kernels  # Built-in algorithms add no user IR.
    artifact = tmp_path / "scan.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_scan_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_scan_export_rejects_changed_executor_and_strided_input():
    values = ti.ndarray(ti.i32, 7)
    sequence = ti.algorithms.primitive_sequence().scan(values)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    sequence._calls[0].workspace.sorting_length = 6
    with pytest.raises(ti.TaichiRuntimeError, match="length"):
        module.add_execution_plan("bad", graph, source_bindings=dict(values=values))
    assert not module._content and not module._execution_plans
    sequence._calls[0].workspace.sorting_length = 7
    module.add_execution_plan("good", graph, source_bindings=dict(values=values))
    graph.close()
    matrix = ti.Vector.field(2, ti.i32, shape=7)
    view = matrix.get_scalar_field(0)
    builder = ti.graph.GraphBuilder()
    builder.append_native(ti.algorithms.primitive_sequence().scan(view))
    graph = builder.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="packed"):
        module.add_execution_plan("strided", graph, snapshot_state=True)
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_scan_snapshot_with_explicit_adjoint(tmp_path):
    n = 17
    values = ti.field(ti.f32, n)
    host = np.arange(n, dtype=np.float32) / 4 - 2
    values.from_numpy(host)

    @ti.kernel
    def read_and_reverse(
        output: ti.types.ndarray(ti.f32, 1), gradient: ti.types.ndarray(ti.f32, 1)
    ):
        for i in range(n):
            output[i] = values[i] * values[i]
            suffix = 0.0
            for j in range(i, n):
                suffix += 2 * values[j]
            gradient[i] = suffix

    builder = ti.graph.GraphBuilder()
    builder.append_native(ti.algorithms.primitive_sequence().scan(values))
    builder.dispatch(
        read_and_reverse,
        *(
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, s, ti.f32, ndim=1)
            for s in ("output", "gradient")
        ),
    )
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("step", graph, snapshot_state=True)
    graph.close()
    artifact = tmp_path / "scan-adjoint.tcm"
    module.archive(str(artifact))
    values.fill(-99)  # Consumer owns the exported initial state.
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        frame = {s: runtime.ndarray("f32", n) for s in ("output", "gradient")}
        with loaded.prepare_execution("step", frame) as call:
            for _ in range(2):
                host = np.cumsum(host)
                call()
                np.testing.assert_array_equal(
                    np.frombuffer(frame["output"].read(), np.float32), host * host
                )
                np.testing.assert_array_equal(
                    np.frombuffer(frame["gradient"].read(), np.float32),
                    np.cumsum((2 * host)[::-1])[::-1],
                )
