"""Ordinary reduction exports share JIT algorithms and retained native scratch."""

import array
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_scan_probe import CODES, values_for


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4
)
def test_reduce_sequence_runtime_only_and_borrowed(tmp_path):
    module = ti.aot.Module()
    arch = ti.lang.impl.current_cfg().arch
    cases = []
    configurations = [
        (dtype, 4099 if dtype in (ti.i32, ti.f32) else 17)
        for dtype in (ti.i32, ti.u32, ti.i64, ti.u64, ti.f32, ti.f64)
    ]
    if arch == ti.cpu:
        configurations.append((ti.f32, 262145))
    for dtype, n in configurations:
        if (
            arch == ti.vulkan
            and not ti.lang.impl.get_runtime().prog.vulkan_reduce_value_type_available(
                ti.algorithms._algorithms._REDUCE_VALUE_TYPE[dtype]
            )
        ):
            continue
        values, output = ti.ndarray(dtype, n), ti.ndarray(dtype, 3)
        for op in ("sum", "min", "max"):
            if n == 262145 and op != "sum":
                continue
            samples = []
            sequence = ti.algorithms.primitive_sequence().reduce(values, output, op=op)
            for iteration in (0, 1):
                host = values_for(str(dtype), n, iteration)
                if n == 262145:
                    host = [0] * n
                    for part in range(4):
                        start, end = n * part // 4, n * (part + 1) // 4
                        host[start : start + 2] = [1e20, 1]
                        host[end - 2 : end] = [-1e20, iteration + 1]
                elif str(dtype).startswith("f") and op != "sum":
                    host[:5] = [float("nan"), -0.0, 0.0, float("inf"), -float("inf")]
                    if iteration:
                        host = [float("nan")] + [
                            -0.0 if i % 2 else 0.0 for i in range(n - 1)
                        ]
                data = array.array(CODES[str(dtype)], host).tobytes()
                values.from_numpy(
                    np.frombuffer(data, dtype=np.dtype(CODES[str(dtype)]))
                )
                output.from_numpy(np.array([37, 38, 39], dtype=values.to_numpy().dtype))
                sequence.run()
                if n == 262145:
                    assert output.to_numpy()[0] == 4 * (iteration + 1)
                samples.append(
                    dict(input=data.hex(), expected=output.to_numpy().tobytes().hex())
                )
            builder = ti.graph.GraphBuilder()
            builder.append_native(sequence)
            graph = builder.compile()
            name = f"reduce_{dtype}_{op}_{n}"
            before = values.to_numpy().tobytes(), output.to_numpy().tobytes()
            plan = module.add_execution_plan(
                name, graph, source_bindings=dict(values=values, output=output)
            )
            assert before == (values.to_numpy().tobytes(), output.to_numpy().tobytes())
            entry = json.loads(plan.to_json())["operations"][0]
            assert entry["operation"] == "reduce"
            if arch == ti.cpu:
                assert entry["parameters"]["cpu_max_threads"] == 4
            cases.append(dict(name=name, dtype=str(dtype), size=n, samples=samples))
            graph.close()
            sequence.clear()
    for name, size in (("prefix", 7),):
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(
                [
                    dict(
                        kind="native",
                        provider="builtin",
                        operation="reduce",
                        version=1,
                        parameters=dict(size=size, dtype="f32", op="sum"),
                        bindings=dict(values="values", output="output"),
                    )
                ]
            ),
        )
    assert not module._exported_kernels
    artifact = tmp_path / "reduce.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_reduce_probe.py")),
                mode,
                backend,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_reduce_snapshot_with_explicit_adjoint(tmp_path):
    n = 17
    values, total = ti.field(ti.f32, n), ti.field(ti.f32, ())
    host = np.arange(n, dtype=np.float32) / 4 - 1.5
    values.from_numpy(host)

    @ti.kernel
    def publish_and_reverse(
        output: ti.types.ndarray(ti.f32, 1), gradient: ti.types.ndarray(ti.f32, 1)
    ):
        output[0] = total[None] * total[None]
        for i in range(n):
            gradient[i] = 2 * total[None]

    sequence = ti.algorithms.primitive_sequence().reduce(values, total)
    sequence.run()  # Also qualify the shared JIT path on scalar field offsets.
    assert total[None] == np.sum(host)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(
        publish_and_reverse,
        *(
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
            for name in ("output", "gradient")
        ),
    )
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("step", graph, snapshot_state=True)
    graph.close()
    artifact = tmp_path / "reduce-adjoint.tcm"
    module.archive(str(artifact))
    values.fill(-99)
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        frame = dict(
            output=runtime.ndarray("f32", 1), gradient=runtime.ndarray("f32", n)
        )
        with loaded.prepare_execution("step", frame) as call:
            for _ in range(2):
                call()
                np.testing.assert_array_equal(
                    np.frombuffer(frame["output"].read(), np.float32),
                    [np.sum(host) ** 2],
                )
                np.testing.assert_array_equal(
                    np.frombuffer(frame["gradient"].read(), np.float32),
                    np.full(n, 2 * np.sum(host)),
                )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_reduce_export_rejects_changed_strategy():
    values, output = ti.ndarray(ti.f32, 7), ti.ndarray(ti.f32, 1)
    sequence = ti.algorithms.primitive_sequence().reduce(values, output)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    sequence._calls[0].kwargs["method"] = "cuda_device"
    with pytest.raises(ti.TaichiRuntimeError, match="strategy"):
        module.add_execution_plan(
            "bad", graph, source_bindings=dict(values=values, output=output)
        )
    assert not module._content and not module._execution_plans
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_reduce_offset_views_preserve_neighbors(tmp_path):
    module = ti.aot.Module()
    dtypes = ("i32", "f32", "i64", "f64")
    if ti.lang.impl.current_cfg().arch == ti.vulkan:
        prog = ti.lang.impl.get_runtime().prog
        dtypes = tuple(dtype for dtype in dtypes if prog.vulkan_reduce_value_type_available(
            ti.algorithms._algorithms._REDUCE_VALUE_TYPE[getattr(ti, dtype)]))
    for dtype in dtypes:
        for n in (17, 4099):
            module.add_execution_plan(
                f"{dtype}_{n}",
                ti.aot.ExecutionPlan(
                    [
                        dict(
                            kind="native",
                            provider="builtin",
                            operation="reduce",
                            version=1,
                            parameters=dict(size=n, dtype=dtype, op=op),
                            bindings=dict(values="source", output=name),
                        )
                        for name, op in (("sum_out", "sum"), ("max_out", "max"))
                    ],
                    resources=[
                        dict(
                            kind="ndarray_view",
                            name="source",
                            array="values",
                            offset=1,
                            shape=[n],
                        ),
                        dict(
                            kind="ndarray_view",
                            name="sum_out",
                            array="output",
                            offset=1,
                            shape=[1],
                        ),
                        dict(
                            kind="ndarray_view",
                            name="max_out",
                            array="output",
                            offset=3,
                            shape=[1],
                        ),
                    ],
                ),
            )
    artifact = tmp_path / "reduce-offset.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        for dtype in dtypes:
            code = CODES[dtype]
            for n in (17, 4099):
                values, output = runtime.ndarray(dtype, n + 2), runtime.ndarray(
                    dtype, 5
                )
                with loaded.prepare_execution(
                    f"{dtype}_{n}", dict(values=values, output=output)
                ) as call:
                    for value in (1, 2):
                        host = array.array(code, [99] + [value] * n + [98]).tobytes()
                        values.write(host)
                        output.write(array.array(code, [-3] * 5))
                        call()
                        assert (
                            output.read()
                            == array.array(
                                code, [-3, n * value, -3, value, -3]
                            ).tobytes()
                        )
                        assert values.read() == host
                values.close()
                output.close()
