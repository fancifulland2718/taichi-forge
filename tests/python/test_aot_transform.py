"""Native transform exports share numerical cores, bindings and owner lifetime."""

import array
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
from tests.python.aot_scan_probe import CODES, normalize, values_for


def _parameters(dtype, n, scale=3, bias=7):
    from taichi_forge_runtime.aot import Scalar

    return dict(
        size=n,
        dtype=dtype,
        scale_bits=Scalar("f64", scale).resource("scale")["bits"],
        bias_bits=Scalar("f64", bias).resource("bias")["bits"],
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4
)
def test_transform_sequence_runtime_only_and_borrowed(tmp_path):
    arch = ti.lang.impl.current_cfg().arch
    module = ti.aot.Module()
    configurations = []
    for dtype in (ti.i32, ti.u32, ti.i64, ti.u64, ti.f32, ti.f64):
        if arch == ti.vulkan and not ti.lang.impl.get_runtime().prog.vulkan_transform_value_type_available(
            ti.algorithms._algorithms._transform_value_type(dtype)
        ):
            continue
        for inplace in (False, True):
            configurations.append(
                (
                    dtype,
                    4099 if dtype == ti.f32 else 17,
                    inplace,
                    2.5 if dtype in (ti.f32, ti.f64) else 3,
                    -7 if dtype in (ti.i32, ti.i64) else 7,
                )
            )
    configurations.extend(
        (ti.f32, 17, False, scale, bias)
        for scale, bias in ((-0.0, -0.0), (float("nan"), 1.0), (float("inf"), -1.0))
    )
    if arch == ti.cpu:
        configurations.append((ti.f32, 262145, False, 2.5, 7))
    cases = []
    for index, (dtype, n, inplace, scale, bias) in enumerate(configurations):
        src = ti.ndarray(dtype, n)
        dst = src if inplace else ti.ndarray(dtype, n)
        sequence = ti.algorithms.primitive_sequence().transform(
            src, dst, scale=scale, bias=bias
        )
        samples = []
        for iteration in (0, 1):
            host = values_for(str(dtype), n, iteration)
            data = array.array(CODES[str(dtype)], host).tobytes()
            src.from_numpy(np.frombuffer(data, dtype=np.dtype(CODES[str(dtype)])))
            sequence.run()
            expected = dst.to_numpy()
            if scale not in (float("inf"),) and not np.isnan(scale):
                reference = np.array(
                    [normalize(v * scale + bias, str(dtype)) for v in host],
                    dtype=expected.dtype,
                )
                np.testing.assert_array_equal(expected, reference)
            samples.append(dict(input=data.hex(), expected=expected.tobytes().hex()))
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()
        name = f"transform_{index}"
        before = src.to_numpy().tobytes(), dst.to_numpy().tobytes()
        plan = module.add_execution_plan(
            name,
            graph,
            source_bindings=dict(src=src) if inplace else dict(src=src, dst=dst),
        )
        assert before == (src.to_numpy().tobytes(), dst.to_numpy().tobytes())
        (operation,) = json.loads(plan.to_json())["operations"]
        assert operation["operation"] == "transform"
        assert len(operation["parameters"]["scale_bits"]) == 16
        cases.append(
            dict(name=name, dtype=str(dtype), size=n, inplace=inplace, samples=samples)
        )
        graph.close()
        sequence.clear()
    assert not module._exported_kernels
    artifact = tmp_path / "transform.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        assert not any(
            name.endswith((".bc", ".spv", ".ptx", ".cubin"))
            for name in archive.namelist()
        )
    artifact.with_suffix(".json").write_text(json.dumps(cases))
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_transform_probe.py")),
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
def test_transform_field_snapshot_and_explicit_adjoint(tmp_path):
    n = 17
    src, dst = ti.field(ti.f32, n), ti.field(ti.f32, n)
    source = np.arange(n, dtype=np.float32) / 4 - 1.5
    src.from_numpy(source)

    @ti.kernel
    def publish(
        output: ti.types.ndarray(ti.f32, 1), gradient: ti.types.ndarray(ti.f32, 1)
    ):
        for i in range(n):
            output[i] = dst[i] * dst[i]
            output[n + i] = src[i]
            gradient[i] = 2.0 * dst[i] * 2.5  # Explicit chain-rule adjoint.

    sequence = ti.algorithms.primitive_sequence().transform(
        src, dst, scale=2.5, bias=-7
    )
    sequence.run()
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(
        publish,
        *(
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
            for name in ("output", "gradient")
        ),
    )
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("step", graph, snapshot_state=True)
    artifact = tmp_path / "transform-snapshot.tcm"
    module.archive(str(artifact))
    graph.close()
    sequence.clear()
    src.fill(99)
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        output, gradient = runtime.ndarray("f32", n * 2), runtime.ndarray("f32", n)
        with loaded.prepare_execution(
            "step", dict(output=output, gradient=gradient)
        ) as call:
            for _ in range(2):
                call()
                np.testing.assert_array_equal(
                    np.frombuffer(output.read(), np.float32),
                    np.concatenate(((source * 2.5 - 7) ** 2, source)),
                )
                np.testing.assert_array_equal(
                    np.frombuffer(gradient.read(), np.float32), 5 * (source * 2.5 - 7)
                )
        output.close()
        gradient.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_transform_export_rejects_changed_strategy():
    src, dst = ti.ndarray(ti.f32, 17), ti.ndarray(ti.f32, 17)
    sequence = ti.algorithms.primitive_sequence().transform(src, dst)
    sequence.run()
    sequence._calls[0].plan.method_name = "unreconstructible"
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="selected cached strategy"):
        module.add_execution_plan("bad", graph, source_bindings=dict(src=src, dst=dst))
    assert not module._exported_kernels
    graph.close()
    sequence.clear()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_transform_offset_prefix_aliases_and_neighbors(tmp_path):
    from tests.python.aot_scan_probe import reject

    module = ti.aot.Module()
    dtypes = ["i32"]
    if (
        ti.lang.impl.current_cfg().arch != ti.vulkan
        or ti.lang.impl.get_runtime().prog.vulkan_transform_value_type_available(4)
    ):
        dtypes.append("i64")
    for dtype in dtypes:
        for n in (17, 4099):
            for overlap in (False, True):
                module.add_execution_plan(
                    f"{dtype}_{n}_{overlap}",
                    ti.aot.ExecutionPlan(
                        [
                            dict(
                                kind="native",
                                provider="builtin",
                                operation="transform",
                                version=1,
                                parameters=_parameters(dtype, n),
                                bindings=dict(src="source", dst="target"),
                            )
                        ],
                        resources=[
                            dict(
                                kind="ndarray_view",
                                name="source",
                                array="src",
                                offset=1,
                                shape=[n],
                            ),
                            dict(
                                kind="ndarray_view",
                                name="target",
                                array="dst",
                                offset=2 if overlap else 1,
                                shape=[n],
                            ),
                        ],
                    ),
                )
    artifact = tmp_path / "transform-views.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        for dtype in dtypes:
            for n in (17, 4099):
                src, dst = runtime.ndarray(dtype, n + 3), runtime.ndarray(dtype, n + 3)
                before = array.array(CODES[dtype], [91] + [2] * n + [92, 93]).tobytes()
                src.write(before)
                reject(
                    lambda: loaded.prepare_execution(
                        f"{dtype}_{n}_True", dict(src=src, dst=src)
                    )
                )
                assert src.read() == before
                for inplace in (False, True):
                    output = src if inplace else dst
                    with loaded.prepare_execution(
                        f"{dtype}_{n}_False", dict(src=src, dst=output)
                    ) as call:
                        for value in (2, 5):
                            data = array.array(
                                CODES[dtype], [91] + [value] * n + [92, 93]
                            ).tobytes()
                            src.write(data)
                            if not inplace:
                                dst.write(array.array(CODES[dtype], [37] * (n + 3)))
                            call()
                            expected = (
                                ([91] if inplace else [37])
                                + [value * 3 + 7] * n
                                + ([92, 93] if inplace else [37, 37])
                            )
                            assert (
                                output.read()
                                == array.array(CODES[dtype], expected).tobytes()
                            )
                            if not inplace:
                                assert src.read() == data
                src.close()
                dst.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_transform_invalid_integer_coefficients_do_not_write(tmp_path):
    from tests.python.aot_scan_probe import reject

    module = ti.aot.Module()
    for i, scale in enumerate((float("nan"), float("inf"), 0.5, 2147483648.0)):
        module.add_execution_plan(
            str(i),
            ti.aot.ExecutionPlan(
                [
                    dict(
                        kind="native",
                        provider="builtin",
                        operation="transform",
                        version=1,
                        parameters=_parameters("i32", 7, scale),
                        bindings=dict(src="src", dst="dst"),
                    )
                ]
            ),
        )
    artifact = tmp_path / "invalid-transform.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        src, dst = runtime.ndarray("i32", 7), runtime.ndarray("i32", 7)
        sentinel = array.array("i", [91] * 7).tobytes()
        dst.write(sentinel)
        for i in range(4):
            reject(lambda: loaded.prepare_execution(str(i), dict(src=src, dst=dst)))
            assert dst.read() == sentinel
        src.close()
        dst.close()
