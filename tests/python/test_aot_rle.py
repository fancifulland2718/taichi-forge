"""RLE closure shares one prefix and deduplicates its existing helper kernels."""

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
def test_rle_sequence_shared_prefix_runtime_only(tmp_path):
    arch = ti.lang.impl.current_cfg().arch
    wide = (
        arch != ti.vulkan
        or ti.lang.impl.get_runtime().prog.vulkan_scan_value_type_available(3)
    )
    module = ti.aot.Module(
        caps=["spirv_has_int64"] if arch == ti.vulkan and wide else None
    )
    cases, dtypes = [], []
    for dtype, capacity in ((ti.i32, 4099), (ti.u32, 17), (ti.i64, 257), (ti.u64, 17)):
        if dtype in (ti.i64, ti.u64) and not wide:
            continue
        dtypes.append(dtype)
        arrays = dict(
            keys=ti.ndarray(dtype, capacity),
            output=ti.ndarray(dtype, capacity + 3),
            lengths=ti.ndarray(ti.i32, capacity + 2),
            count=ti.ndarray(ti.i32, 2),
        )
        arrays["keys"].fill(1)
        for size in (0, 7, capacity):
            sequence = ti.algorithms.primitive_sequence().run_length_encode(
                arrays["keys"],
                arrays["output"],
                arrays["lengths"],
                arrays["count"],
                size=size,
            )
            if size == capacity:
                sequence.run()  # RLE must also export after normal prewarming.
            builder = ti.graph.GraphBuilder()
            builder.append_native(sequence)
            graph = builder.compile()
            before = {
                name: value.to_numpy().tobytes() for name, value in arrays.items()
            }
            name = f"{dtype}_{size}"
            plan = module.add_execution_plan(
                name,
                graph,
                source_bindings=dict(count=arrays["count"]) if size == 0 else arrays,
            )
            assert before == {
                name: value.to_numpy().tobytes() for name, value in arrays.items()
            }
            operations = json.loads(plan.to_json())["operations"]
            if size:
                assert [op["kind"] for op in operations] == [
                    "kernel",
                    "compact",
                    "kernel",
                ]
                assert len(operations[1]["columns"]) == 2
            else:
                assert [op["kind"] for op in operations] == ["fill32"]
            cases.append(
                dict(
                    name=name,
                    dtype=str(dtype),
                    capacity=capacity,
                    size=size,
                    arguments=list(plan.argument_names),
                )
            )
            graph.close()
            sequence.clear()
    # One heads/starts specialization per key type; one shared length helper.
    assert len(module._exported_kernels) == len(dtypes) + 1
    artifact = tmp_path / "rle.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_rle_probe.py")),
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
def test_rle_field_snapshot_and_downstream_explicit_gradient(tmp_path):
    n = 17
    keys, unique, lengths = (ti.field(ti.i32, n) for _ in range(3))
    count = ti.field(ti.i32, ())
    host = np.arange(n, dtype=np.int32) // 3
    keys.from_numpy(host)

    @ti.kernel
    def publish(gain: ti.types.ndarray(ti.f32, 1), result: ti.types.ndarray(ti.f32, 1)):
        total = 0
        for run in range(count[None]):
            total += lengths[run] * lengths[run]
        result[0] = gain[0] * total
        result[1] = total  # Explicit d(result[0])/d(gain[0]); keys are discrete.

    builder = ti.graph.GraphBuilder()
    builder.append_native(
        ti.algorithms.primitive_sequence().run_length_encode(
            keys, unique, lengths, count
        )
    )
    builder.dispatch(
        publish,
        *(
            ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
            for name in ("gain", "result")
        ),
    )
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("step", graph, snapshot_state=True)
    graph.close()
    artifact = tmp_path / "rle-gradient.tcm"
    module.archive(str(artifact))
    keys.fill(99)
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        arrays = dict(gain=runtime.ndarray("f32", 1), result=runtime.ndarray("f32", 2))
        with loaded.prepare_execution("step", arrays) as call:
            for gain in (2, -3):
                arrays["gain"].write(np.array([gain], np.float32).tobytes())
                call()
                np.testing.assert_array_equal(
                    np.frombuffer(arrays["result"].read(), np.float32), [gain * 49, 49]
                )


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_rle_export_validation_is_cold():
    keys, output, count = (
        ti.ndarray(ti.i32, 7),
        ti.ndarray(ti.i32, 7),
        ti.ndarray(ti.i32, 1),
    )
    lengths = ti.ndarray(ti.f32, 7)
    output.fill(91)
    sequence = ti.algorithms.primitive_sequence().run_length_encode(
        keys, output, lengths, count
    )
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence, prewarm=False)
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises((TypeError, ti.TaichiRuntimeError), match="i32"):
        module.add_execution_plan(
            "bad",
            graph,
            source_bindings=dict(
                keys=keys, output=output, lengths=lengths, count=count
            ),
        )
    assert not module._exported_kernels and not module._execution_plans
    np.testing.assert_array_equal(output.to_numpy(), np.full(7, 91))
    graph.close()
