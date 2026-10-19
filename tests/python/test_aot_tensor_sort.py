"""Single-key native sort retains whole tensor payload bits in AOT."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


def graph(sequence):
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_tensor_sort_sequence_and_explicit_aot(tmp_path, monkeypatch):
    arch = (
        "cpu"
        if ti.cfg.arch == ti.cpu
        else "cuda" if ti.cfg.arch == ti.cuda else "vulkan"
    )
    n = 17
    module = ti.aot.Module()
    cases = []
    specs = (
        (ti.f32, np.float32, (3,)),
        (ti.f64, np.float64, (2, 2)),
        (ti.i64, np.int64, (2,)),
        (ti.u64, np.uint64, (1, 3)),
        (ti.i32, np.int32, (3,)),
        (ti.u32, np.uint32, (2, 2)),
    )
    for index, (dtype, npdtype, shape) in enumerate(specs):
        key_dtype, key_np = (
            (ti.i32, np.int32),
            (ti.f32, np.float32),
            (ti.u64, np.uint64),
        )[index % 3]
        keys = ti.ndarray(key_dtype, n)
        values = (
            ti.Vector.ndarray(shape[0], dtype, n)
            if len(shape) == 1
            else ti.Matrix.ndarray(*shape, dtype, n)
        )
        sequence = ti.algorithms.primitive_sequence()
        if index % 2:
            sequence.sort_by_key([keys], values)
        else:
            sequence.sort(keys, values)
        executable = graph(sequence)

        def forbid(*args, **kwargs):
            raise AssertionError("Export must not prewarm tensor sorting")

        with monkeypatch.context() as patch:
            patch.setattr(sequence, "prewarm", forbid)
            plan = module.add_execution_plan(
                f"cold{index}",
                executable,
                source_bindings=dict(keys=keys, values=values),
            )
        assert json.loads(plan.to_json())["operations"] == [
            dict(kind="sort", keys="keys", values="values")
        ]
        assert not module._exported_kernels
        samples = []
        for shift in (0, 4):
            host_keys = np.asarray(
                [(i * 7 + shift) % 5 for i in range(n)], dtype=key_np
            )
            if key_dtype == ti.u64:
                host_keys += np.uint64(1 << 60)
            if key_dtype == ti.f32:
                host_keys[:5] = [np.nan, -0.0, 0.0, np.inf, -np.inf]
            host_values = (
                np.arange(n * np.prod(shape), dtype=npdtype).reshape((n, *shape))
                + shift
            )
            if dtype in (ti.f32, ti.f64):
                bits = host_values.view(
                    np.uint32 if dtype == ti.f32 else np.uint64
                ).reshape(-1)
                patterns = (
                    [0x80000000, 0, 0x7FC00017, 0xFFC00023, 0x7F800000, 0xFF800000]
                    if dtype == ti.f32
                    else [
                        0x8000000000000000,
                        0,
                        0x7FF8000000000017,
                        0xFFF8000000000023,
                        0x7FF0000000000000,
                        0xFFF0000000000000,
                    ]
                )
                bits[: len(patterns)] = patterns
            elif dtype in (ti.i64, ti.u64):
                host_values += npdtype(1 << 60)
            if index == 4:
                host_values[:, 0] = np.arange(n)
            order = np.argsort(host_keys, kind="stable")
            keys.from_numpy(host_keys)
            values.from_numpy(host_values)
            sequence.run()
            assert keys.to_numpy().tobytes() == host_keys[order].tobytes()
            assert values.to_numpy().tobytes() == host_values[order].tobytes()
            samples.append(
                dict(
                    keys=host_keys.tobytes().hex(),
                    values=host_values.tobytes().hex(),
                    sorted_keys=host_keys[order].tobytes().hex(),
                    sorted_values=host_values[order].tobytes().hex(),
                    order=order.tolist(),
                )
            )
        before = values.to_numpy().tobytes()
        module.add_execution_plan(
            f"warm{index}", executable, source_bindings=dict(keys=keys, values=values)
        )
        assert values.to_numpy().tobytes() == before
        cases.append(
            dict(
                index=index,
                key_dtype=str(key_dtype),
                dtype=str(dtype),
                shape=shape,
                n=n,
                samples=samples,
            )
        )
        executable.close()
        sequence.clear()
    assert not module._exported_kernels
    module.add_primitive("explicit", "sort", values="values")

    @ti.kernel
    def seed(
        dx: ti.types.ndarray(ti.f32, 1),
        loss: ti.types.ndarray(ti.f32, 1),
        dl: ti.types.ndarray(ti.f32, 1),
    ):
        for i in dx:
            dx[i] = 0
        loss[0], dl[0] = 0, 1

    @ti.kernel
    def energy(
        order: ti.types.ndarray(ti.types.vector(3, ti.i32), 1),
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in order:
            j = order[i][0]
            loss[0] += (i + 1) * x[j] * x[j]

    module.add_kernel(seed)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    module.add_execution_plan(
        "gradient",
        ti.aot.ExecutionPlan(
            [
                dict(kind="sort", keys="keys", values="values"),
                dict(kind="kernel", entry="seed", arguments=["dx", "loss", "dl"]),
                dict(kind="kernel", entry="energy", arguments=["values", "x", "loss"]),
                dict(
                    kind="kernel", entry="backward", arguments=["values", "x", "loss"]
                ),
            ],
            gradients=dict(x="dx", loss="dl"),
        ),
    )
    artifact = tmp_path / "tensor-sort.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf-8")
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_tensor_sort_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_tensor_sort_does_not_export_vector_keys_or_field_strategy():
    n = 5
    module = ti.aot.Module()
    for keys, values in (
        (ti.Vector.ndarray(2, ti.i32, n), ti.ndarray(ti.i32, n)),
        (ti.field(ti.i32, n), ti.Vector.ndarray(2, ti.f32, n)),
    ):
        sequence = ti.algorithms.primitive_sequence().sort(keys, values)
        executable = graph(sequence)
        with pytest.raises(ti.TaichiRuntimeError, match="scalar.*keys"):
            module.add_execution_plan(
                "bad",
                executable,
                source_bindings=dict(values=values),
                snapshot_state=True,
            )
        assert not module._content and not module._fields and not module._array_fields
        executable.close()
        sequence.clear()
