"""Tensor affine closures preserve each backend's selected native execution."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


def consume(artifact, arch, kind="transform"):
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, *(["-S"] if mode == "standalone" else []),
             str(Path(__file__).with_name("aot_tensor_transform_probe.py")), mode, arch, str(artifact), kind],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90,
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "Validation Error" not in text and "SYNC-HAZARD" not in text, text


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_tensor_transform_cold_warm_and_runtime_only(tmp_path):
    arch = ti.lang.impl.current_cfg().arch
    module = ti.aot.Module()
    cases = []
    configs = [(dtype, (2, 3) if i % 2 else (3,), 277 if dtype == ti.f32 else 17, bool(i % 2))
               for i, dtype in enumerate((ti.i32, ti.u32, ti.i64, ti.u64, ti.f32, ti.f64))]
    configs.append((ti.u32, (1,), 7, False))
    if arch == ti.cpu:
        configs.append((ti.f32, (3,), 21849, False))  # Exercise the actual scalar-work partition boundary.
    for dtype, shape, size, inplace in configs:
        def allocate():
            return ti.Vector.ndarray(shape[0], dtype, size) if len(shape) == 1 else ti.Matrix.ndarray(*shape, dtype, size)

        src = allocate()
        dst = src if inplace else allocate()
        sources = dict(src=src) if inplace else dict(src=src, dst=dst)
        sequence = ti.algorithms.primitive_sequence().transform(src, dst, scale=3, bias=7)
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()
        name = f"tensor_{len(cases)}"
        src.fill(37)
        dst.fill(37)
        plan = module.add_execution_plan(name, graph, source_bindings=sources)
        np.testing.assert_array_equal(dst.to_numpy(), np.full((size, *shape), 37))
        document = json.loads(plan.to_json())
        assert len(document["operations"]) == 1
        assert document["operations"][0]["parameters"]["element_shape"] == list(shape)
        assert not document.get("resources") and not module._exported_kernels
        scalar = np.dtype(str(dtype).replace("i", "int").replace("u", "uint").replace("f", "float")).type
        samples = []
        for iteration in (1, 0, 2):
            host = ((np.arange(size * np.prod(shape)) % 11) * iteration).astype(scalar).reshape(size, *shape)
            if np.issubdtype(scalar, np.integer) and iteration:
                host.flat[:3] = [np.iinfo(scalar).max, np.iinfo(scalar).max - 1, 0]
            elif iteration == 2:
                host.flat[:3] = [-0.0, np.nan, np.inf]
            with np.errstate(over="ignore", invalid="ignore"):
                expected = host * scalar(3) + scalar(7)
            src.from_numpy(host)
            sequence.run()
            np.testing.assert_equal(dst.to_numpy(), expected)
            samples.append(dict(input=host.tobytes().hex(), expected=expected.tobytes().hex()))
        before = dst.to_numpy()
        module.add_execution_plan(name + "_warm", graph, source_bindings=sources)
        np.testing.assert_equal(dst.to_numpy(), before)
        cases.append(dict(name=name, dtype=str(dtype), size=size, shape=shape, inplace=inplace, samples=samples))
        graph.close()
        sequence.clear()
    artifact = tmp_path / "tensor-transform.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf8")
    consume(artifact, "cpu" if arch == ti.cpu else str(arch).split(".")[-1])


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_tensor_transform_export_rejects_changed_cached_strategy():
    src, dst = (ti.Vector.ndarray(3, ti.f32, 7) for _ in range(2))
    sequence = ti.algorithms.primitive_sequence().transform(src, dst)
    src.fill(2)
    sequence.run()
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()
    sequence._calls[0].plan.method_name = "cpu_transform_affine_ndarray"
    with pytest.raises(ti.TaichiRuntimeError, match="selected cached strategy"):
        ti.aot.Module().add_execution_plan("bad", graph, source_bindings=dict(src=src, dst=dst))
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_tensor_transform_downstream_gradient(tmp_path):
    vec = ti.types.vector(3, ti.f32)

    @ti.kernel
    def seed(dx: ti.types.ndarray(vec, 1), loss: ti.types.ndarray(ti.f32, 1),
             dl: ti.types.ndarray(ti.f32, 1)):
        for i in dx:
            dx[i] = ti.Vector.zero(ti.f32, 3)
        loss[0], dl[0] = 0, 1

    @ti.kernel
    def energy(transformed: ti.types.ndarray(vec, 1), x: ti.types.ndarray(vec, 1, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in x:
            loss[0] += (transformed[i] * x[i] * x[i]).sum()

    src, dst = (ti.Vector.ndarray(3, ti.f32, 7) for _ in range(2))
    sequence = ti.algorithms.primitive_sequence().transform(src, dst, scale=3, bias=7)
    arrays = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, vec, ndim=1) for name in ("dst", "x", "dx")}
    arrays.update({name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("loss", "dl")})
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(seed, arrays["dx"], arrays["loss"], arrays["dl"])
    args = tuple(arrays[name] for name in ("dst", "x", "loss"))
    builder.dispatch(energy, *args)
    builder.dispatch(energy.grad, *args)
    graph = builder.compile()
    module = ti.aot.Module()
    module.add_execution_plan("gradient", graph, source_bindings=dict(src=src, dst=dst), gradients=dict(x="dx", loss="dl"))
    artifact = tmp_path / "tensor-gradient.tcm"
    module.archive(str(artifact))
    graph.close()
    arch = ti.lang.impl.current_cfg().arch
    consume(artifact, "cpu" if arch == ti.cpu else str(arch).split(".")[-1], "gradient")
