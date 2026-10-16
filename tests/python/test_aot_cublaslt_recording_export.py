"""Ordinary and captured matmul recordings reconstruct one shared frozen plan."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("capture", (False, True))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cublaslt_recording_export_without_source_plan(tmp_path, capture):
    from taichi_forge.hardware._aot_cublaslt import frozen_parameters
    from taichi_forge.hardware._cublaslt import CublasLtProvider
    from taichi_forge.hardware._cublaslt_algorithms import _MatmulRecipePlan

    adapter = os.environ.get("TI_CUBLASLT_TEST_ADAPTER")
    vendor = os.environ.get("TI_FORGE_TEST_CUBLASLT_LIBRARY_PATH")
    if not adapter or not vendor:
        pytest.skip("set real cuBLASLt adapter/vendor paths")
    provider = CublasLtProvider(vendor)
    m, n, k, batch = (65, 49, 97, 2) if capture else (17, 9, 13, 1)
    semantics = dict(
        m=m,
        n=n,
        k=k,
        batch_count=batch,
        transpose_a=True,
        transpose_b=True,
        alpha=0.75,
        beta=0.25,
        a="a",
        b="b",
        output="output",
    )
    if capture:
        description = _MatmulRecipePlan(
            provider,
            semantics,
            layout="transposed_column_major",
            epilogue="relu",
            workspace_limit_bytes=32 << 20,
            preparation_only=True,
            shortlist_size=33,
        )
        with pytest.raises(RuntimeError, match="materialized"):
            frozen_parameters(description)
        # Exercise private workspace reconstruction when this device exposes it.
        choice = max(description.choices, key=lambda item: item.workspace_bytes)
        description.close()
        plan = _MatmulRecipePlan(
            provider,
            semantics,
            layout="transposed_column_major",
            epilogue="relu",
            workspace_limit_bytes=32 << 20,
            choice=choice,
        )
        recording = plan._capture()
    else:
        plan = provider.plan(**semantics)
        recording = plan
    dimensions = len(plan.output_shape)

    @ti.kernel
    def initialize(seed: ti.types.ndarray(ti.f32, dimensions), output: ti.types.ndarray(ti.f32, dimensions)):
        for index in ti.grouped(seed):
            output[index] = 2 * seed[index]

    @ti.kernel
    def publish(output: ti.types.ndarray(ti.f32, dimensions), result: ti.types.ndarray(ti.f32, dimensions)):
        for index in ti.grouped(output):
            result[index] = 3 * output[index]

    shapes = dict(
        a=plan.a_shape, b=plan.b_shape, seed=plan.output_shape, output=plan.output_shape, result=plan.output_shape
    )
    arrays = {key: ti.ndarray(ti.f32, shape) for key, shape in shapes.items()}
    host = {key: np.full(shape, 0.5, dtype=np.float32) for key, shape in shapes.items()}
    host["a"] = (np.arange(np.prod(plan.a_shape)).reshape(plan.a_shape) % 11 - 5).astype(np.float32) / 16
    host["b"] = (np.arange(np.prod(plan.b_shape)).reshape(plan.b_shape) % 7 - 3).astype(np.float32) / 8
    for key, value in arrays.items():
        value.from_numpy(host[key])
    symbols = {
        key: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, key, ti.f32, ndim=len(shape)) for key, shape in shapes.items()
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, symbols["seed"], symbols["output"])
    builder.append_native(recording)
    builder.append_native(recording)
    builder.dispatch(publish, symbols["output"], symbols["result"])
    graph = builder.compile()
    module = ti.aot.Module()
    exported = module.add_execution_plan("matmul", graph)
    document = json.loads(exported.to_json())
    resources = document["resources"]
    assert len(resources) == 1 and resources[0]["provider"] == "cublaslt"
    assert resources[0]["parameters"] == frozen_parameters(plan)
    assert not module._data_payloads  # No unused matrix or workspace snapshots.
    for key in ("output", "result"):
        np.testing.assert_array_equal(arrays[key].to_numpy(), host[key])
    graph.run(arrays)
    product = 0.75 * (host["a"].swapaxes(-2, -1) @ host["b"].swapaxes(-2, -1))
    expected = 2 * host["seed"]
    for _ in range(2):
        expected = product + 0.25 * expected
        if capture:
            expected = np.maximum(expected, 0)
    np.testing.assert_allclose(arrays["result"].to_numpy(), 3 * expected, rtol=2e-5, atol=2e-5)
    artifact = tmp_path / "cublaslt-recording.tcm"
    module.archive(str(artifact))
    graph.close()
    del graph, builder, recording
    gc.collect()
    plan.close()
    with pytest.raises(RuntimeError, match="materialized"):
        frozen_parameters(plan)
    provider.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_cublaslt_recording_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
