"""Recording lowering keeps compression epochs and configured epilogues."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_cusparselt_recording_probe import M, N, K


@pytest.mark.parametrize("group", (True, False))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparselt_recording_export(group, tmp_path, monkeypatch):
    vendor = os.environ.get("TI_FORGE_TEST_CUSPARSELT_LIBRARY_PATH")
    if not vendor:
        pytest.skip("real cuSPARSELt required")
    from taichi_forge.hardware._bundled_runtime_provider import _bundled_provider_candidates
    from taichi_forge.hardware._cusparselt import DEFINITION
    from taichi_forge.hardware._cusparselt_config import configured_plan
    from taichi_forge.hardware._cusparselt_capture import CusparseLtCaptureRecording
    from taichi_forge.graph._native_definition import NativeOperationBinding, NativeResourceBinding

    adapter = _bundled_provider_candidates(DEFINITION)[0]
    provider = ti.hardware.tensor.CusparseLtProvider(vendor)
    plan = (
        provider.matmul_plan(M, N, K)
        if group
        else configured_plan(provider, M, N, K, configuration=dict(activation="relu"))
    )
    alpha_e = 0.75 if group else -0.5
    records = (
        [
            CusparseLtCaptureRecording(
                plan, a="a", b="b", c="c", d="d", alpha=0.75, beta=0.25, _products=(("b", "c", "d"), ("b2", "c2", "e"))
            )
        ]
        if group
        else [
            plan.record(a="a", b="b", c="c", d="d", alpha=0.75, beta=0.25),
            plan.record(a="a", b="b2", c="c2", d="e", alpha=alpha_e, beta=0.25),
        ]
    )

    @ti.kernel
    def seed(loss: ti.types.ndarray(ti.f32, ndim=1)):
        loss[0] = 0

    @ti.kernel
    def adjoint(
        a: ti.types.ndarray(ti.f16, ndim=2),
        b: ti.types.ndarray(ti.f16, ndim=2),
        b2: ti.types.ndarray(ti.f16, ndim=2),
        d: ti.types.ndarray(ti.f16, ndim=2),
        e: ti.types.ndarray(ti.f16, ndim=2),
        da: ti.types.ndarray(ti.f32, ndim=2),
        db: ti.types.ndarray(ti.f32, ndim=2),
        db2: ti.types.ndarray(ti.f32, ndim=2),
        loss: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i, j in d:
            loss[0] += 0.5 * (ti.cast(d[i, j], ti.f32) ** 2 + ti.cast(e[i, j], ti.f32) ** 2)
        for i, k in da:
            value = 0.0
            if a[i, k] != 0:
                for j in range(b.shape[0]):
                    value += 0.75 * ti.cast(d[i, j], ti.f32) * ti.cast(b[j, k], ti.f32)
                    value += alpha_e * ti.cast(e[i, j], ti.f32) * ti.cast(b2[j, k], ti.f32)
            da[i, k] = value
        for j, k in db:
            left, right = 0.0, 0.0
            for i in range(a.shape[0]):
                left += 0.75 * ti.cast(d[i, j], ti.f32) * ti.cast(a[i, k], ti.f32)
                right += alpha_e * ti.cast(e[i, j], ti.f32) * ti.cast(a[i, k], ti.f32)
            db[j, k], db2[j, k] = left, right

    args = {
        name: ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY,
            name,
            ti.f32 if name in ("da", "db", "db2", "loss") else ti.f16,
            ndim=1 if name == "loss" else 2,
        )
        for name in ("a", "b", "b2", "c", "c2", "d", "e", "da", "db", "db2", "loss")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(seed, args["loss"])
    for record in records:
        builder.append_native(record)
    del record
    builder.dispatch(adjoint, *(args[name] for name in ("a", "b", "b2", "d", "e", "da", "db", "db2", "loss")))
    graph = builder.compile()
    module = ti.aot.Module()
    original_resource = NativeResourceBinding.lower
    original_operation = NativeOperationBinding.lower
    matmuls = []

    def resource(binding, context, **kwargs):
        if binding.definition.provider == "cusparselt":
            assert binding.owner is plan and binding.definition is plan._definition
        return original_resource(binding, context, **kwargs)

    def operation(binding, context, **kwargs):
        if binding.definition.operation == "matmul_f16":
            assert any(binding.definition is record._definition.operation for record in records)
            matmuls.append(binding)
        return original_operation(binding, context, **kwargs)

    monkeypatch.setattr(NativeResourceBinding, "lower", resource)
    monkeypatch.setattr(NativeOperationBinding, "lower", operation)
    assert all(record._cuda_capture_recipe._definition is record._definition for record in records)
    if not group:
        facts = plan._configuration
        facts["activation"] = "identity"
        assert plan._configuration["activation"] == "relu"
    description = json.loads(module.add_execution_plan("step", graph).to_json())
    assert len(matmuls) == 2
    assert len(description["resources"]) == 1
    assert sum(op.get("operation") == "compress_f16" for op in description["operations"]) == (1 if group else 2)
    assert sum(op.get("operation") == "matmul_f16" for op in description["operations"]) == 2
    params = description["resources"][0]["parameters"]
    assert params.get("relu", False) == (not group)
    if not group:
        assert params["algorithm_id"] == plan._configuration["algorithm_id"]
        assert params["split_k"] == plan._configuration["split_k"]
    workspace = plan.compressed_bytes + plan.compression_buffer_bytes + plan.workspace_bytes
    graph.close()
    del graph, builder, records
    gc.collect()
    plan.close()
    provider.close()
    artifact = tmp_path / "cusparselt-recording.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cusparselt_recording_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
                "group" if group else "relu",
                str(workspace),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparselt_snapshot_export_requires_initialization_payload():
    vendor = os.environ.get("TI_FORGE_TEST_CUSPARSELT_LIBRARY_PATH")
    if not vendor:
        pytest.skip("real cuSPARSELt required")
    provider = ti.hardware.tensor.CusparseLtProvider(vendor)
    plan = provider.matmul_plan(16, 16, 16)
    a = ti.ndarray(ti.f16, (16, 16))
    a.fill(0)
    plan.compress(a)
    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record())
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="raw-A initialization payload"):
        module.add_execution_plan("snapshot", graph)
    assert not module._content and not module._exported_kernels and not module._execution_plans
    graph.close()
    del graph, builder
    gc.collect()
    plan.close()
    provider.close()
