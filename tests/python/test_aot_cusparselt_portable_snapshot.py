"""Portable sparse weight epochs use shared data and one initialization."""

import gc
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
from tests.python.aot_cusparselt_recording_probe import M, N, K, inputs


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cusparselt_snapshot_graph_export_is_portable_and_transactional(tmp_path, monkeypatch):
    vendor = os.environ.get("TI_FORGE_TEST_CUSPARSELT_LIBRARY_PATH")
    if not vendor:
        pytest.skip("real cuSPARSELt required")
    from taichi_forge.hardware._bundled_runtime_provider import _bundled_provider_candidates
    from taichi_forge.hardware._cusparselt import DEFINITION

    adapter = _bundled_provider_candidates(DEFINITION)[0]
    provider = ti.hardware.tensor.CusparseLtProvider(vendor)
    plan = provider.matmul_plan(M, N, K)
    a = ti.ndarray(ti.f16, (M, K))
    initial = np.array(inputs(1)["a"], dtype=np.float16).reshape(M, K)
    a.from_numpy(initial)
    assert plan.aot_data_bytes == 0
    with pytest.raises(TypeError, match="boolean"):
        plan.compress(a, retain_aot_data=1)
    before = ti.runtime.stats().transfer.device_to_host_bytes
    plan.compress(a, retain_aot_data=True)
    assert ti.runtime.stats().transfer.device_to_host_bytes - before == initial.nbytes
    assert plan.aot_data_bytes == initial.nbytes
    # Recompression establishes a new epoch and must drop opt-in state by default.
    before = ti.runtime.stats().transfer.device_to_host_bytes
    plan.compress(a)
    assert ti.runtime.stats().transfer.device_to_host_bytes == before
    assert plan.aot_data_bytes == 0
    plan.compress(a, retain_aot_data=True)

    def failed_read(*args, **kwargs):
        raise RuntimeError("snapshot read failed")

    with monkeypatch.context() as failed:
        failed.setattr(type(a), "to_numpy", failed_read)
        with pytest.raises(RuntimeError, match="snapshot read failed"):
            plan.compress(a, retain_aot_data=True)

    @ti.kernel
    def adjoint_c(
        d: ti.types.ndarray(ti.f16, ndim=2),
        e: ti.types.ndarray(ti.f16, ndim=2),
        dc: ti.types.ndarray(ti.f32, ndim=2),
        dc2: ti.types.ndarray(ti.f32, ndim=2),
    ):
        # Explicit derivative of 0.5*(D^2+E^2) with respect to the dense addends;
        # as with other provider adjoints, FP16 rounding is not differentiated.
        for i, j in dc:
            dc[i, j], dc2[i, j] = 0.25 * ti.cast(d[i, j], ti.f32), 0.25 * ti.cast(e[i, j], ti.f32)

    builder = ti.graph.GraphBuilder()
    builder.append_native(plan.record(b="b", c="c", d="d", alpha=0.75, beta=0.25))
    builder.append_native(plan.record(b="b2", c="c2", d="e", alpha=-0.5, beta=0.25))
    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32 if name in ("dc", "dc2") else ti.f16, ndim=2)
        for name in ("d", "e", "dc", "dc2")
    }
    builder.dispatch(adjoint_c, *(args[name] for name in ("d", "e", "dc", "dc2")))
    graph = builder.compile()
    # Changing the original source must not change the compressed or exported epoch.
    a.from_numpy(np.zeros_like(initial))
    invalid_builder = ti.graph.GraphBuilder()
    invalid_builder.append_native(plan.record())
    invalid_builder.observe(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0))
    invalid = invalid_builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="Observation"):
        module.add_execution_plan("invalid", invalid)
    assert not module._content and not module._exported_kernels and not module._execution_plans
    assert not module._data_payloads, "Preflight failure published staged data"
    invalid.close()

    failed = ti.aot.Module()
    native = failed._aot_builder

    class CompileFailure:
        def add(self, name, kernel):
            native.add(name, kernel)
            raise RuntimeError("snapshot compile failed")

    failed._aot_builder = CompileFailure()
    with pytest.raises(RuntimeError, match="snapshot compile failed"):
        failed.add_execution_plan("partial", graph)
    assert not failed._data_payloads and not failed._execution_plans
    with pytest.raises(ti.TaichiRuntimeError, match="new AOT module"):
        failed.archive(str(tmp_path / "partial.tcm"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Export sampled the current A instead of its compressed epoch")

    with monkeypatch.context() as export:
        export.setattr(type(a), "to_numpy", forbidden)
        description = module.add_execution_plan("step", graph)
        module.add_execution_plan("again", graph)
    document = json.loads(description.to_json())
    assert document["version"] == 2
    assert len(document["initialization"]) == 1
    assert document["initialization"][0]["operation"] == "compress_f16"
    assert [op.get("operation") for op in document["operations"]] == ["matmul_f16", "matmul_f16", None]
    assert len(document["resources"]) == 2
    assert len(module._exported_kernels) == 1
    assert len(module._data_payloads) == 1
    workspace = plan.compressed_bytes + plan.compression_buffer_bytes + plan.workspace_bytes
    retained = plan.compressed_bytes + plan.workspace_bytes
    graph.close()
    del invalid, invalid_builder, graph, builder
    gc.collect()
    plan.close()
    assert plan.aot_data_bytes == 0
    provider.close()
    artifact = tmp_path / "portable-snapshot.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        payloads = [name for name in archive.namelist() if name.startswith("data/")]
        assert len(payloads) == 1 and archive.read(payloads[0]) == initial.tobytes()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cusparselt_portable_snapshot_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
                str(workspace),
                str(retained),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
