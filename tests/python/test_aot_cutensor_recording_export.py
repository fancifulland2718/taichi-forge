"""Ordinary cuTENSOR creation and reconstruction share frozen definitions."""

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


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cutensor_recording_definition_and_source_free_instances(tmp_path, monkeypatch):
    from taichi_forge.graph._native_definition import NativeDefinition, NativeOperationBinding
    from taichi_forge.hardware._bundled_runtime_provider import _bundled_provider_candidates
    from taichi_forge.hardware._cutensor import DEFINITION, CutensorProvider

    vendor = os.environ.get("TI_FORGE_TEST_CUTENSOR_LIBRARY_PATH")
    if not vendor:
        pytest.skip("set a real cuTENSOR vendor path")
    adapter = _bundled_provider_candidates(DEFINITION)[0]
    provider = CutensorProvider(vendor)
    shape, modes = [5, 7], [ord("i"), ord("k")]
    plan = provider.contraction_plan(shape, modes, (7, 3), "kj", (5, 3), "ij", (5, 3), "ij")
    shape[0], modes[0] = 1, ord("z")
    a_host = (np.arange(35, dtype=np.float32).reshape(5, 7) % 11 - 5) / 8
    b_host = (np.arange(21, dtype=np.float32).reshape(7, 3) % 7 - 3) / 4
    a, b, output = (ti.ndarray(ti.f32, shape) for shape in ((5, 7), (7, 3), (5, 3)))
    a.from_numpy(a_host)
    b.from_numpy(b_host)
    output.fill(0.5)

    def no_conversion(*args, **kwargs):
        raise AssertionError("warm contraction must not convert its definition")

    # Direct calls retain their per-call coefficients, independently of recordings.
    with monkeypatch.context() as patch:
        patch.setattr(NativeDefinition, "parameter_dict", no_conversion)
        patch.setattr(NativeOperationBinding, "lower", no_conversion)
        plan.execute(a, b, output, output, alpha=0.75, beta=0.5)
    np.testing.assert_allclose(output.to_numpy(), 0.75 * (a_host @ b_host) + 0.25, rtol=3e-5, atol=3e-5)
    output.fill(0.5)
    recording = plan.record(alpha=1.25, beta=0.25)
    recording.alpha, recording.beta = -3.0, 0.0
    builder = ti.graph.GraphBuilder()
    builder.append_native(recording)
    graph = builder.compile()
    module = ti.aot.Module()
    lowered = []
    lower = NativeOperationBinding.lower

    def record_lower(binding, context, **kwargs):
        lowered.append((binding.definition.provider, binding.definition.operation))
        return lower(binding, context, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(NativeOperationBinding, "lower", record_lower)
        exported = module.add_execution_plan("ordinary_contraction", graph)
    assert lowered == [("cutensor", "contract_f32")]
    document = json.loads(exported.to_json())
    resource = document["resources"][0]
    assert resource["parameters"]["a"] == dict(shape=[5, 7], modes=[ord("i"), ord("k")])
    assert resource["parameters"]["workspace_preference"] == "default"
    assert document["operations"][0]["parameters"] == dict(alpha=1.25, beta=0.25)
    np.testing.assert_array_equal(output.to_numpy(), np.full((5, 3), 0.5, dtype=np.float32))
    with monkeypatch.context() as patch:
        patch.setattr(NativeDefinition, "parameter_dict", no_conversion)
        patch.setattr(NativeOperationBinding, "lower", no_conversion)
        graph.run(dict(a=a, b=b, c=output, d=output))
    np.testing.assert_allclose(output.to_numpy(), 1.25 * (a_host @ b_host) + 0.125, rtol=3e-5, atol=3e-5)
    artifact = tmp_path / "cutensor-recording.tcm"
    module.archive(str(artifact))
    graph.close()
    del graph, builder, recording
    gc.collect()
    plan.close()
    provider.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cutensor_recording_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
