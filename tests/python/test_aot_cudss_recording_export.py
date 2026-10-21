"""cuDSS export uses the private seed, selected order and one plan per owner."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.hardware._cudss_capture import CudssCaptureRecording
from taichi_forge.hardware._cudss_config import configured_plan
from tests import test_utils
from tests.python.test_cudss_config import _problem


@pytest.mark.parametrize("phase", ("solve", "factor_solve", "refactor_solve"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cudss_recordings_export_frozen_seed_and_selected_recipe(tmp_path, phase, monkeypatch):
    from taichi_forge.graph._native_definition import NativeOperationBinding, NativeResourceBinding

    matrix, original_values, dense, vendor = _problem()
    owner = configured_plan(matrix, {"version": 1, "reordering": "amd", "solve": "general"},
                            matrix_type="spd", matrix_view="full", library_path=vendor, _graph_owned=True)
    adapter = owner.provider_identity["library_candidate"]
    original_values.fill(0)  # Invalid as a new seed; the existing private seed is valid.

    @ti.kernel
    def produce(source: ti.types.ndarray(ti.f32, 1), rhs: ti.types.ndarray(ti.f32, 1), rhs2: ti.types.ndarray(ti.f32, 1)):
        for i in source:
            rhs[i] = 2 * source[i]
            rhs2[i] = -source[i]

    @ti.kernel
    def adjoint_seed(solution: ti.types.ndarray(ti.f32, 1), adjoint_rhs: ti.types.ndarray(ti.f32, 1)):
        for i in solution:
            adjoint_rhs[i] = solution[i]

    @ti.kernel
    def publish(solution: ti.types.ndarray(ti.f32, 1), adjoint: ti.types.ndarray(ti.f32, 1), result: ti.types.ndarray(ti.f32, 1)):
        for i in result:
            result[i] = solution[i] + adjoint[i] * 0.25

    names = ("source", "rhs", "rhs2", "solution", "solution2", "adjoint_rhs", "adjoint", "result")
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in names}
    frames = {name: ti.ndarray(ti.f32, (4,)) for name in names}
    source = np.arange(1, 5, dtype=np.float32) / 2
    frames["source"].from_numpy(source)
    scale = 1 if phase == "solve" else 2
    if phase != "solve":
        frames["values"] = ti.ndarray(ti.f32, (10,))
        frames["values"].from_numpy(np.array([4, -1, -1, 4, -1, -1, 4, -1, -1, 3], np.float32) * scale)
    forward = CudssCaptureRecording(owner, phase=phase, values="values",
                                    _rhs_pairs=(("rhs", "solution"), ("rhs2", "solution2")))
    adjoint = CudssCaptureRecording(owner, phase=phase, values="values", rhs="adjoint_rhs", solution="adjoint")
    resource_calls, operation_calls = [], []
    original_resource, original_operation = NativeResourceBinding.lower, NativeOperationBinding.lower

    def lower_resource(self, context, **kwargs):
        if self.owner is owner:
            assert self.definition is owner._definition
            resource_calls.append(self)
        return original_resource(self, context, **kwargs)

    def lower_operation(self, context, **kwargs):
        if self.definition.provider == "cudss":
            assert self.definition in (forward._definition.native, adjoint._definition.native)
            operation_calls.append(self)
        return original_operation(self, context, **kwargs)

    monkeypatch.setattr(NativeResourceBinding, "lower", lower_resource)
    monkeypatch.setattr(NativeOperationBinding, "lower", lower_operation)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(produce, symbols["source"], symbols["rhs"], symbols["rhs2"])
    builder.append_native(forward)
    builder.dispatch(adjoint_seed, symbols["solution"], symbols["adjoint_rhs"])
    builder.append_native(adjoint)
    builder.dispatch(publish, symbols["solution"], symbols["adjoint"], symbols["result"])
    graph = builder.compile()
    module = ti.aot.Module()

    def export(name, executable):
        before = owner._program._cuda_cudss_plan_configuration(owner._handle)
        plan = module.add_execution_plan(name, executable)
        after = owner._program._cuda_cudss_plan_configuration(owner._handle)
        assert after["graph_seed_snapshot_count"] - before["graph_seed_snapshot_count"] == 1
        assert after["graph_seed_snapshot_bytes"] - before["graph_seed_snapshot_bytes"] == 100
        resources = json.loads(plan.to_json())["resources"]
        native = [r for r in resources if r.get("operation") == "plan_f32"]
        assert len(native) == 1
        assert native[0]["parameters"] == dict(rows=4, nonzeros=10, matrix_type="spd", matrix_view="full",
                                                phase=phase, reordering="amd", solve="general")

    export("cold", graph)
    graph.run(graph.bind(frames))
    x = np.linalg.solve(dense * scale, 2 * source)
    backward = np.linalg.solve((dense * scale).T, x)
    expected = x + backward * 0.25
    np.testing.assert_allclose(frames["result"].to_numpy(), expected, rtol=3e-5, atol=3e-5)
    export("warm", graph)
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(),))
    catalog.build_compatible_stage(candidate_limit=4)
    assert len(catalog.entries()) == 1
    with definition.materialization_context(provider_set=catalog.provider_set) as context:
        with context.materialize(catalog.entries()[0].recipe) as selected:
            selected.executor.run(selected.executor.bind(frames))
            np.testing.assert_allclose(frames["result"].to_numpy(), expected, rtol=3e-5, atol=3e-5)
            export("selected", selected.executor)
    # Metadata comes from the native owner, not mutable Python display fields.
    owner.matrix_type, owner.matrix_view = "general", "upper"
    export("changed", graph)
    assert len(resource_calls) == 4 and len(operation_calls) == 8
    assert len(module._data_payloads) == 3, "All exports must reuse identical read-only CSR payloads"
    if phase == "solve":
        class Unqualified(CudssCaptureRecording):
            pass

        bad_builder = ti.graph.GraphBuilder()
        bad_builder.append_native(Unqualified(owner, phase=phase))
        bad = bad_builder.compile()
        with pytest.raises(ti.TaichiRuntimeError, match="unavailable"):
            module.add_execution_plan("unqualified", bad)
        bad.close()
        # The native owner freezes the numerical policy when the recording is
        # appended, before materialization or AOT export can reuse its resource.
        conflict = ti.graph.GraphBuilder()
        conflict.append_native(CudssCaptureRecording(owner, phase="solve"))
        with pytest.raises(RuntimeError, match="polic"):
            conflict.append_native(CudssCaptureRecording(owner, phase="factor_solve"))
    graph.close()
    artifact = tmp_path / "cudss-recording.tcm"
    module.archive(str(artifact))
    ti.reset()  # All live producer owners retire before the independent consumer.
    result = subprocess.run(
        [sys.executable, "-S", str(Path(__file__).with_name("aot_cudss_recording_probe.py")),
         str(artifact), adapter, vendor, phase], env=os.environ.copy(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cleanup failed" not in result.stdout + result.stderr
