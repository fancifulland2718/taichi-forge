"""Expert and selected SpMM recordings retain physical choices and one CSR snapshot."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_graph_hardware_recipe import _spmm_definition


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_spmm_recordings_share_csr_snapshot_and_selected_plan(tmp_path, monkeypatch):
    from taichi_forge.graph._native_definition import NativeDefinition, NativeOperationBinding
    from taichi_forge.hardware._cusparse_definitions import SpmmDefinition

    if not ti.hardware.linalg.cusparse_spmm_is_available():
        pytest.skip("External cuSPARSE SpMM is unavailable")
    definition, operation, providers, bindings, expected = _spmm_definition()
    matrix = operation.matrix
    module = ti.aot.Module()
    catalog = definition.recipe_catalog(providers=providers)
    catalog.build_compatible_stage(candidate_limit=32)
    for strategy in ("row_streamed", "tree_direct", "pattern_preprocessed"):
        if strategy not in operation.preparation_report():
            continue
        recipe = next(
            entry.recipe
            for entry in catalog.entries()
            if {f.provider_namespace.rsplit(".", 1)[-1] for f in entry.recipe.fragments}
            == ({"sparse_spmm"} if strategy != "row_streamed" else set())
            and all(
                f.provider_metadata["family_selection"]["choice_id"] == strategy
                for f in entry.recipe.fragments
                if f.provider_namespace.endswith(".sparse_spmm")
            )
        )
        with definition.materialization_context(provider_set=catalog.provider_set) as context:
            with context.materialize(recipe) as result:
                graph = result.executor
                graph.run(graph.bind(bindings))
                np.testing.assert_allclose(bindings["output"].to_numpy(), expected, rtol=2e-5, atol=2e-5)
                plan = module.add_execution_plan(strategy, graph)
                native = [op for op in json.loads(plan.to_json())["operations"] if op["kind"] == "native"]
                assert len(native) == 1
                p = native[0]["parameters"]
                assert p["algorithm"] == ("csr_alg2" if strategy == "row_streamed" else "csr_alg3")
                assert p["preprocess"] == ("required" if strategy == "pattern_preprocessed" else "none")
                assert p["expected_workspace_bytes"] == operation.preparation_report()[strategy]["workspace_bytes"]
                live_plan = module.add_execution_plan(
                    strategy + "_live", graph, source_bindings=dict(coefficients=matrix)
                )
                live_document = json.loads(live_plan.to_json())
                resource = next(r for r in live_document["resources"] if r.get("operation") == "csr_f32")
                assert resource["bindings"]["values"] == "coefficients"

    legacy = ti.hardware.linalg.CusparseSpmmRecording(matrix, 8, algorithm="deterministic", output="product")
    builder = ti.graph.GraphBuilder()
    builder.append_native(legacy)
    builder.append_native(ti.hardware.linalg.CusparseSpmmRecording(matrix, 8))
    builder.append_native(ti.hardware.linalg.CusparseSpmvRecording(matrix, input="vector", output="result"))
    graph = builder.compile()
    assert not legacy.plan_info()["prepared"]

    def export(name):
        before = matrix._debug_runtime_stats()["transfers"]["device_to_host_bytes"]
        lowered = []
        lower = NativeOperationBinding.lower

        def observe(binding, context, **kwargs):
            lowered.append(binding.definition.operation)
            return lower(binding, context, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(NativeOperationBinding, "lower", observe)
            plan = module.add_execution_plan(name, graph)
        assert lowered == ["spmm_f32", "spmm_f32", "spmv_f32"]
        after = matrix._debug_runtime_stats()["transfers"]["device_to_host_bytes"]
        assert after - before == 144, "SpMV and both SpMM calls must sample this CSR only once"
        document = json.loads(plan.to_json())
        resources = [r for r in document["resources"] if r["kind"] == "native"]
        assert len(resources) == 1
        ops = [op for op in document["operations"] if op["kind"] == "native"]
        assert len({op["bindings"]["matrix"] for op in ops}) == 1
        return document

    original = export("legacy_unprepared")
    first = original["operations"][0]["parameters"]
    assert first["preprocess"] == "optional" and "expected_workspace_bytes" not in first
    assert not legacy.plan_info()["prepared"], "Export must not create a live SpMM plan"
    live = dict(bindings, vector=ti.ndarray(ti.f32, 5), result=ti.ndarray(ti.f32, 7))
    live["vector"].fill(1)
    graph.run(graph.bind(live))

    def no_conversion(*args, **kwargs):
        raise AssertionError("warm SpMM must not construct or lower a definition")

    with monkeypatch.context() as patch:
        patch.setattr(SpmmDefinition, "operation", no_conversion)
        patch.setattr(NativeDefinition, "parameter_dict", no_conversion)
        patch.setattr(NativeOperationBinding, "lower", no_conversion)
        graph.run(graph.bind(live))
    prepared = export("legacy_prepared")
    assert prepared["operations"][0]["parameters"]["preprocess"] == (
        "required" if legacy.plan_info()["preprocessed"] else "none"
    )
    values = [2 * ((row + 1) * 0.125 + col * 0.0625) for row in range(7) for col in sorted({row % 5, (row + 2) % 5})]
    update = ti.ndarray(ti.f32, len(values))
    update.from_numpy(np.array(values, dtype=np.float32))
    matrix.update_values(update)
    export("legacy_updated")
    assert len(module._data_payloads) == 4

    class Unqualified(ti.hardware.linalg.CusparseSpmmRecording):
        pass

    bad = ti.graph.GraphBuilder()
    bad.append_native(Unqualified(matrix, 8))
    bad_graph = bad.compile()
    with pytest.raises(ti.TaichiRuntimeError, match="unavailable"):
        module.add_execution_plan("bad", bad_graph)
    assert len(module._data_payloads) == 4
    bad_graph.close()
    graph.close()
    operation.close()
    empty = ti.linalg.SparseMatrixBuilder(7, 5, max_num_triplets=1).build()
    empty_builder = ti.graph.GraphBuilder()
    empty_builder.append_native(ti.hardware.linalg.CusparseSpmmRecording(empty, 8))
    empty_graph = empty_builder.compile()
    module.add_execution_plan("empty", empty_graph)
    empty_graph.close()
    artifact = tmp_path / "spmm-recording.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_spmm_recording_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
