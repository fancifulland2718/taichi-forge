"""Selected matmul export preserves packing, epilogues and public constraints."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("batch", (1, 2))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_matmul_aot_closure(tmp_path, monkeypatch, batch):
    from taichi_forge.hardware._cublaslt_algorithms import _AlgorithmApi
    from taichi_forge.linalg._packing_kernels import TILED_PACKING_IMPLEMENTATION

    adapter = os.environ.get("TI_CUBLASLT_TEST_ADAPTER")
    vendor = os.environ.get("TI_FORGE_TEST_CUBLASLT_LIBRARY_PATH")
    if not adapter or not vendor:
        pytest.skip("set real cuBLASLt adapter/vendor paths")
    monkeypatch.setenv("TI_CUBLASLT_LIBRARY_PATH", vendor)
    m, n, k = (17, 17, 17) if batch == 1 else (65, 49, 97)
    rank = 2 if batch == 1 else 3
    operation = ti.linalg.record_matmul(
        m,
        n,
        k,
        batch_count=batch,
        transpose_a=True,
        transpose_b=True,
        alpha=0.75,
        beta=0.25,
        activation="relu",
        absolute_tolerance=2e-5,
        relative_tolerance=2e-5,
    )
    preparation = operation.prepare(heuristic_limit=1)
    owner = operation._provider_owner

    @ti.kernel
    def initialize(
        seed: ti.types.ndarray(ti.f32, rank),
        output: ti.types.ndarray(ti.f32, rank),
        doutput: ti.types.ndarray(ti.f32, rank),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
    ):
        for index in ti.grouped(seed):
            output[index] = seed[index]
            doutput[index] = 0
        loss[0] = 0
        dloss[0] = 1

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, rank, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for index in ti.grouped(output):
            loss[0] += 0.5 * output[index] * output[index]

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1 if name in ("loss", "dloss") else rank)
        for name in ("a", "b", "seed", "output", "doutput", "loss", "dloss")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, *(symbols[name] for name in ("seed", "output", "doutput", "loss", "dloss")))
    builder.append_native(operation)
    builder.dispatch(energy, symbols["output"], symbols["loss"])
    builder.dispatch(energy.grad, symbols["output"], symbols["loss"])
    definition = builder.freeze()
    providers = (*ti.graph.default_recipe_providers(), ti.hardware.linalg.MatmulRecipeProvider())
    catalog = definition.recipe_catalog(providers=providers)
    selected = {"baseline": (catalog.baseline.recipe, preparation["choices"][preparation["baseline"]])}
    targets = {
        (("a", "b"), "fused", TILED_PACKING_IMPLEMENTATION): "packed_fused",
        (("a", "b"), "separate", "direct-f32-v1"): "packed_separate",
    }
    for fragment in catalog.fragments:
        if fragment.provider_namespace != "taichi_forge.graph.matmul":
            continue
        choice = fragment.provider_metadata["family_selection"]["materialization_choice"]
        config = preparation["choices"][choice]
        name = targets.get((tuple(config["packed_inputs"]), config["epilogue"], config["packing_lowering"]))
        if name is not None and name not in selected:
            selected[name] = (catalog.compose((fragment.fragment_id,), stage="aot-matmul").recipe, config)
    assert set(selected) == {"baseline", "packed_fused", "packed_separate"}
    module = ti.aot.Module()
    ordinary = builder.compile()
    module.add_execution_plan("ordinary", ordinary, gradients=dict(output="doutput", loss="dloss"))
    ordinary.close()
    del ordinary
    operation.close()

    def no_search(*_args, **_kwargs):
        raise AssertionError("Frozen materialization/export repeated algorithm search")

    with monkeypatch.context() as frozen:
        frozen.setattr(_AlgorithmApi, "shortlist", no_search)
        with definition.materialization_context(providers=providers) as context:
            for name, (recipe, config) in selected.items():
                with context.materialize(recipe) as product:
                    plan = module.add_execution_plan(
                        name, product.executor, gradients=dict(output="doutput", loss="dloss")
                    )
                    document = json.loads(plan.to_json())
                    native = [r for r in document["resources"] if r["kind"] == "native"]
                    scratch = [r for r in document["resources"] if r["kind"] == "ndarray"]
                    assert len(native) == 1 and len(scratch) == len(config["packed_inputs"])
                    assert native[0]["parameters"]["algorithm"] == config["algorithm"]
                    assert native[0]["parameters"]["epilogue"] == (
                        "relu" if config["epilogue"] == "fused" else "identity"
                    )
                    assert native[0]["parameters"]["transpose_a"] == ("a" not in config["packed_inputs"])
                    assert len(document["operations"]) == 4 + len(config["packed_inputs"]) + (
                        config["epilogue"] == "separate"
                    )
                    for key in ("a", "b"):
                        assert any(
                            c["kind"] == "disjoint" and {c["left"], c["right"]} == {key, "output"}
                            for c in document["constraints"]
                        )
    assert not module._data_payloads
    artifact = tmp_path / "selected-matmul.tcm"
    module.archive(str(artifact))
    del context, product, builder, definition, catalog
    gc.collect()
    owner.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_matmul_recipe_probe.py")),
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
