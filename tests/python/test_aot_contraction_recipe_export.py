"""Selected contractions retain packing, epilogues and public bindings in AOT."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_cutensor_probe import contract, native_plan


@pytest.mark.parametrize("batch", (1, 2))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_contraction_aot_closure(tmp_path, monkeypatch, batch):
    from taichi_forge.hardware._bundled_runtime_provider import _bundled_provider_candidates
    from taichi_forge.hardware._cutensor import DEFINITION
    from taichi_forge.linalg._packing_kernels import TILED_PACKING_IMPLEMENTATION

    vendor = os.environ.get("TI_FORGE_TEST_CUTENSOR_LIBRARY_PATH")
    if not vendor:
        pytest.skip("set a real cuTENSOR vendor path")
    candidates = _bundled_provider_candidates(DEFINITION)
    assert candidates, "The requested cuTENSOR adapter was not staged"
    adapter = candidates[0]
    monkeypatch.setenv("TI_CUTENSOR_LIBRARY_PATH", vendor)
    m, n, k = (17, 17, 17) if batch == 1 else (17, 9, 13)
    prefix = () if batch == 1 else (batch,)
    label = "" if batch == 1 else "s"
    shapes = [(*prefix, k, m), (*prefix, k, n), (*prefix, m, n)]
    modes = [label + value for value in ("ki", "kj", "ij")]
    rank = len(shapes[0])
    operation = ti.linalg.record_contraction(
        shapes[0],
        modes[0],
        shapes[1],
        modes[1],
        modes[2],
        alpha=0.75,
        beta=0.25,
        activation="relu",
        compute="f32",
        absolute_tolerance=3e-5,
        relative_tolerance=3e-5,
    )
    preparation = operation.prepare()
    owner = operation._provider_owner

    @ti.kernel
    def initialize(
        seed: ti.types.ndarray(ti.f32, rank),
        c: ti.types.ndarray(ti.f32, rank),
        doutput: ti.types.ndarray(ti.f32, rank),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
    ):
        for index in ti.grouped(seed):
            c[index] = seed[index]
            doutput[index] = 0
        loss[0] = 0
        dloss[0] = 1

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, rank, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for index in ti.grouped(output):
            loss[0] += 0.5 * output[index] * output[index]

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1 if name in ("loss", "dloss") else rank)
        for name in ("a", "b", "seed", "c", "output", "doutput", "loss", "dloss")
    }
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, *(symbols[name] for name in ("seed", "c", "doutput", "loss", "dloss")))
    builder.append_native(operation)
    builder.dispatch(energy, symbols["output"], symbols["loss"])
    builder.dispatch(energy.grad, symbols["output"], symbols["loss"])
    definition = builder.freeze()
    providers = (*ti.graph.default_recipe_providers(), ti.hardware.tensor.ContractionRecipeProvider())
    catalog = definition.recipe_catalog(providers=providers)
    selected = {"baseline": (catalog.baseline.recipe, preparation["choices"][preparation["baseline"]])}
    for fragment in catalog.fragments:
        if fragment.provider_namespace != "taichi_forge.graph.contraction":
            continue
        choice = fragment.provider_metadata["family_selection"]["materialization_choice"]
        config = preparation["choices"][choice]
        if not all(tuple(config["permutations"][key]) != tuple(range(rank)) for key in ("a", "b")):
            continue
        name = {
            ("vendor", TILED_PACKING_IMPLEMENTATION): "packed_vendor",
            ("separate", "direct-f32-v1"): "packed_separate",
        }.get((config["epilogue"], config["packing_lowering"]))
        if name is not None and name not in selected:
            selected[name] = (catalog.compose((fragment.fragment_id,), stage="aot-contraction").recipe, config)
    assert set(selected) == {"baseline", "packed_vendor", "packed_separate"}
    module = ti.aot.Module()
    ordinary = builder.compile()
    module.add_execution_plan("ordinary", ordinary, gradients=dict(output="doutput", loss="dloss"))
    ordinary.close()
    del ordinary
    operation.close()

    def no_search(*_args, **_kwargs):
        raise AssertionError("Frozen export rediscovered contraction dataflows")

    import taichi_forge.linalg._contraction as contraction_module

    with monkeypatch.context() as frozen:
        frozen.setattr(contraction_module, "_dataflows", no_search)
        with definition.materialization_context(providers=providers) as context:
            for name, (recipe, config) in selected.items():
                with context.materialize(recipe) as product:
                    plan = module.add_execution_plan(
                        name, product.executor, gradients=dict(output="doutput", loss="dloss")
                    )
                    document = json.loads(plan.to_json())
                    native = [r for r in document["resources"] if r["kind"] == "native"]
                    scratch = [r for r in document["resources"] if r["kind"] == "ndarray"]
                    packed = sum(tuple(config["permutations"][key]) != tuple(range(rank)) for key in ("a", "b"))
                    assert len(native) == 1 and len(scratch) == packed + (config["epilogue"] == "separate")
                    parameters = native[0]["parameters"]
                    assert parameters["expected_workspace_bytes"] == config["workspace_bytes"]
                    for key, shape in zip(("a", "b"), shapes):
                        permutation = config["permutations"][key]
                        assert parameters[key]["shape"] == [shape[i] for i in permutation]
                    assert len(document["operations"]) == 5 + packed
                    for key in ("a", "b"):
                        assert any(
                            c["kind"] == "disjoint" and {c["left"], c["right"]} == {key, "output"}
                            for c in document["constraints"]
                        )
    # Explicit provider adjoints consume the real downstream energy.grad output.
    # ReLU's inactive outputs already have zero loss derivative in this fixture.
    module.add_execution_plan(
        "adjoint",
        ti.aot.ExecutionPlan(
            [
                contract("da_plan", "doutput", "b", "da", "da", alpha=0.75),
                contract("db_plan", "a", "doutput", "db", "db", alpha=0.75),
            ],
            resources=[
                native_plan(
                    "da_plan", [shapes[2], shapes[1], shapes[0], shapes[0]], [modes[2], modes[1], modes[0], modes[0]]
                ),
                native_plan(
                    "db_plan", [shapes[0], shapes[2], shapes[1], shapes[1]], [modes[0], modes[2], modes[1], modes[1]]
                ),
            ],
        ),
    )
    assert not module._data_payloads
    artifact = tmp_path / "selected-contraction.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            assert adapter.encode() not in data and vendor.encode() not in data
    del context, product, builder, definition, catalog
    gc.collect()
    owner.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_contraction_recipe_probe.py")),
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
