"""Frozen shared-A sparse matmuls export complete physical recipes."""

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
from tests.python.test_graph_sparse_matmul_recipe import _operation, _providers


@pytest.mark.parametrize("products", (2, 3))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_sparse_matmul_aot_closure(products, tmp_path, monkeypatch):
    from taichi_forge.hardware._bundled_runtime_provider import _bundled_provider_candidates
    from taichi_forge.hardware._cusparselt import DEFINITION

    operation = _operation(monkeypatch, products=products, shape=(16, 32, 32))
    preparation = operation.prepare(max_algorithms=2)
    owner = operation._provider_owner
    adapter = _bundled_provider_candidates(DEFINITION)[0]
    vendor = os.environ["TI_FORGE_TEST_CUSPARSELT_LIBRARY_PATH"]

    @ti.kernel
    def initialize(loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def publish(
        d: ti.types.ndarray(ti.f16, 2),
        output: ti.types.ndarray(ti.f32, 2),
        doutput: ti.types.ndarray(ti.f32, 2),
    ):
        for i, j in d:
            output[i, j], doutput[i, j] = ti.cast(d[i, j], ti.f32), 0

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i, j in output:
            loss[0] += 0.5 * output[i, j] ** 2

    @ti.kernel
    def adjoint(
        a: ti.types.ndarray(ti.f16, 2),
        b: ti.types.ndarray(ti.f16, 2),
        doutput: ti.types.ndarray(ti.f32, 2),
        da: ti.types.ndarray(ti.f32, 2),
        db: ti.types.ndarray(ti.f32, 2),
        dc: ti.types.ndarray(ti.f32, 2),
    ):
        for i, k in da:
            value = 0.0
            if a[i, k] != 0:
                for j in range(b.shape[0]):
                    value += 0.75 * doutput[i, j] * ti.cast(b[j, k], ti.f32)
            da[i, k] = value
        for j, k in db:
            value = 0.0
            for i in range(a.shape[0]):
                value += 0.75 * doutput[i, j] * ti.cast(a[i, k], ti.f32)
            db[j, k] = value
        for i, j in dc:
            dc[i, j] = 0.25 * doutput[i, j]

    def arg(name, dtype=ti.f16):
        return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1 if name in ("loss", "dloss") else 2)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, arg("loss", ti.f32), arg("dloss", ti.f32))
    builder.append_native(operation)
    gradients = dict(loss="dloss")
    for index in range(products):
        output, doutput = f"output{index}", f"doutput{index}"
        gradients[output] = doutput
        builder.dispatch(publish, arg(f"d{index}"), arg(output, ti.f32), arg(doutput, ti.f32))
        builder.dispatch(energy, arg(output, ti.f32), arg("loss", ti.f32))
        builder.dispatch(energy.grad, arg(output, ti.f32), arg("loss", ti.f32))
        builder.dispatch(
            adjoint,
            arg("a"),
            arg(f"b{index}"),
            *(arg(name, ti.f32) for name in (doutput, f"da{index}", f"db{index}", f"dc{index}")),
        )
    definition = builder.freeze()
    providers = _providers()
    catalog = definition.recipe_catalog(providers=providers)
    selected = {"baseline": (catalog.baseline.recipe, preparation["choices"][preparation["baseline"]])}
    for fragment in catalog.fragments:
        if not fragment.provider_namespace.endswith(".sparse_matmul"):
            continue
        key = fragment.provider_metadata["family_selection"]["materialization_choice"]
        config = preparation["choices"][key]
        name = "separate" if config["epilogue"] == "separate_relu" else "vendor"
        if name not in selected:
            selected[name] = (catalog.compose((fragment.fragment_id,), stage="aot-sparse-matmul").recipe, config)
    assert set(selected) == {"baseline", "vendor", "separate"}
    module = ti.aot.Module()
    ordinary = builder.compile()
    module.add_execution_plan("ordinary", ordinary, gradients=gradients)
    ordinary.close()
    del ordinary
    operation.close()

    def no_search(*_args, **_kwargs):
        raise AssertionError("Frozen export repeated sparse-matmul preparation")

    with monkeypatch.context() as frozen:
        frozen.setattr(type(operation), "prepare", no_search)
        with definition.materialization_context(providers=providers) as context:
            for name, (recipe, config) in selected.items():
                with context.materialize(recipe) as product:
                    plan = module.add_execution_plan(name, product.executor, gradients=gradients)
                    document = json.loads(plan.to_json())
                    assert len(document["resources"]) == 1
                    parameters = document["resources"][0]["parameters"]
                    assert parameters["expected_resources"] == config["resources"]
                    for key in ("algorithm_id", "split_k", "split_k_mode", "split_k_buffers"):
                        assert parameters[key] == config["algorithm"][key]
                    assert parameters["relu"] == (config["epilogue"] == "vendor")
                    assert sum(op.get("operation") == "compress_f16" for op in document["operations"]) == 1
                    assert sum(op.get("operation") == "matmul_f16" for op in document["operations"]) == products
                    assert len(document["operations"]) == 2 + 5 * products + (products if name == "separate" else 0)
                    groups = [c for c in document["constraints"] if c["kind"] == "disjoint_writes"]
                    assert len(groups) == 1 and len(groups[0]["writes"]) == products
    assert not module._data_payloads
    artifact = tmp_path / "selected-sparse-matmul.tcm"
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
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_sparse_matmul_recipe_probe.py")),
                mode,
                str(artifact),
                adapter,
                vendor,
                str(products),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
