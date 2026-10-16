"""Selected sparse solves retain their phase, configuration and shared RHS group."""

import copy
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
from tests.python.test_graph_sparse_solve_recipe import _operation, _problem, _providers


@pytest.mark.parametrize("fixed", (False, True))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_sparse_solve_aot_closure(tmp_path, fixed):
    from taichi_forge.hardware._cudss import resolve_cudss_provider

    problem = _problem(n=9)
    operation = _operation(problem, fixed=fixed)
    preparation = operation.prepare(max_plans=2)
    adapter = resolve_cudss_provider(problem[-1]).adapter_path

    @ti.kernel
    def initialize(
        dx0: ti.types.ndarray(ti.f32, 1),
        dx1: ti.types.ndarray(ti.f32, 1),
        dx2: ti.types.ndarray(ti.f32, 1),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
    ):
        for i in dx0:
            dx0[i], dx1[i], dx2[i] = 0, 0, 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        x0: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        x1: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        x2: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in x0:
            loss[0] += 0.5 * (x0[i] * x0[i] + x1[i] * x1[i] + x2[i] * x2[i])

    names = ("b0", "b1", "b2", "x0", "x1", "x2", "dx0", "dx1", "dx2", "loss", "dloss")
    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in names}
    builder = ti.graph.GraphBuilder()
    builder.dispatch(initialize, *(symbols[name] for name in ("dx0", "dx1", "dx2", "loss", "dloss")))
    builder.append_native(operation)
    arguments = [symbols[name] for name in ("x0", "x1", "x2", "loss")]
    builder.dispatch(energy, *arguments)
    builder.dispatch(energy.grad, *arguments)
    definition = builder.freeze()
    providers = _providers()
    catalog = definition.recipe_catalog(providers=providers)
    selected_phase = "solve" if fixed else "refactor_solve"
    fragment = next(
        fragment
        for fragment in catalog.fragments
        if fragment.provider_namespace.endswith(".sparse_solve")
        and (
            config := preparation["choices"][fragment.provider_metadata["family_selection"]["materialization_choice"]]
        )["numeric_phase"]
        == selected_phase
        and config["configuration"]["reordering"] == "amd"
    )
    selected_config = preparation["choices"][fragment.provider_metadata["family_selection"]["materialization_choice"]]
    baseline_config = preparation["choices"][preparation["baseline"]]
    recipe = catalog.compose((fragment.fragment_id,), stage="aot-sparse-solve").recipe
    # The catalog and each materialized plan own immutable copies of this seed.
    problem[1].fill(0)
    module = ti.aot.Module()
    gradients = dict(x0="dx0", x1="dx1", x2="dx2", loss="dloss")

    def export(name, graph, config):
        plan = module.add_execution_plan(name, graph, gradients=gradients)
        document = json.loads(plan.to_json())
        native = [resource for resource in document["resources"] if resource["kind"] == "native"]
        assert len(native) == 1
        p = native[0]["parameters"]
        assert p["phase"] == config["numeric_phase"]
        assert (p["reordering"], p["solve"]) == (
            config["configuration"]["reordering"],
            config["configuration"]["solve"],
        )
        groups = [item for item in document["operations"] if item["kind"] == "native"]
        assert len(groups) == 1 and groups[0]["parameters"] == dict(rhs_count=3, disjoint_bindings=True)
        assert [groups[0]["bindings"][f"rhs_{i}"] for i in range(3)] == ["b0", "b1", "b2"]
        assert len(document["operations"]) == 4
        # This SPD fixture has A^T == A. Reuse the exported configuration and
        # immutable CSR seed for a separate explicit adjoint solve of dx/db.
        adjoint = copy.deepcopy(groups[0])
        for i in range(3):
            adjoint["bindings"][f"rhs_{i}"] = f"dx{i}"
            adjoint["bindings"][f"output_{i}"] = f"db{i}"
        module.add_execution_plan(name + "_adjoint", ti.aot.ExecutionPlan([adjoint], resources=document["resources"]))

    ordinary = builder.compile()
    export("ordinary", ordinary, baseline_config)
    ordinary.close()
    del ordinary
    operation.close()
    with definition.materialization_context(providers=providers) as context:
        for name, chosen, config in (
            ("baseline", catalog.baseline.recipe, baseline_config),
            ("selected", recipe, selected_config),
        ):
            with context.materialize(chosen) as product:
                export(name, product.executor, config)
    assert len(module._data_payloads) == 3  # One shared CSR seed, not one copy per RHS or recipe.
    artifact = tmp_path / "sparse-solve.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            assert adapter.encode() not in data and problem[-1].encode() not in data
    del context, product, builder, definition, catalog, operation
    gc.collect()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_sparse_solve_recipe_probe.py")),
                mode,
                str(artifact),
                adapter,
                problem[-1],
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
