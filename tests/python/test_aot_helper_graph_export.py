"""Reconstruct selected helper kernels and their cold binding certificates."""

import gc
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("dtype", [ti.i32, ti.f32])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_selected_reduction_and_explicit_adjoint(tmp_path, dtype):
    count = 4097 if dtype == ti.i32 else 257
    builder = ti.graph.GraphBuilder()
    values = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "values", dtype, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", dtype, ndim=1)
    gradients = None
    if dtype == ti.f32:

        @ti.kernel
        def square(
            x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), z: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
        ):
            for i in range(count):
                z[i] = x[i] * x[i]

        @ti.kernel
        def seed(dx: ti.types.ndarray(ti.f32, ndim=1), dz: ti.types.ndarray(ti.f32, ndim=1)):
            for i in range(count):
                dx[i] = 0
                dz[i] = 1  # Explicit adjoint of sum(z); no derivative synthesis for the helper.

        intermediate = builder.private_ndarray("intermediate", ti.f32, count)
        dz = builder.private_ndarray("dz", ti.f32, count)
        alias = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "reduction_input", ti.f32, ndim=1)
        builder._bind_internal_runtime_arg(alias, builder._declared_private_bindings["intermediate"][1])
        dx = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dx", ti.f32, ndim=1)
        builder.dispatch(square, values, intermediate)
        builder.reduce(alias, output, count=count, absolute_tolerance=1e-4, relative_tolerance=1e-5)
        builder.dispatch(seed, dx, dz)
        builder.dispatch(square.grad, values, intermediate)
        gradients = dict(values="dx", intermediate="dz")
    else:
        builder.reduce(values, output, count=count)
    graph = builder.compile()
    search = ti.graph.compileiq_recipe_search(graph)
    direct = search.baseline_recipe_id
    staged = next(
        recipe
        for recipe in search.recipe_ids
        if search.recipe_manifest(recipe).get("reduction_recipe_manifest", {}).get("strategy")
        == "block_partial_finalize"
    )
    module = ti.aot.Module()
    for name, recipe in (("direct", direct), ("staged", staged)):
        parameters = dict(domain_fingerprint=search.domain_fingerprint, recipe_id=recipe)
        with search.materialize(parameters) as materialized:
            search.verify_materialized_graph(parameters, materialized)
            selected = materialized.executor
            plan = module.add_execution_plan(name, selected, gradients=gradients)
            document = json.loads(plan.to_json())
            assert len(document["operations"]) == selected.physical_plan()["physical_dispatch_count"]
            assert len(document["constraints"]) == 3
            assert plan.argument_names == (("dx", "output", "values") if dtype == ti.f32 else ("output", "values"))
            if dtype == ti.f32:
                layout = next(
                    c for c in document["constraints"] if c["kind"] == "array_layout" and c["array"] != "output"
                )
                assert layout["array"] in {r["name"] for r in document["resources"] if r["kind"] == "ndarray"}
                assert "reduction_input" not in plan.argument_names
            previous = len(module._exported_kernels)
            module.add_execution_plan(name + "_again", selected, gradients=gradients)
            assert len(module._exported_kernels) == previous
    graph.close()
    del graph, builder, search, materialized, selected
    gc.collect()
    artifact = tmp_path / "helper.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_helper_graph_probe.py")),
                mode,
                str(dtype),
                str(count),
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_aot_helper_rejects_unknown_binding_contract_before_publication():
    @ti.kernel
    def copy(x: ti.types.ndarray(ti.f32, ndim=1), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in y:
            y[i] = x[i]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(copy, *(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y")))
    graph = builder.compile()
    module = ti.aot.Module()
    # These are malformed/internal contracts, not alternate supported layouts.
    for requirement in (("x", 17, 4), ("x", 17, 4, 4, (), "affine"), ("x", (2, 3), 4, 4, (), "aos")):
        graph._spec.memory_layout_requirements = (requirement,)
        with pytest.raises(ti.TaichiRuntimeError, match="layout certificate"):
            module.add_execution_plan("invalid", graph)
        assert not module._content and not module._exported_kernels and not module._execution_plans
        assert not module._graph_export_failed
    graph.close()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_helper_packed_layout_certificates(tmp_path):
    from taichi_forge_runtime.aot import AotError

    vector_type = ti.types.vector(2, ti.f32)

    @ti.kernel
    def vector_copy(x: ti.types.ndarray(vector_type, ndim=1), y: ti.types.ndarray(vector_type, ndim=1)):
        for i in y:
            y[i] = x[i]

    @ti.kernel
    def matrix_copy(x: ti.types.ndarray(ti.f32, ndim=2), y: ti.types.ndarray(ti.f32, ndim=2)):
        for i, j in y:
            y[i, j] = x[i, j]

    module = ti.aot.Module()
    for name, kernel, dtype, ndim, extent, elements, record, layout in (
        ("aos", vector_copy, vector_type, 1, 17, (2,), 8, "aos"),
        ("matrix", matrix_copy, ti.f32, 2, (2, 3), (), 4, "row_major_2d"),
    ):
        builder = ti.graph.GraphBuilder()
        args = [ti.graph.Arg(ti.graph.ArgKind.NDARRAY, key, dtype, ndim=ndim) for key in ("x", "y")]
        builder.dispatch(kernel, *args)
        # Install the six-field certificate used by selected memory recipes.
        # This checks lowering, not qualification of every recipe producing it.
        builder._pending_ir_nodes[-1] = replace(
            builder._pending_ir_nodes[-1],
            memory_layout_requirements=tuple((key, extent, record, 4, elements, layout) for key in ("x", "y")),
            memory_disjoint_pairs=(("x", "y"),),
        )
        graph = builder.compile()
        module.add_execution_plan(name, graph)
        graph.close()
    artifact = tmp_path / "layouts.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        for name, shape, elements, short_shape in (
            ("aos", (17,), (2,), (16,)),
            ("matrix", (3, 5), (), (3, 2)),
        ):
            arrays = {key: runtime.ndarray("f32", shape, element_shape=elements) for key in ("x", "y")}
            payload = b"\x00\x00\x80\x3f" * (arrays["x"].nbytes // 4)
            arrays["x"].write(payload)
            prepared = loaded.prepare_execution(name, arrays)
            prepared()
            assert arrays["y"].read() == payload
            short = runtime.ndarray("f32", short_shape, element_shape=elements)
            with pytest.raises(AotError, match="smaller"):
                loaded.prepare_execution(name, dict(arrays, x=short))
            short.close()
            prepared.close()
