"""Selected CUTLASS recipes export forward work and explicit matrix adjoints."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_cutlass_source_capability_is_backend_specific():
    with ti.aot.borrow_runtime() as runtime:
        runtime.configure_provider("cutlass_source", adapter_path="unused", library_path="unused")
        assert runtime.probe_native_operation("cutlass_source_execute") == {
            "available": False,
            "capabilities": frozenset(),
            "library_version": None,
        }


@pytest.mark.parametrize(
    "strategy,transpose,activation",
    [
        ("direct_fused", (False, False), "identity"),
        ("split_k_reduce_fused", (True, False), "relu"),
        ("split_k_wide_reduce_fused", (False, True), "relu"),
    ],
)
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_cutlass_source_and_adjoint(tmp_path, strategy, transpose, activation):
    from tests.python.test_graph_cutlass_recipe import _provider
    from taichi_forge.graph._native_definition import NativeOperationBinding
    from taichi_forge.hardware._source_provider import SourceProviderManifest

    if not os.environ.get("TI_CUTLASS_SOURCE_TEST_ADAPTER"):
        pytest.skip("requires the CUTLASS common adapter")
    provider = _provider()
    m, n, k = 7, 9, 129
    ta, tb = transpose
    arg = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=2)
    output, dprevious = arg("output"), arg("dprevious")
    loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)

    @ti.kernel
    def seed(
        dy: ti.types.ndarray(ti.f32, 2), loss: ti.types.ndarray(ti.f32, 1), dl: ti.types.ndarray(ti.f32, 1)
    ):
        for I in ti.grouped(dy):
            dy[I] = 0
        loss[0], dl[0] = 0, 1

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 2, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for I in ti.grouped(output):
            loss[0] += 0.5 * output[I] * output[I]

    @ti.kernel
    def epilogue_adjoint(
        output: ti.types.ndarray(ti.f32, 2),
        dy: ti.types.ndarray(ti.f32, 2),
        previous: ti.types.ndarray(ti.f32, 2),
    ):
        for I in ti.grouped(output):
            if ti.static(activation == "relu"):
                if output[I] <= 0:
                    dy[I] = 0
            previous[I] = 0.25 * dy[I]

    builder = ti.graph.GraphBuilder()
    dy = builder.private_ndarray("dy", ti.f32, (m, n))
    dl = builder.private_ndarray("dl", ti.f32, 1)

    def append(rows, columns, reduction, a, b, result, trans_a, trans_b, beta=0, epilogue="identity"):
        recording = ti.linalg.record_matmul(
            rows,
            columns,
            reduction,
            a=a,
            b=b,
            output=result,
            transpose_a=trans_a,
            transpose_b=trans_b,
            alpha=0.75,
            beta=beta,
            activation=epilogue,
            absolute_tolerance=5e-5,
            relative_tolerance=5e-5,
        )
        recording.prepare(heuristic_limit=1)
        builder.append_native(recording)

    append(m, n, k, "a", "b", "output", ta, tb, beta=0.25, epilogue=activation)
    builder.dispatch(seed, dy, loss, dl)
    builder.dispatch(energy, output, loss)
    builder.dispatch(energy.grad, output, loss)
    builder.dispatch(epilogue_adjoint, output, dy, dprevious)
    if ta:
        append(k, m, n, "b", "dy", "da", tb, True)
    else:
        append(m, k, n, "dy", "b", "da", False, not tb)
    if tb:
        append(n, k, m, "dy", "a", "db", True, ta)
    else:
        append(k, n, m, "a", "dy", "db", not ta, False)
    definition = builder.freeze()
    providers = (*ti.graph.default_recipe_providers(), provider)
    catalog = definition.recipe_catalog(providers=providers)
    selected = [
        f
        for f in catalog.fragments
        if f.provider_namespace == provider.descriptor.namespace and f.fragment_key.endswith(":" + strategy)
    ]
    assert len(selected) == 3
    recipe = catalog.composer.compose(selected)
    module = ti.aot.Module()
    with definition.materialization_context(providers=providers) as context:
        with context.materialize(recipe) as materialized:
            lowered = []
            original_lower = NativeOperationBinding.lower

            def observe_lower(binding, *args, **kwargs):
                if binding.definition.provider == "cutlass_source":
                    lowered.append(binding)
                return original_lower(binding, *args, **kwargs)

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(NativeOperationBinding, "lower", observe_lower)
                patch.setattr(
                    SourceProviderManifest,
                    "native_requirements",
                    lambda *_: pytest.fail("export rebuilt provider identity"),
                )
                plan = module.add_execution_plan("step", materialized.executor, gradients=dict(output="dy", loss="dl"))
            assert len(lowered) == 3
            for binding in lowered:
                resource = binding.resources[0][1]
                assert resource.definition is resource.owner._definition
                assert resource.definition is resource.owner._cuda_capture_recipe._definition
            assert set(plan.argument_names) == {"a", "b", "output", "da", "db", "dprevious", "loss"}
            document = json.loads(plan.to_json())
            resources = [r for r in document["resources"] if r["kind"] == "native"]
            assert len(resources) == 3
            assert all(
                r["provider"] == "cutlass_source" and r["parameters"]["strategy"] == strategy for r in resources
            )

    @ti.kernel
    def condition(
        counter: ti.types.ndarray(ti.i32, 1),
        target: ti.types.ndarray(ti.i32, 1),
        predicate: ti.types.ndarray(ti.i32, 1),
    ):
        predicate[0] = counter[0] < target[0]

    @ti.kernel
    def increment(counter: ti.types.ndarray(ti.i32, 1)):
        counter[0] += 1

    module.add_kernel(condition, name="cutlass_condition")
    module.add_kernel(increment, name="cutlass_increment")
    loop = json.loads(json.dumps(document))
    loop["version"] = max(3, loop["version"])
    loop["resources"] += [
        dict(kind="ndarray", name=name, dtype="i32", shape=[1]) for name in ("counter", "predicate")
    ]
    loop["operations"] = [
        dict(kind="fill32", destination="counter", value=0),
        dict(
            kind="while",
            predicate="predicate",
            max_iterations=3,
            condition=[
                dict(kind="kernel", entry="cutlass_condition", arguments=["counter", "target", "predicate"])
            ],
            body=document["operations"]
            + [dict(kind="kernel", entry="cutlass_increment", arguments=["counter"])],
        ),
        dict(kind="copy", source="counter", destination="iterations", size_bytes=4),
    ]
    module.add_execution_plan("loop", ti.aot.ExecutionPlan.from_json(json.dumps(loop)))
    artifact = tmp_path / "cutlass.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cutlass_source_probe.py")),
                mode,
                str(artifact),
                str(int(ta)),
                str(int(tb)),
                activation,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Unable to release execution resource" not in result.stdout + result.stderr
