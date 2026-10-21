"""Selected source-addon scan reconstructs its topology and provider in AOT."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.vulkan])
def test_cub_source_capability_is_backend_specific():
    with ti.aot.borrow_runtime() as runtime:
        runtime.configure_provider("cub_source", adapter_path="unused", library_path="unused")
        assert runtime.probe_native_operation("cub_source_execute") == {
            "available": False,
            "capabilities": frozenset(),
            "library_version": None,
        }


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cub_source_native_primitive_inventory(tmp_path):
    from taichi_forge.hardware._aot_cub_source import frozen_parameters
    from taichi_forge.hardware._cub_source_provider import _OPERATION_SPECS, load_cub_source_provider

    manifest = os.environ.get("TI_FORGE_TEST_CUB_SOURCE_PROVIDER_MANIFEST")
    if not manifest or not os.environ.get("TI_CUB_SOURCE_TEST_ADAPTER"):
        pytest.skip("requires the CUB source addon and common adapter")
    provider = load_cub_source_provider(manifest)
    module = ti.aot.Module()
    for operation, n in [
        *((key, 37) for key in _OPERATION_SPECS),
        ("select_flagged_u32", 0),
        ("segmented_exclusive_scan_u32", 0),
    ]:
        source = provider.plan(operation, n)
        ports = (
            ("input0", "output0")
            if operation == "exclusive_scan_u32"
            else (
                ("input0", "input1", "output0")
                if operation.startswith("segmented_")
                else ("input0", "input1", "output0", "output1")
            )
        )
        dtype = "u64" if operation == "radix_sort_pairs_u64" else "u32"
        plan = ti.aot.ExecutionPlan(
            [
                dict(
                    kind="native",
                    provider="cub_source",
                    operation="execute",
                    version=1,
                    parameters={},
                    bindings=dict(plan="plan", **{port: port for port in ports}),
                )
            ],
            resources=[
                dict(
                    kind="native",
                    name="plan",
                    provider="cub_source",
                    operation="plan",
                    version=1,
                    parameters=frozen_parameters(source, dtype),
                    bindings={},
                )
            ],
        )
        module.add_execution_plan(f"{operation}_{n}", plan)
    artifact = tmp_path / "cub-primitives.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cub_primitives_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dtype,inclusive,n", ((ti.u32, True, 37), (ti.i32, False, 4099)))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_selected_cub_source_scan_and_downstream_gradient(tmp_path, dtype, inclusive, n):
    from taichi_forge.hardware.source_providers import CubSegmentedScanRecipeProvider
    from taichi_forge.graph._native_definition import NativeOperationBinding
    from taichi_forge.hardware._source_provider import SourceProviderManifest

    manifest = os.environ.get("TI_FORGE_TEST_CUB_SOURCE_PROVIDER_MANIFEST")
    if not manifest or not os.environ.get("TI_CUB_SOURCE_TEST_ADAPTER"):
        pytest.skip("requires the CUB source addon and common adapter")
    capacity = n + 7
    values, output = (ti.ndarray(dtype, capacity) for _ in range(2))
    offsets = [0, 0, 1, 32, 32, n, n]
    layout = ti.algorithms.SegmentedLayout.from_offsets(np.array(offsets, np.int32), capacity=capacity)
    arg = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
    w, dw, loss = (arg(name) for name in ("w", "dw", "loss"))
    result_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", dtype, ndim=1)

    @ti.kernel
    def seed(
        dw: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dl: ti.types.ndarray(ti.f32, 1)
    ):
        for i in dw:
            dw[i] = 0
        loss[0], dl[0] = 0, 1

    @ti.kernel
    def energy(
        scan: ti.types.ndarray(dtype, 1),
        w: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in range(n):
            loss[0] += 0.5 * ti.cast(scan[i] & 7, ti.f32) * w[i] * w[i]

    builder = ti.graph.GraphBuilder()
    dloss = builder.private_ndarray("dloss", ti.f32, 1)
    builder.dispatch(seed, dw, loss, dloss)
    builder.segmented_scan(values, layout, output, inclusive=inclusive)
    builder.dispatch(energy, result_arg, w, loss)
    builder.dispatch(energy.grad, result_arg, w, loss)
    definition = builder.freeze()
    provider = CubSegmentedScanRecipeProvider(manifest)
    providers = (*ti.graph.default_recipe_providers(), provider)
    catalog = definition.recipe_catalog(providers=providers)
    selected = [f for f in catalog.fragments if f.provider_namespace == provider.descriptor.namespace]
    assert len(selected) == 1
    recipe = catalog.composer.compose(selected)
    module = ti.aot.Module()
    with definition.materialization_context(providers=providers) as context:
        with context.materialize(recipe) as materialized:
            lowered = []
            original_lower = NativeOperationBinding.lower

            def observe_lower(binding, *args, **kwargs):
                if binding.definition.provider == "cub_source":
                    lowered.append(binding)
                return original_lower(binding, *args, **kwargs)

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(NativeOperationBinding, "lower", observe_lower)
                patch.setattr(
                    SourceProviderManifest,
                    "native_requirements",
                    lambda *_: pytest.fail("export rebuilt provider identity"),
                )
                plan = module.add_execution_plan(
                    "step",
                    materialized.executor,
                    source_bindings=dict(input=values, result=output),
                    gradients=dict(w="dw", loss="dloss"),
                )
            assert len(lowered) == 1
            resource = lowered[0].resources[0][1]
            assert all(
                resource.definition.parameters[key] == value
                for key, value in resource.owner._definition.parameters.items()
            )
            document = json.loads(plan.to_json())
            assert set(plan.argument_names) == {"input", "result", "w", "dw", "loss"}
            native = [r for r in document["resources"] if r["kind"] == "native"]
            assert len(native) == 1 and native[0]["provider"] == "cub_source"
            assert native[0]["parameters"]["num_items"] == n

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

    module.add_kernel(condition, name="cub_condition")
    module.add_kernel(increment, name="cub_increment")
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
                dict(kind="kernel", entry="cub_condition", arguments=["counter", "target", "predicate"])
            ],
            body=document["operations"] + [dict(kind="kernel", entry="cub_increment", arguments=["counter"])],
        ),
        dict(kind="copy", source="counter", destination="iterations", size_bytes=4),
    ]
    module.add_execution_plan("loop", ti.aot.ExecutionPlan.from_json(json.dumps(loop)))
    artifact = tmp_path / "cub.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cub_source_probe.py")),
                mode,
                str(artifact),
                str(dtype),
                str(n),
                str(int(inclusive)),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Unable to release execution resource" not in result.stdout + result.stderr
