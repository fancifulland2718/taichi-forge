"""Hardware recordings retain their place and resources in whole-Graph recipes."""

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.graph._recipes.map_fusion import GraphMapFusionRecipeProvider
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("native_first", (True, False))
def test_captured_fft_survives_baseline_and_fused_recipe_reconstruction(native_first):
    if not ti.hardware.fft.is_available():
        pytest.skip("a compatible optional cuFFT library is unavailable")
    length = 64

    @ti.kernel
    def scale(source: ti.types.ndarray(dtype=ti.f32, ndim=1), output: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(length * 2):
            output[i] = source[i] * 2.0

    @ti.kernel
    def shift(source: ti.types.ndarray(dtype=ti.f32, ndim=1), output: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(length * 2):
            output[i] = source[i] + 0.25

    layout = ti.hardware.fft.CufftLayout(embed=(8, 8))
    plan = ti.hardware.fft.CufftPlanND((8, 8), input_layout=layout, output_layout=layout)
    symbols = [
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("input", "first", "second", "output")
    ]
    builder = ti.graph.GraphBuilder()
    if native_first:
        builder.append_native(plan.record(input="input", output="first"), admission="auto")
        builder.dispatch(scale, symbols[1], symbols[2])
        builder.dispatch(shift, symbols[2], symbols[3])
    else:
        builder.dispatch(scale, symbols[0], symbols[1])
        builder.dispatch(shift, symbols[1], symbols[2])
        builder.append_native(plan.record(input="second", output="output"), admission="auto")
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(), GraphMapFusionRecipeProvider()))
    assert len(catalog.entries()) == 2, "baseline and an adjacent two-map fusion must both be reachable"
    arrays = {symbol.name: ti.ndarray(ti.f32, shape=length * 2) for symbol in symbols}
    host = np.linspace(-1, 2, length * 2, dtype=np.float32)
    fft_input = host if native_first else host * 2 + 0.25
    reference = np.fft.fft2((fft_input[0::2] + 1j * fft_input[1::2]).reshape(8, 8)).reshape(-1)
    expected = np.stack((reference.real, reference.imag), axis=1).reshape(-1)
    if native_first:
        expected = expected * 2 + 0.25
    identities = set()
    try:
        with definition.materialization_context(provider_set=catalog.provider_set) as context:
            for entry in catalog.entries():
                arrays["input"].from_numpy(host)
                for name in ("first", "second", "output"):
                    arrays[name].fill(-123)
                with context.materialize(entry.recipe) as materialized:
                    graph = materialized.executor
                    assert graph._debug_info["native_count"] == 1
                    manifest = materialized.manifest
                    assert not manifest.task_topology_exact
                    native_tasks = [task for task in manifest.tasks if task.kind == "native_action"]
                    assert len(native_tasks) == 1
                    assert manifest.tasks[0 if native_first else -1] is native_tasks[0]
                    assert native_tasks[0].kernel_indices == ()
                    bindings = graph.bind(arrays)
                    for _ in range(3):
                        graph.run(bindings)
                    np.testing.assert_allclose(arrays["output"].to_numpy(), expected, rtol=3e-5, atol=3e-5)
                    identities.add(materialized.materialized_physical_id)
            assert len(identities) == 2
    finally:
        plan.close()
