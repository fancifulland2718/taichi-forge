"""Public construction contracts select existing complete recipes."""

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_public_execution_selection_reuses_bindings_and_preserves_baseline():
    @ti.kernel
    def step(data: ti.types.ndarray(dtype=ti.i32, ndim=1), delta: ti.i32):
        for i in data:
            data[i] += delta

    builder = ti.graph.GraphBuilder()
    data_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "data", ti.i32, ndim=1)
    delta = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.i32)
    builder.dispatch(step, data_arg, delta)
    builder.dispatch(step, data_arg, delta)
    definition = builder.freeze()
    baseline_id = definition.baseline_recipe.recipe_id
    selected = definition.select_execution_recipe(binding_reuse="prefer")
    assert isinstance(selected, ti.graph.GraphRecipeHandle)
    assert selected.recipe_id == definition.select_execution_recipe(binding_reuse="prefer").recipe_id
    if ti.lang.impl.current_cfg().arch == ti.cpu:
        assert selected.manifest.is_baseline
        with pytest.raises(ti.graph.GraphRecipeProviderError) as failure:
            definition.select_execution_recipe(binding_reuse="require")
        assert failure.value.error_key == "execution_contract_unavailable"
    else:
        assert not selected.manifest.is_baseline
        assert selected.recipe_id == definition.select_execution_recipe().recipe_id
    assert definition.baseline_recipe.recipe_id == baseline_id

    first = ti.ndarray(ti.i32, 128)
    second = ti.ndarray(ti.i32, 128)
    first.fill(0)
    second.fill(10)
    with definition.materialize(selected) as materialized:
        graph = materialized.executor
        bindings = graph.bind({"data": first, "delta": 2})
        graph.submit(bindings).wait()
        bindings.update(data=second, delta=3)
        graph.submit(bindings).wait()
        np.testing.assert_array_equal(first.to_numpy(), np.full(128, 4, dtype=np.int32))
        np.testing.assert_array_equal(second.to_numpy(), np.full(128, 16, dtype=np.int32))
    assert graph.execution_stats().lifecycle_state == "runtime_invalid"


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_graphics_requirement_never_falls_back_and_field_admission_stays_explicit():
    values = ti.field(ti.i32, shape=32)

    @ti.kernel
    def update():
        for i in values:
            values[i] += 1

    builder = ti.graph.GraphBuilder()
    builder.dispatch(update)
    builder.dispatch(update)
    definition = builder.freeze()
    for binding_reuse in ("prefer", "require"):
        with pytest.raises(ti.graph.GraphRecipeProviderError) as failure:
            definition.select_execution_recipe(queue="graphics", binding_reuse=binding_reuse)
        assert failure.value.error_key == "execution_contract_unavailable"
    selected = definition.select_execution_recipe(binding_reuse="prefer")
    assert selected.manifest.is_baseline
    with pytest.raises(ti.graph.GraphRecipeProviderError):
        definition.select_execution_recipe()
    with definition.materialize(selected) as materialized:
        materialized.executor.submit({}).wait()
    np.testing.assert_array_equal(values.to_numpy(), np.full(32, 2, dtype=np.int32))
    with pytest.raises(ValueError, match="execution queue"):
        definition.select_execution_recipe(queue="invented")
    with pytest.raises(ValueError, match="binding_reuse"):
        definition.select_execution_recipe(binding_reuse="invented")
