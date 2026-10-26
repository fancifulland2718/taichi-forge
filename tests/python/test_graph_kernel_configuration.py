"""Whole-Graph configuration trials retain the compiler and deployment ABI."""

import numpy as np
import pytest
import taichi_forge as ti

from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.graph._recipes.kernel_configuration import (
    GraphKernelConfigurationRecipeProvider,
)
from tests import test_utils


PROVIDERS = (GraphRuntimeAssemblyProvider(), GraphKernelConfigurationRecipeProvider())


def choose(definition, catalog, predicate):
    source = definition._runtime_spec._graph_kernel_configuration_sources[0]
    choice = next(
        manifest.recipe_id for manifest, spec in source.choices() if predicate(spec)
    )
    return next(
        fragment
        for fragment in catalog.fragments
        if fragment.provider_metadata["family_selection"]["choice_id"] == choice
    )


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, compile_tier="fast"
)
def test_configuration_discovery_is_lazy_and_repeated_calls_share_code():
    field = ti.field(ti.i32, shape=257)

    @ti.kernel
    def add(offset: ti.i32):
        for i in range(257):
            field[i] += i + offset

    builder = ti.graph.GraphBuilder()
    builder.dispatch(add, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "first", ti.i32))
    builder.dispatch(add, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "second", ti.i32))
    definition = builder.freeze()
    before = len(add._primal.compiled_kernels)
    catalog = definition.recipe_catalog(providers=PROVIDERS)
    assert len(add._primal.compiled_kernels) == before
    if ti.lang.impl.current_cfg().arch == ti.cpu:
        assert not catalog.fragments
        with definition.materialization_context(providers=PROVIDERS) as context:
            baseline = context.materialize(catalog.baseline.recipe)
            baseline.executor.run(dict(first=2, second=7))
        np.testing.assert_array_equal(field.to_numpy(), np.arange(257) * 2 + 9)
        return
    fragment = choose(
        definition, catalog, lambda spec: spec.backend.workgroup_size is not None
    )
    source = definition._runtime_spec._graph_kernel_configuration_sources[0]
    block = next(
        spec.backend.workgroup_size
        for _, spec in source.choices()
        if spec.backend.workgroup_size is not None
    )
    assert len(fragment.coverage_region_ids) == 2
    entry = catalog.compose((fragment.fragment_id,), stage="configuration")
    with definition.materialization_context(providers=PROVIDERS) as context:
        baseline = context.materialize(catalog.baseline.recipe)
        selected = context.materialize(entry.recipe)
        again = context.materialize(entry.recipe)
        assert again.cache_hit
        assert len(add._primal.compiled_kernels) == before + 1
        assert selected.materialized_physical_id != baseline.materialized_physical_id
        assert all(
            task.selected_block_size == block
            for task in selected.executor.task_manifest()
            if task.task_type == "range_for"
        )
        selected.executor.run(dict(first=2, second=7))
        np.testing.assert_array_equal(field.to_numpy(), np.arange(257) * 2 + 9)
        again.close()
        selected.close()
        field.fill(0)
        baseline.executor.run(dict(first=3, second=8))
        np.testing.assert_array_equal(field.to_numpy(), np.arange(257) * 2 + 11)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False, compile_tier="fast")
def test_configuration_keeps_source_block_and_template_owner():
    @ti.data_oriented
    class Model:
        def __init__(self):
            self.values = ti.field(ti.i32, shape=65)

        @ti.kernel
        def run(self, factor: ti.template(), offset: ti.i32):
            ti.loop_config(block_dim=64)
            for i in range(65):
                self.values[i] = i * factor + offset

    model = Model()
    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        model.run,
        ti.graph.Arg(ti.graph.ArgKind.SCALAR, "offset", ti.i32),
        template_args={"self": model, "factor": 7},
    )
    definition = builder.freeze()
    source = definition._runtime_spec._graph_kernel_configuration_sources[0]
    # LLVM exposes source-owned geometry in the manifest. Older SPIR-V
    # metadata does not, so the native compiler must still reject the request.
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        assert all(spec.backend.workgroup_size is None for _, spec in source.choices())
    else:
        block_choices = tuple(
            manifest
            for manifest, spec in source.choices()
            if spec.backend.workgroup_size is not None
        )
        if block_choices:
            with pytest.raises(Exception, match="explicit ti.loop_config"):
                source.materialize(block_choices[0].recipe_id)
    catalog = definition.recipe_catalog(providers=PROVIDERS)
    fragment = choose(definition, catalog, lambda spec: spec.ir.compile_tier == "full")
    entry = catalog.compose((fragment.fragment_id,), stage="tier")
    with definition.materialization_context(providers=PROVIDERS) as context:
        selected = context.materialize(entry.recipe)
        assert all(
            task.selected_block_size == 64
            for task in selected.executor.task_manifest()
            if task.task_type == "range_for"
        )
        selected.executor.run(dict(offset=9))
        np.testing.assert_array_equal(model.values.to_numpy(), np.arange(65) * 7 + 9)
