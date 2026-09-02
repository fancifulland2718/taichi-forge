from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._graph import _graph_definition_semantic_root
from taichi_forge.graph._ir import (
    BoundedDomain,
    DispatchNode,
    RuntimeBinding,
    SequentialRegion,
    WhileRegion,
)
from taichi_forge.graph._recipes import GraphDefinition

from tests import test_utils


def _static_spec(root, *, runtime_arg_names=()):
    return SimpleNamespace(
        pre_optimization_ir_root=root,
        definition_semantic_root=_graph_definition_semantic_root(root),
        definition_semantic_sources=(),
        ir_root=root,
        runtime_arg_names=frozenset(runtime_arg_names),
        fixed_runtime_args={},
        temporary_runtime_arg_names=frozenset(),
        derived_runtime_arg_names=frozenset(),
        execution_definition={
            "nodes": (),
            "dispatch_count": 1,
            "native_count": 0,
            "observation_count": 0,
            "structured_control_count": 0,
            "max_structured_depth": 0,
            "runtime_arg_count": len(runtime_arg_names),
            "fixed_runtime_arg_count": 0,
            "internal_storage_bytes": 0,
            "temporary_memory_plan": {},
        },
    )


@test_utils.test(arch=ti.cpu)
def test_graph_builder_freeze_creates_stable_complete_baseline_definition():
    @ti.kernel
    def scale(
        value: ti.i32,
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = value * source[i]

    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)

    def freeze():
        builder = ti.graph.GraphBuilder()
        builder.dispatch(scale, value, source, output)
        return builder.freeze()

    first = freeze()
    second = freeze()

    assert isinstance(first, ti.graph.GraphDefinition)
    assert first.semantic_graph_id == second.semantic_graph_id
    assert first.baseline_recipe.recipe_id == second.baseline_recipe.recipe_id
    assert (
        first.baseline_recipe.planned_physical_id
        == second.baseline_recipe.planned_physical_id
    )
    assert first.baseline_recipe.coverage_region_ids == tuple(
        region.region_id for region in first.regions
    )
    assert tuple(region.kind for region in first.regions) == (
        "sequential_region",
        "sequential_region",
        "dispatch",
    )
    assert all("0x" not in region.region_id for region in first.regions)
    assert len(first.sources) == 1
    assert first.sources[0].region_id == first.regions[-1].region_id
    assert {item.name: item.scope for item in first.binding_abi} == {
        "output": "public",
        "source": "public",
        "value": "public",
    }
    assert first.compile_provenance.core_commit
    with pytest.raises(FrozenInstanceError):
        first.backend = "changed"

    graph = first.compile()
    assert graph.definition is first
    source_array = ti.ndarray(ti.i32, shape=4)
    output_array = ti.ndarray(ti.i32, shape=4)
    source_array.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))
    graph.run({"value": 3, "source": source_array, "output": output_array})
    assert output_array.to_numpy().tolist() == [3, 6, 9, 12]


@test_utils.test(arch=ti.cpu)
def test_graph_builder_compile_uses_a_frozen_definition_without_changing_api():
    @ti.kernel
    def fill(value: ti.i32, output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in output:
            output[i] = value

    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(fill, value, output)

    graph = builder.compile()
    manifest = graph.definition.to_dict()
    assert manifest["semantic_graph_id"].startswith("semantic-graph:")
    assert manifest["baseline_recipe"]["kind"] == "baseline"
    assert manifest["baseline_recipe"]["fragments"] == ()
    assert manifest["planned_physical_manifest"]["backend"] == "cpu"

    output_array = ti.ndarray(ti.i32, shape=3)
    graph.run({"value": 7, "output": output_array})
    assert output_array.to_numpy().tolist() == [7, 7, 7]


@test_utils.test(arch=ti.cpu)
def test_graph_definition_identity_tracks_semantic_kernel_change():
    @ti.kernel
    def add_one(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = source[i] + 1

    @ti.kernel
    def add_two(
        source: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in source:
            output[i] = source[i] + 2

    source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.i32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)

    def freeze(kernel):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(kernel, source, output)
        return builder.freeze()

    first = freeze(add_one)
    second = freeze(add_two)
    assert first.semantic_graph_id != second.semantic_graph_id
    assert first.baseline_recipe.recipe_id != second.baseline_recipe.recipe_id


def test_graph_definition_normalizes_recipe_physical_fields_and_has_golden_id():
    def root(block_dim, physical_grid, chunk_size, *, memory_requirement):
        dispatch = DispatchNode(
            name=f"physical_{physical_grid}",
            bindings=(
                RuntimeBinding("extent", "ArgKind.NDARRAY"),
                RuntimeBinding("output", "ArgKind.NDARRAY"),
            ),
            bounded_domain=BoundedDomain(
                extent="extent",
                capacity=4096,
                block_dim=block_dim,
                block_mode="require",
                physical_grid_policy=physical_grid,
                physical_grid_requirement=(
                    "adaptive_grid" if physical_grid == "extent" else "fixed_capacity"
                ),
                update_policy=("grouped_stateful" if physical_grid == "extent" else ""),
                semantic_kernel_identity="kernel:stable",
                publication_epoch=7,
            ),
            logical_kernel_identity=f"physical-kernel:{physical_grid}",
            fusion_blocker=f"physical:{physical_grid}",
            memory_layout_requirements=(memory_requirement,),
        )
        return SequentialRegion(
            (
                WhileRegion(
                    predicate="continue_flag",
                    max_iterations=4,
                    condition=SequentialRegion((), name="condition"),
                    body=SequentialRegion((dispatch,), name="body"),
                    chunk_size=chunk_size,
                    compound_chunk_size=chunk_size,
                    vulkan_first_chunk_strategy="coarse_conditional",
                    masked_execution=chunk_size != 1,
                    lowering_mode="native_required",
                ),
            ),
            name="graph",
        )

    exact = GraphDefinition._from_graph_spec(
        _static_spec(
            root(64, "capacity", 1, memory_requirement=("output", 4096, 1, 4)),
            runtime_arg_names=("extent", "output"),
        ),
        "cuda",
        core_commit="first-build",
    )
    adaptive = GraphDefinition._from_graph_spec(
        _static_spec(
            root(256, "extent", 8, memory_requirement=("output", 8192, 1, 16)),
            runtime_arg_names=("extent", "output"),
        ),
        "cuda",
        core_commit="second-build",
    )

    assert exact.semantic_graph_id == adaptive.semantic_graph_id
    assert (
        exact.baseline_recipe.planned_physical_id
        != adaptive.baseline_recipe.planned_physical_id
    )
    assert exact.regions == adaptive.regions
    assert exact.binding_abi == adaptive.binding_abi
    assert exact.compile_provenance != adaptive.compile_provenance
    assert exact.semantic_graph_id == (
        "semantic-graph:0faeef408cf1c7359a21d21ead3001ced67e1c33a62f4a40e91830a9408271f5"
    )
