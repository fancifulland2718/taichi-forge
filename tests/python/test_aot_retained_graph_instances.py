"""Retained instances share code but keep their own declarative private state."""

import json
import struct
from contextlib import ExitStack

import pytest
import taichi_forge as ti
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from taichi_forge_runtime.aot import ExecutionPlan, Runtime

from tests import test_utils


class _RetainedGraph(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, graph, output):
        self.graph = graph
        self.bindings = graph.bind(dict(output=output))

    def compile(self):
        return self

    def run(self):
        self.graph.run(self.bindings)

    def _export_aot(self, context):
        context.bound_graph(self.graph, self.bindings)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_retained_graph_instances_keep_private_state_and_aliases(
    tmp_path, pooled=False
):
    @ti.kernel
    def advance(
        state: ti.types.ndarray(ti.i32, ndim=1),
        alias: ti.types.ndarray(ti.i32, ndim=1),
        output: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in state:
            state[i] += 1
            output[i] = alias[i]

    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i]

    builder = ti.graph.GraphBuilder()
    private = builder.private_ndarray("state", ti.i32, 5)
    alias = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "alias", ti.i32, ndim=1)
    builder._bind_internal_runtime_arg(
        alias, builder._declared_private_bindings["state"][1]
    )
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder.dispatch(advance, private, alias, output)
    definition = builder.freeze()
    contexts = ExitStack()
    if pooled:
        from tests.python.test_graph_resource_lifetime_recipe import _providers

        catalog = definition.recipe_catalog(providers=_providers())
        fragment = next(
            f
            for f in catalog.fragments
            if f.fragment_key.endswith(":generation-owned-storage")
        )
        recipe = catalog.composer.compose((fragment,))
        first = contexts.enter_context(
            definition.materialization_context(provider_set=catalog.provider_set)
        )
        second = contexts.enter_context(
            definition.materialization_context(provider_set=catalog.provider_set)
        )
        a = contexts.enter_context(first.materialize(recipe))
        b = contexts.enter_context(second.materialize(recipe))
        cached = contexts.enter_context(first.materialize(recipe))
        one, two = a.executor, b.executor
        assert cached.executor._instance is one._instance
        assert one._instance._storage_owners[0] is not two._instance._storage_owners[0]
    else:
        one, two = definition.compile(), definition.compile()
    if not pooled:
        assert one._spec is two._spec
    assert (
        one._spec.fixed_runtime_args["state"] is two._spec.fixed_runtime_args["state"]
    )
    source = {name: ti.ndarray(ti.f32, 5) for name in ("first", "second", "again")}
    retained = [
        _RetainedGraph(one, source["first"]),
        _RetainedGraph(two, source["second"]),
        _RetainedGraph(one, source["again"]),
    ]
    parent = ti.graph.GraphBuilder()
    for child in retained:
        parent.append_native(child)
    graph = parent.compile()
    graph.run({})
    assert [source[name].to_numpy().tolist() for name in source] == [
        [1] * 5,
        [1] * 5,
        [2] * 5,
    ]
    module = ti.aot.Module()
    plan = module.add_execution_plan("forward", graph, source_bindings=source)
    document = json.loads(plan.to_json())
    arrays = [r for r in document["resources"] if r["kind"] == "ndarray"]
    assert len(arrays) == 2
    pools = [
        r
        for r in document["resources"]
        if r["kind"] == "native" and r["provider"] == "memory"
    ]
    assert len(pools) == (2 if pooled else 0)
    if pooled:
        assert {r["allocator"] for r in arrays} == {r["name"] for r in pools}
    assert len(module._exported_kernels) == 1
    operations = document["operations"]
    assert len({op["entry"] for op in operations}) == 1
    assert operations[0]["arguments"][:2] == operations[2]["arguments"][:2]
    assert operations[0]["arguments"][0] != operations[1]["arguments"][0]
    assert all(op["arguments"][0] == op["arguments"][1] for op in operations)

    # Re-export snapshots a changed binding, preserving instance identity and
    # its own alias set without adding another copy of the kernel.
    replacement = ti.ndarray(ti.f32, 5)
    retained[1].bindings.update(output=replacement)
    changed = module.add_execution_plan(
        "changed", graph, source_bindings=dict(source, second=replacement)
    )
    assert changed.argument_names == plan.argument_names
    assert len(module._exported_kernels) == 1
    assert (
        len(
            [
                r
                for r in json.loads(changed.to_json())["resources"]
                if r["kind"] == "ndarray"
            ]
        )
        == 2
    )

    module.add_kernel(square)
    module.add_kernel(square.grad, name="reverse")
    module.add_execution_plan(
        "step",
        ExecutionPlan(
            operations
            + [
                dict(kind="fill32", destination="dfirst", value=0),
                dict(kind="fill32", destination="dy", value=0x3F800000),
                dict(kind="kernel", entry="square", arguments=["first", "y"]),
                dict(kind="kernel", entry="reverse", arguments=["first", "y"]),
            ],
            resources=document["resources"],
            gradients=dict(first="dfirst", y="dy"),
            constraints=document.get("constraints"),
        ),
    )
    graph.close()
    one.close()
    two.close()
    contexts.close()
    rejected = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError):
        rejected.add_execution_plan("closed", graph, source_bindings=source)
    assert not rejected._exported_kernels
    artifact = tmp_path / "instances.tcm"
    module.archive(str(artifact))
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"

    for make_runtime in (lambda: Runtime(arch), ti.aot.borrow_runtime):
        with make_runtime() as runtime:
            loaded = runtime.load_module(artifact)
            frames = []
            for _ in range(2):
                outputs = {
                    name: runtime.ndarray("f32", 5)
                    for name in ("first", "second", "again", "y", "dfirst", "dy")
                }
                call = loaded.prepare_execution("step", outputs)
                assert call.info["resource_bytes"] == 40
                frames.append((call, outputs))
            for index, repeats in ((0, 3), (1, 1)):
                call, outputs = frames[index]
                for _ in range(repeats):
                    call()
                expected = {
                    "first": 2 * repeats - 1,
                    "second": repeats,
                    "again": 2 * repeats,
                    "y": (2 * repeats - 1) ** 2,
                    "dfirst": 2 * (2 * repeats - 1),
                }
                for name, value in expected.items():
                    assert (
                        list(struct.unpack("<5f", outputs[name].read())) == [value] * 5
                    )
            frames[0][0]()
            frames[0][0].close()
            frames[1][0]()
            loaded.close()
            assert list(struct.unpack("<5f", frames[1][1]["again"].read())) == [4] * 5
            with pytest.raises(RuntimeError):
                frames[1][0]()
            for call, outputs in frames:
                call.close()
                for value in outputs.values():
                    value.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_retained_graph_instances_keep_selected_pools(tmp_path):
    test_retained_graph_instances_keep_private_state_and_aliases(tmp_path, pooled=True)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_retained_pool_export_rejects_unmatched_owners_and_unqualified_arenas(
    monkeypatch,
):
    from dataclasses import replace
    from tests.python.test_graph_resource_lifetime_recipe import (
        _definition,
        _providers,
        _TemporaryWork,
    )

    definition = _definition(groups=1, size=5)
    catalog = definition.recipe_catalog(providers=_providers())
    fragment = next(
        f
        for f in catalog.fragments
        if f.fragment_key.endswith(":generation-owned-storage")
    )
    with definition.materialization_context(
        provider_set=catalog.provider_set
    ) as context:
        with context.materialize(catalog.composer.compose((fragment,))) as selected:
            graph = selected.executor
            rejected = ti.aot.Module()
            with monkeypatch.context() as altered:
                altered.setattr(graph._instance, "_storage_owners", (object(),))
                with pytest.raises(ti.TaichiRuntimeError, match="owner differs"):
                    rejected.add_execution_plan("wrong_owner", graph)
            with monkeypatch.context() as altered:
                altered.setattr(
                    graph._spec,
                    "_storage_plans",
                    tuple(
                        replace(plan, factory=lambda: None)
                        for plan in graph._spec._storage_plans
                    ),
                )
                with pytest.raises(
                    ti.TaichiRuntimeError, match="selected storage/binding"
                ):
                    rejected.add_execution_plan("unknown_factory", graph)
            assert not rejected._exported_kernels and not rejected._graph_export_failed
            rejected.add_execution_plan("valid", graph)
    monkeypatch.setenv("TI_GRAPH_TEMPORARY_ARENA_SLOTS", "3")
    builder = ti.graph.GraphBuilder()
    builder.append_native(_TemporaryWork(5))
    definition = builder.freeze()
    catalog = definition.recipe_catalog(providers=_providers())
    with definition.materialization_context(
        provider_set=catalog.provider_set
    ) as context:
        for flavor in (":generation-owned-storage", ":queue-ordered-storage"):
            fragment = next(
                f for f in catalog.fragments if f.fragment_key.endswith(flavor)
            )
            with context.materialize(catalog.composer.compose((fragment,))) as selected:
                output = ti.ndarray(ti.i32, 5)
                parent = ti.graph.GraphBuilder()
                parent.append_native(_RetainedGraph(selected.executor, output))
                graph = parent.compile()
                rejected = ti.aot.Module()
                with monkeypatch.context() as altered:
                    altered.setattr(
                        selected.executor._spec,
                        "native_execution_observer_leases",
                        (object(),),
                    )
                    with pytest.raises(
                        (ti.TaichiRuntimeError, ValueError),
                        match="qualified ordered CUDA|single-stream CUDA",
                    ):
                        rejected.add_execution_plan(
                            "unqualified_arena",
                            graph,
                            source_bindings=dict(output=output),
                        )
                assert (
                    not rejected._exported_kernels and not rejected._graph_export_failed
                )
                exported = rejected.add_execution_plan(
                    "nested_arena", graph, source_bindings=dict(output=output)
                )
                assert len(exported.kernel_entries) == 2
                recipe = json.loads(exported.to_json()).get("storage_recipe")
                if flavor == ":generation-owned-storage":
                    assert recipe["capacity"] == 3
                else:
                    assert recipe is None
                graph.close()
