"""Allocation ownership at assembly/instance boundaries, with real Graph work."""

from types import SimpleNamespace
from math import prod

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.graph._graph import _materialize_graph_internal_bindings
from taichi_forge.graph._ir import GraphAccess, ResourceEffect, RuntimeBinding, TemporaryRequirement
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider
from taichi_forge.graph._recipes.runtime_assembly import GraphRuntimeRecipeAssembly
from taichi_forge.graph._recipes.runtime_storage import GraphRuntimeStoragePlan, validate_storage_plans
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import ScalarNdarray
from tests import test_utils


class _Owner:
    def __init__(self, events, name, *, fail_allocation=False, fail_close=False, fail_after=None):
        self.events, self.name = events, name
        self.fail_allocation, self.fail_close = fail_allocation, fail_close
        self.fail_after = fail_after
        self.allocations = []
        self.closed = False
        self.pool = None
        if impl.current_cfg().arch == ti.cuda:
            impl.get_runtime().materialize()
            if not core._CudaGraphMemoryPool.available():
                pytest.skip("CUDA Graph-owned pools are unavailable")
            self.pool = core._CudaGraphMemoryPool(impl.get_runtime().prog, 32 << 20)
        events.append(("create", name))

    def allocate(self, dtype, shape):
        assert not self.closed
        if self.fail_allocation or (self.fail_after is not None and len(self.allocations) >= self.fail_after):
            raise RuntimeError("injected allocation failure")
        self.allocations.append((dtype, tuple(shape)))
        if self.pool is not None:
            return ScalarNdarray._graph_pool_storage(dtype, shape, self.pool)
        return ScalarNdarray(dtype, shape)

    def close(self):
        assert not self.closed
        self.closed = True
        if self.pool is not None:
            self.pool.close()
        self.events.append(("close", self.name))
        if self.fail_close:
            raise RuntimeError("injected cleanup failure")


def _plan(name, bindings, owners, events, *, arena=False, **owner_options):
    def factory():
        owner = _Owner(events, name, **owner_options)
        owners.append(owner)
        return owner

    return GraphRuntimeStoragePlan(name, tuple(bindings), arena, factory)


def _private_definition(size=131):
    @ti.kernel
    def stage(value: ti.i32, first: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in first:
            first[i] = value + i * 3

    @ti.kernel
    def transform(first: ti.types.ndarray(dtype=ti.i32, ndim=1), second: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in second:
            second[i] = first[i] * 2

    @ti.kernel
    def finish(second: ti.types.ndarray(dtype=ti.i32, ndim=1), output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in output:
            output[i] = second[i] + 7

    builder = ti.graph.GraphBuilder()
    first = builder.private_ndarray("first", ti.i32, size)
    second = builder.private_ndarray("second", ti.i32, size)
    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.i32)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.i32, ndim=1)
    builder.dispatch(stage, value, first)
    builder.dispatch(transform, first, second)
    builder.dispatch(finish, second, output)
    return builder.freeze()


def _assembly(definition, plans):
    assembly = GraphRuntimeRecipeAssembly(definition)
    for plan in plans:
        assembly.select_storage(plan)
    return assembly


def _materialize(definition, assembly, *, lanes=1):
    baseline = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(),)).baseline.recipe
    return definition._runtime_spec.materialize_complete_recipe(
        definition, baseline, assembly, workspace_lanes=lanes, workspace_saturation="wait"
    )


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_storage_owners_are_per_instance_and_never_called_by_replay():
    definition = _private_definition()
    owners, events = [], []
    assembly = _assembly(
        definition,
        (_plan("first", ("first",), owners, events), _plan("second", ("second",), owners, events)),
    )
    assert assembly.executor_release is not None
    graph = _materialize(definition, assembly, lanes=2)
    left, right = graph._workspace_pool.materialize_pair()
    assert len(owners) == 4
    assert left._internal_storages[0] is not right._internal_storages[0]
    assert all(owner.allocations == [(ti.i32, (131,))] for owner in owners)
    output = ti.ndarray(ti.i32, 131)
    binding = graph.bind(dict(value=5, output=output))
    graph.run(binding)
    ti.sync()
    allocation_counts = tuple(len(owner.allocations) for owner in owners)
    # Fail immediately if replay ever tries to consult the allocation factory.
    for owner in owners:
        owner.fail_allocation = True
    for _ in range(19):
        graph.run(binding)
    np.testing.assert_array_equal(output.to_numpy(), (5 + np.arange(131, dtype=np.int32) * 3) * 2 + 7)
    assert tuple(len(owner.allocations) for owner in owners) == allocation_counts
    assert graph.execution_stats().memory.persistent_internal_storage_bytes == 2 * 2 * 131 * 4
    held_storage = left._internal_storages[0]
    assembly.executor_release(graph)
    assert all(owner.closed for owner in owners)
    assert [name for event, name in events if event == "close"] == ["second", "first", "second", "first"]
    # Closing the factory is not revocation of an outstanding storage lease.
    np.testing.assert_array_equal(held_storage.to_numpy(), 5 + np.arange(131, dtype=np.int32) * 3)
    assembly.executor_release(graph)
    assert len([event for event in events if event[0] == "close"]) == 4


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_storage_alias_ownership_is_validated_before_allocation():
    requirement = ti.graph.GraphOwnedNdarray(ti.i32, (17,))
    spec = SimpleNamespace(
        fixed_runtime_args={"scratch": requirement, "alias": requirement, "external": object()},
        temporary_memory_plan=SimpleNamespace(allocations=()),
    )
    owners, events = [], []
    complete = _plan("complete", ("scratch", "alias"), owners, events)
    validate_storage_plans(spec, (complete,))
    for bindings, message in ((["scratch"], "aliases"), (["external"], "private bindings")):
        with pytest.raises(ValueError, match=message):
            validate_storage_plans(spec, (_plan("invalid", bindings, owners, events),))
    with pytest.raises(ValueError, match="multiple allocation owners"):
        validate_storage_plans(spec, (complete, _plan("duplicate", ("scratch", "alias"), owners, events)))
    with pytest.raises(ValueError, match="absent temporary arena"):
        validate_storage_plans(spec, (_plan("missing-arena", (), owners, events, arena=True),))
    assert events == []
    owner = complete.factory()
    bindings, storage = _materialize_graph_internal_bindings(
        spec.fixed_runtime_args, {"scratch": owner.allocate, "alias": owner.allocate}
    )
    assert len(storage) == len(owner.allocations) == 1
    assert bindings["scratch"] is bindings["alias"]
    owner.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_partial_storage_and_executor_failures_retire_all_owners_without_mutating_baseline():
    definition = _private_definition()
    for fail_at in ("allocation", "executor", "cleanup"):
        owners, events = [], []
        assembly = _assembly(
            definition,
            (
                _plan("first", ("first",), owners, events),
                _plan(
                    "second",
                    ("second",),
                    owners,
                    events,
                    fail_allocation=fail_at == "allocation",
                    fail_close=fail_at == "cleanup",
                ),
            ),
        )
        if fail_at != "allocation":

            def fail_executor(instance):
                assert len(instance._internal_storages) == 2
                raise RuntimeError("injected executor failure")

            assembly.select_binding_executor(fail_executor)
        with pytest.raises(RuntimeError, match="injected") as failure:
            _materialize(definition, assembly)
        if fail_at == "cleanup":
            assert "injected executor failure" in str(failure.value.__cause__)
        assert all(owner.closed for owner in owners)
        assert events == [("create", "first"), ("create", "second"), ("close", "second"), ("close", "first")]
        assert not hasattr(definition._runtime_spec, "_storage_plans")
    baseline = definition.compile()
    output = ti.ndarray(ti.i32, 131)
    baseline.run(dict(value=11, output=output))
    np.testing.assert_array_equal(output.to_numpy(), (11 + np.arange(131, dtype=np.int32) * 3) * 2 + 7)


class _TemporaryWork(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, kernel, size):
        self.kernel, self.size = kernel, size

    def compile(self):
        return self

    @property
    def runtime_arg_schema(self):
        return (RuntimeBinding("output", "ndarray"),)

    @property
    def resource_effects(self):
        return (ResourceEffect("output", GraphAccess.WRITE),)

    @property
    def temporary_requirements(self):
        return (
            TemporaryRequirement("raw", self.size * 4, 16),
            TemporaryRequirement("typed", self.size * 4, 16, "f32"),
        )

    def run_with_graph_temporaries(self, temporaries, runtime_args=None):
        raw, typed = temporaries["raw"], temporaries["typed"]
        self.kernel(raw.storage, raw.offset // 4, typed.storage, runtime_args["output"])


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_selected_temporary_ring_is_prepared_eagerly_and_reuses_registered_storage(monkeypatch):
    monkeypatch.setenv("TI_GRAPH_TEMPORARY_ARENA_SLOTS", "3")

    @ti.kernel
    def work(
        raw: ti.types.ndarray(dtype=ti.i32, ndim=1),
        offset: ti.i32,
        typed: ti.types.ndarray(dtype=ti.f32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in output:
            raw[offset + i] = i * 3 + 5
            typed[i] = raw[offset + i] * 1.5
            output[i] = ti.cast(typed[i] * 2, ti.i32) + 1

    builder = ti.graph.GraphBuilder()
    builder.append_native(_TemporaryWork(work, 131))
    definition = builder.freeze()
    baseline = definition.compile()
    assert baseline.execution_stats().memory.temporary_arena_slots == 0
    owners, events = [], []
    assembly = _assembly(definition, (_plan("arena", (), owners, events, arena=True),))
    graph = _materialize(definition, assembly)
    memory = graph.execution_stats().memory
    assert memory.temporary_arena_slots == memory.temporary_arena_capacity == 3
    assert memory.temporary_arena_allocations == 3
    assert len(owners[0].allocations) == 6
    assert memory.persistent_temporary_bytes == sum(
        prod(shape) * core.data_type_size(dtype) for dtype, shape in owners[0].allocations
    )
    output = ti.ndarray(ti.i32, 131)
    for _ in range(13):
        graph.run(dict(output=output))
    np.testing.assert_array_equal(output.to_numpy(), (np.arange(131, dtype=np.int32) * 3 + 5) * 3 + 1)
    assert graph.execution_stats().memory.temporary_arena_allocations == 3
    assert len(owners[0].allocations) == 6
    if owners[0].pool is not None:
        assert owners[0].pool.snapshot()["used_current_bytes"] >= 3 * 2 * 131 * 4
    assembly.executor_release(graph)
    assert owners[0].closed
    failed_owners, failed_events = [], []
    failed_assembly = _assembly(
        definition, (_plan("partial-arena", (), failed_owners, failed_events, arena=True, fail_after=3),)
    )
    with pytest.raises(RuntimeError, match="injected allocation failure"):
        _materialize(definition, failed_assembly)
    assert failed_owners[0].closed
    assert len(failed_owners[0].allocations) == 3
    # Failure half-way through the second slot does not alter the source spec.
    baseline.run(dict(output=output))
    np.testing.assert_array_equal(output.to_numpy(), (np.arange(131, dtype=np.int32) * 3 + 5) * 3 + 1)
