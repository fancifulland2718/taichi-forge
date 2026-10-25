"""Dependency-derived fork/join is shared by selected JIT and AOT plans."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.graph._ir import (
    DispatchNode, GraphAccess, ResourceEffect, TemporaryRequirement,
    contiguous_parallel_groups,
)
from tests import test_utils


def test_dependency_partition_preserves_interleaved_hazards_and_shared_reads():
    def node(reads=(), writes=(), **kwargs):
        return DispatchNode("stage", effects=tuple(
            ResourceEffect(name, access, False)
            for names, access in ((reads, GraphAccess.READ), (writes, GraphAccess.WRITE))
            for name in names
        ), opaque=False, **kwargs)

    # A WAR edge also connects a chain; independent shared readers stay apart.
    assert contiguous_parallel_groups((node(("a",)), node(writes=("a",)),
                                       node(("shared",)), node(("shared",)))) == ((0, 1), (2,), (3,))
    # Interleaved chains cannot become contiguous branches by reordering them.
    assert contiguous_parallel_groups((node(writes=("a",)), node(writes=("b",)),
                                       node(("a",)), node(("b",)))) == ((0, 1, 2, 3),)
    assert contiguous_parallel_groups((node(writes=("a",)), node(writes=("a",)))) == ((0, 1),)
    scratch = (TemporaryRequirement("scratch", 16),)
    assert contiguous_parallel_groups((node(temporaries=scratch), node(), node(temporaries=scratch))) == ((0, 1, 2),)
    assert not contiguous_parallel_groups((node(side_effects=("random",)), node()))
    # Structural partitions have no guessed branch-count admission ceiling.
    assert len(contiguous_parallel_groups(tuple(node(writes=(f"a{i}",)) for i in range(65)))) == 65


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_selected_parallel_binding_alias_failure_is_cold():
    @ti.kernel
    def copy(x: ti.types.ndarray(ti.i32, 1), y: ti.types.ndarray(ti.i32, 1)):
        ti.loop_config(block_dim=64)
        for i in range(17):
            y[i] = x[i] + 1

    @ti.kernel
    def publish(header: ti.types.ndarray(ti.i32, 1)):
        header[0] = 91

    symbols = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
               for name in ("x0", "x1", "y0", "y1", "header")}
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    builder.dispatch(copy, symbols["x0"], symbols["y0"])
    builder.dispatch(copy, symbols["x1"], symbols["y1"])
    builder.dispatch(publish, symbols["header"])
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragment = next(f for f in catalog.fragments if f.provider_namespace == "taichi_forge.graph.branch_join_schedule")
    recipe = catalog.compose((fragment.fragment_id,), stage="binding-alias")
    arrays = {name: ti.ndarray(ti.i32, 17 if name != "header" else 1) for name in symbols}
    for name, value in arrays.items():
        value.fill(3 if name.startswith("x") else 0)
    queue = ti.graph.execution_domain(instance_capacity=1)
    with definition.materialization_context() as context:
        with context.materialize(recipe.recipe) as product:
            # Admission must precede upload or capacity consumption.
            with pytest.raises((ValueError, RuntimeError), match="overlap|disjoint"):
                product.executor.prepare(dict(arrays, y1=arrays["y0"]), execution_domain=queue)
            assert queue.latest_completion().info["sequence"] == 0
            assert not np.any(arrays["header"].to_numpy())
            call = product.executor.prepare(arrays, execution_domain=queue)
            call.initialize()
            assert all(kernel.actual_block_size == 64 for kernel in product.manifest.kernels
                       if kernel.properties["dispatch_index"] < 2)
            done = call.submit()
            assert done is not None
            done.wait()
            for name in ("y0", "y1"):
                np.testing.assert_array_equal(arrays[name].to_numpy(), np.full(17, 4))
            assert arrays["header"].to_numpy()[0] == 91
            done.close()
            call.close()
    queue.close()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_cpu_does_not_discard_a_parallel_schedule():
    @ti.kernel
    def write(out: ti.types.ndarray(ti.i32, 1)):
        out[0] = 7

    out = ti.ndarray(ti.i32, 1)
    out.fill(0)
    symbol = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.i32, ndim=1)
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    for _ in range(3):
        builder.dispatch(write, symbol)
    builder._set_parallel_dispatch_groups(((0,), (1,)))
    graph = builder.compile()
    assert not graph.definition._runtime_spec._graph_branch_join_sources
    with pytest.raises(RuntimeError, match="prepare.*execution_domain"):
        graph.run(dict(out=out))
    assert out.to_numpy()[0] == 0
    graph.close()


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_selected_parallel_field_copy_publication_jit_and_aot(tmp_path):
    backend = ti.lang.impl.current_cfg().arch
    backend_name = "cuda" if backend == ti.cuda else "vulkan"
    count, lanes = 17, 3
    sources = [ti.Vector.field(3, ti.f32, shape=count) for _ in range(lanes)]
    outputs = [ti.Vector.field(3, ti.f32, shape=(2, count)) for _ in range(lanes)]
    counters = [ti.field(ti.i32, shape=()) for _ in range(lanes)]
    epochs = ti.field(ti.u64, shape=2)
    host = [np.arange(count * 3, dtype=np.float32).reshape(count, 3) + lane * 100 for lane in range(lanes)]
    for source, values in zip(sources, host):
        source.from_numpy(values)
    for output in outputs:
        output.fill(-1)
    epochs.fill(0)

    def copier(source, output, counter):
        @ti.kernel
        def copy(slot: ti.i32):
            counter[None] = slot + 10
            for i in range(count):
                output[slot, i] = source[i]
        return copy

    @ti.kernel
    def publish(slot: ti.i32, epoch: ti.u64):
        epochs[slot] = epoch

    @ti.kernel
    def readback(out: ti.types.ndarray(ti.f32, 1), header: ti.types.ndarray(ti.u64, 1), slot: ti.i32):
        for i in range(count):
            for lane in ti.static(range(lanes)):
                for component in ti.static(range(3)):
                    out[(lane * count + i) * 3 + component] = outputs[lane][slot, i][component]
        header[0] = epochs[slot]
        for lane in ti.static(range(lanes)):
            header[lane + 1] = counters[lane][None]

    slot = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "slot", ti.i32)
    epoch = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "epoch", ti.u64)
    builder = ti.graph.GraphBuilder(_explicit_map_source_groups=())
    for lane in range(lanes):
        builder.dispatch(copier(sources[lane], outputs[lane], counters[lane]), slot)
    builder.dispatch(publish, slot, epoch)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragment = next(f for f in catalog.fragments if f.provider_namespace == "taichi_forge.graph.branch_join_schedule")
    assert len(fragment.tasks) == 4
    recipe = catalog.compose((fragment.fragment_id,), stage="snapshot-publication")
    artifact = tmp_path / "parallel.tcm"
    module = ti.aot.Module(caps=["spirv_has_int64"] if backend == ti.vulkan else None)
    queue = ti.graph.execution_domain(capacity=2, instance_capacity=2, gpu_timing=True)
    with definition.materialization_context() as context:
        with context.materialize(recipe.recipe) as product:
            graph = product.executor
            tasks = product.manifest.tasks
            # Each branch has a serial offload followed by a range offload;
            # all physical tasks, not only the first, retain the branch queue.
            for lane in range(lanes):
                branch = [task for task in tasks if task.properties["dispatch_index"] == lane]
                assert len(branch) >= 2
                assert all(task.queue == f"{backend_name}_branch:{lane}" for task in branch)
            join = next(task for task in tasks if task.properties["dispatch_index"] == lanes)
            assert len(join.depends_on) == lanes
            before = graph._spec.pipeline_definition
            plan = module.add_execution_plan("step", graph, snapshot_state=True)
            data = json.loads(plan.to_json())
            assert data["operations"][0]["kind"] == f"{backend_name}_parallel"
            assert len(data["operations"][0]["branches"]) == lanes
            assert len(data["operations"]) == 2
            module.add_kernel(readback)
            module.archive(str(artifact))
            module.archive(str(tmp_path / "again.tcm"))
            assert graph._spec.pipeline_definition == before
            args = dict(slot=0, epoch=1)
            if backend == ti.vulkan:
                with pytest.raises(RuntimeError, match="prepare.*execution_domain|flat fixed"):
                    graph.run(args)
                assert not np.any(epochs.to_numpy())
            calls = [graph.prepare(dict(args, slot=slot_index), execution_domain=queue, mutable_parameters=("epoch",))
                     for slot_index in range(2)]
            for call in calls:
                call.initialize()
            graph.close()
            for turn in range(2):
                values = dict(epoch=(1 << 63) + turn)
                done = calls[turn].submit(values=values, sample_gpu=True)
                values["epoch"] = 0
                assert done is not None
                done.wait()
                assert done.gpu_timing["status"] == "ready"
                for output, want in zip(outputs, host):
                    np.testing.assert_array_equal(output.to_numpy()[turn], want)
                assert epochs[turn] == (1 << 63) + turn
                assert all(counter[None] == turn + 10 for counter in counters)
                done.close()
            last = calls[0].submit(values=dict(epoch=7))
            ti.reset()
            assert last.done()
            with pytest.raises(RuntimeError, match="reset|closed|retired"):
                calls[0].submit(values=dict(epoch=8))
            for call in calls:
                call.close()
            queue.close()
            last.close()
    result = subprocess.run(
        [sys.executable, "-S", str(Path(__file__).with_name("aot_parallel_field_probe.py")), backend_name, str(artifact)],
        capture_output=True, text=True, env=os.environ.copy(), timeout=90,
    )
    (tmp_path / "consumer.log").write_text(result.stdout + result.stderr, encoding="utf8")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PARALLEL_FIELD_OK" in result.stdout
    assert not any(error in result.stderr for error in ("VUID-", "SYNC-HAZARD", "cleanup failed"))
