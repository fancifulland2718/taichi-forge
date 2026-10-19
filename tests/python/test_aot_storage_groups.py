"""Independent arena allocations share one whole-plan frame schedule."""

from contextlib import ExitStack
import json
import struct
import time
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests import test_utils
from tests.python.test_aot_retained_graph_instances import _RetainedGraph
from tests.python.test_aot_temporary_arena import _selection
from tests.python.test_graph_resource_lifetime_recipe import _TaggedTemporaryWork


def _read(array):
    return list(struct.unpack("<" + "f" * array.shape[0], array.read()))


def _square(module):
    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i]

    module.add_kernel(square)
    module.add_kernel(square.grad, name="reverse")


@test_utils.test(arch=[ti.cpu, ti.vulkan], offline_cache=False)
def test_grouped_storage_uses_declared_frames_and_independent_allocations(tmp_path):
    @ti.kernel
    def step(
        a: ti.types.ndarray(ti.f32, ndim=1),
        b: ti.types.ndarray(ti.f32, ndim=1),
        old_a: ti.types.ndarray(ti.f32, ndim=1),
        old_b: ti.types.ndarray(ti.f32, ndim=1),
        x: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in x:
            old_a[i], old_b[i] = a[i], b[i]
            a[i] += 1
            b[i] += 1
            x[i] = a[i] + 10 * b[i]

    exported = ti.aot.Module()
    exported.add_kernel(step)
    _square(exported)
    recipe = dict(
        kind="ordered_temporary_ring",
        version=2,
        groups=[dict(capacity=3, arrays=["a"]), dict(capacity=5, arrays=["b"])],
    )
    exported.add_execution_plan(
        "step",
        ExecutionPlan(
            [
                dict(
                    kind="kernel",
                    entry="step",
                    arguments=["a", "b", "old_a", "old_b", "x"],
                ),
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="fill32", destination="dy", value=0x3F800000),
                dict(kind="kernel", entry="square", arguments=["x", "y"]),
                dict(kind="kernel", entry="reverse", arguments=["x", "y"]),
            ],
            resources=[
                dict(
                    kind="ndarray",
                    name=name,
                    dtype="f32",
                    shape=[5],
                    zero_initialize=True,
                )
                for name in ("a", "b")
            ],
            gradients=dict(x="dx", y="dy"),
            storage_recipe=recipe,
        ),
    )
    artifact = tmp_path / "groups.tcm"
    exported.archive(str(artifact))
    arch = "vulkan" if ti.lang.impl.current_cfg().arch == ti.vulkan else "cpu"
    for factory in (lambda: Runtime(arch), ti.aot.borrow_runtime):
        with factory() as runtime:
            module = runtime.load_module(artifact)
            arrays = {
                name: runtime.ndarray("f32", 5)
                for name in ("old_a", "old_b", "x", "y", "dx", "dy")
            }
            call = module.prepare_execution("step", arrays)
            assert call.info["resource_bytes"] == (3 + 5) * 5 * 4
            assert call.info["dispatch_count"] == 3
            counters = [[0] * 3, [0] * 5]
            for epoch in range(16):
                phase = epoch % 5
                before = [counter[phase % len(counter)] for counter in counters]
                for counter in counters:
                    counter[phase % len(counter)] += 1
                call()
                if epoch < 6 or epoch == 15:
                    x = before[0] + 1 + 10 * (before[1] + 1)
                    assert _read(arrays["old_a"]) == [before[0]] * 5
                    assert _read(arrays["old_b"]) == [before[1]] * 5
                    assert _read(arrays["y"]) == [x * x] * 5
                    assert _read(arrays["dx"]) == [2 * x] * 5
            call()
            module.close()
            with pytest.raises(RuntimeError):
                call()
            call.close()


class _RetainedTagged(_RetainedGraph):
    def __init__(self, graph, output, tag):
        self.graph = graph
        self.bindings = graph.bind(dict(output=output, tag=tag))

    def _export_aot(self, context):
        context.bound_graph(self.graph, self.bindings)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_parent_arena_rejects_image_dispatch_before_publishing(monkeypatch):
    @ti.kernel
    def sample(
        image: ti.types.texture(num_dimensions=2), out: ti.types.ndarray(ti.f32, 1)
    ):
        out[0] = image.fetch(ti.Vector([0, 0]), 0).x

    monkeypatch.setenv("TI_GRAPH_TEMPORARY_ARENA_SLOTS", "3")
    builder = ti.graph.GraphBuilder()
    builder.append_native(_TaggedTemporaryWork(1, ti.f32))
    definition = builder.freeze()
    catalog, recipe = _selection(definition)
    output = ti.ndarray(ti.f32, 1)
    image = ti.Texture(ti.Format.r32f, (1, 1))
    with definition.materialization_context(
        provider_set=catalog.provider_set
    ) as context:
        with context.materialize(recipe) as selected:
            child = _RetainedTagged(selected.executor, output, 3)
            for image_first in (False, True):
                parent = ti.graph.GraphBuilder()
                if not image_first:
                    parent.append_native(child)
                parent.dispatch(
                    sample,
                    ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2),
                    ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "result", ti.f32, ndim=1),
                )
                if image_first:
                    parent.append_native(child)
                graph = parent.compile()
                try:
                    module = ti.aot.Module()
                    with pytest.raises(
                        ti.TaichiRuntimeError, match="ordinary buffer dispatches"
                    ):
                        module.add_execution_plan(
                            "unsupported",
                            graph,
                            source_bindings=dict(image=image, output=output),
                            resource_bindings={image: "public_image"},
                        )
                    assert (
                        not module._exported_kernels and not module._graph_export_failed
                    )
                finally:
                    graph.close()


@pytest.mark.parametrize("capacity,size", [(5, 17), (65, 1)])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_parent_arenas_preserve_instances_capacity_code_and_downstream_ad(
    tmp_path, monkeypatch, capacity, size
):
    builder = ti.graph.GraphBuilder()
    builder.append_native(_TaggedTemporaryWork(size, ti.f32))
    definition = builder.freeze()
    sources = {name: ti.ndarray(ti.f32, size) for name in ("first", "second", "again")}
    exported = ti.aot.Module()
    with ExitStack() as scope:
        children = []
        for count in (3, capacity):
            monkeypatch.setenv("TI_GRAPH_TEMPORARY_ARENA_SLOTS", str(count))
            catalog, recipe = _selection(definition)
            context = scope.enter_context(
                definition.materialization_context(provider_set=catalog.provider_set)
            )
            selected = scope.enter_context(context.materialize(recipe))
            assert selected.executor._instance._temporary_arena.capacity == count
            children.append(selected.executor)
        nodes = [
            _RetainedTagged(children[0], sources["first"], -3),
            _RetainedTagged(children[1], sources["second"], 7),
            _RetainedTagged(children[0], sources["again"], 11),
        ]
        parent = ti.graph.GraphBuilder()
        for node in nodes:
            parent.append_native(node)
        graph = parent.compile()
        scope.callback(graph.close)
        graph.run({})
        expected = {
            name: [(i * 3 + tag) * 2 + 5 for i in range(size)]
            for name, tag in zip(sources, (-3, 7, 11))
        }
        assert {
            name: value.to_numpy().tolist() for name, value in sources.items()
        } == expected
        began = time.perf_counter_ns()
        plan = exported.add_execution_plan("forward", graph, source_bindings=sources)
        export_ms = (time.perf_counter_ns() - began) / 1e6
        document = json.loads(plan.to_json())
        assert document["version"] == 22
        groups = document["storage_recipe"]["groups"]
        assert [group["capacity"] for group in groups] == [3, capacity]
        assert all(len(group["arrays"]) == 1 for group in groups)
        assert groups[0]["arrays"] != groups[1]["arrays"]
        assert len(exported._exported_kernels) == 2
        operations = document["operations"]
        assert len(operations) == 6
        assert operations[0]["arguments"][0] == operations[4]["arguments"][0]
        assert operations[0]["arguments"][0] != operations[2]["arguments"][0]
        assert len({operation["entry"] for operation in operations}) == 2
        assert len([r for r in document["resources"] if r["kind"] == "ndarray"]) == 2
        assert len([r for r in document["resources"] if r["kind"] == "native"]) == 2
        # Rebinding a retained child snapshots new input data without conflating
        # its arena or duplicating the already exported code.
        nodes[1].bindings.update(tag=19)
        changed = exported.add_execution_plan("changed", graph, source_bindings=sources)
        assert (
            json.loads(changed.to_json())["storage_recipe"]
            == document["storage_recipe"]
        )
        assert len(exported._exported_kernels) == 2
        graph.run({})
        assert sources["second"].to_numpy().tolist() == [
            (i * 3 + 19) * 2 + 5 for i in range(size)
        ]
    _square(exported)
    document["operations"].extend(
        [
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="kernel", entry="square", arguments=["first", "y"]),
            dict(kind="kernel", entry="reverse", arguments=["first", "y"]),
        ]
    )
    document["gradients"] = dict(first="dx", y="dy")
    exported.add_execution_plan(
        "with_ad", ExecutionPlan.from_json(json.dumps(document))
    )
    artifact = tmp_path / "parent.tcm"
    exported.archive(str(artifact))
    with ZipFile(artifact) as archive:
        code = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.endswith((".bc", ".cubin", ".ptx", ".spv"))
        }
        assert len(code) == 4
    measurements = []
    for mode, factory in (
        ("standalone", lambda: Runtime("cuda")),
        ("borrowed", ti.aot.borrow_runtime),
    ):
        with factory() as runtime:
            module = runtime.load_module(artifact)
            frames = []
            for _ in range(2):
                arrays = {
                    name: runtime.ndarray("f32", size)
                    for name in (*sources, "y", "dx", "dy")
                }
                started = time.perf_counter_ns()
                call = module.prepare_execution("with_ad", arrays)
                measurements.append(
                    dict(
                        mode=mode,
                        prepare_ms=(time.perf_counter_ns() - started) / 1e6,
                        info=call.info,
                    )
                )
                assert call.info["resource_bytes"] == (3 + capacity) * size * 4
                assert call.info["dispatch_count"] == 8
                frames.append((call, arrays))
            for _ in range(capacity + 2):
                for call, _ in frames:
                    call()
            for call, arrays in frames:
                assert {name: _read(arrays[name]) for name in sources} == expected
                assert _read(arrays["y"]) == [x * x for x in expected["first"]]
                assert _read(arrays["dx"]) == [2 * x for x in expected["first"]]
            changed = module.prepare_execution(
                "changed", {name: frames[0][1][name] for name in sources}
            )
            changed()
            assert _read(frames[0][1]["second"]) == [
                (i * 3 + 19) * 2 + 5 for i in range(size)
            ]
            changed.close()
            call, arrays = frames[0]
            call()
            runtime.wait()
            samples = []
            for _ in range(3):
                began = time.perf_counter_ns()
                call()
                submitted = time.perf_counter_ns()
                runtime.wait()
                samples.append(
                    dict(
                        host_us=(submitted - began) / 1e3,
                        complete_us=(time.perf_counter_ns() - began) / 1e3,
                    )
                )
            measurements[-2]["samples"] = samples
            call()
            arrays["first"].close()
            with pytest.raises(RuntimeError):
                call()
            call.close()
            frames[1][0]()
            module.close()
            with pytest.raises(RuntimeError):
                frames[1][0]()
            frames[1][0].close()
    (tmp_path / "measurements.json").write_text(
        json.dumps(
            dict(
                capacity=capacity,
                size=size,
                export_ms=export_ms,
                artifact_bytes=artifact.stat().st_size,
                code_entries=len(code),
                code_bytes=sum(map(len, code.values())),
                measurements=measurements,
            )
        )
    )
