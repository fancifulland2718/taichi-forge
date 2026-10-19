"""Selected single-queue scratch exports preserve every slot and code sharing."""

import json
import struct
import time
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar
from tests import test_utils
from tests.python.test_graph_resource_lifetime_recipe import (
    _TaggedTemporaryWork,
    _TemporaryAction,
    _providers,
)


def _selection(definition):
    catalog = definition.recipe_catalog(providers=_providers())
    fragment = next(
        f
        for f in catalog.fragments
        if f.fragment_key.endswith(":generation-owned-storage")
    )
    return catalog, catalog.composer.compose((fragment,))


@pytest.mark.parametrize(
    "capacity,size,dtype", [(3, 257, ti.i32), (3, 257, ti.f32), (65, 1, ti.f32)]
)
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_selected_temporary_arena(tmp_path, monkeypatch, capacity, size, dtype):
    monkeypatch.setenv("TI_GRAPH_TEMPORARY_ARENA_SLOTS", str(capacity))
    builder = ti.graph.GraphBuilder()
    builder.append_native(_TaggedTemporaryWork(size, dtype))
    definition = builder.freeze()
    catalog, recipe = _selection(definition)
    module = ti.aot.Module()
    with definition.materialization_context(
        provider_set=catalog.provider_set
    ) as context:
        with context.materialize(recipe) as selected:
            graph = selected.executor
            output = ti.ndarray(dtype, size)
            bound = graph.bind(dict(output=output, tag=-3))
            graph.run(bound)
            bound.update(tag=11)
            graph.run(bound)
            expected = [(i * 3 + 11) * 2 + 5 for i in range(size)]
            assert output.to_numpy().tolist() == expected
            began = time.perf_counter_ns()
            exported = module.add_execution_plan("step", graph)
            export_ms = (time.perf_counter_ns() - began) / 1e6
            document = json.loads(exported.to_json())
            assert document["storage_recipe"]["capacity"] == capacity
            assert len(document["storage_recipe"]["arrays"]) == 1
            assert len(module._exported_kernels) == 2

            # Warm replay must not revisit any temporary mapping callback.
            def forbidden(*args, **kwargs):
                raise AssertionError("Warm replay resolved temporary bindings")

            with monkeypatch.context() as replay:
                replay.setattr(_TemporaryAction, "bind_graph_temporaries", forbidden)
                for _ in range(capacity + 2):
                    graph.run(bound)
            assert output.to_numpy().tolist() == expected
        ordered = next(
            f
            for f in catalog.fragments
            if f.fragment_key.endswith(":queue-ordered-storage")
        )
        with context.materialize(catalog.composer.compose((ordered,))) as selected:
            module.add_execution_plan("ordered_reference", selected.executor)
            assert len(module._exported_kernels) == 2
    if dtype == ti.f32:

        @ti.kernel
        def square(
            x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
            y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        ):
            for i in x:
                y[i] = x[i] * x[i]

        module.add_kernel(square)
        module.add_kernel(square.grad, name="reverse")
        # Compose real downstream AD without changing the exported scratch recipe.
        document["operations"].extend(
            [
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="fill32", destination="dy", value=0x3F800000),
                dict(kind="kernel", entry="square", arguments=["output", "y"]),
                dict(kind="kernel", entry="reverse", arguments=["output", "y"]),
            ]
        )
        document["gradients"] = dict(output="dx", y="dy")
        module.add_execution_plan(
            "with_ad", ExecutionPlan.from_json(json.dumps(document))
        )
    artifact = tmp_path / "arena.tcm"
    module.archive(str(artifact))
    measurements = []
    for mode, factory in (
        ("standalone", lambda: Runtime("cuda")),
        ("borrowed", ti.aot.borrow_runtime),
    ):
        with factory() as runtime:
            loaded = runtime.load_module(artifact)
            frames = []
            for tag in (-3, 41):
                names = ("output", "y", "dx", "dy") if dtype == ti.f32 else ("output",)
                arrays = {name: runtime.ndarray(str(dtype), size) for name in names}
                began = time.perf_counter_ns()
                call = loaded.prepare_execution(
                    "with_ad" if dtype == ti.f32 else "step",
                    dict(arrays, tag=Scalar("i32", tag)),
                )
                measurements.append(
                    dict(mode=mode, prepare_ms=(time.perf_counter_ns() - began) / 1e6)
                )
                assert call.info["resource_bytes"] == capacity * size * 4
                assert call.info["dispatch_count"] == (4 if dtype == ti.f32 else 2)
                frames.append((call, arrays, tag))
            for _ in range(capacity + 2):
                for call, _, _ in frames:
                    call()
            for call, arrays, tag in frames:
                expected = [(i * 3 + tag) * 2 + 5 for i in range(size)]

                def read(name):
                    return list(
                        struct.unpack(
                            "<" + ("f" if dtype == ti.f32 else "i") * size,
                            arrays[name].read(),
                        )
                    )

                assert read("output") == expected
                if dtype == ti.f32:
                    assert read("y") == [v * v for v in expected]
                    assert read("dx") == [2 * v for v in expected]
            samples = []
            for _ in range(3):
                began = time.perf_counter_ns()
                frames[0][0]()
                submitted = time.perf_counter_ns()
                runtime.wait()
                samples.append(
                    dict(
                        host_us=(submitted - began) / 1e3,
                        complete_us=(time.perf_counter_ns() - began) / 1e3,
                    )
                )
            measurements[-1]["samples"] = samples
            fresh = loaded.prepare_execution(
                "step",
                dict(output=frames[0][1]["output"], tag=Scalar("i32", 19)),
            )
            fresh()
            packed = frames[0][1]["output"].read()
            assert list(
                struct.unpack("<" + ("f" if dtype == ti.f32 else "i") * size, packed)
            ) == [(i * 3 + 19) * 2 + 5 for i in range(size)]
            fresh.close()
            frames[0][0]()
            frames[0][0].close()
            frames[1][0]()
            loaded.close()
            with pytest.raises(RuntimeError):
                frames[1][0]()
            for call, arrays, _ in frames:
                call.close()
                for value in arrays.values():
                    value.close()
    with ZipFile(artifact) as archive:
        code = [
            archive.read(n)
            for n in archive.namelist()
            if n.endswith((".bc", ".ptx", ".cubin"))
        ]
    (tmp_path / "measurements.json").write_text(
        json.dumps(
            dict(
                capacity=capacity,
                dtype=str(dtype),
                export_ms=export_ms,
                artifact_bytes=artifact.stat().st_size,
                code_entries=len(code),
                code_bytes=sum(map(len, code)),
                measurements=measurements,
            )
        ),
        encoding="utf-8",
    )


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_temporary_arena_checks_all_slots_before_publication(monkeypatch):
    monkeypatch.setenv("TI_GRAPH_TEMPORARY_ARENA_SLOTS", "3")
    builder = ti.graph.GraphBuilder()
    builder.append_native(_TaggedTemporaryWork(5, ti.f32))
    definition = builder.freeze()
    catalog, recipe = _selection(definition)
    with definition.materialization_context(
        provider_set=catalog.provider_set
    ) as context:
        with context.materialize(recipe) as selected:
            graph = selected.executor
            arena, spec = graph._instance._temporary_arena, graph._spec
            resolve = spec.bind_temporary_args
            foreign = ti.ndarray(ti.f32, 5)

            def bad_last_slot(bindings):
                resolved = dict(resolve(bindings))
                if bindings is arena._slots[-1]["bindings"]:
                    resolved[next(iter(resolved))] = foreign
                return resolved

            module = ti.aot.Module()
            with monkeypatch.context() as altered:
                altered.setattr(spec, "bind_temporary_args", bad_last_slot)
                with pytest.raises(
                    ti.TaichiRuntimeError, match="whole ordered storage"
                ):
                    module.add_execution_plan("bad_mapping", graph)
            with monkeypatch.context() as altered:
                altered.setattr(arena, "_slots", arena._slots[:-1])
                with pytest.raises(ti.TaichiRuntimeError, match="eagerly materialized"):
                    module.add_execution_plan("partial_arena", graph)
            with monkeypatch.context() as altered:
                altered.setattr(spec.nodes[0], "parallel_dispatch_groups", ((0,), (1,)))
                with pytest.raises(
                    ti.TaichiRuntimeError, match="qualified ordered CUDA"
                ):
                    module.add_execution_plan("not_ordered", graph)
            with pytest.raises(ValueError, match="cannot own gradient"):
                module.add_execution_plan(
                    "scratch_grad", graph, gradients={"__arena_scratch": "d_scratch"}
                )
            assert (
                not module._content
                and not module._exported_kernels
                and not module._graph_export_failed
            )
            module.add_execution_plan("valid", graph)
