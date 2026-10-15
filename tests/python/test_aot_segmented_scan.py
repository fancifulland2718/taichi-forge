"""Selected scan recipes export their actual stages and retained storage."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_segmented_scan_probe import expected, inputs


def consume(artifact, kind):
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_segmented_scan_probe.py")),
                mode,
                str(artifact),
                kind,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_retained_scan_native_extent_alias_and_retirement(tmp_path):
    from taichi_forge.graph._retained_scan import scan_workspace_bytes

    module = ti.aot.Module()
    for dtype in (ti.i32, ti.u32):
        for size in (0, 1, 31, 129, 1025, 4099):
            module.add_execution_plan(
                f"{dtype}_{size}",
                ti.aot.ExecutionPlan(
                    [
                        dict(
                            kind="native",
                            provider="cuda_driver",
                            operation="retained_scan",
                            version=1,
                            parameters=dict(size=size, dtype=str(dtype)),
                            bindings=dict(values="values", scratch="scratch"),
                        )
                    ],
                    resources=[
                        dict(
                            kind="ndarray", name="scratch", dtype="u32", shape=[scan_workspace_bytes(size, dtype) // 4]
                        )
                    ],
                ),
            )
    artifact = tmp_path / "scan-native.tcm"
    module.archive(str(artifact))
    consume(artifact, "native")


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("dtype,inclusive", ((ti.i32, False), (ti.u32, True)))
def test_selected_segmented_scan_recipes(dtype, inclusive, tmp_path):
    offsets = [0, 0, 1, 32, 32, 1057, 4099, 4099]
    capacity = offsets[-1] + 7
    values, output = (ti.ndarray(dtype, capacity) for _ in range(2))
    layout = ti.algorithms.SegmentedLayout.from_offsets(np.asarray(offsets, np.int32), capacity=capacity)
    builder = ti.graph.GraphBuilder()
    builder.segmented_scan(values, layout, output, inclusive=inclusive)
    graph = builder.compile()
    search = ti.graph.compileiq_recipe_search(graph)
    module = ti.aot.Module()
    strategies = []
    for recipe_id in search.recipe_ids:
        strategy = search.recipe_manifest(recipe_id)["native_algorithm_recipe_manifest"]["strategy"]
        with search.materialize(dict(domain_fingerprint=search.domain_fingerprint, recipe_id=recipe_id)) as selected:
            values.from_numpy(
                np.asarray(inputs(capacity, str(dtype), 1), dtype=np.int32 if dtype == ti.i32 else np.uint32)
            )
            output.fill(73)
            plan = module.add_execution_plan(
                strategy, selected.executor, source_bindings=dict(input=values, result=output)
            )
            np.testing.assert_array_equal(output.to_numpy(), np.full(capacity, 73))
            document = json.loads(plan.to_json())
            natives = [op for op in document["operations"] if op["kind"] == "native"]
            assert len(natives) == int(strategy == "global_scan_segment_correction")
            if natives:
                assert natives[0]["operation"] == "retained_scan"
            selected.executor.run({})
            np.testing.assert_array_equal(
                output.to_numpy(), expected(offsets, inputs(capacity, str(dtype), 1), str(dtype), inclusive)
            )
        strategies.append(strategy)
    assert set(strategies) == {
        "segment_local_serial",
        "global_scan_segment_correction",
        "warp_chunked_carry",
        "block_chunked_carry",
        "warp_shuffle_carry",
        "block_hierarchical_carry",
        "length_bucket_hybrid",
    }
    graph.close()
    artifact = tmp_path / "segmented.tcm"
    module.archive(str(artifact))
    (tmp_path / "segmented.json").write_text(
        json.dumps(
            dict(dtype=str(dtype), inclusive=inclusive, offsets=offsets, capacity=capacity, strategies=strategies)
        ),
        encoding="utf8",
    )
    consume(artifact, "segmented")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_scan_scopes_and_zero_items(tmp_path):
    module = ti.aot.Module()
    builder = ti.graph.GraphBuilder()
    sources = {}
    for dtype, names, inclusive in ((ti.i32, ("input", "result"), False), (ti.u32, ("values", "scanned"), True)):
        values, output = (ti.ndarray(dtype, 130) for _ in range(2))
        sources.update(zip(names, (values, output)))
        layout = ti.algorithms.SegmentedLayout.from_offsets(np.array([0, 0, 31, 129], np.int32), capacity=130)
        builder.segmented_scan(values, layout, output, inclusive=inclusive)
    definition = builder.freeze()
    catalog = definition.recipe_catalog()
    fragments = [
        fragment
        for fragment in catalog.fragments
        if any(task.kind == "global_inclusive_scan" for task in fragment.tasks)
    ]
    assert len(fragments) == 2
    recipe = catalog.compose(tuple(fragment.fragment_id for fragment in fragments), stage="scan-scopes").recipe
    with definition.materialize(recipe) as selected:
        plan = module.add_execution_plan("pair", selected.executor, source_bindings=sources)
        doc = json.loads(plan.to_json())
        assert len([r for r in doc["resources"] if "initial_data" in r]) == 1  # shared frozen offsets
        assert len([op for op in doc["operations"] if op["kind"] == "native"]) == 2
    empty_values, empty_output = (ti.ndarray(ti.u32, 7) for _ in range(2))
    layout = ti.algorithms.SegmentedLayout.from_offsets(np.zeros(4, np.int32), capacity=7)
    empty = ti.graph.GraphBuilder()
    empty.segmented_scan(empty_values, layout, empty_output)
    graph = empty.compile()
    search = ti.graph.compileiq_recipe_search(graph)
    empty_names = []
    for recipe_id in search.recipe_ids:
        name = "empty_" + str(len(empty_names))
        with search.materialize(dict(domain_fingerprint=search.domain_fingerprint, recipe_id=recipe_id)) as selected:
            module.add_execution_plan(
                name, selected.executor, source_bindings=dict(input=empty_values, result=empty_output)
            )
        empty_names.append(name)
    artifact = tmp_path / "scopes.tcm"
    module.archive(str(artifact))
    graph.close()
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        args = {name: runtime.ndarray(str(owner.dtype), 130) for name, owner in sources.items()}
        with loaded.prepare_execution("pair", args) as call:
            for dtype, names, inclusive in (("i32", ("input", "result"), False), ("u32", ("values", "scanned"), True)):
                from tests.python.aot_segmented_scan_probe import write

                write(args[names[0]], inputs(130, dtype, 1), dtype)
                write(args[names[1]], [73] * 130, dtype)
            call()
            from tests.python.aot_segmented_scan_probe import read

            for dtype, names, inclusive in (("i32", ("input", "result"), False), ("u32", ("values", "scanned"), True)):
                assert read(args[names[1]], dtype) == expected([0, 0, 31, 129], inputs(130, dtype, 1), dtype, inclusive)
        args = {name: runtime.ndarray("u32", 7) for name in ("input", "result")}
        for name in empty_names:
            write(args["result"], [73] * 7, "u32")
            with loaded.prepare_execution(name, args) as call:
                call()
                assert read(args["result"], "u32") == [73] * 7
