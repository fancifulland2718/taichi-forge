"""Keep selected segmented reduction code, topology and scratch in its AOT closure."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.aot_segmented_reduce_probe import expected
from tests.python.aot_segmented_scan_probe import inputs, read, write
from tests.python.test_graph_segmented_reduce_recipe import _recipes


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize('dtype', (ti.i32, ti.u32))
def test_selected_segmented_reduce_runtime_only_and_borrowed(dtype, tmp_path, monkeypatch):
    offsets = [0, 0, 1, 32, 32, 1057, 4099, 4099]
    capacity = offsets[-1] + 7
    layout = ti.algorithms.SegmentedLayout.from_offsets(np.asarray(offsets, np.int32), capacity=capacity)
    values = ti.ndarray(dtype, capacity)
    output = ti.ndarray(dtype, len(offsets) - 1)
    builder = ti.graph.GraphBuilder()
    builder.segmented_reduce(values, layout, output)
    definition = builder.freeze()
    module = ti.aot.Module()
    strategies = []
    for recipe in _recipes(definition):
        with definition.materialize(recipe) as selected:
            executable = selected.executor._spec.nodes[0].executable
            strategy = executable._manifest.strategy
            output.fill(73)
            with monkeypatch.context() as cold:
                # Export must use the frozen host description, never read
                # device topology or run the selected reduction to capture it.
                cold.setattr(type(layout._offsets), 'to_numpy', lambda *a, **k: pytest.fail('device read during export'))
                plan = module.add_execution_plan(
                    strategy, selected.executor, source_bindings=dict(input=values, result=output)
                )
            np.testing.assert_array_equal(output.to_numpy(), [73] * output.shape[0])
            document = json.loads(plan.to_json())
            resources = document['resources']
            frozen = [r for r in resources if 'initial_data' in r]
            scratch = [r for r in resources if r['kind'] == 'ndarray' and 'initial_data' not in r]
            partial = strategy == 'chunk_partial_finalize'
            assert len(frozen) == (2 if partial else 1)
            assert len(scratch) == int(partial)
            assert len(document['operations']) == (2 if partial else 1)
            assert all(op['kind'] == 'kernel' for op in document['operations'])
            assert all(r.get('shape') != [capacity] for r in resources)  # No unused per-item IDs.
            if partial:
                assert sorted(r['shape'][0] for r in frozen) == sorted(len(b) for b in executable._source.partial_layout)
                assert scratch[0]['shape'] == [len(executable._source.partial_layout[0]) - 1]
            host = inputs(capacity, str(dtype), 1)
            values.from_numpy(np.asarray(host, dtype=np.int32 if dtype == ti.i32 else np.uint32))
            selected.executor.run({})
            np.testing.assert_array_equal(output.to_numpy(), expected(offsets, host, str(dtype)))
            strategies.append(strategy)
    assert set(strategies) == {'segment_serial', 'warp_segment', 'block_segment', 'chunk_partial_finalize'}
    artifact = tmp_path / 'segmented-reduce.tcm'
    module.archive(str(artifact))
    artifact.with_suffix('.json').write_text(
        json.dumps(dict(dtype=str(dtype), capacity=capacity, offsets=offsets, strategies=strategies)), encoding='utf8'
    )
    for mode in ('standalone', 'borrowed'):
        result = subprocess.run(
            [sys.executable, *(['-S'] if mode == 'standalone' else []),
             str(Path(__file__).with_name('aot_segmented_reduce_probe.py')), mode, str(artifact)],
            env=os.environ.copy(), capture_output=True, text=True, timeout=45,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_segmented_reduce_scopes_empty_and_ordinary_consumer(tmp_path):
    module = ti.aot.Module()
    builder = ti.graph.GraphBuilder()
    sources = {}
    offsets = [0, 0, 31, 129]
    layout = ti.algorithms.SegmentedLayout.from_offsets(np.asarray(offsets, np.int32), capacity=130)
    for names in (('input', 'result'), ('values', 'reduced')):
        values, output = ti.ndarray(ti.i32, 130), ti.ndarray(ti.i32, 3)
        sources.update(zip(names, (values, output)))
        builder.segmented_reduce(values, layout, output)

    @ti.kernel
    def consume(a: ti.types.ndarray(ti.i32, 1), b: ti.types.ndarray(ti.i32, 1), c: ti.types.ndarray(ti.i32, 1)):
        for i in a:
            c[i] = a[i] ^ b[i]

    builder.dispatch(consume, *(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1)
                                for name in ('result', 'reduced', 'combined')))
    definition = builder.freeze()
    with definition.materialize(definition.recipe_catalog().baseline.recipe) as selected:
        plan = module.add_execution_plan('pair', selected.executor, source_bindings=sources)
        doc = json.loads(plan.to_json())
        assert len([r for r in doc['resources'] if 'initial_data' in r]) == 1  # Shared frozen topology.
        assert len(doc['operations']) == 3
    layout = ti.algorithms.SegmentedLayout.from_offsets(np.zeros(4, np.int32), capacity=7)
    empty_values, empty_output = ti.ndarray(ti.u32, 7), ti.ndarray(ti.u32, 3)
    empty = ti.graph.GraphBuilder()
    empty.segmented_reduce(empty_values, layout, empty_output)
    definition = empty.freeze()
    with definition.materialize(definition.recipe_catalog().baseline.recipe) as selected:
        module.add_execution_plan('empty', selected.executor, source_bindings=dict(input=empty_values, result=empty_output))
    artifact = tmp_path / 'scopes.tcm'
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        args = {name: runtime.ndarray('i32', owner.shape[0]) for name, owner in sources.items()}
        args['combined'] = runtime.ndarray('i32', 3)
        with loaded.prepare_execution('pair', args) as call:
            for iteration in (1, 0):
                left, right = inputs(130, 'i32', iteration), inputs(130, 'i32', -iteration)
                write(args['input'], left, 'i32')
                write(args['values'], right, 'i32')
                call()
                assert read(args['combined'], 'i32') == [a ^ b for a, b in zip(expected(offsets, left, 'i32'), expected(offsets, right, 'i32'))]
        args = dict(input=runtime.ndarray('u32', 7), result=runtime.ndarray('u32', 3))
        write(args['result'], [73] * 3, 'u32')
        with loaded.prepare_execution('empty', args) as call:
            call()
            assert read(args['result'], 'u32') == [0] * 3
