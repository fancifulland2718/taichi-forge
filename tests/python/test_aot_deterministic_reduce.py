"""Deploy the selected stable reduction dispatch and its current topology."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("kind", ["scalar", "vector", "field", "empty"])
def test_deterministic_retained_export(kind, tmp_path):
    keys = np.array([-1] * 6 if kind == "empty" else [1, 0, 1, 2, -1, 1], np.int32)
    vector = kind == "vector"
    make = (
        (lambda n: ti.Vector.ndarray(2, ti.f32, n))
        if vector
        else ((lambda n: ti.field(ti.f32, shape=n)) if kind == "field" else (lambda n: ti.ndarray(ti.f32, n)))
    )
    values, output = make(6), make(3)
    data = np.array([1e20, 3, 1, 7, 99, -1e20], np.float32)
    if vector:
        data = np.stack((data, np.arange(6, dtype=np.float32)), axis=1)
    values.from_numpy(data)
    topology = ti.algorithms.DeterministicScatterReducePlan(keys, 3)
    binding = topology.bind(values, output)
    builder = ti.graph.GraphBuilder()
    builder.append_native(binding.graph_action())
    graph = builder.compile()
    assert graph.physical_plan()["physical_dispatch_count"] == 1
    # A valid changed topology must be sampled from actual dispatch inputs,
    # never silently reconstructed from the original host keys.
    permutation = [0] if kind == "empty" else [5, 2, 0, 1, 3]
    offsets = [0, 0, 0, 0] if kind == "empty" else [0, 0, 3, 5]
    topology.permutation.from_numpy(np.array(permutation, np.int32))
    topology.layout._offsets.from_numpy(np.array(offsets, np.int32))
    graph.run({})
    expected = output.to_numpy()
    reference = [[0, 0], [0, 7], [10, 4]] if vector else ([0, 0, 0] if kind == "empty" else [0, 0, 10])
    np.testing.assert_array_equal(expected, reference)
    output.fill(73)
    module = ti.aot.Module()
    if kind == "field":
        with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state"):
            module.add_execution_plan("missing", graph)
        assert not module._execution_plans
    for name in ("first", "second"):
        plan = module.add_execution_plan(
            name,
            graph,
            snapshot_state=kind == "field",
            source_bindings=None if kind == "field" else dict(values=values, output=output),
        )
        doc = json.loads(plan.to_json())
        assert len(doc["operations"]) == 1 and doc["operations"][0]["kind"] == "kernel"
        assert len([r for r in doc["resources"] if "initial_data" in r]) >= 2
        np.testing.assert_array_equal(output.to_numpy(), np.full_like(expected, 73))
    # Export cannot change the chosen route or poison its producer.
    graph.run({})
    np.testing.assert_array_equal(output.to_numpy(), expected)
    if kind == "field":

        @ti.kernel
        def read_field(dst: ti.types.ndarray(dtype=ti.f32, ndim=1)):
            for i in range(3):
                dst[i] = output[i]

        module.add_kernel(read_field)
    artifact = tmp_path / "deterministic.tcm"
    module.archive(str(artifact))
    if kind == "field":
        values.fill(99)
        with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
            result = runtime.ndarray("f32", 3)
            with loaded.prepare_execution("first", {}) as call, loaded.kernel("read_field").prepare(result) as read:
                call()
                read()
                np.testing.assert_array_equal(np.frombuffer(result.read(), np.float32), expected)
    else:
        artifact.with_suffix(".json").write_text(
            json.dumps(
                dict(
                    vector=vector,
                    values=data.tolist(),
                    expected=expected.tolist(),
                    permutation=permutation,
                    offsets=offsets,
                )
            )
        )
        arch = ti.lang.impl.current_cfg().arch
        backend = "cpu" if arch == ti.cpu else str(arch).split(".")[-1]
        topology.permutation.fill(0)
        topology.layout._offsets.fill(0)
        graph.close()
        ti.reset()
        for mode in ("standalone", "borrowed"):
            result = subprocess.run(
                [
                    sys.executable,
                    *(["-S"] if mode == "standalone" else []),
                    str(Path(__file__).with_name("aot_deterministic_reduce_probe.py")),
                    mode,
                    backend,
                    str(artifact),
                ],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=60,
            )
            text = result.stdout + result.stderr
            assert result.returncode == 0, text
            assert not any(w in text for w in ("Validation Error", "SYNC-HAZARD", "Exception ignored")), text


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_deterministic_export_rejects_changed_source_owner():
    values, output = ti.ndarray(ti.i32, 3), ti.ndarray(ti.i32, 2)
    binding = ti.algorithms.DeterministicScatterReducePlan(np.array([1, 0, 1], np.int32), 2).bind(values, output)
    executable = binding.graph_action().compile()
    executable._array_sources[next(iter(executable._array_sources))] = output
    # Exercise the actual lowering adapter through a frozen graph lease.
    builder = ti.graph.GraphBuilder()
    builder.append_native(binding.graph_action())
    graph = builder.compile()
    lease = next(x for x in graph._spec.lifetime_leases if type(x) is type(executable))
    lease._array_sources[next(iter(lease._array_sources))] = output
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="exact source ndarray"):
        module.add_execution_plan("bad", graph, source_bindings=dict(values=values, output=output))
    assert not module._execution_plans and not module._exported_kernels
