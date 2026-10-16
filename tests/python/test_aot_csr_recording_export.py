"""CSR recordings freeze once per export and share the native AOT resource."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_materialized_mixed_graph_preserves_recording_order(tmp_path):
    if not ti.hardware.linalg.cusparse_is_available():
        pytest.skip("External cuSPARSE is unavailable")
    arrays = []
    for dtype, data in ((ti.i32, [0, 1, 2]), (ti.i32, [0, 1]), (ti.f32, [2, 3])):
        array = ti.ndarray(dtype, len(data))
        array.from_numpy(np.array(data, dtype=np.int32 if dtype == ti.i32 else np.float32))
        arrays.append(array)
    matrix = ti.linalg.SparsePattern.csr(2, 2, arrays[0], arrays[1]).matrix(arrays[2])

    @ti.kernel
    def shift(x: ti.types.ndarray(ti.f32, ndim=1), y: ti.types.ndarray(ti.f32, ndim=1)):
        for i in x:
            y[i] = x[i] + 1

    args = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "tmp", "product", "y")}
    builder = ti.graph.GraphBuilder()
    sequence = builder.create_sequential()
    sequence.dispatch(shift, args["x"], args["tmp"], label="before")
    builder.append(sequence)
    builder.append_native(ti.hardware.linalg.CusparseSpmvRecording(matrix, input="tmp", output="product"))
    builder.dispatch(shift, args["product"], args["y"], label="after")
    definition = builder.freeze()
    module = ti.aot.Module()
    with definition.materialize() as selected:
        graph = selected.executor
        plan = module.add_execution_plan("selected", graph)
        operations = json.loads(plan.to_json())["operations"]
        assert [op["kind"] for op in operations] == ["kernel", "native", "kernel"]
        assert [op["label"] for op in operations if op["kind"] == "kernel"] == ["before", "after"]
        # A broken physical description must still fail before publishing data.
        node = graph._spec.nodes[0]
        complete = node.recording_operations
        try:
            node.recording_operations = complete[:-1]
            invalid = ti.aot.Module()
            with pytest.raises(ti.TaichiRuntimeError, match="does not cover"):
                invalid.add_execution_plan("incomplete", graph)
            assert not invalid._execution_plans and not invalid._data_payloads
        finally:
            node.recording_operations = complete
    artifact = tmp_path / "selected-csr.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        bound = {name: runtime.ndarray("f32", (2,)) for name in args}
        bound["x"].write(np.array([1, 2], np.float32).tobytes())
        prepared = loaded.prepare_execution("selected", bound)
        prepared()
        np.testing.assert_array_equal(np.frombuffer(bound["y"].read(), np.float32), [5, 10])


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_csr_recording_snapshot_composition_and_reverse(tmp_path):
    from taichi_forge_runtime.aot import Runtime

    with Runtime("cuda") as runtime:
        if not runtime.probe_native_operation("cusparse_spmv_f32")["available"]:
            pytest.skip("Compatible external cuSPARSE required")

    def uploaded(dtype, data):
        result = ti.ndarray(dtype, len(data))
        result.from_numpy(np.asarray(data, dtype=np.int32 if dtype == ti.i32 else np.float32))
        return result

    offsets, columns = uploaded(ti.i32, [0, 2, 4]), uploaded(ti.i32, [0, 1, 0, 1])
    values = uploaded(ti.f32, [2, -1, 0.5, 3])
    transpose_values = uploaded(ti.f32, [2, 0.5, -1, 3])
    pattern = ti.linalg.SparsePattern.csr(rows=2, cols=2, row_offsets=offsets, column_indices=columns)
    matrix, transposed = pattern.matrix(values), pattern.matrix(transpose_values)

    @ti.kernel
    def seed(
        dy: ti.types.ndarray(ti.f32, ndim=1),
        loss: ti.types.ndarray(ti.f32, ndim=1),
        dloss: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in dy:
            dy[i] = 0
        loss[0] = 0
        dloss[0] = 1

    @ti.kernel
    def energy(
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), loss: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        for i in y:
            loss[0] += 0.5 * y[i] * y[i]

    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("x", "tmp", "y", "dy", "loss", "dloss", "dtmp", "dx")
    }
    builder = ti.graph.GraphBuilder()
    child = builder.create_sequential()
    recording = ti.hardware.linalg.CusparseSpmvRecording
    child.dispatch(seed, args["dy"], args["loss"], args["dloss"])
    builder.append(child)
    builder.append_native(recording(matrix, input="x", output="tmp"))
    builder.append_native(recording(matrix, input="tmp", output="y"))
    builder.dispatch(energy, args["y"], args["loss"])
    builder.dispatch(energy.grad, args["y"], args["loss"])
    builder.append_native(recording(transposed, input="dy", output="dtmp"))
    builder.append_native(recording(transposed, input="dtmp", output="dx"))
    graph = builder.compile()
    module = ti.aot.Module()

    def export(name):
        before = [m._debug_runtime_stats()["transfers"]["device_to_host_bytes"] for m in (matrix, transposed)]
        plan = module.add_execution_plan(name, graph, gradients=dict(y="dy", loss="dloss"))
        after = [m._debug_runtime_stats()["transfers"]["device_to_host_bytes"] for m in (matrix, transposed)]
        assert [b - a for a, b in zip(before, after)] == [44, 44]
        doc = json.loads(plan.to_json())
        resources = [r for r in doc["resources"] if r["kind"] == "native"]
        assert len(resources) == 2
        assert resources[0]["bindings"]["row_offsets"] == resources[1]["bindings"]["row_offsets"]
        assert resources[0]["bindings"]["column_indices"] == resources[1]["bindings"]["column_indices"]
        assert resources[0]["bindings"]["values"] != resources[1]["bindings"]["values"]
        operations = [op for op in doc["operations"] if op["kind"] == "native"]
        matrices = [op["bindings"]["matrix"] for op in operations]
        assert matrices[0] == matrices[1] and matrices[2] == matrices[3] and matrices[0] != matrices[2]
        return plan

    first = export("original")
    first_json = first.to_json()
    # The producer still has live coefficients; a later export takes a new snapshot.
    values.from_numpy(np.array([4, -2, 1, 6], np.float32))
    transpose_values.from_numpy(np.array([4, 1, -2, 6], np.float32))
    matrix.update_values(values)
    transposed.update_values(transpose_values)
    export("updated")
    assert first.to_json() == first_json
    assert len(module._exported_kernels) == 3
    # Equal topology payloads across the two matrices/exports share archive data.
    assert len(module._data_payloads) == 6
    empty_matrix = ti.linalg.SparseMatrixBuilder(2, 2, max_num_triplets=1).build()
    empty_builder = ti.graph.GraphBuilder()
    empty_builder.append_native(recording(empty_matrix, input="x", output="y"))
    empty_graph = empty_builder.compile()
    module.add_execution_plan("empty", empty_graph)
    module.add_execution_plan("empty_live", empty_graph, source_bindings=dict(coefficients=empty_matrix))
    empty_graph.close()
    del empty_graph, empty_builder, empty_matrix
    live = {
        name: ti.ndarray(ti.f32, 1 if name in ("loss", "dloss") else 2, needs_grad=name in ("y", "loss"))
        for name in args
    }
    live["dy"], live["dloss"] = live["y"].grad, live["loss"].grad
    live["x"].from_numpy(np.array([1, -2], np.float32))
    # JIT CGraph does not bind ndarray gradients; validate live coefficients
    # through its direct recording path. AOT executes the actual grad below.
    recording(matrix).execute(dict(input=live["x"], output=live["tmp"]))
    recording(matrix).execute(dict(input=live["tmp"], output=live["y"]))
    a = np.array([[4, -2], [1, 6]], np.float32)
    expected = a @ a @ np.array([1, -2], np.float32)
    np.testing.assert_allclose(live["y"].to_numpy(), expected)
    # A failure after a valid recording must publish no partial data or plan.
    bad_builder = ti.graph.GraphBuilder()
    bad_builder.append_native(recording(matrix))
    predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0)
    bad_builder.observe(predicate)
    bad = bad_builder.compile()
    failed_module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="Observation"):
        failed_module.add_execution_plan("invalid", bad)
    assert not failed_module._data_payloads and not failed_module._execution_plans
    bad.close()
    graph.close()
    del graph, builder, child, bad, bad_builder, matrix, transposed, pattern, live
    gc.collect()
    artifact = tmp_path / "csr-recordings.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_csr_recording_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "returned status" not in result.stdout + result.stderr
