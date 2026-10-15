"""BSR native resources and automatic recording export preserve block storage."""

import array
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
from tests.python.aot_bsr_probe import fixture, native


def require_bsr():
    with ti.aot.borrow_runtime() as runtime:
        if not runtime.probe_native_operation("cusparse_bsr_spmv_f32")["available"]:
            pytest.skip("Compatible external cuSPARSE BSR provider required")


def consume(artifact, kind):
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_bsr_probe.py")),
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
        assert "returned status" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_bsr_live_resources_and_explicit_adjoint(tmp_path):
    require_bsr()

    @ti.kernel
    def energy(y: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in y:
            loss[0] += 0.5 * y[i] * y[i]

    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="energy_grad")
    offsets, columns, _, toffsets, tcolumns, _, _ = fixture(4)
    for dtype, code in (("i32", "i"), ("i64", "q")):
        resources = [
            dict(
                kind="ndarray",
                name=name,
                dtype=dtype,
                shape=[len(data)],
                initial_data=module.add_data(array.array(code, data)),
            )
            for name, data in (("r", offsets), ("c", columns), ("tr", toffsets), ("tc", tcolumns))
        ]
        resources += [
            dict(
                native(
                    "bsr_f32",
                    dict(row_offsets=r, column_indices=c, values=v),
                    block_rows=rows,
                    block_columns=cols,
                    block_size=4,
                ),
                name=name,
            )
            for name, rows, cols, r, c, v in (
                ("a", 3, 4, "r", "c", "values"),
                ("at", 4, 3, "tr", "tc", "transpose_values"),
            )
        ]
        resources += [
            dict(kind="ndarray", name=name, dtype="f32", shape=[size]) for name, size in (("dy", 12), ("dloss", 1))
        ]
        ops = [dict(kind="fill32", destination=name, value=0) for name in ("dy", "loss")]
        ops += [
            dict(kind="fill32", destination="dloss", value=0x3F800000),
            dict(kind="copy", source="y0", destination="y"),
            native("bsr_spmv_f32", dict(matrix="a", input="x", output="y"), alpha=1.5, beta=-0.25),
            dict(kind="kernel", entry="energy", arguments=["y", "loss"]),
            dict(kind="kernel", entry="energy_grad", arguments=["y", "loss"]),
            native("bsr_spmv_f32", dict(matrix="at", input="dy", output="dx"), alpha=1.5),
        ]
        module.add_execution_plan(
            dtype, ti.aot.ExecutionPlan(ops, resources=resources, gradients=dict(y="dy", loss="dloss"))
        )
    artifact = tmp_path / "bsr-live.tcm"
    module.archive(str(artifact))
    consume(artifact, "live")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_bsr_recording_snapshots_reexport_and_selected_graph(tmp_path):
    require_bsr()
    offsets, columns, values, toffsets, tcolumns, tvalues, dense = fixture(2, ordered=True)

    def upload(dtype, data):
        result = ti.ndarray(dtype, len(data))
        result.from_numpy(np.asarray(data, np.int32 if dtype == ti.i32 else np.float32))
        return result

    value_arrays = [upload(ti.f32, v) for v in (values, tvalues)]
    patterns = [
        ti.linalg.SparsePattern.bsr(r, c, 2, upload(ti.i32, off), upload(ti.i32, col))
        for r, c, off, col in ((3, 4, offsets, columns), (4, 3, toffsets, tcolumns))
    ]
    matrix, transpose = [pattern.matrix(v) for pattern, v in zip(patterns, value_arrays)]

    @ti.kernel
    def seed(dy: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in dy:
            dy[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(y: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in y:
            loss[0] += 0.5 * y[i] * y[i]

    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
        for name in ("x", "y", "check", "dy", "loss", "dloss", "dx")
    }
    recording = ti.hardware.linalg.CusparseSpmvRecording
    builder = ti.graph.GraphBuilder()
    builder.dispatch(seed, args["dy"], args["loss"], args["dloss"])
    builder.append_native(recording(matrix, input="x", output="y"))
    builder.append_native(recording(matrix, input="x", output="check"))
    builder.dispatch(energy, args["y"], args["loss"])
    builder.dispatch(energy.grad, args["y"], args["loss"])
    builder.append_native(recording(transpose, input="dy", output="dx"))
    definition = builder.freeze()
    graph = builder.compile()
    module = ti.aot.Module()

    def export(name, selected):
        before = [m._debug_runtime_stats()["transfers"]["device_to_host_bytes"] for m in (matrix, transpose)]
        plan = module.add_execution_plan(name, selected, gradients=dict(y="dy", loss="dloss"))
        after = [m._debug_runtime_stats()["transfers"]["device_to_host_bytes"] for m in (matrix, transpose)]
        assert [b - a for a, b in zip(before, after)] == [116, 120]
        document = json.loads(plan.to_json())
        resources = [r for r in document["resources"] if r["kind"] == "native"]
        assert len(resources) == 2 and all(r["operation"] == "bsr_f32" for r in resources)
        assert resources[0]["parameters"] == dict(block_rows=3, block_columns=4, block_size=2)
        operations = [op for op in document["operations"] if op["kind"] == "native"]
        assert all(op["operation"] == "bsr_spmv_f32" for op in operations)
        assert operations[0]["bindings"]["matrix"] == operations[1]["bindings"]["matrix"]
        return plan

    frozen = export("original", graph).to_json()
    for m, v, host in zip((matrix, transpose), value_arrays, (values, tvalues)):
        v.from_numpy(2 * np.asarray(host, np.float32))
        m.update_values(v)
    export("updated", graph)
    with definition.materialize() as selected:
        export("selected", selected.executor)
    assert module._execution_plans["original"].to_json() == frozen
    assert len(module._exported_kernels) == 3
    x, y = upload(ti.f32, np.arange(8)), ti.ndarray(ti.f32, 6)
    recording(matrix).execute(dict(input=x, output=y))
    np.testing.assert_allclose(y.to_numpy(), 2 * np.asarray(dense) @ np.arange(8), atol=2e-6)
    graph.close()
    del graph, builder, definition, matrix, transpose, patterns, value_arrays, x, y
    gc.collect()
    artifact = tmp_path / "bsr-frozen.tcm"
    module.archive(str(artifact))
    consume(artifact, "frozen")
