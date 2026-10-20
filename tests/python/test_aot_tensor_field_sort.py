"""Cold/warm tensor Field sort export shares native records and module roots."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.algorithms import _algorithms as algorithms
from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode
from tests import test_utils


class _Parent(NativeGraphNode, NativeGraphExecutable):
    def __init__(self, graph, frame):
        self.graph, self.bindings = graph, graph.bind(frame)

    def compile(self):
        return self

    def run(self):
        self.graph.run(self.bindings)

    def _export_aot(self, context):
        context.bound_graph(self.graph, self.bindings)


@pytest.mark.parametrize("case", ["matrix", "vector64", "wide_integer"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_tensor_field_sort_cold_warm_parent_bits_and_gradient(tmp_path, case, monkeypatch):
    vector = case == "vector64"
    rows, cols = (3, 1) if vector else (8, 8) if case == "wide_integer" else (2, 3)
    n = 5 if case == "wide_integer" else 17
    shape, lanes = ((rows,) if vector else (rows, cols)), rows * cols
    dtype = ti.i64 if vector else ti.i32 if case == "wide_integer" else ti.f32
    unsigned = ti.u64 if vector else ti.u32
    np_dtype = np.int64 if vector else np.int32 if case == "wide_integer" else np.float32
    bits_dtype = np.uint64 if vector else np.uint32
    keys = ti.field(ti.f32, n)
    values = ti.Vector.field(rows, dtype, n) if vector else ti.Matrix.field(rows, cols, dtype, n)
    host_keys = ((np.arange(n) * 17) % 7 - 3).astype(np.float32)
    host_keys[:5] = [np.nan, -0.0, 0.0, -np.inf, np.nan]
    host_keys.view(np.uint32)[0] = 0xFFC00013
    host = np.arange(n * lanes, dtype=np_dtype).reshape((n, *shape))
    if vector:
        host += np.int64(1 << 60)
    if case == "matrix":
        host.view(np.uint32).reshape(-1)[1:5] = [0x7FC12345, 0xFFC54321, 0x80000000, 0]
    host.reshape(n, lanes)[:, 0] = np.arange(n)
    keys.from_numpy(host_keys)
    values.from_numpy(host)
    frame = dict(out_keys=ti.ndarray(ti.u32, n), out_values=ti.ndarray(unsigned, n * lanes), ids=ti.ndarray(ti.i32, n))

    @ti.kernel
    def observe(
        out_keys: ti.types.ndarray(ti.u32, 1),
        out_values: ti.types.ndarray(unsigned, 1),
        ids: ti.types.ndarray(ti.i32, 1),
    ):
        for i in range(n):
            out_keys[i] = ti.bit_cast(keys[i], ti.u32)
            if ti.static(vector):
                ids[i] = ti.cast(values[i][0], ti.i32)
            else:
                ids[i] = ti.cast(values[i][0, 0], ti.i32)
        for i, lane in ti.ndrange(n, lanes):
            if ti.static(vector):
                out_values[i * lanes + lane] = ti.bit_cast(values[i][lane], unsigned)
            else:
                out_values[i * lanes + lane] = ti.bit_cast(values[i][lane // cols, lane % cols], unsigned)

    def no_host(*args, **kwargs):
        raise AssertionError("Tensor Field export/replay fell back to host sorting")

    monkeypatch.setattr(algorithms, "_host_stable_sort", no_host)
    sequence = ti.algorithms.primitive_sequence().sort_by_key([keys], values)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    builder.dispatch(
        observe, *(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, a.dtype, ndim=1) for name, a in frame.items())
    )
    graph = builder.compile()
    module = ti.aot.Module(caps=["spirv_has_int64"] if vector and ti.cfg.arch == ti.vulkan else [])
    cold = module.add_execution_plan("cold", graph, snapshot_state=True)
    document = json.loads(cold.to_json())
    sort, observation = document["operations"]
    assert sort["kind"] == "sort" and len(module._exported_kernels) == 1
    assert sequence._calls[0].workspace._prepared_storage_sort_plan is None
    assert keys.to_numpy().tobytes() == host_keys.tobytes()
    assert values.to_numpy().tobytes() == host.tobytes()
    graph.run(frame)
    order = np.argsort(host_keys, kind="stable")
    np.testing.assert_array_equal(frame["out_keys"].to_numpy(), host_keys[order].view(np.uint32))
    np.testing.assert_array_equal(frame["out_values"].to_numpy(), host[order].view(bits_dtype).ravel())
    cached = sequence._calls[0].workspace._prepared_storage_sort_plan
    assert cached is not None and not cached.report()["closed"]
    keys.from_numpy(host_keys)
    values.from_numpy(host)
    module.add_execution_plan("warm", graph, snapshot_state=True)
    assert sequence._calls[0].workspace._prepared_storage_sort_plan is cached
    parent_builder = ti.graph.GraphBuilder()
    parent_builder.append_native(_Parent(graph, frame))
    parent = parent_builder.compile()
    module.add_execution_plan("parent", parent, source_bindings=frame, snapshot_state=True)
    assert len(module._exported_kernels) == 1
    module.add_execution_plan("read", ti.aot.ExecutionPlan([observation]))
    module.add_execution_plan(
        "reset",
        ti.aot.ExecutionPlan(
            [
                dict(kind="copy", source="input_keys", destination=sort["keys"]),
                dict(kind="copy", source="input_values", destination=sort["values"]),
            ],
            resources=document["resources"],
        ),
    )

    @ti.kernel
    def energy(
        ids: ti.types.ndarray(ti.i32, 1),
        source: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        out: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in ids:
            out[i] = (i + 1) * source[ids[i]] ** 2

    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="reverse")
    module.add_execution_plan(
        "suffix",
        ti.aot.ExecutionPlan(
            document["operations"]
            + [
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="kernel", entry="energy", arguments=["ids", "source", "energy"]),
                dict(kind="kernel", entry="reverse", arguments=["ids", "source", "energy"]),
            ],
            resources=document["resources"],
            gradients=dict(source="dx", energy="dy"),
        ),
    )
    source = np.arange(n, dtype=np.float32) / 8 - 1
    cases = []
    for initial in (host_keys, host_keys[::-1].copy()):
        order = np.argsort(initial, kind="stable")
        gradient = np.zeros(n, np.float32)
        gradient[order] = 2 * np.arange(1, n + 1) * source[order]
        cases.append(
            dict(
                keys=initial.view(np.uint32).tolist(),
                sorted_keys=initial[order].view(np.uint32).tolist(),
                sorted_values=host[order].view(bits_dtype).ravel().tolist(),
                ids=order.tolist(),
                gradient=gradient.tolist(),
                energy=((np.arange(n) + 1) * source[order] ** 2).tolist(),
            )
        )
    artifact = tmp_path / f"{case}.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(
        json.dumps(
            dict(
                n=n,
                shape=shape,
                lanes=lanes,
                dtype=str(dtype),
                bits_dtype=str(unsigned),
                values=host.view(bits_dtype).ravel().tolist(),
                source=source.tolist(),
                cases=cases,
            )
        )
    )
    parent.close()
    graph.close()
    sequence.clear()
    assert cached.report()["closed"]
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("aot_tensor_field_sort_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert not any(
            marker in text for marker in ("VUID-", "Validation Error", "SYNC-HAZARD", "cleanup failed")
        ), text
