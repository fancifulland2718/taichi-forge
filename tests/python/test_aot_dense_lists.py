"""Non-demoted dense loops own lists independently of consumer compile options."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False, demote_dense_struct_fors=False)
def test_aot_non_demoted_dense_lists(tmp_path):
    _dense_lists_module(tmp_path)


@test_utils.test(arch=ti.cuda, offline_cache=False, demote_dense_struct_fors=False, cuda_listgen_reuse=False)
def test_aot_dense_lists_without_cuda_reuse(tmp_path):
    _dense_lists_module(tmp_path)


def _dense_lists_module(tmp_path):
    unrelated = ti.field(ti.i32, shape=1)
    unrelated[0] = 91
    values = ti.field(ti.f32, needs_grad=True)
    first = ti.FieldsBuilder()
    first.dense(ti.i, 4).dense(ti.i, 2).place(values, values.grad)
    first.finalize()
    bias = ti.field(ti.f32)
    second = ti.FieldsBuilder()
    second.dense(ti.i, 1).place(bias)
    second.finalize()

    @ti.kernel
    def seed(scale: ti.f32):
        for i in range(8):
            values[i] = scale * (i + 1)
            values.grad[i] = 0
        bias[0] = 3

    @ti.kernel
    def energy(loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in values:
            loss[0] += 0.5 * values[i] * values[i] * bias[0]

    @ti.kernel
    def gradient(out: ti.types.ndarray(ti.f32, 1)):
        for i in range(8):
            out[i] = values.grad[i]

    start = time.perf_counter()
    arch = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    module = ti.aot.Module(artifact_format="ptx" if arch == "cuda" else "bc")
    for kernel in (seed, energy, gradient):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    graph_builder = ti.graph.GraphBuilder()
    graph_builder.dispatch(seed, ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32))
    loss_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    graph_builder.dispatch(energy, loss_arg)
    graph_builder.dispatch(energy.grad, loss_arg)
    graph = graph_builder.compile()
    module.add_graph("step", graph)
    # Snapshot a separately prepared graph so list-based roots also exercise
    # the existing dense initial-state and storage-view contract.
    seed(2)
    module.add_execution_plan("snapshot", graph, gradients={"loss": "dloss"}, snapshot_state=True)
    artifact = tmp_path / "dense-lists.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        metadata = json.loads(archive.read("aot_metadata.json"))
    assert metadata["resource_binding"] == "list_v1"
    assert metadata["schema_version"] == 17
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(export_ms=(time.perf_counter() - start) * 1000, bytes=artifact.stat().st_size))
    )
    graph.close()
    ti.reset()
    for mode in ("standalone", "borrowed", "borrowed_lists"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_dense_lists_probe.py")),
                arch,
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Failed to retire" not in result.stdout + result.stderr
