"""Mesh relation tables and attributes use the ordinary dense root contract."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False, require=ti.extension.mesh)
def test_aot_mesh_snapshot_isolation_and_suffix_gradients(tmp_path):
    unrelated = ti.field(ti.i32, shape=1)
    unrelated[0] = 13
    source = Path(__file__).with_name("ell.json")
    metadata = json.loads(source.read_text())
    builder = ti.lang.mesh._TetMesh()
    builder.verts.place({"value": ti.f32})
    model = builder.build(ti.Mesh.load_meta(str(source)))
    values = np.arange(1, len(model.verts) + 1, dtype=np.float32)
    model.verts.value.from_numpy(values)
    relation = next(r for r in metadata["relations"] if r["from_order"] == 3 and r["to_order"] == 0)
    expected = values[np.array(relation["value"]).reshape(-1, 4)].sum(axis=1).tolist()

    @ti.kernel
    def gather(y: ti.types.ndarray(ti.f32, 1)):
        for cell in model.cells:
            total = 0.0
            for j in range(cell.verts.size):
                total += cell.verts[j].value
            y[cell.id] = total

    @ti.kernel
    def energy(y: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in y:
            loss[i] = y[i] * y[i]

    @ti.kernel
    def advance(delta: ti.f32):
        for vertex in model.verts:
            vertex.value += delta

    reference = ti.ndarray(ti.f32, len(expected))
    gather(reference)
    assert reference.to_numpy().tolist() == expected

    graph_builder = ti.graph.GraphBuilder()
    y = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1)
    loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    delta = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "delta", ti.f32)
    graph_builder.dispatch(gather, y)
    graph_builder.dispatch(energy, y, loss)
    graph_builder.dispatch(energy.grad, y, loss)
    graph_builder.dispatch(advance, delta)
    graph = graph_builder.compile()
    dependencies = list(graph._compiled_graph._snode_tree_dependency_info)
    assert len(dependencies) == 2 and all(entry[0] > 0 for entry in dependencies)
    start = time.perf_counter()
    arch = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    module = ti.aot.Module(artifact_format="ptx" if arch == "cuda" else "bc")
    module.add_execution_plan("step", graph, gradients={"y": "dy", "loss": "dloss"}, snapshot_state=True)
    artifact = tmp_path / "mesh.tcm"
    module.archive(str(artifact))
    export_ms = (time.perf_counter() - start) * 1000
    with ZipFile(artifact) as archive:
        saved = json.loads(archive.read("aot_metadata.json"))
    assert saved["resource_binding"] == "dense_v1"
    assert {entry["tree_id"] for entry in saved["initial_root_data"]} == {0, 1}
    (tmp_path / "producer.json").write_text(json.dumps(dict(export_ms=export_ms, bytes=artifact.stat().st_size)))
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_mesh_resources_probe.py")),
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
