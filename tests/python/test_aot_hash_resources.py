"""Hash AOT resources retain producer layout and independent runtime storage."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_hash_snode_cpu import _collision_keys


@pytest.mark.parametrize("nested", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_hash_module_isolation_and_gradients(tmp_path, nested):
    cfg = ti.lang.impl.current_cfg()
    cfg.hash_snode_active_list = nested
    cfg.hash_snode_diagnostics = nested
    cfg.hash_snode_compact_child_pool = nested
    unused = ti.field(ti.i32, shape=1)
    unused[0] = 17
    values = ti.field(ti.f32, needs_grad=True)
    builder = ti.FieldsBuilder()
    outer = builder.hash(ti.i, 64, expected_active=4) if nested else builder.hash(ti.i, 64, capacity=8)
    if nested:
        outer.hash(ti.j, 8, capacity=4).dense(ti.k, 2).place(values, values.grad)
    else:
        outer.place(values, values.grad)
    builder.finalize()
    first, second, replacement = _collision_keys(8, 3, domain=64)
    key_a = (first, 1, 0) if nested else (first,)
    key_b = (second, 3, 1) if nested else (second,)
    key_c = (replacement, 2, 0) if nested else (replacement,)

    @ti.kernel
    def clear():
        for i in range(64):
            ti.deactivate(outer, i)

    @ti.kernel
    def seed(scale: ti.f32):
        values[key_a] = 2 * scale
        values[key_b] = 3 * scale
        values.grad[key_a] = 0
        values.grad[key_b] = 0

    @ti.kernel
    def churn(scale: ti.f32):
        ti.deactivate(outer, first)
        values[key_c] = 4 * scale
        values.grad[key_c] = 0
        values.grad[key_b] = 0

    @ti.kernel
    def energy(loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for index in ti.grouped(values):
            loss[0] += 0.5 * values[index] * values[index]

    @ti.kernel
    def gather(out: ti.types.ndarray(ti.f32, 1)):
        for index in ti.grouped(values):
            out[0] += values[index]
            out[1] += values.grad[index]
            out[2] += 1

    start = time.perf_counter()
    arch = {ti.cpu: "cpu", ti.cuda: "cuda", ti.vulkan: "vulkan"}[cfg.arch]
    module = ti.aot.Module(artifact_format={"cpu": "bc", "cuda": "ptx", "vulkan": None}[arch])
    for kernel in (clear, seed, churn, energy, gather):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    graph_builder = ti.graph.GraphBuilder()
    scale_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32)
    loss_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    out_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1)
    graph_builder.dispatch(clear)
    graph_builder.dispatch(seed, scale_arg)
    graph_builder.dispatch(churn, scale_arg)
    graph_builder.dispatch(energy, loss_arg)
    graph_builder.dispatch(energy.grad, loss_arg)
    graph_builder.dispatch(gather, out_arg)
    graph = graph_builder.compile()
    module.add_graph("step", graph)
    artifact = tmp_path / "hash.tcm"
    module.archive(str(artifact))
    (tmp_path / "producer.json").write_text(
        json.dumps(dict(export_ms=(time.perf_counter() - start) * 1000, bytes=artifact.stat().st_size))
    )
    graph.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_hash_resources_probe.py")),
                arch,
                mode,
                str(artifact),
                str(int(nested)),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Failed to retire" not in result.stdout + result.stderr
    with zipfile.ZipFile(artifact) as archive:
        metadata = json.loads(archive.read("metadata.json" if arch == "vulkan" else "aot_metadata.json"))
    if arch == "vulkan":
        assert metadata["metadata_version"] == 17
        assert any(tree["storage_kind"] == 2 for tree in metadata["snode_tree_resources"])
    else:
        assert metadata["resource_binding"] == ("sparse_v2" if arch == "cuda" else "sparse_v1")
