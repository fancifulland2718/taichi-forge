"""Legacy CGraphs and execution plans reuse exact published kernel identities."""

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


@pytest.mark.parametrize("order", ["explicit_first", "graph_first"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_graph_reuses_exact_exports(tmp_path, order):
    x = ti.field(ti.f32, shape=1, needs_grad=True)
    y = ti.field(ti.f32, shape=1, needs_grad=True)

    @ti.kernel
    def seed(value: ti.f32):
        x[0] = value
        y[0] = 0
        x.grad[0] = 0
        y.grad[0] = 1

    @ti.kernel
    def step(scale: ti.f32):
        for i in range(1):
            y[i] = scale * x[i] * x[i]

    @ti.kernel
    def readback(out: ti.types.ndarray(ti.f32, 1)):
        out[0] = y[0]
        out[1] = x.grad[0]

    start = time.perf_counter()
    builder = ti.graph.GraphBuilder()
    value = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "value", ti.f32)
    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32)
    builder.dispatch(seed, value)
    builder.dispatch(step, scale)
    builder.dispatch(step.grad, scale)
    graph = builder.compile()
    arch = "cuda" if ti.cfg.arch == ti.cuda else "vulkan" if ti.cfg.arch == ti.vulkan else "cpu"
    module = ti.aot.Module(artifact_format="ptx" if arch == "cuda" else None)
    if order == "explicit_first":
        module.add_kernel(seed, name="initialize")
        module.add_kernel(step, name="evaluate")
        module.add_kernel(step.grad, name="differentiate")
    module.add_graph("first", graph)
    module.add_graph("second", graph)
    # The new execution-plan exporter must see entries published by add_graph.
    module.add_execution_plan("third", graph, snapshot_state=True)
    if order == "graph_first":
        # A new explicitly requested public alias remains callable.
        module.add_kernel(step, name="evaluate")
    module.add_kernel(readback)
    artifact = tmp_path / "reuse.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        if arch == "vulkan":
            metadata = json.loads(archive.read("metadata.json"))
            payloads = len(metadata["kernels"])
            aliases = len(metadata.get("kernel_aliases", []))
        else:
            payloads = sum(name.endswith(".ptx" if arch == "cuda" else ".bc") for name in archive.namelist())
            aliases = 0
    metrics = dict(
        order=order,
        arch=arch,
        export_ms=(time.perf_counter() - start) * 1000,
        bytes=artifact.stat().st_size,
        payloads=payloads,
        aliases=aliases,
    )
    artifact.with_suffix(".json").write_text(json.dumps(metrics))
    assert payloads == (5 if order == "graph_first" and arch != "vulkan" else 4), metrics
    if arch == "vulkan":
        assert aliases == (1 if order == "graph_first" else 0), metrics
    assert len(module._exported_kernels) == (5 if order == "graph_first" else 4)
    # Export rewrites only its own copy, never the source Graph's dispatches.
    graph.run(dict(value=3.0, scale=2.0))
    assert y[0] == 18
    assert x.grad[0] == 12
    graph.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_graph_export_reuse_probe.py")),
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
