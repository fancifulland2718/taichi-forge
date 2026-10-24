"""Vulkan cooperative MMA keeps its exact deployment and resource contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_cooperative_matrix_aot_graph_and_explicit_adjoint(tmp_path):
    specs = ti.hardware.matrix.cooperative_matrix_specs(executable_only=True)
    if not specs:
        pytest.skip("no executable Vulkan cooperative matrix tuple")
    spec = next(
        (item for item in specs if (item.m, item.n, item.k) == (16, 16, 16)), specs[0]
    )
    a_type = ti.types.matrix(spec.m, spec.k, ti.f16)
    b_type = ti.types.matrix(spec.k, spec.n, ti.f16)
    o_type = ti.types.matrix(spec.m, spec.n, ti.f32)
    da_type = ti.types.matrix(spec.m, spec.k, ti.f32)
    db_type = ti.types.matrix(spec.k, spec.n, ti.f32)

    @ti.kernel
    def cooperative(
        a: ti.types.ndarray(a_type, 1),
        b: ti.types.ndarray(b_type, 1),
        c: ti.types.ndarray(o_type, 1),
        output: ti.types.ndarray(o_type, 1),
    ):
        ti.loop_config(block_dim=spec.subgroup_size)
        for lane in range(spec.subgroup_size):
            ti.hardware.matrix.cooperative_mma_f16_f32(a, b, c, output, lane, spec)

    @ti.kernel
    def seed(
        dy: ti.types.ndarray(o_type, 1),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
    ):
        for i, j in ti.ndrange(spec.m, spec.n):
            dy[0][i, j] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        output: ti.types.ndarray(o_type, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i, j in ti.ndrange(spec.m, spec.n):
            loss[0] += 0.5 * output[0][i, j] * output[0][i, j]

    @ti.kernel
    def adjoint(
        a: ti.types.ndarray(a_type, 1),
        b: ti.types.ndarray(b_type, 1),
        dy: ti.types.ndarray(o_type, 1),
        da: ti.types.ndarray(da_type, 1),
        db: ti.types.ndarray(db_type, 1),
        dc: ti.types.ndarray(o_type, 1),
    ):
        for i, t in ti.ndrange(spec.m, spec.k):
            value = 0.0
            for j in range(spec.n):
                value += dy[0][i, j] * ti.cast(b[0][t, j], ti.f32)
            da[0][i, t] = value
        for t, j in ti.ndrange(spec.k, spec.n):
            value = 0.0
            for i in range(spec.m):
                value += ti.cast(a[0][i, t], ti.f32) * dy[0][i, j]
            db[0][t, j] = value
        for i, j in ti.ndrange(spec.m, spec.n):
            dc[0][i, j] = dy[0][i, j]

    builder = ti.graph.GraphBuilder()
    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)
        for name, dtype in (
            ("a", a_type),
            ("b", b_type),
            ("c", o_type),
            ("output", o_type),
            ("da", da_type),
            ("db", db_type),
            ("dc", o_type),
            ("dy", o_type),
            ("loss", ti.f32),
        )
    }
    dy = args["dy"]
    dloss = builder.private_ndarray("dloss", ti.f32, 1)
    builder.dispatch(seed, dy, args["loss"], dloss)
    builder.dispatch(cooperative, args["a"], args["b"], args["c"], args["output"])
    builder.dispatch(energy, args["output"], args["loss"])
    builder.dispatch(energy.grad, args["output"], args["loss"])
    builder.dispatch(
        adjoint, args["a"], args["b"], dy, args["da"], args["db"], args["dc"]
    )
    graph = builder.compile()
    module = ti.aot.Module(
        caps=[
            "spirv_has_cooperative_matrix",
            "spirv_has_float16",
            "spirv_has_int16",
            "spirv_has_int16_storage",
        ]
    )
    module.add_kernel(cooperative)
    module.add_execution_plan("step", graph, gradients=dict(output="dy", loss="dloss"))
    graph.close()
    artifact = tmp_path / "cooperative.tcm"
    module.archive(str(artifact))
    with ZipFile(artifact) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["metadata_version"] == 10
        requirements = [
            requirement
            for kernel in metadata["kernels"]
            for task in kernel["tasks_attribs"]
            for requirement in task["cooperative_matrix_requirements"]
        ]
        assert requirements
        assert all(
            item == dict(m=spec.m, n=spec.n, k=spec.k, subgroup_size=spec.subgroup_size)
            for item in requirements
        )
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_cooperative_matrix_probe.py")),
                mode,
                str(artifact),
                str(spec.m),
                str(spec.n),
                str(spec.k),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "COOPERATIVE_AOT_OK" in result.stdout
        for marker in ("VUID-", "Validation Error", "SYNC-HAZARD", "Failed to retire"):
            assert marker not in result.stdout + result.stderr
