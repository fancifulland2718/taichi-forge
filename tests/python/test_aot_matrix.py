"""Driver MMA recordings compose with actual AOT gradients and explicit adjoints."""

import gc
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("batch", (1, 3))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_mma_recording_and_adjoint(tmp_path, batch):
    if not ti.hardware.matrix.is_available():
        pytest.skip("CUDA WMMA requires compute capability 7.0 or newer")
    shape = (16, 16) if batch == 1 else (batch, 16, 16)
    rank = len(shape)

    @ti.kernel
    def seed(
        dy: ti.types.ndarray(ti.f32, rank),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
    ):
        for I in ti.grouped(dy):
            dy[I] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        c: ti.types.ndarray(ti.f32, rank, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for I in ti.grouped(c):
            loss[0] += 0.5 * c[I] * c[I]

    @ti.kernel
    def transpose(
        a: ti.types.ndarray(ti.f16, rank),
        b: ti.types.ndarray(ti.f16, rank),
        dy: ti.types.ndarray(ti.f32, rank),
        at: ti.types.ndarray(ti.f16, rank),
        bt: ti.types.ndarray(ti.f16, rank),
        dh: ti.types.ndarray(ti.f16, rank),
    ):
        if ti.static(batch == 1):
            for i, j in ti.ndrange(16, 16):
                at[j, i], bt[j, i] = a[i, j], b[i, j]
                dh[i, j] = ti.cast(dy[i, j], ti.f16)
        else:
            for k, i, j in ti.ndrange(batch, 16, 16):
                at[k, j, i], bt[k, j, i] = a[k, i, j], b[k, i, j]
                dh[k, i, j] = ti.cast(dy[k, i, j], ti.f16)

    args = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=rank)
        for name, dtype in (("a", ti.f16), ("b", ti.f16), ("c", ti.f32), ("da", ti.f32), ("db", ti.f32))
    }
    loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    private = {
        name: builder.private_ndarray(name, dtype, size)
        for name, dtype, size in (
            ("dy", ti.f32, shape),
            ("dloss", ti.f32, 1),
            ("at", ti.f16, shape),
            ("bt", ti.f16, shape),
            ("dh", ti.f16, shape),
        )
    }
    builder.dispatch(seed, private["dy"], loss, private["dloss"])
    builder.append_native(ti.hardware.matrix.CudaMatrixMmaRecording(batch, a="a", b="b", output="c"))
    builder.dispatch(energy, args["c"], loss)
    builder.dispatch(energy.grad, args["c"], loss)
    builder.dispatch(
        transpose, args["a"], args["b"], private["dy"], private["at"], private["bt"], private["dh"]
    )
    builder.append_native(ti.hardware.matrix.CudaMatrixMmaRecording(batch, a="dh", b="bt", output="da"))
    builder.append_native(ti.hardware.matrix.CudaMatrixMmaRecording(batch, a="at", b="dh", output="db"))
    graph = builder.compile()
    module = ti.aot.Module()
    plan = module.add_execution_plan("step", graph, gradients=dict(c="dy", loss="dloss"))
    assert set(plan.argument_names) == {"a", "b", "c", "da", "db", "loss"}
    graph.close()
    del graph, builder
    gc.collect()
    artifact = tmp_path / "mma.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_matrix_probe.py")),
                mode,
                str(artifact),
                str(batch),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Unable to release execution resource" not in result.stdout + result.stderr
