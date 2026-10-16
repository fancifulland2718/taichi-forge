"""Named sparse values stay live across recording export and independent replay."""

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
@pytest.mark.parametrize("kind", ("csr", "bsr", "spmm2", "spmm3", "spmm_preprocessed", "spsv", "spsm"))
def test_sparse_live_values_export(tmp_path, kind):
    from tests.python.aot_sparse_live_probe import case

    n, rhs, dense, compressed, transposed = case(kind)
    triangular = kind in ("spsv", "spsm")
    probe = (
        ti.hardware.linalg.cusparse_spsv_is_available
        if kind == "spsv"
        else (
            ti.hardware.linalg.cusparse_spsm_is_available
            if kind == "spsm"
            else ti.hardware.linalg.cusparse_spmm_is_available if rhs > 1 else ti.hardware.linalg.cusparse_is_available
        )
    )
    if not probe():
        pytest.skip("Compatible external cuSPARSE required")

    def upload(dtype, data):
        array = ti.ndarray(dtype, len(data))
        array.from_numpy(np.asarray(data, np.float32 if dtype == ti.f32 else np.int32))
        return array

    width = n // 2 if kind == "bsr" else n
    offsets = upload(ti.i32, [i * width for i in range(width + 1)])
    columns = upload(ti.i32, list(range(width)) * width)
    pattern = (
        ti.linalg.SparsePattern.bsr(width, width, 2, offsets, columns)
        if kind == "bsr"
        else ti.linalg.SparsePattern.csr(n, n, offsets, columns)
    )
    original = upload(ti.f32, compressed)
    matrix = pattern.matrix(original)
    adjoint_matrix = matrix if triangular else pattern.matrix(upload(ti.f32, transposed))
    owners = [matrix] if triangular else [matrix, adjoint_matrix]
    source_bindings = dict(coefficients=matrix)
    if not triangular:
        source_bindings["adjoint_coefficients"] = adjoint_matrix

    def recording(owner, source, output, transpose=False):
        if triangular:
            constructor = (
                ti.hardware.linalg.CusparseSpsvRecording if rhs == 1 else ti.hardware.linalg.CusparseSpsmRecording
            )
            args = (owner,) if rhs == 1 else (owner, rhs)
            return constructor(*args, transpose=transpose, input=source, output=output)
        if rhs > 1:
            algorithm = dict(spmm2="row_major", spmm3="csr3_direct", spmm_preprocessed="csr3_preprocessed")[kind]
            return ti.hardware.linalg.CusparseSpmmRecording(
                owner, rhs, algorithm=algorithm, input=source, output=output
            )
        return ti.hardware.linalg.CusparseSpmvRecording(owner, input=source, output=output)

    dim = 1 if rhs == 1 else 2
    shape = (n,) if rhs == 1 else (n, rhs)

    @ti.kernel
    def seed(dy: ti.types.ndarray(ti.f32, dim), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in ti.grouped(dy):
            dy[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(y: ti.types.ndarray(ti.f32, dim, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in ti.grouped(y):
            loss[0] += y[i] * y[i] * 0.5

    names = ("x", "y", "check", "dy", "loss", "dloss", "dx")
    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1 if name in ("loss", "dloss") else dim)
        for name in names
    }
    forward = recording(matrix, "x", "y")
    reverse = recording(adjoint_matrix, "dy", "dx", transpose=triangular)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(seed, symbols["dy"], symbols["loss"], symbols["dloss"])
    builder.append_native(forward)
    builder.append_native(recording(matrix, "x", "check"))
    builder.dispatch(energy, symbols["y"], symbols["loss"])
    builder.dispatch(energy.grad, symbols["y"], symbols["loss"])
    builder.append_native(reverse)
    graph = builder.compile()
    module = ti.aot.Module()
    gradients = dict(y="dy", loss="dloss")

    # A copied source ndarray is not the matrix's numeric allocation.
    with pytest.raises(ValueError, match="unused"):
        module.add_execution_plan("wrong_owner", graph, source_bindings=dict(coefficients=original))
    with pytest.raises(ValueError, match="one public binding"):
        module.add_execution_plan("duplicate", graph, source_bindings=dict(a=matrix, b=matrix))
    assert not module._execution_plans and not module._data_payloads
    module.add_execution_plan("frozen", graph, gradients=gradients)

    def export(name, executable):
        before = [owner._debug_runtime_stats()["transfers"]["device_to_host_bytes"] for owner in owners]
        plan = module.add_execution_plan(name, executable, gradients=gradients, source_bindings=source_bindings)
        after = [owner._debug_runtime_stats()["transfers"]["device_to_host_bytes"] for owner in owners]
        assert [b - a for a, b in zip(before, after)] == [(width + 1 + width * width) * 4] * len(owners)
        document = json.loads(plan.to_json())
        native = [r for r in document["resources"] if r["kind"] == "native"]
        matrices = [r for r in native if r["operation"] in ("csr_f32", "bsr_f32")]
        assert len(matrices) == len(owners)
        assert {r["bindings"]["values"] for r in matrices} == set(source_bindings)
        assert all(r["dtype"] == "i32" for r in document["resources"] if r["kind"] == "ndarray")
        if triangular:
            assert all(r["parameters"]["mutable_values"] for r in native if r["operation"] == "triangular_plan_f32")
            ops = [op["operation"] for op in document["operations"] if op["kind"] == "native"]
            assert ops == ["triangular_update_f32", "triangular_solve_f32"] * 3
        return plan

    export("live", graph)
    with builder.freeze().materialize() as selected:
        export("selected", selected.executor)

    # Producer live-value behavior and the real DSL backward remain unchanged.
    x, y, dx = ti.ndarray(ti.f32, shape), ti.ndarray(ti.f32, shape, needs_grad=True), ti.ndarray(ti.f32, shape)
    loss = ti.ndarray(ti.f32, 1, needs_grad=True)
    host = np.arange(n * rhs, dtype=np.float32).reshape(shape) / 3 - 1
    x.from_numpy(host)
    a = np.asarray(dense, np.float32)
    for scale in (1.0, 1.75):
        original.from_numpy(np.asarray(compressed, np.float32) * scale)
        matrix.update_values(original)
        if not triangular:
            adjoint_matrix.update_values(upload(ti.f32, np.asarray(transposed, np.float32) * scale))
        seed(y.grad, loss, loss.grad)
        forward.execute(dict(x=x, y=y))
        energy(y, loss)
        energy.grad(y, loss)
        reverse.execute(dict(dy=y.grad, dx=dx))
        expected = np.linalg.solve(a * scale, host) if triangular else a * scale @ host
        derivative = np.linalg.solve((a * scale).T, expected) if triangular else (a * scale).T @ expected
        np.testing.assert_allclose(y.to_numpy(), expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(dx.to_numpy(), derivative, rtol=2e-5, atol=2e-5)
    graph.close()
    artifact = tmp_path / "sparse-live.tcm"
    module.archive(str(artifact))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_sparse_live_probe.py")),
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
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=[ti.cpu, ti.vulkan], offline_cache=False)
def test_sparse_live_binding_rejects_other_backends():
    from taichi_forge.aot._graph_source_bindings import SourceArrayBindings

    matrix = ti.linalg.SparseMatrixBuilder(2, 2, max_num_triplets=1).build()
    with pytest.raises(ti.TaichiRuntimeError, match="CUDA CSR/BSR"):
        SourceArrayBindings(dict(values=matrix), {})


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_sparse_live_binding_rejects_retired_matrix():
    from taichi_forge.aot._graph_source_bindings import SourceArrayBindings

    matrix = ti.linalg.SparseMatrixBuilder(2, 2, max_num_triplets=1).build()
    ti.reset()
    ti.init(arch=ti.cuda)
    with pytest.raises(ti.TaichiRuntimeError, match="reset"):
        SourceArrayBindings(dict(values=matrix), {})
