"""Export declarative Graph storage, including alias and gradient bindings."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_private_graph_aliases_and_reverse(tmp_path):
    @ti.kernel
    def advance(
        x: ti.types.ndarray(ti.f32, ndim=1), state: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), gain: ti.f32
    ):
        for i in range(17):
            state[i] = state[i] + gain * x[i]

    @ti.kernel
    def square(
        state: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        for i in range(17):
            y[i] = state[i] * state[i]

    @ti.kernel
    def seed(ds: ti.types.ndarray(ti.f32, ndim=1), dy: ti.types.ndarray(ti.f32, ndim=1)):
        for i in ds:
            ds[i] = 0
            dy[i] = 1

    @ti.kernel
    def report(ds: ti.types.ndarray(ti.f32, ndim=1), output: ti.types.ndarray(ti.f32, ndim=1)):
        for i in output:
            output[i] = ds[i]

    builder = ti.graph.GraphBuilder(_map_recipe="map2")
    private = {key: builder.private_ndarray(key, ti.f32, 17) for key in ("state", "ds", "dy")}
    public = {key: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, key, ti.f32, ndim=1) for key in ("x", "y", "dstate")}
    gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "gain", ti.f32)
    alias = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state_alias", ti.f32, ndim=1)
    # Providers use this same cold binding mechanism for shared private owners.
    owner = builder._declared_private_bindings["state"][1]
    builder._bind_internal_runtime_arg(alias, owner)
    builder.dispatch(advance, public["x"], private["state"], gain)
    nested = builder.create_sequential()
    nested.dispatch(square, alias, public["y"])
    builder.append(nested)
    builder.dispatch(seed, private["ds"], private["dy"])
    builder.dispatch(square.grad, alias, public["y"])
    builder.dispatch(report, private["ds"], public["dstate"])
    graph = builder.compile()
    assert graph._spec.fixed_runtime_args["state"] is graph._spec.fixed_runtime_args["state_alias"]
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="same gradient"):
        module.add_execution_plan("conflict", graph, gradients=dict(state="ds", state_alias="dy", y="dy"))
    assert not module._content and not module._exported_kernels and not module._execution_plans
    gradients = dict(state="ds", state_alias="ds", y="dy")
    plan = module.add_execution_plan("step", graph, gradients=gradients)
    assert plan.argument_names == ("dstate", "gain", "x", "y")
    document = json.loads(plan.to_json())
    assert len(document["resources"]) == 3, "Aliases must share the same resource"
    assert all(r["zero_initialize"] for r in document["resources"])
    assert len(document["gradients"]) == 2
    assert len(document["operations"]) == graph.physical_plan()["physical_dispatch_count"]
    kernels = len(module._exported_kernels)
    module.add_execution_plan("again", graph, gradients=gradients)
    assert len(module._exported_kernels) == kernels
    graph.close()
    del graph, builder, nested, owner
    gc.collect()
    artifact = tmp_path / "private-graph.tcm"
    module.archive(str(artifact))
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_private_graph_probe.py")), mode, arch, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_aot_private_graph_rejects_live_storage_before_publication():
    @ti.kernel
    def copy(source: ti.types.ndarray(ti.f32, ndim=1), output: ti.types.ndarray(ti.f32, ndim=1)):
        for i in output:
            output[i] = source[i]

    source = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "source", ti.f32, ndim=1)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder._bind_internal_runtime_arg(source, ti.ndarray(ti.f32, 17))
    builder.dispatch(copy, source, output)
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="live arrays/scalars"):
        module.add_execution_plan("invalid", graph)
    assert not module._content and not module._exported_kernels and not module._execution_plans
    graph.close()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_private_array_feeds_native_recording(tmp_path):
    if not ti.hardware.linalg.cublas_is_available():
        pytest.skip("cuBLAS required")

    @ti.kernel
    def produce(x: ti.types.ndarray(ti.f32, ndim=2), a: ti.types.ndarray(ti.f32, ndim=2)):
        for i, k in a:
            a[i, k] = 2 * x[i, k]

    @ti.kernel
    def adjoint(b: ti.types.ndarray(ti.f32, ndim=2), dx: ti.types.ndarray(ti.f32, ndim=2)):
        for i, k in dx:
            value = 0.0
            for j in range(3):
                value += 2 * b[k, j]
            dx[i, k] = value

    builder = ti.graph.GraphBuilder()
    private = builder.private_ndarray("a", ti.f32, (5, 7))
    public = {key: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, key, ti.f32, ndim=2) for key in ("x", "b", "dx")}
    builder.dispatch(produce, public["x"], private)
    recording = ti.hardware.linalg.CublasGemmRecording(5, 3, 7, output="y")
    builder.append_native(recording)
    builder.dispatch(adjoint, public["b"], public["dx"])
    graph = builder.compile()
    module = ti.aot.Module()
    plan = module.add_execution_plan("gemm", graph)
    assert plan.argument_names == ("b", "dx", "x", "y")
    document = json.loads(plan.to_json())
    assert len(document["operations"]) == 3
    array = next(r for r in document["resources"] if r["kind"] == "ndarray")
    native = next(op for op in document["operations"] if op["kind"] == "native")
    assert native["bindings"]["a"] == array["name"]
    graph.close()
    del graph, builder, recording
    gc.collect()
    artifact = tmp_path / "private-gemm.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_private_graph_probe.py")),
                mode,
                "cuda",
                str(artifact),
                "gemm",
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
