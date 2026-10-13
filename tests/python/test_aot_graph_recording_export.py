"""Export complete sequential Graphs, including registered native recordings."""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils


def consume(tmp_path, module, kind, arch, *provider):
    artifact = tmp_path / f"{kind}.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [
                *command,
                str(Path(__file__).with_name("aot_graph_recording_probe.py")),
                mode,
                arch,
                str(artifact),
                kind,
                *provider,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
    return artifact


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_graph_kernel_closure_and_reverse(tmp_path):
    @ti.kernel
    def seed(dx: ti.types.ndarray(ti.f32, ndim=1), dy: ti.types.ndarray(ti.f32, ndim=1)):
        for i in dx:
            dx[i] = 0
            dy[i] = 1

    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        gain: ti.f32,
    ):
        for i in x:
            y[i] = gain * x[i] * x[i]

    arguments = {name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y", "dx", "dy")}
    gain = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "gain", ti.f32)
    builder = ti.graph.GraphBuilder()
    prefix = builder.create_sequential()
    prefix.dispatch(seed, arguments["dx"], arguments["dy"])
    builder.append(prefix)
    builder.dispatch(square, arguments["x"], arguments["y"], gain)
    builder.dispatch(square.grad, arguments["x"], arguments["y"], gain)
    graph = builder.compile()
    module = ti.aot.Module()
    plan = module.add_execution_plan("step", graph, gradients=dict(x="dx", y="dy"))
    assert isinstance(plan, ti.aot.ExecutionPlan)
    assert len(module._exported_kernels) == 3
    module.add_execution_plan("again", graph, gradients=dict(x="dx", y="dy"))
    assert len(module._exported_kernels) == 3, "Repeated Graph export duplicated its kernel closure"
    before = (tuple(module._content), tuple(module._exported_kernels))
    with pytest.raises(ValueError, match="Duplicate"):
        module.add_execution_plan("step", graph)
    assert before == (tuple(module._content), tuple(module._exported_kernels))
    graph.close()
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    consume(tmp_path, module, "square", arch)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_export_rejects_missing_contract_without_partial_publication(tmp_path):
    @ti.kernel
    def noop(x: ti.types.ndarray(ti.f32, ndim=1)):
        for i in x:
            x[i] += 1

    x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)
    predicate = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "predicate", ti.i32, ndim=0)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(noop, x)
    builder.observe(predicate)
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="Observation"):
        module.add_execution_plan("invalid", graph)
    assert not module._content and not module._exported_kernels and not module._execution_plans
    graph.close()
    valid = ti.graph.GraphBuilder()
    valid.dispatch(noop, x)
    good = valid.compile()
    module.add_execution_plan("valid", good)
    good.close()
    module.archive(str(tmp_path / "valid.tcm"))


def test_graph_export_resource_identity_is_local_and_configuration_stable():
    from types import SimpleNamespace
    from taichi_forge.aot._graph_export import _ExportContext

    context = _ExportContext(SimpleNamespace(_exported_kernels={}), {"__forge_aot_resource_0"})
    owner, other = object(), object()
    options = dict(parameters=dict(rows=5))
    name = context.resource(owner, "test", "plan", **options)
    assert name == "__forge_aot_resource_1"
    assert name == context.resource(owner, "test", "plan", **options)
    assert name != context.resource(other, "test", "plan", **options)
    options["parameters"]["rows"] = 6
    with pytest.raises(ValueError, match="changed"):
        context.resource(owner, "test", "plan", **options)
    assert context.resources[0]["parameters"] == dict(rows=5)


def test_graph_export_stages_frozen_data_and_resource_initialization():
    from types import SimpleNamespace
    from taichi_forge.aot._graph_export import _ExportContext

    context = _ExportContext(SimpleNamespace(_exported_kernels={}), set())
    owner = object()
    data = b"\0\0\x80\x3f"
    name = context.data_resource(owner, data, dtype="f32", shape=[1])
    assert context.data_resource(owner, data, dtype="f32", shape=[1]) == name
    assert len(context.data_payloads) == 1
    context.initialize_native_once(name, "test", "setup", bindings=dict(array=name))
    context.initialize_native_once(name, "test", "setup", bindings=dict(array=name))
    assert len(context.initialization) == 1 and not context.operations
    with pytest.raises(ValueError, match="initialization changed"):
        context.initialize_native_once(name, "test", "setup", bindings=dict(array="other"))
    with pytest.raises(ValueError, match="reconstruction changed"):
        context.data_resource(owner, b"\0\0\0\0", dtype="f32", shape=[1])
    with pytest.raises(TypeError, match="immutable"):
        context.data_resource(object(), bytearray(data), dtype="f32", shape=[1])


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_export_compile_failure_cannot_publish_partial_artifact(tmp_path):
    @ti.kernel
    def first(x: ti.types.ndarray(ti.f32, ndim=1)):
        for i in x:
            x[i] = 1

    @ti.kernel
    def second(x: ti.types.ndarray(ti.f32, ndim=1)):
        for i in x:
            x[i] += 2

    x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)
    builder = ti.graph.GraphBuilder()
    builder.dispatch(first, x)
    builder.dispatch(second, x)
    graph = builder.compile()
    module = ti.aot.Module()
    native_builder = module._aot_builder

    class FailedCompilation:
        count = 0

        def add(self, name, kernel):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("backend compile failure")
            # Actually publish the first kernel into the native builder, so
            # rejecting only the Python metadata would not prevent a partial archive.
            native_builder.add(name, kernel)

    module._aot_builder = FailedCompilation()
    with pytest.raises(RuntimeError, match="backend compile failure"):
        module.add_execution_plan("step", graph)
    assert not module._content and not module._exported_kernels and not module._execution_plans
    module._aot_builder = native_builder
    for save, path in ((module.save, tmp_path / "partial"), (module.archive, tmp_path / "partial.tcm")):
        with pytest.raises(ti.TaichiRuntimeError, match="new AOT module"):
            save(str(path))
        assert not path.exists()
    graph.close()


def build_products(recordings):
    @ti.kernel
    def initialize(seed: ti.types.ndarray(ti.f32, ndim=2), c: ti.types.ndarray(ti.f32, ndim=2)):
        for i, j in c:
            c[i, j] = seed[i, j]

    @ti.kernel
    def adjoint(
        a: ti.types.ndarray(ti.f32, ndim=2),
        b: ti.types.ndarray(ti.f32, ndim=2),
        d: ti.types.ndarray(ti.f32, ndim=2),
        e: ti.types.ndarray(ti.f32, ndim=2),
        f: ti.types.ndarray(ti.f32, ndim=2),
        da: ti.types.ndarray(ti.f32, ndim=2),
        db: ti.types.ndarray(ti.f32, ndim=2),
    ):
        for i, k in da:
            value = 0.0
            for j in range(b.shape[1]):
                value += (0.75 * d[i, j] - 0.5 * e[i, j] + 0.125 * f[i, j]) * b[k, j]
            da[i, k] = value
        for k, j in db:
            value = 0.0
            for i in range(a.shape[0]):
                value += (0.75 * d[i, j] - 0.5 * e[i, j] + 0.125 * f[i, j]) * a[i, k]
            db[k, j] = value

    args = {
        key: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, key, ti.f32, ndim=2)
        for key in ("a", "b", "seed", "c", "d", "e", "f", "da", "db")
    }
    builder = ti.graph.GraphBuilder()
    sequential = builder.create_sequential()
    sequential.dispatch(initialize, args["seed"], args["c"])
    builder.append(sequential)
    for recording in recordings:
        builder.append_native(recording)
    builder.dispatch(adjoint, *(args[key] for key in ("a", "b", "d", "e", "f", "da", "db")))
    return builder.compile()


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cublas_recording_export_and_unknown_subclass(tmp_path):
    if not ti.hardware.linalg.cublas_is_available():
        pytest.skip("cuBLAS required")
    Recording = ti.hardware.linalg.CublasGemmRecording
    # cuBLAS writes its C input in place; first copy/initialize each output.
    # Here beta is zero and the initializer still tests the ordinary prefix.
    records = [Recording(5, 3, 7, alpha=alpha, output=key) for alpha, key in ((0.75, "d"), (-0.5, "e"), (0.125, "f"))]
    graph = build_products(records)
    module = ti.aot.Module()
    plan = module.add_execution_plan("step", graph)
    assert len(json.loads(plan.to_json())["operations"]) == 5
    assert len(module._exported_kernels) == 2
    graph.close()

    class Unqualified(Recording):
        pass

    bad = build_products([*records[:2], Unqualified(5, 3, 7, output="f")])
    before = tuple(module._content), tuple(module._exported_kernels)
    with pytest.raises(ti.TaichiRuntimeError, match="Unqualified"):
        module.add_execution_plan("invalid", bad)
    assert before == (tuple(module._content), tuple(module._exported_kernels))
    bad.close()
    consume(tmp_path, module, "cublas", "cuda")


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_cutensor_recording_exports_shared_and_independent_plans(tmp_path):
    vendor = os.environ.get("TI_FORGE_TEST_CUTENSOR_LIBRARY_PATH")
    if not vendor:
        pytest.skip("set TI_FORGE_TEST_CUTENSOR_LIBRARY_PATH for cuTENSOR")
    from taichi_forge.hardware._bundled_runtime_provider import _bundled_provider_candidates
    from taichi_forge.hardware._cutensor import DEFINITION

    adapter = _bundled_provider_candidates(DEFINITION)[0]
    provider = ti.hardware.tensor.CutensorProvider(vendor)
    options = dict(workspace_preference="min", workspace_limit_bytes=1 << 20)
    dimensions = ((5, 7), "ik", (7, 3), "kj", (5, 3), "ij", (5, 3), "ij")
    first = provider.contraction_plan(*dimensions, **options)
    second = provider.contraction_plan(*dimensions, **options)
    records = [
        first.record(d="d", alpha=0.75, beta=0.25),
        first.record(d="e", alpha=-0.5, beta=0.5),
        second.record(d="f", workspace="second_workspace", alpha=0.125),
    ]
    graph = build_products(records)
    module = ti.aot.Module()
    plan = module.add_execution_plan("step", graph)
    description = json.loads(plan.to_json())
    assert len(description["resources"]) == 2
    assert all(resource["parameters"]["workspace_preference"] == "min" for resource in description["resources"])
    native = [op for op in description["operations"] if op["kind"] == "native"]
    assert native[0]["bindings"]["plan"] == native[1]["bindings"]["plan"] != native[2]["bindings"]["plan"]
    graph.close()
    del graph, records
    gc.collect()
    first.close()
    second.close()
    provider.close()
    artifact = consume(tmp_path, module, "cutensor", "cuda", adapter, vendor)
    with zipfile.ZipFile(artifact) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            assert vendor.encode() not in data and adapter.encode() not in data
