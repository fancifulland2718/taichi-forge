"""Dense module initialization, independent of Graph snapshot discovery."""

import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge._lib import utils as runtime_utils
from tests import test_utils


def export_and_check(tmp_path, artifact_format=None, *, automatic=False):
    if runtime_utils._loaded_native_runtime_path is None:
        pytest.skip("Requires a split AOT runtime")
    unused = ti.field(ti.i32, shape=3)
    unused.fill(17)  # Force non-local producer tree IDs.
    state = ti.field(ti.f32, needs_grad=True)
    first = ti.FieldsBuilder()
    first.dense(ti.i, 4).place(state, state.grad)
    first.finalize()
    vectors = ti.Vector.field(2, ti.f32)
    second = ti.FieldsBuilder()
    second.dense(ti.i, 4).place(vectors)
    second.finalize()

    @ti.kernel
    def seed(base: ti.f32):
        for i in range(4):
            state[i] = base + i
            state.grad[i] = 0
            vectors[i] = ti.Vector([10 + i, 20 + i])

    @ti.kernel
    def forward(y: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(4):
            y[i] = state[i] * state[i] + vectors[i][0]

    @ti.kernel
    def clear_grad():
        for i in range(4):
            state.grad[i] = 0

    @ti.kernel
    def read_grad(dx: ti.types.ndarray(ti.f32, 1)):
        for i in range(4):
            dx[i] = state.grad[i]

    @ti.kernel
    def advance():
        for i in range(4):
            state[i] += 1
            vectors[i][0] += 2

    graph_builder = ti.graph.GraphBuilder()
    y_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1)
    dx_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dx", ti.f32, ndim=1)
    graph_builder.dispatch(clear_grad)
    graph_builder.dispatch(forward, y_arg)
    graph_builder.dispatch(forward.grad, y_arg)
    graph_builder.dispatch(read_grad, dx_arg)
    graph_builder.dispatch(advance)
    graph = graph_builder.compile()
    dependencies = list(graph._compiled_graph._snode_tree_dependency_info)
    assert len(dependencies) == 2 and dependencies[0][0] > 0
    caps = (
        [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
        if artifact_format == "cubin"
        else None
    )
    module = ti.aot.Module(
        artifact_format=artifact_format,
        caps=caps,
        compile_options={"advanced_optimization": False, "llvm_opt_level": 0} if automatic else None,
    )
    for name, kernel in (
        ("forward", forward),
        ("backward", forward.grad),
        ("clear_grad", clear_grad),
        ("read_grad", read_grad),
        ("advance", advance),
    ):
        if not automatic:
            module.add_kernel(kernel, name=name)
    if not automatic:
        native = module._aot_builder
        assert native._used_snode_tree_ids(["forward", "advance", "forward"]) == [item[0] for item in dependencies]
        assert native._used_snode_tree_ids(["clear_grad"]) == [dependencies[0][0]]
        assert native._used_snode_tree_ids([]) == []
        with pytest.raises(RuntimeError, match="Unknown AOT kernel"):
            native._used_snode_tree_ids(["missing"])
    plan = ti.aot.ExecutionPlan(
        [
            dict(kind="kernel", entry="clear_grad", arguments=[]),
            dict(kind="kernel", entry="forward", arguments=["y"]),
            dict(kind="kernel", entry="backward", arguments=["y"]),
            dict(kind="kernel", entry="read_grad", arguments=["dx"]),
            dict(kind="kernel", entry="advance", arguments=[]),
        ],
        gradients={"y": "dy"},
    )
    seed(1)
    program = ti.lang.impl.get_runtime().prog
    snapshot = program._snapshot_dense_state(dependencies, [])
    if automatic:
        with pytest.raises(ti.TaichiRuntimeError, match="snapshot_state=True"):
            module.add_execution_plan("missing_choice", graph)
        assert not module._data_payloads and not module._exported_kernels
        exported = None
        for name in ("step", "same_state"):
            module.add_execution_plan(name, graph, gradients={"y": "dy"}, snapshot_state=True)
            current = tuple(module._exported_kernels)
            if exported is not None:
                assert current == exported, "Repeated export duplicated the selected physical kernels"
            exported = current
    else:
        module.add_execution_plan("step", plan)
        module._add_initial_root_snapshots(dependencies, snapshot["roots"])
        module._add_initial_root_snapshots(dependencies, snapshot["roots"])
    assert len(module._initial_root_data) == len(module._data_payloads) == 2
    module.add_data(b"unused data must not be exported")
    artifact = tmp_path / "initial-state.tcm"
    module.archive(str(artifact))
    seed(100)
    with pytest.raises(ValueError, match="conflicts"):
        if automatic:
            module.add_execution_plan("conflicting_state", graph, gradients={"y": "dy"}, snapshot_state=True)
        else:
            module._add_initial_root_snapshots(dependencies, program._snapshot_dense_state(dependencies, [])["roots"])
    assert len(module._initial_root_data) == 2 and "conflicting_state" not in module._execution_plans
    if automatic:
        with pytest.raises(ti.TaichiRuntimeError, match="export failed"):
            module.save(tmp_path / "partial")
    graph.close()
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name
    with ZipFile(artifact) as archive:
        metadata_name = "metadata.json" if arch == "vulkan" else "aot_metadata.json"
        document = json.loads(archive.read(metadata_name))
        # Vulkan ndarray arguments use the descriptor-relative offset contract;
        # the initial-root-data contract alone only required metadata 3.
        assert document["metadata_version" if arch == "vulkan" else "schema_version"] == (12 if arch == "vulkan" else 13)
        entries = document["initial_root_data"]
        assert {entry["tree_id"] for entry in entries} == {0, 1}
        assert len([name for name in archive.namelist() if name.startswith("data/")]) == 2
        assert sum(entry["size"] for entry in entries) == 64
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_initial_root_data_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Failed to retire" not in result.stdout + result.stderr


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_dense_module_initial_state_and_retirement(tmp_path):
    export_and_check(tmp_path)


@pytest.mark.parametrize("artifact_format", ["ptx", "cubin"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_dense_initial_state_cuda_device_artifacts(tmp_path, artifact_format):
    export_and_check(tmp_path, artifact_format)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, advanced_optimization=True)
def test_graph_dense_initial_state_and_actual_aot_dependencies(tmp_path):
    export_and_check(tmp_path, automatic=True)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_state_snapshot_rejects_sparse_and_partial_publication(tmp_path):
    state = ti.field(ti.f32)
    fields = ti.FieldsBuilder()
    fields.pointer(ti.i, 2).dense(ti.i, 4).place(state)
    tree = fields.finalize()

    @ti.kernel
    def read_sparse(out: ti.types.ndarray(ti.f32, 1)):
        for i in range(8):
            out[i] = state[i]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(read_sparse, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1))
    graph = builder.compile()
    module = ti.aot.Module()
    with pytest.raises(RuntimeError, match="purely dense"):
        module.add_execution_plan("unsupported", graph, snapshot_state=True)
    assert not module._execution_plans and not module._data_payloads
    with pytest.raises(ti.TaichiRuntimeError, match="export failed"):
        module.archive(str(tmp_path / "partial.tcm"))
    assert not (tmp_path / "partial.tcm").exists()
    graph.close()
    tree.destroy()


@pytest.mark.parametrize("advanced_jit", [False, True])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_graph_snapshot_uses_aot_closure_when_optimization_changes(tmp_path, advanced_jit):
    # Advanced alias analysis sees that the ndarray store cannot modify this
    # Field, so both loads become one and their subtraction disappears. The
    # conservative configuration retains them and still binds the root.
    ti.lang.impl.current_cfg().advanced_optimization = advanced_jit
    state = ti.field(ti.f32, shape=4)
    state.fill(3)

    @ti.kernel
    def apply(x: ti.types.ndarray(ti.f32, 1), y: ti.types.ndarray(ti.f32, 1)):
        for i in range(4):
            first = state[i]
            y[i] = x[i]
            second = state[i]
            y[i] = first - second + x[i]

    builder = ti.graph.GraphBuilder()
    builder.dispatch(apply, *(ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1) for name in ("x", "y")))
    graph = builder.compile()
    assert bool(graph._compiled_graph._snode_tree_dependency_info) == (not advanced_jit)
    options = dict(advanced_optimization=not advanced_jit, llvm_opt_level=0)
    if advanced_jit:
        rejected = ti.aot.Module(compile_options=options)
        with pytest.raises(ti.TaichiRuntimeError, match="compiled Field state.*snapshot_state=True"):
            rejected.add_execution_plan("missing_choice", graph)
        with pytest.raises(ti.TaichiRuntimeError, match="export failed"):
            rejected.save(tmp_path / "partial")
    module = ti.aot.Module(compile_options=options)
    before = ti.runtime.stats().transfer.device_to_host_bytes
    plan = module.add_execution_plan("apply", graph, snapshot_state=True)
    assert ti.runtime.stats().transfer.device_to_host_bytes - before == (16 if advanced_jit else 0)
    assert bool(module._aot_builder._used_snode_tree_ids(plan.kernel_entries)) == advanced_jit
    assert len(module._initial_root_data) == len(module._data_payloads) == int(advanced_jit)
    if advanced_jit:
        import struct

        assert struct.unpack("4f", next(iter(module._data_payloads.values()))) == (3, 3, 3, 3)
    module.archive(str(tmp_path / "actual-closure.tcm"))
    graph.close()
