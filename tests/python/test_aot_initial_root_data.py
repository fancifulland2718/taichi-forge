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


def export_and_check(tmp_path, artifact_format=None):
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
    graph_builder.dispatch(forward, ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "y", ti.f32, ndim=1))
    graph = graph_builder.compile()
    dependencies = list(graph._compiled_graph._snode_tree_dependency_info)
    assert len(dependencies) == 2 and dependencies[0][0] > 0
    caps = (
        [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
        if artifact_format == "cubin"
        else None
    )
    module = ti.aot.Module(artifact_format=artifact_format, caps=caps)
    for name, kernel in (
        ("forward", forward),
        ("backward", forward.grad),
        ("clear_grad", clear_grad),
        ("read_grad", read_grad),
        ("advance", advance),
    ):
        module.add_kernel(kernel, name=name)
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
    module.add_execution_plan("step", plan)
    seed(1)
    program = ti.lang.impl.get_runtime().prog
    snapshot = program._snapshot_dense_state(dependencies, [])
    module._add_initial_root_snapshots(dependencies, snapshot["roots"])
    module._add_initial_root_snapshots(dependencies, snapshot["roots"])
    assert len(module._initial_root_data) == len(module._data_payloads) == 2
    seed(100)
    with pytest.raises(ValueError, match="conflicts"):
        module._add_initial_root_snapshots(dependencies, program._snapshot_dense_state(dependencies, [])["roots"])
    assert len(module._initial_root_data) == len(module._data_payloads) == 2
    module.add_data(b"unused data must not be exported")
    artifact = tmp_path / "initial-state.tcm"
    module.archive(str(artifact))
    graph.close()
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else ti.lang.impl.current_cfg().arch.name
    with ZipFile(artifact) as archive:
        metadata_name = "metadata.json" if arch == "vulkan" else "aot_metadata.json"
        document = json.loads(archive.read(metadata_name))
        assert document["metadata_version" if arch == "vulkan" else "schema_version"] == (3 if arch == "vulkan" else 7)
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
