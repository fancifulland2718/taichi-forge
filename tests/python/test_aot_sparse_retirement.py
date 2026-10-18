"""Sparse/list storage is retired before the same module is loaded again."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_sparse_list_close_reload(tmp_path):
    _sparse_module(tmp_path, "bc")


@pytest.mark.parametrize("artifact_format", ["ptx", "cubin"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_sparse_device_artifact_isolation(tmp_path, artifact_format):
    if artifact_format == "cubin" and not (os.environ.get("TI_CUDA_PTXAS_PATH") or shutil.which("ptxas")):
        pytest.skip("External ptxas is required for cubin export")
    _sparse_module(tmp_path, artifact_format)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_sparse_multilevel_fixed(tmp_path):
    _sparse_module(tmp_path, "ptx", multilevel=True)


@pytest.mark.parametrize("pool", ["inline", "independent", "chunked"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_vulkan_sparse_resource_reconstruction(tmp_path, pool):
    ti.cfg.vulkan_pointer_independent_pool = pool != "inline"
    ti.cfg.vulkan_pointer_allocator_kind = "chunked" if pool == "chunked" else "bump"
    ti.cfg.vulkan_pointer_max_chunks = 2 if pool == "chunked" else 1
    _sparse_module(tmp_path, None, consumer="aot_sparse_retirement_probe.py")


@pytest.mark.parametrize("pool", ["independent", "chunked"])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_vulkan_sparse_fixed_execution(tmp_path, pool):
    ti.cfg.vulkan_pointer_independent_pool = True
    ti.cfg.vulkan_pointer_allocator_kind = "chunked" if pool == "chunked" else "bump"
    ti.cfg.vulkan_pointer_max_chunks = 2 if pool == "chunked" else 1
    export_growth = None
    if pool == "independent":
        # A separate module grows the shared list after the consumer has
        # recorded its first module. Only one sparse cell is ever touched.
        extra = ti.field(ti.i32)
        builder = ti.FieldsBuilder()
        builder.pointer(ti.i, 4096).place(extra)
        builder.finalize()

        @ti.kernel
        def touch(out: ti.types.ndarray(ti.f32, 1)):
            extra[4095] = 91
            out[0] = 0
            for i in extra:
                out[0] += extra[i]

        def export_growth(_module):
            growth = ti.aot.Module()
            growth.add_kernel(touch)
            growth.archive(str(tmp_path / "list-growth.tcm"))

    _sparse_module(tmp_path, None, consumer="aot_sparse_fixed_probe.py", export_extra=export_growth)


def _sparse_module(tmp_path, artifact_format, multilevel=False, *, export_extra=None, consumer=None):
    # Ensure the producer tree ID differs from the compact artifact ID.
    unrelated = ti.field(ti.i32, shape=1)
    unrelated[0] = 7
    values = ti.field(ti.f32, needs_grad=True)
    blocks = ti.root.pointer(ti.i, 4)
    leaves = blocks.dense(ti.i, 2) if multilevel else blocks
    leaves.bitmasked(ti.i, 2).place(values, values.grad)
    second = 13 if multilevel else 5
    entries = ti.field(ti.f32)
    dynamic = ti.root.dynamic(ti.i, 8, chunk_size=4)
    dynamic.place(entries)

    @ti.kernel
    def activate(scale: ti.f32):
        values[2] = scale
        values[second] = 2 * scale
        values.grad[2] = 0
        values.grad[second] = 0
        ti.append(dynamic, [], scale)
        ti.append(dynamic, [], 3 * scale)

    @ti.kernel
    def gather(out: ti.types.ndarray(ti.f32, 1)):
        for i in values:
            ti.atomic_add(out[0], values[i])
        for i in entries:
            ti.atomic_add(out[1], entries[i])
        out[2] = ti.length(dynamic, [])

    @ti.kernel
    def deactivate():
        for i in range(4):
            ti.deactivate(blocks, i)
        ti.deactivate(dynamic, [])

    @ti.kernel
    def energy(out: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in range(2):
            loss[0] += 0.5 * out[i] * out[i]

    @ti.kernel
    def sparse_energy(loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in values:
            loss[0] += 0.5 * values[i] * values[i]

    @ti.kernel
    def gather_grad(out: ti.types.ndarray(ti.f32, 1)):
        for i in values:
            out[0] += values.grad[i]

    arch = {ti.cpu: "cpu", ti.cuda: "cuda", ti.vulkan: "vulkan"}[ti.cfg.arch]
    start = time.perf_counter()
    caps = (
        [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
        if artifact_format == "cubin"
        else None
    )
    module = ti.aot.Module(artifact_format=artifact_format, caps=caps)
    for kernel in (activate, gather, deactivate, energy, sparse_energy, gather_grad):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    module.add_kernel(sparse_energy.grad, name="sparse_backward")
    out_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "out", ti.f32, ndim=1)
    loss_arg = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    scale_arg = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32)
    parent = ti.graph.GraphBuilder()
    child = parent.create_sequential()
    child.dispatch(gather, out_arg)
    parent.dispatch(deactivate)
    parent.dispatch(activate, scale_arg)
    parent.append(child)
    parent.dispatch(sparse_energy, loss_arg)
    parent.dispatch(sparse_energy.grad, loss_arg)
    module.add_graph("sparse_step", parent.compile())
    if export_extra is not None:
        export_extra(module)
    artifact = tmp_path / "sparse.tcm"
    module.archive(str(artifact))
    with zipfile.ZipFile(artifact) as archive:
        metadata = json.loads(archive.read("metadata.json" if arch == "vulkan" else "aot_metadata.json"))
    if arch == "vulkan":
        assert metadata["metadata_version"] == 7
        assert len(metadata["snode_tree_resources"]) == len(metadata["root_buffer_sizes"])
        assert any(tree["storage_kind"] == 2 for tree in metadata["snode_tree_resources"])
    else:
        assert metadata["resource_binding"] == ("sparse_v2" if arch == "cuda" else "sparse_v1")
    legacy = None
    if artifact_format == "ptx" and consumer is None:
        # v1 keeps ordinary loading, but cannot promise the new device listgen
        # publication contract. Never silently admit it to fixed replay.
        legacy = tmp_path / "sparse-v1.tcm"
        old_metadata = dict(metadata, resource_binding="sparse_v1")
        with zipfile.ZipFile(artifact) as source, zipfile.ZipFile(legacy, "w") as target:
            for name in source.namelist():
                target.writestr(name, json.dumps(old_metadata) if name == "aot_metadata.json" else source.read(name))
    (tmp_path / "producer.json").write_text(
        json.dumps(
            dict(
                compile_export_ms=(time.perf_counter() - start) * 1000,
                bytes=artifact.stat().st_size,
                format=artifact_format,
            )
        )
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        if consumer is not None:
            result = subprocess.run(
                [
                    sys.executable,
                    *(["-S"] if mode == "standalone" else []),
                    str(Path(__file__).with_name(consumer)),
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
            assert "Failed to retire" not in result.stdout + result.stderr
            continue
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_sparse_retirement_probe.py")),
                arch,
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Failed to retire" not in result.stdout + result.stderr
        fixed = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_sparse_fixed_probe.py")),
                arch,
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert fixed.returncode == 0, fixed.stdout + fixed.stderr
        if legacy is not None:
            old = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("aot_sparse_fixed_probe.py")),
                    arch,
                    mode,
                    str(legacy),
                    "legacy",
                ],
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert old.returncode == 0, old.stdout + old.stderr
