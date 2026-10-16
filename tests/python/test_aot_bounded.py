"""Selected bounded payloads retain launch recipes, private packets and gradients."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import core as ti_core
from tests import test_utils


def _export(tmp_path, indirect=False, exact=False, artifact_format=None, adaptive_policy=None):
    capacity = 65

    @ti.kernel
    def prepare(
        raw: ti.types.ndarray(ti.i32, 1),
        extent: ti.types.ndarray(ti.i32, 1),
        y: ti.types.ndarray(ti.f32, 1),
        dy: ti.types.ndarray(ti.f32, 1),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
        visited: ti.types.ndarray(ti.i32, 1),
    ):
        ti.device_extent_publish(extent, capacity, raw[0])
        for i in y:
            y[i], dy[i] = 0, 0
        loss[0], dloss[0] = 0, 1
        visited[0] = 0

    @ti.kernel
    def payload(
        extent: ti.types.ndarray(ti.i32, 1),
        x: ti.types.ndarray(ti.f32, 1),
        y: ti.types.ndarray(ti.f32, 1),
        visited: ti.types.ndarray(ti.i32, 1),
    ):
        for i in range(capacity):
            ti.atomic_add(visited[0], 1)
            if i < ti.device_extent_count(extent):
                y[i] = x[i] * x[i]

    @ti.kernel
    def energy(y: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in y:
            loss[0] += 0.5 * y[i] * y[i]

    @ti.kernel
    def adjoint(
        extent: ti.types.ndarray(ti.i32, 1),
        x: ti.types.ndarray(ti.f32, 1),
        dy: ti.types.ndarray(ti.f32, 1),
        dx: ti.types.ndarray(ti.f32, 1),
    ):
        for i in dx:
            dx[i] = 0
            if i < ti.device_extent_count(extent):
                dx[i] = 2 * x[i] * dy[i]

    names = ("raw", "extent", "x", "y", "dy", "dx", "loss", "dloss", "visited")
    symbols = {
        name: ti.graph.Arg(
            ti.graph.ArgKind.NDARRAY, name, ti.i32 if name in ("raw", "extent", "visited") else ti.f32, ndim=1
        )
        for name in names
    }

    def builder(backward):
        graph = ti.graph.GraphBuilder()
        graph.dispatch(prepare, *(symbols[name] for name in ("raw", "extent", "y", "dy", "loss", "dloss", "visited")))
        for _ in range(2 if adaptive_policy else 1):
            graph.dispatch_bounded(
                payload,
                symbols["extent"],
                symbols["x"],
                symbols["y"],
                symbols["visited"],
                extent=symbols["extent"],
                capacity=capacity,
                block_dim=32,
                physical_grid=(
                    "extent"
                    if adaptive_policy
                    else (
                        ("auto" if ti.lang.impl.current_cfg().arch == ti.cuda else "extent")
                        if exact
                        else ("extent" if indirect else "capacity")
                    )
                ),
            )
        if backward:
            graph.dispatch(energy, symbols["y"], symbols["loss"])
            graph.dispatch(energy.grad, symbols["y"], symbols["loss"])
            graph.dispatch(adjoint, symbols["extent"], symbols["x"], symbols["dy"], symbols["dx"])
        return graph

    frames = {
        name: ti.ndarray(
            ti.i32 if name in ("raw", "visited") else ti.f32,
            1 if name in ("raw", "loss", "dloss", "visited") else capacity,
        )
        for name in names
        if name != "extent"
    }
    frames["extent"] = ti.DeviceExtent(capacity)
    frames["x"].from_numpy(np.arange(capacity, dtype=np.float32) / 16)
    forward = builder(False).compile()
    args = {name: value for name, value in frames.items() if name != "dx"}
    for count in (33, 0, 65):
        frames["raw"].from_numpy(np.array([count], np.int32))
        forward.run(args)
        expected = np.arange(capacity, dtype=np.float32) ** 2 / 256
        expected[count:] = 0
        np.testing.assert_allclose(frames["y"].to_numpy(), expected)
        assert frames["visited"].to_numpy()[0] == (
            2 * count if adaptive_policy else count if exact else min(65, ((count + 31) // 32) * 32) if indirect else 65
        )
    forward.close()
    export_builder = builder(True)
    graph = export_builder.compile()
    caps = (
        [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
        if artifact_format == "cubin"
        else None
    )
    module = ti.aot.Module(artifact_format=artifact_format, caps=caps)
    if adaptive_policy == "grouped_stateful":
        from taichi_forge.graph._recipes.families import GraphRuntimeAssemblyProvider

        definition = export_builder.freeze()
        catalog = definition.recipe_catalog(providers=(GraphRuntimeAssemblyProvider(),))
        catalog.build_compatible_stage(candidate_limit=4)
        assert len(catalog.entries()) == 1
        with definition.materialization_context(provider_set=catalog.provider_set) as context:
            with context.materialize(catalog.entries()[0].recipe) as selected:
                plan = module.add_execution_plan("step", selected.executor, gradients=dict(y="dy", loss="dloss"))
    else:
        plan = module.add_execution_plan("step", graph, gradients=dict(y="dy", loss="dloss"))
    document = json.loads(plan.to_json())
    launched = [op for op in document["operations"] if "launch" in op]
    assert len(launched) == (2 if adaptive_policy else int(indirect or exact))
    if adaptive_policy:
        assert document["version"] == 8
        assert launched[0]["entry"] == launched[1]["entry"]
        assert all(
            op["launch"]
            == dict(kind="cuda_adaptive", extent="extent", capacity=65, block_dim=32, update_policy=adaptive_policy)
            for op in launched
        )
    elif exact:
        assert document["version"] == 7
        assert launched[0]["launch"] == dict(
            kind="cuda_logical_exact" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu_exact",
            extent="extent",
            capacity=capacity,
        )
    if indirect:
        assert document["version"] == 6
        packet = launched[0]["launch"]["packet"]
        assert packet not in plan.argument_names
        assert any(r["name"] == packet and r["dtype"] == "u32" and r["shape"] == [3] for r in document["resources"])
        external = dict(document, resources=[r for r in document["resources"] if r["name"] != packet])
        module.add_execution_plan("external_packet", ti.aot.ExecutionPlan.from_json(json.dumps(external)))
    assert any(c.get("array") == "extent" and c.get("exact_shape") == [2] for c in document["constraints"])
    artifact = tmp_path / "bounded.tcm"
    module.archive(str(artifact))
    graph.close()
    arch = (
        "vulkan"
        if ti.lang.impl.current_cfg().arch == ti.vulkan
        else ("cuda" if ti.lang.impl.current_cfg().arch == ti.cuda else "cpu")
    )
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_bounded_probe.py")),
                mode,
                arch,
                str(artifact),
                *(["adaptive"] if adaptive_policy else []),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_bounded_fixed_capacity(tmp_path):
    _export(tmp_path, False)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_bounded_indirect(tmp_path):
    _export(tmp_path, True)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_bounded_exact(tmp_path):
    _export(tmp_path, exact=True)


@pytest.mark.parametrize("artifact_format", ["ptx", "cubin"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_bounded_exact_device_artifact(tmp_path, artifact_format):
    if artifact_format == "cubin" and not (os.environ.get("TI_CUDA_PTXAS_PATH") or shutil.which("ptxas")):
        pytest.skip("External ptxas is required for cubin export")
    _export(tmp_path, exact=True, artifact_format=artifact_format)


@pytest.mark.parametrize("policy", ["per_node", "grouped_stateful"])
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_adaptive_bounded(tmp_path, monkeypatch, policy):
    probe = dict(ti_core.cuda_bounded_dispatch_probe())
    if not probe["exact_device_grid_available"]:
        pytest.skip(probe["unavailable_reason"])
    monkeypatch.setenv("TI_CUDA_BOUNDED_DISPATCH_MODE", "device_update")
    monkeypatch.setenv("TI_GRAPH_CUDA_BOUNDED_UPDATE_POLICY", policy)
    _export(tmp_path, exact=True, adaptive_policy=policy, artifact_format="ptx" if policy == "per_node" else None)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_ordered_indirect_packet_reuse(tmp_path):
    @ti.kernel
    def clear(output: ti.types.ndarray(ti.i32, 1)):
        for i in output:
            output[i] = 0

    @ti.kernel
    def segment(
        offsets: ti.types.ndarray(ti.i32, 1),
        extent: ti.types.ndarray(ti.i32, 1),
        output: ti.types.ndarray(ti.i32, 1),
        state: ti.types.ndarray(ti.i32, 1),
    ):
        for local in range(65):
            if local < ti.graph.segmented_dispatch_count(state):
                begin = ti.graph.segmented_dispatch_begin(state)
                value = ti.graph.segmented_dispatch_index(state) + 1
                if begin > 0:
                    value += output[begin - 1]
                output[begin + local] = value

    symbols = {
        name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.i32, ndim=1) for name in ("offsets", "extent", "output")
    }
    builder = ti.graph.GraphBuilder()
    prefix = builder.create_sequential()
    prefix.dispatch(clear, symbols["output"])
    builder.append(prefix)
    builder.dispatch_ordered_segments(
        segment,
        *symbols.values(),
        offsets=symbols["offsets"],
        extent=symbols["extent"],
        capacity=65,
        segment_count=3,
        block_dim=32,
    )
    graph = builder.compile()
    module = ti.aot.Module()
    plan = module.add_execution_plan("step", graph)
    ops = json.loads(plan.to_json())["operations"]
    launches = [op for op in ops if "launch" in op]
    assert len(launches) == 3
    assert len({op["entry"] for op in launches}) == 1
    assert len({op["launch"]["packet"] for op in launches}) == 1
    artifact = tmp_path / "segments.tcm"
    module.archive(str(artifact))
    graph.close()
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_bounded_probe.py")),
                mode,
                "vulkan",
                str(artifact),
                "segments",
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
