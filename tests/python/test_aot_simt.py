"""Consume compiled SIMT instructions after the producer runtime has gone."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_simt_exchange_shared_storage_and_downstream_gradient(tmp_path):
    cuda = ti.cfg.arch == ti.cuda
    arch = "cuda" if cuda else "vulkan"
    caps = None
    if not cuda:
        subgroup = ti.simt.subgroup
        if not all(
            subgroup.is_shuffle_supported(name)
            for name in ("shuffle", "shuffle_up", "shuffle_down", "shuffle_xor")
        ):
            pytest.skip("Vulkan device lacks subgroup shuffle instructions")
        caps = [
            "spirv_has_subgroup_basic",
            "spirv_has_subgroup_shuffle",
            "spirv_has_subgroup_shuffle_relative",
        ]

    @ti.kernel
    def exchange(out: ti.types.ndarray(ti.i32, 2), delta: ti.u32, scale: ti.i32):
        ti.loop_config(block_dim=64)
        for i in range(137):
            value = (i * 3 - 17) * scale
            if ti.static(cuda):
                lane = i % 32
                out[i, 0] = lane
                out[i, 1] = 32
                mask = ti.simt.warp.active_mask()
                shift = ti.cast(delta, ti.i32)
                out[i, 2] = ti.simt.warp.shfl_sync_i32(mask, value, lane ^ shift)
                out[i, 3] = ti.simt.warp.shfl_up_i32(mask, value, shift)
                out[i, 4] = ti.simt.warp.shfl_down_i32(mask, value, shift)
                out[i, 5] = ti.simt.warp.shfl_xor_i32(mask, value, shift)
            else:
                lane = ti.simt.subgroup.invocation_id()
                out[i, 0] = lane
                out[i, 1] = ti.simt.subgroup.group_size()
                out[i, 2] = ti.simt.subgroup.shuffle(value, ti.u32(lane) ^ delta)
                out[i, 3] = ti.simt.subgroup.shuffle_up(value, delta)
                out[i, 4] = ti.simt.subgroup.shuffle_down(value, delta)
                out[i, 5] = ti.simt.subgroup.shuffle_xor(value, delta)

    @ti.kernel
    def shared(out: ti.types.ndarray(ti.f32, 1), scale: ti.i32):
        ti.loop_config(block_dim=32)
        for i in range(128):
            total = ti.simt.block.SharedArray((1,), ti.i32)
            if i % 32 == 0:
                total[0] = 0
            ti.simt.block.sync()
            ti.atomic_add(total[0], i * scale)
            ti.simt.block.sync()
            out[i] = ti.cast(total[0], ti.f32) / 32

    @ti.kernel
    def energy(
        values: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in values:
            loss[0] += values[i] * values[i] * 0.5

    start = time.perf_counter()
    module = ti.aot.Module(caps=caps)
    for kernel in (exchange, shared, energy):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    module.add_execution_plan(
        "simt",
        ti.aot.ExecutionPlan(
            [
                dict(kind="fill32", destination="gradient", value=0),
                dict(kind="fill32", destination="loss", value=0),
                dict(kind="fill32", destination="seed", value=0x3F800000),
                dict(
                    kind="kernel", entry="exchange", arguments=["out", "delta", "scale"]
                ),
                dict(kind="kernel", entry="shared", arguments=["values", "scale"]),
                dict(kind="kernel", entry="energy", arguments=["values", "loss"]),
                dict(kind="kernel", entry="backward", arguments=["values", "loss"]),
            ],
            resources=[dict(kind="ndarray", name="seed", dtype="f32", shape=[1])],
            gradients={"values": "gradient", "loss": "seed"},
        ),
    )
    artifact = tmp_path / "simt.tcm"
    module.archive(str(artifact))
    export_ms = (time.perf_counter() - start) * 1000
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_simt_probe.py")),
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
        assert not any(
            marker in result.stdout + result.stderr
            for marker in (
                "VUID-",
                "Validation Error",
                "SYNC-HAZARD",
                "Failed to retire",
            )
        )
    (tmp_path / "producer.json").write_text(
        json.dumps(
            dict(
                arch=arch,
                export_ms=export_ms,
                artifact_bytes=artifact.stat().st_size,
            )
        ),
        encoding="utf-8",
    )
