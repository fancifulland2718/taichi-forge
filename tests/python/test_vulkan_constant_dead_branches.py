"""Fast compilation must not emit invalid addresses from constant dead branches."""

import json
import subprocess
import sys

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


@pytest.mark.parametrize("dynamic_column", [False, True])
@pytest.mark.parametrize("invert", [False, True])
@test_utils.test(arch=ti.vulkan, compile_tier="fast", advanced_optimization=False, offline_cache=False)
def test_constant_dead_shared_access(dynamic_column, invert, tmp_path, monkeypatch):
    monkeypatch.setenv("TI_VULKAN_VALIDATE_SPIRV", "1")
    output = ti.field(ti.f32, shape=32)
    arch = ti.lang.impl.current_cfg().arch

    @ti.kernel
    def dead_shared():
        ti.loop_config(block_dim=32)
        for i in range(32):
            shared = ti.simt.block.SharedArray((1, 12), ti.f32)
            slot = -1
            if ti.static(arch == ti.cuda):
                slot = 0
            if ti.static(invert):
                if slot >= 0:
                    if i < 6:
                        col = 0
                        if ti.static(dynamic_column):
                            col = i
                        shared[slot, col] = shared[slot, col] + 1
                else:
                    output[i] = i + 1
            else:
                if slot < 0:
                    output[i] = i + 1
                else:
                    if i < 6:
                        col = 0
                        if ti.static(dynamic_column):
                            col = i
                        shared[slot, col] = shared[slot, col] + 1

    result = ti.tools.export_vulkan_kernel(dead_shared, directory=tmp_path / "jit", validate=True)
    assert all(shader["validation"]["returncode"] == 0 for shader in result["shaders"])
    dead_shared()
    np.testing.assert_array_equal(output.to_numpy(), np.arange(1, 33, dtype=np.float32))


@test_utils.test(arch=ti.vulkan, compile_tier="fast", advanced_optimization=False, offline_cache=False)
def test_runtime_branches_loops_and_shared_memory_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("TI_VULKAN_VALIDATE_SPIRV", "1")
    output = ti.field(ti.i32, shape=32)

    @ti.kernel
    def dynamic(flag: ti.i32):
        ti.loop_config(block_dim=32)
        for i in range(32):
            shared = ti.simt.block.SharedArray(32, ti.i32)
            shared[i] = i + 10
            ti.simt.block.sync()
            slot = 0
            if flag:
                slot = i
            acc = 0
            j = 0
            while j < 4:
                j += 1
                if j == 2:
                    continue
                if j == 4:
                    break
                acc += j
            output[i] = shared[slot] + acc
            # Trigger the correctness cleanup in the same task as the live
            # dynamic branch, shared-memory exchange and loop-carried values.
            dead_slot = -1
            if dead_slot >= 0:
                output[i] = shared[dead_slot]

    ti.tools.export_vulkan_kernel(dynamic, 0, directory=tmp_path / "jit", validate=True)
    for flag in (0, 1, 0):
        dynamic(flag)
        expected = np.arange(32, dtype=np.int32) + 14 if flag else np.full(32, 14, np.int32)
        np.testing.assert_array_equal(output.to_numpy(), expected)


@test_utils.test(arch=ti.vulkan, compile_tier="fast", advanced_optimization=False, offline_cache=False)
def test_reachable_invalid_access_is_rejected_before_submission(tmp_path, monkeypatch):
    monkeypatch.setenv("TI_VULKAN_VALIDATE_SPIRV", "1")
    monkeypatch.setenv("TI_VULKAN_DIAGNOSTICS_DIR", str(tmp_path / "capture"))
    output = ti.field(ti.f32, shape=32)

    @ti.kernel
    def invalid():
        for i in range(32):
            shared = ti.simt.block.SharedArray(12, ti.f32)
            slot = -1
            output[i] = shared[slot]

    # Export does not submit. The runtime validator still rejects a live bad
    # access; test in a child process so a native failure cannot abort pytest.
    ti.tools.export_vulkan_kernel(invalid, directory=tmp_path / "invalid")
    script = tmp_path / "invalid.py"
    script.write_text(
        """import taichi_forge as ti
ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=False,
        compile_tier="fast", advanced_optimization=False)
out = ti.field(ti.f32, shape=32)
@ti.kernel
def invalid():
    for i in range(32):
        shared = ti.simt.block.SharedArray(12, ti.f32)
        slot = -1
        out[i] = shared[slot]
invalid()
""",
        encoding="utf8",
    )
    check = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert check.returncode != 0
    events = [
        json.loads(line)
        for path in (tmp_path / "capture").glob("*/events.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert any(event["phase"] == "validation_failed" and "negative" in event["message"] for event in events)
    assert all(event["phase"] != "pipeline_call_started" and event["vk_result"] is None for event in events)
