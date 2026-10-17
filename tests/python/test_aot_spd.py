"""Frozen SPD code and owner-local counters without a consumer SDK/compiler."""

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_hardware_spd import _dependencies
from tests.python.aot_spd_probe import plan_parts


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_spd_backend_isolation():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_resource_capabilities("spd_plan")
        assert not runtime.native_operation_capabilities("spd_run")


@pytest.mark.parametrize("shape,reduction", [((3, 5), "min"), ((129, 131), "mean")])
@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_spd_resources_pyramid_and_downstream_gradients(tmp_path, shape, reduction):
    from taichi_forge.hardware._spd_jit import compile_shader

    dependencies = _dependencies()
    levels = min(shape).bit_length() - 1
    sizes = [(shape[0] >> (i + 1), shape[1] >> (i + 1)) for i in range(levels)]
    offsets = [sum(w * h for w, h in sizes[:i]) for i in range(levels)]

    @ti.kernel
    def sample(image: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for level in ti.static(range(levels)):
            for i in range(sizes[level][0] * sizes[level][1]):
                output[offsets[level] + i] = image.fetch(
                    ti.Vector([i % sizes[level][0], i // sizes[level][0]]), level
                ).x

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in output:
            loss[0] += output[i] * output[i] * 0.5

    start = time.perf_counter()
    words, _ = compile_shader(
        dependencies["compiler_path"], dependencies["source_path"], "r32f", "r32f", levels, reduction
    )
    code = struct.pack(f"<{len(words)}I", *words)
    shader_ms = (time.perf_counter() - start) * 1000
    digest = hashlib.sha256()

    def text(value):
        encoded = value.encode()
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)

    text("forge.spd.plan.v1")
    text(reduction)
    for width, height, count in ((*shape, 1), (*sizes[0], levels)):
        text("r32f")
        digest.update(struct.pack("<4Q", width, height, 1, count))
    text(hashlib.sha256(code).hexdigest())
    spec = dict(shape=shape, levels=levels, reduction=reduction, expected_plan=digest.hexdigest())
    (tmp_path / "shader.spv").write_bytes(code)
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(sample)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    key = module.add_data(code)
    for policy in ("private", "public"):
        module.add_execution_plan(policy, ti.aot.ExecutionPlan(**plan_parts(spec, key, len(code), policy == "private")))
    build_ms = (time.perf_counter() - start) * 1000
    artifact = tmp_path / "spd.tcm"
    module.archive(str(artifact))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        run = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_spd_probe.py")),
                mode,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        assert "Validation Error" not in run.stdout + run.stderr
    (tmp_path / "producer.json").write_text(
        json.dumps(
            dict(shader_ms=shader_ms, build_ms=build_ms, shader_bytes=len(code), artifact_bytes=artifact.stat().st_size)
        )
    )
