"""Explicit graphics plans use the shared raster core and owner contracts."""

import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import time

import taichi_forge as ti
from tests import test_utils
from tests.python.aot_vulkan_graphics_probe import native, pipeline_spec, image_resources, operations
from tests.python.test_hardware_graphics import _spirv_header


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_graphics_backend_isolation():
    with ti.aot.borrow_runtime() as runtime:
        assert not runtime.native_operation_capabilities("vulkan_graphics_pass_direct")
        assert not runtime.native_operation_capabilities("vulkan_graphics_pipeline")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_vulkan_graphics_reconstruction_and_gradients(tmp_path):
    @ti.kernel
    def sample(image: ti.types.texture(num_dimensions=2), output: ti.types.ndarray(ti.f32, 1)):
        for i in range(64):
            value = image.fetch(ti.Vector([i % 8, i // 8]), 0)
            for j in ti.static(range(4)):
                output[4 * i + j] = value[j]

    @ti.kernel
    def energy(
        values: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in values:
            loss[0] += 0.5 * values[i] * values[i]

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(sample)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    pipelines = []
    for instanced in (False, True):
        bindings = {}
        for i, stage in enumerate(("vert", "frag")):
            name = f"shader_{int(instanced)}_{i}"
            code = _spirv_header(f"{2 if instanced else 3}_triangle.{stage}.spv.h")
            (tmp_path / (name + ".spv")).write_bytes(code)
            pipelines.append(dict(kind="host_blob", name=name, data=module.add_data(code), byte_size=len(code)))
            bindings[f"shader_{i}"] = name
        pipelines.append(
            dict(
                native("vulkan_graphics", "pipeline", bindings, **pipeline_spec(instanced)),
                name="instance_pipeline" if instanced else "pipeline",
            )
        )
    arrays = []
    for name, dtype, usage, values in [
        ("vertices", "f32", "vertex", [v for xy in [(-1, -1), (3, -1), (-1, 3)] for v in [*xy, 1, 0, 0, 0.5, 0.5]]),
        ("positions_storage", "f32", "vertex", [99, 99, -1, -1, 3, -1, -1, 3]),
        ("indices", "u32", "index", [0, 1, 2]),
        ("uniform", "f32", "uniform", [1, 0, 0, 0]),
    ]:
        payload = struct.pack(f"<{len(values)}{'I' if dtype == 'u32' else 'f'}", *values)
        arrays.append(
            dict(
                kind="ndarray",
                name=name,
                dtype=dtype,
                shape=[len(values)],
                usage=[usage],
                initial_data=module.add_data(payload),
            )
        )
    arrays.append(dict(kind="ndarray_view", name="positions", array="positions_storage", offset=2, shape=[6]))
    for name in ("public", "private", "private_arrays", "contract"):
        resources = [dict(kind="ndarray", name="dloss", dtype="f32", shape=[1])]
        if name != "public":
            resources += pipelines + image_resources()
        if name == "private_arrays":
            resources += arrays
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(
                operations(contract=name == "contract"),
                resources=resources,
                gradients=dict(features="dfeatures", loss="dloss"),
            ),
        )
    costs = dict(compile_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "graphics.tcm"
    (tmp_path / "push_constant.spv").write_bytes(
        (Path(__file__).parent / "assets/hardware_graphics_push_constant.vert.spv").read_bytes()
    )
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_vulkan_graphics_probe.py")),
                mode,
                str(artifact),
                str(tmp_path / f"{mode}.json"),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Validation Error" not in result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
