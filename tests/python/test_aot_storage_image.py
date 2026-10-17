"""AOT storage images reuse typed kernels, raster passes and resource owners."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import taichi_forge as ti
from tests import test_utils
from tests.python.aot_storage_image_probe import image_resources, mip_parts, native, operations, pipeline_parameters
from tests.python.test_hardware_graphics import _spirv_header


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_storage_images_backend_isolation():
    with ti.aot.borrow_runtime() as runtime:
        with pytest.raises(RuntimeError, match="Native resource provider unavailable"):
            runtime.native_resource("image", "storage", parameters=dict(shape=[2, 2], format="r32u"))


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_storage_images_integer_mrt_and_mips(tmp_path):
    @ti.kernel
    def update(
        unsigned: ti.types.rw_texture(2, fmt=ti.Format.r32u),
        signed: ti.types.rw_texture(2, fmt=ti.Format.r32i),
        delta: ti.types.ndarray(ti.i32, 1),
    ):
        for x, y in ti.ndrange(8, 4):
            xy = ti.Vector([x, y])
            unsigned.store(xy, ti.Vector([unsigned.load(xy).x + ti.cast(delta[0], ti.u32), 0, 0, 0]))
            signed.store(xy, ti.Vector([signed.load(xy).x - delta[0], 0, 0, 0]))

    @ti.kernel
    def consume(
        color: ti.types.texture(2),
        unsigned: ti.types.rw_texture(2, fmt=ti.Format.r32u),
        signed: ti.types.rw_texture(2, fmt=ti.Format.r32i),
        ids: ti.types.ndarray(ti.u32, 1),
        primitives: ti.types.ndarray(ti.i32, 1),
        features: ti.types.ndarray(ti.f32, 1),
    ):
        for i in ids:
            xy = ti.Vector([i % 8, i // 8])
            ids[i] = unsigned.load(xy).x
            primitives[i] = signed.load(xy).x
            value = color.fetch(xy, 0)
            for j in ti.static(range(4)):
                features[4 * i + j] = value[j]

    @ti.kernel
    def energy(
        features: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in features:
            loss[0] += 0.5 * features[i] * features[i]

    @ti.kernel
    def mip_update(image: ti.types.rw_texture(2, fmt=ti.Format.r32f, lod=1), output: ti.types.ndarray(ti.f32, 1)):
        output[0] = image.shape[0]
        output[1] = image.shape[1]
        for y in range(image.shape[1]):
            xy = ti.Vector([0, y])
            image.store(xy, ti.Vector([image.load(xy).x + 3, 0, 0, 0]))

    @ti.kernel
    def mip_read(image: ti.types.rw_texture(2, fmt=ti.Format.r32f, lod=1), output: ti.types.ndarray(ti.f32, 1)):
        for y in range(2):
            output[2 + y] = image.load(ti.Vector([0, y])).x

    @ti.kernel
    def mip_last(image: ti.types.rw_texture(2, fmt=ti.Format.r32f, lod=2), output: ti.types.ndarray(ti.f32, 1)):
        output[4] = image.load(ti.Vector([0, 0])).x

    @ti.kernel
    def seeded_update(image: ti.types.rw_texture(2, fmt=ti.Format.r32u), output: ti.types.ndarray(ti.u32, 1)):
        xy = ti.Vector([0, 0])
        value = image.load(xy).x + 1
        image.store(xy, ti.Vector([value, 0, 0, 0]))
        output[0] = value

    start = time.perf_counter()
    module = ti.aot.Module()
    for kernel in (update, consume, energy, mip_update, mip_read, mip_last, seeded_update):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    resources = []
    for i, code in enumerate(
        [
            _spirv_header("2_triangle.vert.spv.h"),
            (Path(__file__).parent / "assets/hardware_graphics_mrt.frag.spv").read_bytes(),
        ]
    ):
        (tmp_path / f"shader_{i}.spv").write_bytes(code)
        resources.append(dict(kind="host_blob", name=f"shader_{i}", data=module.add_data(code), byte_size=len(code)))
    resources.append(
        dict(
            native(
                "vulkan_graphics", "pipeline", dict(shader_0="shader_0", shader_1="shader_1"), **pipeline_parameters()
            ),
            name="pipeline",
        )
    )
    resources += image_resources()
    for name in ("public", "private"):
        module.add_execution_plan(
            name,
            ti.aot.ExecutionPlan(
                operations(),
                resources=[dict(kind="ndarray", name="dloss", dtype="f32", shape=[1])]
                + (resources if name == "private" else []),
                gradients=dict(features="dfeatures", loss="dloss"),
            ),
        )
    parts = mip_parts()
    parts["operations"].append(dict(kind="kernel", entry="mip_last", arguments=["image", "mip_output"]))
    module.add_execution_plan("mip", ti.aot.ExecutionPlan(**parts))
    costs = dict(compile_ms=(time.perf_counter() - start) * 1000)
    artifact = tmp_path / "storage-images.tcm"
    module.archive(str(artifact))
    costs["artifact_bytes"] = artifact.stat().st_size
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_storage_image_probe.py")),
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
