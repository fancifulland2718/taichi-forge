"""Ordinary Graph transfer recordings reconstruct their shared image owners."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_graph_integer_image_full_copy(tmp_path):
    from tests.python.aot_optix_probe import read, write

    first, second = [ti.Texture(ti.Format.r32u, (4, 2)) for _ in range(2)]
    builder = ti.graph.GraphBuilder()
    image = ti.hardware.image
    for recording in (
        image.VulkanBufferToImageRecording(source="input", destination="first"),
        image.VulkanImageCopyRecording(source="first", destination="second"),
        image.VulkanImageToBufferRecording(source="second", destination="output"),
    ):
        builder.append_native(recording, admission="auto")
    graph = builder.compile()
    module = ti.aot.Module()
    plan = module.add_execution_plan("copy", graph, source_bindings=dict(first=first, second=second))
    doc = json.loads(plan.to_json())
    assert not doc.get("initialization")
    assert [r["operation"] for r in doc["resources"]] == ["storage", "storage"]
    graph.close()
    artifact = tmp_path / "integer-transfer.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        source, output = [runtime.ndarray("u32", (8,)) for _ in range(2)]
        call = loaded.execution_plan("copy").prepare(loaded, dict(input=source, output=output))
        for values in ([0xFFFFFFFF, 2, 3, 4, 5, 6, 7, 8], [0] * 8):
            write(source, values, "I")
            call()
            assert read(output, "I") == values
        call.close()
        loaded.close()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_aot_graph_image_transfer_export(tmp_path):
    image = ti.hardware.image
    region = image.VulkanImageRegion
    layout = image.VulkanBufferImageLayout
    a, b, c = [ti.Texture(ti.Format.r32f, shape) for shape in ((3, 2), (4, 3), (4, 4))]
    source, padded = [ti.ndarray(ti.f32, 32) for _ in range(2)]
    source_view = ti.experimental.ndarray_view(source, slices=(slice(2, 11),))
    target_view = ti.experimental.ndarray_view(padded, slices=(slice(2, 26),))
    builder = ti.graph.GraphBuilder()
    arg = lambda name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
    builder._bind_internal_runtime_arg(arg("input_view"), source_view)
    builder._bind_internal_runtime_arg(arg("output_view"), target_view)
    builder.append_native(
        image.VulkanBufferToImageRecording(
            source="input_view", destination="a", buffer_layout=layout(byte_offset=4, row_length=5, image_height=4)
        ),
        admission="auto",
    )
    builder.append_native(
        image.VulkanImageCopyRecording(
            source="a",
            destination="b",
            source_region=region(offset=(1, 0), extent=(2, 2)),
            destination_region=region(offset=(1, 1)),
        ),
        admission="auto",
    )
    builder.append_native(
        image.VulkanImageBlitRecording(source="a", destination="c", source_region=region(offset=(1, 0), extent=(2, 2))),
        admission="auto",
    )
    builder.append_native(
        image.VulkanImageToBufferRecording(
            source="c", destination="output_view", buffer_layout=layout(byte_offset=4, row_length=6)
        ),
        admission="auto",
    )
    builder.append_native(image.VulkanImageToBufferRecording(source="b", destination="copied"), admission="auto")

    @ti.kernel
    def sample(texture: ti.types.texture(2), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = texture.fetch(ti.Vector([i % 4, i // 4]), 0).x

    @ti.kernel
    def seed(dout: ti.types.ndarray(ti.f32, 1), loss: ti.types.ndarray(ti.f32, 1), dloss: ti.types.ndarray(ti.f32, 1)):
        for i in dout:
            dout[i] = 0
        loss[0], dloss[0] = 0, 1

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 1, needs_grad=True), loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)
    ):
        for i in output:
            loss[0] += 0.5 * output[i] ** 2

    texture_arg = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "c", ndim=2)
    builder.dispatch(sample, texture_arg, arg("output"))
    builder.dispatch(seed, arg("doutput"), arg("loss"), arg("dloss"))
    builder.dispatch(energy, arg("output"), arg("loss"))
    builder.dispatch(energy.grad, arg("output"), arg("loss"))
    graph = builder.compile()
    module = ti.aot.Module()
    start = time.perf_counter()
    sources = dict(a=a, b=b, c=c, input=source, padded=padded)
    options = dict(source_bindings=sources, gradients=dict(output="doutput", loss="dloss"))
    with pytest.raises(ti.TaichiRuntimeError, match="name in source_bindings"):
        module.add_execution_plan(
            "missing_parent",
            graph,
            source_bindings={name: value for name, value in sources.items() if name != "input"},
            resource_bindings={b: dict(pixels="seed")},
        )
    assert not module._exported_kernels and not module._graph_export_failed
    private = module.add_execution_plan("private", graph, **options, resource_bindings={b: dict(pixels="seed")})
    document = json.loads(private.to_json())
    images = [r for r in document["resources"] if r.get("provider") == "image"]
    assert len(images) == 3 and len(document["initialization"]) == 1
    views = [r for r in document["resources"] if r["kind"] == "ndarray_view"]
    assert sorted((v["offset"], v["shape"][0]) for v in views) == [(2, 9), (2, 24)]
    assert not module._data_payloads  # No producer texture snapshot.
    module.add_execution_plan("public", graph, **options, resource_bindings={a: "a", b: "b", c: "c"})
    export_ms = (time.perf_counter() - start) * 1000
    graph.close()
    artifact = tmp_path / "image-transfer-graph.tcm"
    module.archive(str(artifact))
    ti.reset()
    for mode in ("standalone", "borrowed"):
        run = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_image_transfer_export_probe.py")),
                mode,
                str(artifact),
                str(tmp_path / f"{mode}.json"),
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
            dict(
                export_ms=export_ms, artifact_bytes=artifact.stat().st_size, plan_bytes=len(private.to_json().encode())
            ),
            indent=2,
        )
    )
