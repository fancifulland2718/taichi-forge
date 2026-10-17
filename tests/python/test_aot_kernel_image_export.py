"""Graph Texture arguments lower to shared consumer-owned sampled images."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils
from tests.python.test_aot_binding_recipe_export import selected


@pytest.mark.parametrize("recipe", ["ordinary", "selected"])
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_kernel_image_graph_export(tmp_path, recipe):
    arch = ti.cfg.arch.name

    @ti.kernel
    def sample(
        image: ti.types.texture(num_dimensions=2),
        alias: ti.types.texture(num_dimensions=2),
        output: ti.types.ndarray(ti.f32, 2),
    ):
        for i, j in output:
            output[i, j] = (
                image.fetch(ti.Vector([i, j]), 0).x
                + alias.fetch(ti.Vector([i, j]), 0).x
            )

    @ti.kernel
    def seed(
        dout: ti.types.ndarray(ti.f32, 2),
        loss: ti.types.ndarray(ti.f32, 1),
        dloss: ti.types.ndarray(ti.f32, 1),
    ):
        for i, j in dout:
            dout[i, j] = 0
        loss[0] = 0
        dloss[0] = 1

    @ti.kernel
    def energy(
        output: ti.types.ndarray(ti.f32, 2, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i, j in output:
            loss[0] += 0.5 * output[i, j] ** 2

    @ti.kernel
    def analytic(
        output: ti.types.ndarray(ti.f32, 2),
        dout: ti.types.ndarray(ti.f32, 2),
        loss: ti.types.ndarray(ti.f32, 1),
    ):
        for i, j in output:
            loss[0] += 0.5 * output[i, j] ** 2
            dout[i, j] = output[i, j]

    builder = ti.graph.GraphBuilder()
    image = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "image", ndim=2)
    alias = ti.graph.Arg(ti.graph.ArgKind.TEXTURE, "alias", ndim=2)
    output = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "output", ti.f32, ndim=2)
    dout = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "dout", ti.f32, ndim=2)
    loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    dloss = builder.private_ndarray("dloss", ti.f32, 1)
    texture = ti.Texture(
        ti.Format.r32f,
        (3, 5),
        sampler=ti.hardware.sampling.SamplerConfig(
            min_filter="nearest", mag_filter="nearest"
        ),
    )
    pixels = ti.ndarray(ti.f32, (3, 5))
    pixels.fill(99)  # The consumer supplies different data; export copies none.
    texture.from_ndarray(pixels)
    if recipe == "ordinary":
        builder._bind_internal_runtime_arg(alias, texture)
    builder.dispatch(sample, image, alias, output)
    builder.dispatch(seed, dout, loss, dloss)
    if recipe == "ordinary":
        builder.dispatch(energy, output, loss)
        builder.dispatch(energy.grad, output, loss)
    else:
        builder.dispatch(analytic, output, dout, loss)
    module = ti.aot.Module()

    def export(graph):
        arguments = dict(
            source_bindings=(
                dict(image=texture)
                if recipe == "ordinary"
                else dict(image=texture, alias=texture)
            ),
            resource_bindings={texture: dict(pixels="pixels")},
        )
        if recipe == "ordinary":
            arguments["gradients"] = dict(output="dout", loss="dloss")
            # Reject a mismatched source before publishing compiled entries.
            wrong = ti.Texture(ti.Format.r32f, (3,))
            with pytest.raises(ti.TaichiRuntimeError, match="dimension"):
                module.add_execution_plan(
                    "bad", graph, source_bindings=dict(image=wrong)
                )
            assert not module._exported_kernels and not module._graph_export_failed
            if arch == "vulkan":
                extended = ti.Texture(
                    ti.Format.r32f,
                    (3, 5),
                    sampler=ti.hardware.sampling.SamplerConfig(min_filter="nearest"),
                )
                with pytest.raises(ti.TaichiRuntimeError, match="basic sampler"):
                    module.add_execution_plan(
                        "sampler", graph, source_bindings=dict(image=extended)
                    )
                assert not module._exported_kernels and not module._graph_export_failed
        else:
            arrays = dict(
                output=ti.ndarray(ti.f32, (3, 5)),
                dout=ti.ndarray(ti.f32, (3, 5)),
                loss=ti.ndarray(ti.f32, 1),
            )
            graph.submit(graph.bind(dict(arrays, image=texture, alias=texture))).wait()
            np.testing.assert_array_equal(
                arrays["output"].to_numpy(), np.full((3, 5), 198)
            )
            np.testing.assert_array_equal(
                arrays["dout"].to_numpy(), np.full((3, 5), 198)
            )
        start = time.perf_counter()
        plan = module.add_execution_plan("step", graph, **arguments)
        elapsed = (time.perf_counter() - start) * 1000
        assert plan.argument_names == ("dout", "loss", "output", "pixels")
        document = json.loads(plan.to_json())
        images = [r for r in document["resources"] if r.get("provider") == "image"]
        assert len(images) == 1
        assert images[0]["parameters"]["shape"] == [3, 5]
        assert images[0]["parameters"]["filter"] == "nearest"
        assert document["operations"][0]["arguments"][:2] == [images[0]["name"]] * 2
        assert plan.binding_strategy == (
            "cuda_immutable_argument_frames"
            if recipe == "selected" and arch == "cuda"
            else None
        )
        count = len(module._exported_kernels)
        module.add_execution_plan("again", graph, **arguments)
        assert len(module._exported_kernels) == count
        return dict(export_ms=elapsed, kernels=count)

    if recipe == "selected":
        with selected(builder.freeze()) as graph:
            costs = export(graph)
    else:
        graph = builder.compile()
        costs = export(graph)
        # Direct JIT numerical reference for the real downstream gradient.
        values = ti.ndarray(ti.f32, (3, 5), needs_grad=True)
        total = ti.ndarray(ti.f32, 1, needs_grad=True)
        sample(texture, texture, values)
        values.grad.fill(0)
        total.fill(0)
        total.grad.fill(1)
        energy(values, total)
        energy.grad(values, total)
        np.testing.assert_array_equal(values.grad.to_numpy(), np.full((3, 5), 198))
        graph.close()
    artifact = tmp_path / "images.tcm"
    module.archive(str(artifact))
    costs["bytes"] = artifact.stat().st_size
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_kernel_image_export_probe.py")),
                mode,
                str(artifact),
                recipe,
                str(tmp_path / f"{mode}.json"),
                arch,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
    (tmp_path / "producer.json").write_text(json.dumps(costs, indent=2))
