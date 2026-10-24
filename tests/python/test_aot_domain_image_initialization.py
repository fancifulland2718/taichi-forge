"""Image construction observes pixels at prepare, before explicit initialization."""

import array
import os
from contextlib import ExitStack

import pytest
import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils
from tests.python.aot_kernel_image_probe import native


@pytest.mark.parametrize("private_pixels", [False, True])
@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_initial_pixels_are_consumed_before_prepare_returns(tmp_path, private_pixels):
    @ti.kernel
    def sample(image: ti.types.texture(1), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = image.fetch(ti.Vector([i]), 0).x

    @ti.kernel
    def energy(output: ti.types.ndarray(ti.f32, 1, needs_grad=True),
               loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in output:
            loss[0] += output[i] * output[i]

    arch = ti.cfg.arch.name
    module = ti.aot.Module()
    module.add_kernel(sample)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    expected = ([3, -2, 7, 11], [13, 17, -19, 23])
    for index, values in enumerate(expected):
        resources = []
        if private_pixels:
            payload = module.add_data(array.array("f", [99, *values, -99]).tobytes())
            resources += [dict(kind="ndarray", name="seed", dtype="f32", shape=[6], initial_data=payload),
                          dict(kind="ndarray_view", name="pixels", array="seed", offset=1, shape=[4])]
        resources.append(dict(native("image", "sampled", dict(pixels="pixels"),
                                     shape=[4], format="r32f"), name="image"))
        module.add_execution_plan(f"step{index}", ti.aot.ExecutionPlan(
            [dict(kind="kernel", entry="sample", arguments=["image", "output"]),
             dict(kind="fill32", destination="doutput", value=0),
             dict(kind="fill32", destination="loss", value=0),
             dict(kind="fill32", destination="dloss", value=1065353216),
             dict(kind="kernel", entry="energy", arguments=["output", "loss"]),
             dict(kind="kernel", entry="backward", arguments=["output", "loss"])],
            resources=resources, gradients=dict(output="doutput", loss="dloss")))
    path = tmp_path / "frozen-image-pixels.tcm"
    module.archive(str(path))
    ti.reset()

    def write(buffer, values):
        buffer.write(array.array("f", values).tobytes())

    def read(buffer):
        values = array.array("f")
        values.frombytes(buffer.read())
        return list(values)

    with ExitStack() as scope:
        runtime = scope.enter_context(aot.Runtime(
            arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0))
            if arch == "vulkan" else 0))
        loaded = scope.enter_context(runtime.load_module(path))
        pixels = runtime.ndarray("f32", 4)
        instances = []
        for index, values in enumerate(expected):
            write(pixels, values)
            output = runtime.ndarray("f32", 4)
            doutput = runtime.ndarray("f32", 4)
            bindings = dict(output=output, doutput=doutput, loss=runtime.ndarray("f32", 1),
                            dloss=runtime.ndarray("f32", 1))
            if not private_pixels:
                bindings["pixels"] = pixels
            domain = scope.enter_context(runtime.execution_domain(capacity=1, instance_capacity=1))
            prepared = scope.enter_context(loaded.prepare_execution(
                f"step{index}", bindings,
                execution_domain=domain))
            instances.append((prepared, bindings))
        # Initialization hasn't run. Mutating the shared construction input now
        # cannot change either image's already consumed initial state.
        write(pixels, [-91] * 4)
        runtime.wait()
        for prepared, _ in instances:
            prepared.initialize()
        with ExitStack() as tickets:
            pending = [tickets.enter_context(prepared.submit()) for prepared, _ in instances]
            for ticket in pending:
                ticket.wait()
        for (_, bindings), values in zip(instances, expected):
            assert read(bindings["output"]) == values
            assert read(bindings["doutput"]) == [2 * v for v in values]
            assert read(bindings["loss"]) == [sum(v * v for v in values)]
        instances[0][0].close()
        with instances[1][0].submit() as ticket:
            ticket.wait()
        assert read(instances[1][1]["output"]) == expected[1]
        assert read(instances[1][1]["doutput"]) == [2 * v for v in expected[1]]
