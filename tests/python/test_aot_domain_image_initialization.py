"""Image construction observes pixels at prepare, before explicit initialization."""

import array
import os
from contextlib import ExitStack

import taichi_forge as ti
from taichi_forge_runtime import aot
from tests import test_utils
from tests.python.aot_kernel_image_probe import native


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_initial_pixels_are_consumed_before_prepare_returns(tmp_path):
    @ti.kernel
    def sample(image: ti.types.texture(1), output: ti.types.ndarray(ti.f32, 1)):
        for i in output:
            output[i] = image.fetch(ti.Vector([i]), 0).x

    arch = ti.cfg.arch.name
    module = ti.aot.Module()
    module.add_kernel(sample)
    image = dict(native("image", "sampled", dict(pixels="pixels"),
                        shape=[4], format="r32f"), name="image")
    module.add_execution_plan("step", ti.aot.ExecutionPlan(
        [dict(kind="kernel", entry="sample", arguments=["image", "output"])],
        resources=[image]))
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
        expected = ([3, -2, 7, 11], [13, 17, -19, 23])
        for values in expected:
            write(pixels, values)
            output = runtime.ndarray("f32", 4)
            domain = scope.enter_context(runtime.execution_domain(capacity=1, instance_capacity=1))
            prepared = scope.enter_context(loaded.prepare_execution(
                "step", dict(pixels=pixels, output=output), execution_domain=domain))
            instances.append((prepared, output))
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
        for (_, output), values in zip(instances, expected):
            assert read(output) == values
        instances[0][0].close()
        with instances[1][0].submit() as ticket:
            ticket.wait()
        assert read(instances[1][1]) == expected[1]
