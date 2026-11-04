"""Refit/query a frozen component with new geometry and instance topology."""

import array
from contextlib import ExitStack, closing
import os
import sys

from taichi_forge_runtime.aot import Runtime
from aot_component_ray_probe import rejected, write


def main(artifact):
    with ExitStack() as life:
        runtime = life.enter_context(
            Runtime(
                "vulkan",
                device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")),
            )
        )
        module = life.enter_context(runtime.load_module(artifact))
        component = module.component("moving")
        records = []
        for count, height in ((3, 0.25), (7, -0.5)):
            vertices = life.enter_context(closing(runtime.ndarray("f32", (3, 3))))
            indices = life.enter_context(closing(runtime.ndarray("i32", (1, 3))))
            poses = life.enter_context(closing(runtime.ndarray("f32", (count, 3, 4))))
            shift = life.enter_context(closing(runtime.ndarray("f32", 1)))
            write(vertices, [-2, -2, height, 2, -2, height, 0, 2, height])
            write(indices, [0, 1, 2], "i")
            write(poses, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] * count)
            blas = life.enter_context(
                closing(
                    runtime.native_resource(
                        "vulkan_ray",
                        "triangle_blas",
                        parameters=dict(vertex_count=3, triangle_count=1),
                        bindings=dict(vertices=vertices, indices=indices),
                    )
                )
            )
            scene = life.enter_context(
                closing(
                    runtime.native_resource(
                        "vulkan_ray",
                        "instance_tlas",
                        parameters=dict(instance_count=count),
                        bindings=dict(
                            transforms=poses,
                            **{f"blas_{i}": blas for i in range(count)},
                        ),
                    )
                )
            )
            instance = life.enter_context(component.instantiate(dict(count=count)))
            bad_poses = life.enter_context(
                closing(runtime.ndarray("f32", (count + 1, 3, 4)))
            )
            rejected(
                lambda: instance.prepare(
                    arguments=dict(scene=scene, transforms=bad_poses, shift=shift)
                )
            )
            records.append((instance, scene, poses, shift, count, height))
        for use_domain in (False, True):
            with ExitStack() as calls:
                prepared = []
                for instance, scene, poses, shift, _, _ in records:
                    domain = (
                        calls.enter_context(runtime.execution_domain())
                        if use_domain
                        else None
                    )
                    call = calls.enter_context(
                        instance.prepare(
                            arguments=dict(scene=scene, transforms=poses, shift=shift),
                            execution_domain=domain,
                        )
                    )
                    call.initialize()
                    prepared.append(call)
                for offset in (0.25, -0.25, 0.0):
                    for record in records:
                        write(record[3], [offset])
                    tickets = [calls.enter_context(call.submit()) for call in prepared]
                    for ticket in tickets:
                        ticket.wait()
                    for instance, _, _, _, count, height in records:
                        values = array.array("f")
                        values.frombytes(instance.resource("output").read())
                        assert list(values) == [
                            2 - offset - 0.125 * i - height for i in range(count)
                        ], values
                if use_domain:
                    ticket = calls.enter_context(prepared[0].submit())
                    records[0][1].close()
                    ticket.wait()
                    rejected(prepared[0].submit)
                    with prepared[1].submit() as other:
                        other.wait()
        assert "taichi_forge" not in sys.modules
        print("COMPONENT_REFIT_OK")


if __name__ == "__main__":
    main(*sys.argv[1:])
