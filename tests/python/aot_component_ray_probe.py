"""Frozen component consumption using only the public Runtime and new geometry."""

import array
from contextlib import ExitStack, closing
import os
import sys

from taichi_forge_runtime.aot import Runtime


def write(value, data, kind="f"):
    value.write(array.array(kind, data).tobytes())


def rejected(operation):
    try:
        operation()
    except (RuntimeError, ValueError, TypeError):
        return
    raise AssertionError("Invalid AS binding was accepted")


def main(artifact):
    device = int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))
    with ExitStack() as scope:
        runtime = scope.enter_context(Runtime("vulkan", device_index=device))
        module = scope.enter_context(runtime.load_module(artifact))
        component = module.component("rays")
        records = []
        for count, height in ((5, 0.25), (11, -0.5)):
            vertices = scope.enter_context(closing(runtime.ndarray("f32", (3, 3))))
            indices = scope.enter_context(closing(runtime.ndarray("i32", (1, 3))))
            transforms = scope.enter_context(closing(runtime.ndarray("f32", (1, 12))))
            write(vertices, [-2, -2, height, 2, -2, height, 0, 2, height])
            write(indices, [0, 1, 2], "i")
            write(transforms, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
            blas = scope.enter_context(
                closing(
                    runtime.native_resource(
                        "vulkan_ray",
                        "triangle_blas",
                        parameters=dict(vertex_count=3, triangle_count=1),
                        bindings=dict(vertices=vertices, indices=indices),
                    )
                )
            )
            scene = scope.enter_context(
                closing(
                    runtime.native_resource(
                        "vulkan_ray",
                        "instance_tlas",
                        parameters=dict(instance_count=1),
                        bindings=dict(blas_0=blas, transforms=transforms),
                    )
                )
            )
            instance = scope.enter_context(component.instantiate(dict(count=count)))
            rejected(lambda: instance.prepare(arguments=dict(scene=vertices)))
            rejected(lambda: instance.prepare(arguments=dict(scene=blas)))
            records.append((instance, scene, height, count))
        for use_domain in (False, True):
            with ExitStack() as calls:
                prepared = []
                for instance, scene, _, _ in records:
                    domain = (
                        calls.enter_context(runtime.execution_domain())
                        if use_domain
                        else None
                    )
                    call = calls.enter_context(
                        instance.prepare(
                            arguments=dict(scene=scene), execution_domain=domain
                        )
                    )
                    call.initialize()
                    prepared.append(call)
                tickets = [calls.enter_context(call.submit()) for call in prepared]
                for ticket in tickets:
                    ticket.wait()
                for instance, _, height, count in records:
                    values = array.array("f")
                    values.frombytes(instance.resource("output").read())
                    assert list(values) == [
                        1 + 0.125 * i - height for i in range(count)
                    ], values
                if use_domain:
                    # Accepted work retains the exact scene until completion;
                    # closing one scene cannot change another instance's binding.
                    ticket = calls.enter_context(prepared[0].submit())
                    records[0][1].close()
                    ticket.wait()
                    rejected(prepared[0].submit)
                    with prepared[1].submit() as other:
                        other.wait()
        assert "taichi_forge" not in sys.modules
        print("COMPONENT_RAY_OK")


if __name__ == "__main__":
    main(*sys.argv[1:])
