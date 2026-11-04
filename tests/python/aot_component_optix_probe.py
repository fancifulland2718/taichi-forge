"""Runtime-only OptiX component consumption on new geometry and query extents."""

import array
from contextlib import ExitStack, closing
import os
import sys
from taichi_forge_runtime.aot import Runtime
from aot_component_ray_probe import rejected, write


def main(artifact):
    with ExitStack() as life:
        runtime = life.enter_context(Runtime("cuda"))
        runtime.configure_provider(
            "optix", adapter_path=os.environ["TAICHI_FORGE_TEST_OPTIX_PROVIDER"]
        )
        context = life.enter_context(
            closing(
                runtime.native_resource("optix", "context", parameters={}, bindings={})
            )
        )
        module = life.enter_context(runtime.load_module(artifact))
        component = module.component("optix")
        records = []
        for count, rays, height in ((3, 5, 0.25), (7, 11, -0.5)):
            arrays = {
                name: life.enter_context(closing(runtime.ndarray(dtype, shape)))
                for name, dtype, shape in (
                    ("vertices", "f32", (3, 3)),
                    ("indices", "i32", (1, 3)),
                    ("transforms", "f32", (count, 3, 4)),
                    ("shift", "f32", 1),
                    ("rays", "f32", (rays, 8)),
                    ("hits", "f32", (rays, 4)),
                    ("hit_indices", "i32", (rays, 4)),
                    ("occluded", "i32", rays),
                )
            }
            write(arrays["vertices"], [-2, -2, height, 2, -2, height, 0, 2, height])
            write(arrays["indices"], [0, 1, 2], "i")
            write(arrays["transforms"], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] * count)
            gas = life.enter_context(
                closing(
                    runtime.native_resource(
                        "optix",
                        "triangle_gas",
                        parameters=dict(
                            vertex_count=3, triangle_count=1, allow_update=True
                        ),
                        bindings=dict(
                            context=context,
                            vertices=arrays["vertices"],
                            indices=arrays["indices"],
                        ),
                    )
                )
            )
            scene = life.enter_context(
                closing(
                    runtime.native_resource(
                        "optix",
                        "instance_scene",
                        parameters=dict(instance_count=count, allow_update=True),
                        bindings=dict(
                            context=context,
                            transforms=arrays["transforms"],
                            **{f"gas_{i}": gas for i in range(count)},
                        ),
                    )
                )
            )
            instance = life.enter_context(component.instantiate(dict(count=rays)))
            args = dict(
                scene=scene,
                **{
                    key: value
                    for key, value in arrays.items()
                    if key not in ("vertices", "indices")
                },
            )
            bad = life.enter_context(closing(runtime.ndarray("f32", (rays + 1, 4))))
            rejected(lambda: instance.prepare(arguments=dict(args, hits=bad)))
            rejected(lambda: instance.prepare(arguments=dict(args, scene=gas)))
            records.append((instance, scene, args, count, rays, height))
        for use_domain in (False, True):
            with ExitStack() as calls:
                prepared = []
                for instance, scene, args, _, _, _ in records:
                    domain = (
                        calls.enter_context(runtime.execution_domain())
                        if use_domain
                        else None
                    )
                    call = calls.enter_context(
                        instance.prepare(arguments=args, execution_domain=domain)
                    )
                    call.initialize()
                    prepared.append(call)
                for offset in (0.25, -0.25, 0.0):
                    for _, _, args, *_ in records:
                        write(args["shift"], [offset])
                    tickets = [calls.enter_context(call.submit()) for call in prepared]
                    for ticket in tickets:
                        ticket.wait()
                    for instance, _, _, count, rays, height in records:
                        actual = array.array("f")
                        actual.frombytes(instance.resource("output").read())
                        assert list(actual) == [
                            2 - offset - 0.125 * (i % count) - height
                            for i in range(rays)
                        ], actual
                if use_domain:
                    ticket = calls.enter_context(prepared[0].submit())
                    records[0][1].close()
                    ticket.wait()
                    rejected(prepared[0].submit)
                    with prepared[1].submit() as other:
                        other.wait()
        assert "taichi_forge" not in sys.modules
        print("COMPONENT_OPTIX_OK")


if __name__ == "__main__":
    main(*sys.argv[1:])
