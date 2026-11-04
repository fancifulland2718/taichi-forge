"""Runtime-only instance/primitive face partitions, with an analytic ray oracle."""

import array
from contextlib import ExitStack, closing
import os
import sys

from taichi_forge_runtime.aot import Runtime
from aot_component_ray_probe import rejected, write


def geometry(count, primitives, phase):
    vertices, indices = [], []
    for p in range(primitives):
        x = 4 * p
        vertices.extend((x - 1, -1, 0, x + 1, -1, 0, x, 1, 0))
        indices.extend(
            (3 * p, 3 * p + 1, 3 * p + 2) if p % 2 else (3 * p + 2, 3 * p + 1, 3 * p)
        )
    transforms, rays, faces, expected = [], [], [], []
    for i in range(count):
        transforms.extend((1, 0, 0, 100 * i, 0, 1, 0, 0, 0, 0, 1, 0.125 * i))
        # Different GAS primitive counts in the same IAS; instances reuse GAS.
        for p in range(primitives if i % 2 else 1):
            rule = (i + p + phase) % 2
            faces.append(rule)
            rays.extend((100 * i + 4 * p, 0, 2, 0.001, 0, 0, -1, 5))
            expected.append(2 - 0.125 * i if p % 2 or not rule else -1)
    return vertices, indices, transforms, rays, faces, expected


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
        component = module.component("faces")
        records = []
        for count, primitives, phase in ((3, 3, 1), (7, 4, 0)):
            vertices, indices, transforms, rays, faces, expected = geometry(
                count, primitives, phase
            )

            def allocate(dtype, shape, data=None, kind="f"):
                value = life.enter_context(closing(runtime.ndarray(dtype, shape)))
                if data is not None:
                    write(value, data, kind)
                return value

            v = allocate("f32", (3 * primitives, 3), vertices)
            ix = allocate("i32", (primitives, 3), indices, "i")
            v_one = allocate("f32", (3, 3), vertices[:9])
            ix_one = allocate("i32", (1, 3), indices[:3], "i")
            gas = []
            for n, vd, ind in ((1, v_one, ix_one), (primitives, v, ix)):
                gas.append(
                    life.enter_context(
                        closing(
                            runtime.native_resource(
                                "optix",
                                "triangle_gas",
                                parameters=dict(vertex_count=3 * n, triangle_count=n),
                                bindings=dict(
                                    context=context, vertices=vd, indices=ind
                                ),
                            )
                        )
                    )
                )
            tx = allocate("f32", (count, 3, 4), transforms)
            scene = life.enter_context(
                closing(
                    runtime.native_resource(
                        "optix",
                        "instance_scene",
                        parameters=dict(instance_count=count),
                        bindings=dict(
                            context=context,
                            transforms=tx,
                            **{f"gas_{i}": gas[i % 2] for i in range(count)},
                        ),
                    )
                )
            )
            n = len(faces)
            args = dict(
                scene=scene,
                rays=allocate("f32", (n, 8), rays),
                hits=allocate("f32", (n, 4)),
                hit_indices=allocate("i32", (n, 4)),
                occluded=allocate("i32", n),
                faces=allocate("i32", n, faces, "i"),
            )
            instance = life.enter_context(component.instantiate(dict(count=n)))
            rejected(
                lambda: instance.prepare(
                    arguments=dict(args, faces=allocate("i32", n + 1))
                )
            )
            rejected(
                lambda: instance.prepare(arguments=dict(args, faces=allocate("f32", n)))
            )
            records.append((instance, args, expected))
        for use_domain in (False, True):
            with ExitStack() as calls:
                prepared = []
                for instance, args, _ in records:
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
                tickets = [calls.enter_context(call.submit()) for call in prepared]
                for ticket in tickets:
                    ticket.wait()
                for instance, _, expected in records:
                    actual = array.array("f")
                    actual.frombytes(instance.resource("output").read())
                    assert list(actual) == expected, (actual, expected)
                if use_domain:
                    ticket = calls.enter_context(prepared[0].submit())
                    records[0][1]["faces"].close()
                    ticket.wait()
                    rejected(prepared[0].submit)
                    with prepared[1].submit() as other:
                        other.wait()
        assert "taichi_forge" not in sys.modules
        print("COMPONENT_OPTIX_FACES_OK")


if __name__ == "__main__":
    main(*sys.argv[1:])
