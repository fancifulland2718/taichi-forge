"""Source-free consumer of ordinary Graph OptiX program exports."""

import json
from pathlib import Path
import statistics
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_optix_probe import raises, read, write


def main(mode, artifact, adapter):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    code = (Path(__file__).with_name("assets") / "hardware_optix_program.ptx").read_bytes()
    records = []
    with runtime:
        runtime.configure_provider("optix", adapter_path=adapter)
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        frame = {
            name: runtime.ndarray(dtype, shape)
            for name, dtype, shape in (
                ("vertices", "f32", (3, 3)),
                ("indices", "i32", (1, 3)),
                ("transforms", "f32", (1, 12)),
                ("out", "u32", (8,)),
                ("input", "u32", (8,)),
                ("pixels", "f32", (2, 2)),
                ("values", "f32", (8,)),
                ("dvalues", "f32", (8,)),
                ("loss", "f32", (1,)),
            )
        }
        write(frame["vertices"], [-1, -1, 0, 1, -1, 0, 0, 1, 0])
        write(frame["indices"], [0, 1, 2], "i")
        identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        write(frame["transforms"], identity)
        write(frame["pixels"], [9] * 4)
        write(frame["input"], range(8), "I")

        def check(call, expected):
            call()
            assert read(frame["out"], "I") == expected
            assert read(frame["values"]) == expected
            assert read(frame["dvalues"]) == expected
            assert read(frame["loss"]) == [sum(v * v / 2 for v in expected)]

        retained = []
        for prefix in ("trace", "indirect"):
            private = module.execution_plan(prefix + "_private")
            document = json.loads(private.to_json())
            spec = next(r["parameters"] for r in document["resources"] if r.get("operation") == "program")
            context = runtime.native_resource("optix", "context")

            def create_program(owner):
                with runtime.host_blob(code) as blob:
                    return runtime.native_resource(
                        "optix", "program", parameters=spec, bindings=dict(context=owner, module_0=blob)
                    )

            program = create_program(context)
            owners = [program, context]
            public_frame = dict(frame, shared_program=program)
            if prefix == "trace":
                gas = runtime.native_resource(
                    "optix",
                    "triangle_gas",
                    parameters=dict(vertex_count=3, triangle_count=1),
                    bindings=dict(context=context, vertices=frame["vertices"], indices=frame["indices"]),
                )
                scene = runtime.native_resource(
                    "optix",
                    "instance_scene",
                    parameters=dict(instance_count=1, custom_indices=[17], allow_update=True),
                    bindings=dict(context=context, gas_0=gas, transforms=frame["transforms"]),
                )
                public_frame["shared_scene"] = scene
                owners[:0] = [scene, gas]
            else:
                write(frame["pixels"], [2] * 4)
                image = runtime.native_resource(
                    "image",
                    "sampled",
                    parameters=dict(shape=[2, 2], format="r32f", filter="nearest", address_modes=["clamp_to_edge"] * 2),
                    bindings=dict(pixels=frame["pixels"]),
                )
                public_frame["shared_image"] = image
                owners.insert(0, image)
                write(frame["pixels"], [9] * 4)
            calls = []
            for suffix in ("private", "public"):
                plan = module.execution_plan(prefix + "_" + suffix)
                args = {name: public_frame[name] for name in plan.argument_names}
                if prefix == "trace" and suffix == "public":
                    other_context = runtime.native_resource("optix", "context")
                    other_program = create_program(other_context)
                    raises(lambda: plan.prepare(module, dict(args, shared_program=other_program)), "alias mismatch")
                    other_program.close()
                    other_context.close()
                start = time.perf_counter()
                call = plan.prepare(module, args)
                prepare_ms = (time.perf_counter() - start) * 1000
                calls.append(call)
                assert call.info["execution_kind"] == "cuda_ordered"
                for shift in (0, 5, 0):
                    if prefix == "trace":
                        transform = identity.copy()
                        transform[3] = shift
                        write(frame["transforms"], transform)
                        expected = [12] * 8 if shift else [122, 12, 12, 12] * 2
                    else:
                        write(frame["input"], [i + shift for i in range(8)], "I")
                        expected = [i + shift + (9 if suffix == "private" else 2) for i in range(8)]
                    check(call, expected)
                submits, completes = [], []
                for _ in range(5):
                    start = time.perf_counter_ns()
                    call()
                    submitted = time.perf_counter_ns()
                    runtime.wait()
                    submits.append((submitted - start) / 1000)
                    completes.append((time.perf_counter_ns() - start) / 1000)
                records.append(
                    dict(
                        name=prefix + "_" + suffix,
                        prepare_ms=prepare_ms,
                        submit_us=statistics.median(submits),
                        complete_us=statistics.median(completes),
                        info=call.info,
                    )
                )
            if prefix == "indirect":
                # Public texture data changes through the same native owner, without re-export.
                upload = ExecutionPlan(
                    [
                        dict(
                            kind="native",
                            provider="image",
                            operation="upload",
                            version=1,
                            parameters={},
                            bindings=dict(image="image", pixels="pixels"),
                        )
                    ]
                ).prepare(module, dict(image=image, pixels=frame["pixels"]))
                write(frame["pixels"], [3] * 4)
                upload()
                check(calls[1], [i + 3 for i in range(8)])
                check(calls[0], [i + 9 for i in range(8)])
                upload.close()
            calls[1]()
            program.close()
            raises(calls[1], "retired dependency")
            calls[1].close()
            check(calls[0], [i + 9 for i in range(8)] if prefix == "indirect" else [122, 12, 12, 12] * 2)
            retained.append(calls[0])
            for owner in owners:
                owner.close()
        retained[-1]()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for call in retained:
            raises(call)
            call.close()
        for value in frame.values():
            value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(artifact).with_suffix(f".{mode}.json").write_text(json.dumps(dict(load_ms=load_ms, records=records), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
