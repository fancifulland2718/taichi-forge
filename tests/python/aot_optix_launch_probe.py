"""Loaded OptiX launches, mutable resources and downstream numeric gradients."""

import copy
import json
from pathlib import Path
import statistics
import struct
import sys
import time

from taichi_forge_runtime.aot import ExecutionPlan, Runtime
from tests.python.aot_optix_probe import raises, read, write
from tests.python.aot_optix_program_resource_probe import contract, parameters


def native(operation, bindings, parameters=None, provider="optix", **extra):
    return dict(
        kind="native",
        provider=provider,
        operation=operation,
        version=1,
        bindings=bindings,
        parameters=parameters or {},
        **extra,
    )


def program_parameters(code):
    p = parameters(code)
    p["groups"].append(dict(name="indirect", kind=0, entry=dict(module=0, name="__raygen__indirect")))
    p["expected_program"] = contract(p)
    return p


def layout(access):
    return dict(dtype="u32", shape=[8], element_shape=[], record_bytes=4, alignment=4, access=access)


def payload(offset, size, group=None, references=()):
    result = dict(offset=offset, size=size, references=list(references))
    if group is not None:
        result["group"] = group
    return result


def reference(kind, index, offset):
    return dict(kind=kind, index=index, offset=offset)


def packet(p, indirect=False):
    if indirect:
        data = bytes(48)
        params = payload(0, 24)
        raygen = payload(24, 24, 3, [reference("buffer", 0, 0), reference("buffer", 1, 8), reference("texture", 0, 16)])
        miss, hit = [], []
    else:
        data = struct.pack("<QQIIIIIII", 0, 0, 5, 8, 7, 100, 1, 200, 0)
        params = payload(0, 24, references=[reference("scene", 0, 0), reference("buffer", 0, 8)])
        raygen = payload(24, 0, 0)
        miss, hit = [payload(24, 4, 1)] * 2, [payload(28, 8, 2), payload(36, 8, 2)]
    desc = dict(
        expected_program=p["expected_program"],
        dimensions=[8, 1, 1],
        parameters=params,
        raygen=raygen,
        miss=miss,
        hit=hit,
        array_count=2 if indirect else 1,
        image_count=int(indirect),
        scene_count=int(not indirect),
        array_layouts=[layout("write")] + ([layout("read")] if indirect else []),
    )
    bindings = dict(program="program", data="packet", array_0="out")
    bindings.update(dict(array_1="input", image_0="image") if indirect else dict(scene_0="scene"))
    return data, native("prepared_launch", bindings, desc, name="launch")


def plan_parts(p, indirect=False, public=False):
    operations = []
    if indirect:
        operations.append(native("upload", dict(image="image", pixels="pixels"), provider="image"))
    else:
        operations.append(native("instance_refit", dict(scene="scene", transforms="transforms")))
    operations += [
        native("launch", dict(launch="launch")),
        dict(kind="kernel", entry="convert", arguments=["out", "values"]),
        dict(kind="fill32", destination="dvalues", value=0),
        dict(kind="fill32", destination="loss", value=0),
        dict(kind="fill32", destination="dloss", value=1065353216),
        dict(kind="kernel", entry="energy", arguments=["values", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["values", "loss"]),
    ]
    resources = [dict(kind="ndarray", name="dloss", dtype="f32", shape=[1])]
    if not public:
        resources += [
            native("context", {}, name="context"),
            native("program", dict(context="context", module_0="ptx"), p, name="program"),
        ]
        if indirect:
            resources.append(
                native(
                    "sampled",
                    {},
                    dict(shape=[2, 2], format="r32f", filter="nearest", address_modes=["clamp_to_edge"] * 2),
                    provider="image",
                    name="image",
                )
            )
        else:
            resources += [
                native(
                    "triangle_gas",
                    dict(context="context", vertices="vertices", indices="indices"),
                    dict(vertex_count=3, triangle_count=1),
                    name="gas",
                ),
                native(
                    "instance_scene",
                    dict(context="context", gas_0="gas", transforms="transforms"),
                    dict(instance_count=1, custom_indices=[17], opaque=[False], allow_update=True),
                    name="scene",
                ),
            ]
        resources.append(packet(p, indirect)[1])
    return operations, resources


def main(mode, artifact, adapter):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    records = []
    with runtime:
        runtime.configure_provider("optix", adapter_path=adapter)
        assert runtime.native_operation_capabilities("optix_program_launch") == {"cuda_ordered"}
        assert runtime.native_resource_capabilities("optix_prepared_launch") == {"cuda_ordered"}
        start = time.perf_counter()
        module = runtime.load_module(artifact)
        load_ms = (time.perf_counter() - start) * 1000
        frame = {
            name: runtime.ndarray(dtype, shape)
            for name, dtype, shape in (
                ("vertices", "f32", (3, 3)),
                ("indices", "i32", (1, 3)),
                ("transforms", "f32", (1, 12)),
                ("pixels", "f32", (2, 2)),
                ("out", "u32", (8,)),
                ("input", "u32", (8,)),
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
        calls = []

        def check(call, expected):
            call()
            assert read(frame["out"], "I") == expected
            assert read(frame["values"]) == expected
            assert read(frame["dvalues"]) == expected
            assert read(frame["loss"]) == [sum(v * v / 2 for v in expected)]

        for name in ("trace", "indirect"):
            plan = module.execution_plan(name)
            if name == "indirect":
                document = json.loads(plan.to_json())
                document["operations"].pop(0)
                uninitialized = ExecutionPlan.from_json(json.dumps(document))
                raises(
                    lambda: uninitialized.prepare(module, {key: frame[key] for key in uninitialized.argument_names}),
                    "initial pixels or an unconditional upload",
                )
            write(frame["out"], [0xFFFFFFFF] * 8, "I")
            start = time.perf_counter()
            call = plan.prepare(module, {key: frame[key] for key in plan.argument_names})
            prepare_ms = (time.perf_counter() - start) * 1000
            assert read(frame["out"], "I") == [0xFFFFFFFF] * 8  # Preparation launches no rays.
            assert call.info["execution_kind"] == "cuda_ordered"
            calls.append(call)
            for change in (0, 1, 0):
                if name == "trace":
                    transform = identity.copy()
                    transform[3] = change * 5
                    write(frame["transforms"], transform)
                    expected = [12] * 8 if change else [122, 12, 12, 12] * 2
                else:
                    write(frame["pixels"], [9 - change * 7] * 4)
                    write(frame["input"], [i + change * 40 for i in range(8)], "I")
                    expected = [i + change * 40 + 9 - change * 7 for i in range(8)]
                check(call, expected)
            submit, complete = [], []
            for _ in range(7):
                start = time.perf_counter_ns()
                call()
                queued = time.perf_counter_ns()
                runtime.wait()
                submit.append((queued - start) / 1000)
                complete.append((time.perf_counter_ns() - start) / 1000)
            records.append(
                dict(
                    name=name,
                    prepare_ms=prepare_ms,
                    submit_us=statistics.median(submit),
                    complete_us=statistics.median(complete),
                    info=call.info,
                )
            )

        # Explicit public owners share the same pipeline and scene across plans.
        code = (Path(__file__).with_name("assets") / "hardware_optix_program.ptx").read_bytes()
        p = program_parameters(code)
        context = runtime.native_resource("optix", "context")
        with runtime.host_blob(code) as ptx:
            program = runtime.native_resource(
                "optix", "program", parameters=p, bindings=dict(context=context, module_0=ptx)
            )
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
        data, description = packet(p)

        def prepare(desc=None, data_bytes=data, output=None, scene_owner=None):
            with runtime.host_blob(data_bytes) as blob:
                return runtime.native_resource(
                    "optix",
                    "prepared_launch",
                    parameters=description["parameters"] if desc is None else desc,
                    bindings=dict(
                        program=program,
                        data=blob,
                        scene_0=scene if scene_owner is None else scene_owner,
                        array_0=frame["out"] if output is None else output,
                    ),
                )

        for change, message in (
            (dict(expected_program="0" * 64), "contract mismatch"),
            (dict(dimensions=[0, 1, 1]), "positive"),
            (dict(raygen=payload(24, 0, 1)), "wrong group kind"),
            (dict(raygen=payload(24, 0, 99)), "out of range"),
            (dict(hit=[]), "SBT does not cover"),
        ):
            desc = dict(description["parameters"], **change)
            raises(lambda: prepare(desc), message)
        desc = copy.deepcopy(description["parameters"])
        desc["parameters"]["references"][1]["offset"] = 4
        raises(lambda: prepare(desc), "misaligned")
        raises(lambda: prepare(data_bytes=data[:32]), "host blob")
        raises(lambda: prepare(data_bytes=bytes([1]) + data[1:]), "must be zero")
        invalid = runtime.ndarray("f32", 8)
        try:
            raises(lambda: prepare(output=invalid), "scalar dtype")
        finally:
            invalid.close()
        invalid = runtime.ndarray("u32", (4, 2))
        try:
            raises(lambda: prepare(output=invalid), "shape")
        finally:
            invalid.close()
        raises(lambda: prepare(scene_owner=gas), "triangle scene or IAS")
        other_context = runtime.native_resource("optix", "context")
        other_scene = runtime.native_resource(
            "optix",
            "triangle_scene",
            parameters=dict(vertex_count=3, triangle_count=1),
            bindings=dict(context=other_context, vertices=frame["vertices"], indices=frame["indices"]),
        )
        raises(lambda: prepare(scene_owner=other_scene), "another context")
        other_scene.close()
        other_context.close()
        launch = prepare()
        public = module.execution_plan("public")
        public_frame = dict(frame, launch=launch, scene=scene)
        first = public.prepare(module, {name: public_frame[name] for name in public.argument_names})
        second = public.prepare(module, {name: public_frame[name] for name in public.argument_names})
        check(first, [122, 12, 12, 12] * 2)
        rebound_data = data[:16] + struct.pack("<I", 9) + data[20:]
        rebound_launch = prepare(data_bytes=rebound_data)
        rebound_frame = dict(public_frame, launch=rebound_launch)
        rebound = public.prepare(module, {name: rebound_frame[name] for name in public.argument_names})
        check(rebound, [126, 16, 16, 16] * 2)
        check(first, [122, 12, 12, 12] * 2)
        rebound.close()
        rebound_launch.close()
        first.close()
        check(second, [122, 12, 12, 12] * 2)
        # Exactly aliased SBT buffers retain one physical range; this is legal in JIT too.
        indirect_data, indirect_desc = packet(p, True)
        image = runtime.native_resource(
            "image",
            "sampled",
            parameters=dict(shape=[2, 2], format="r32f", filter="nearest", address_modes=["clamp_to_edge"] * 2),
        )
        with runtime.host_blob(indirect_data) as blob:
            alias_launch = runtime.native_resource(
                "optix",
                "prepared_launch",
                parameters=indirect_desc["parameters"],
                bindings=dict(program=program, data=blob, array_0=frame["out"], array_1=frame["out"], image_0=image),
            )
        operations, resources = plan_parts(p, indirect=True, public=True)
        alias_plan = ExecutionPlan(operations, resources=resources, gradients=dict(values="dvalues", loss="dloss"))
        alias_frame = dict(frame, launch=alias_launch, image=image)
        alias = alias_plan.prepare(module, {key: alias_frame[key] for key in alias_plan.argument_names})
        write(frame["pixels"], [2] * 4)
        write(frame["out"], range(8), "I")
        check(alias, [i + 2 for i in range(8)])
        check(alias, [i + 4 for i in range(8)])
        image.close()
        raises(alias, "retired dependency")
        alias.close()
        alias_launch.close()
        second()
        program.close()  # Retires dependent plans while their launch is in flight.
        raises(second, "retired dependency")
        second.close()
        launch.close()
        gas.close()
        scene.close()
        context.close()
        # Closing a public input retires calls before releasing its device storage.
        calls[-1]()
        frame["out"].close()
        raises(calls[-1])
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        for call in calls:
            call.close()
        for value in frame.values():
            value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(artifact).with_suffix(f".{mode}.json").write_text(json.dumps(dict(load_ms=load_ms, records=records), indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
