"""Compile immutable OptiX program resources without importing the producer."""

import copy
import gc
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import weakref

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_optix_probe import raises


def parameters(code):
    return dict(
        module_count=1,
        module_sha256=[hashlib.sha256(code).hexdigest()],
        groups=[
            dict(name="render", kind=0, entry=dict(module=0, name="__raygen__render")),
            dict(name="miss", kind=1, entry=dict(module=0, name="__miss__value")),
            dict(
                name="hit",
                kind=2,
                entry=dict(module=0, name="__closesthit__value"),
                any_hit=dict(module=0, name="__anyhit__mask"),
            ),
        ],
        parameter_name="params",
        parameter_size=24,
        payload_count=1,
        attribute_count=2,
        max_trace_depth=1,
    )


def contract(p):
    # Independent byte-level oracle for the durable code/layout identity.
    data = bytearray()

    def number(value):
        data.extend(struct.pack("<Q", value))

    def text(value):
        value = value.encode()
        number(len(value))
        data.extend(value)

    text("forge-optix-program-v1")
    number(p["module_count"])
    for digest in p["module_sha256"]:
        text(digest)
    number(len(p["groups"]))
    for group in p["groups"]:
        text(group["name"])
        number(group["kind"])
        for key in ("entry", "any_hit"):
            entry = group.get(key) or {}
            number(entry.get("module", 0))
            text(entry.get("name", ""))
    text(p["parameter_name"])
    for key in ("parameter_size", "payload_count", "attribute_count", "max_trace_depth", "allow_opacity_micromaps"):
        number(p.get(key, 0))
    return hashlib.sha256(data).hexdigest()


def main(mode, adapter, output):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    code = (Path(__file__).with_name("assets") / "hardware_optix_program.ptx").read_bytes()
    p = parameters(code)
    p["expected_program"] = contract(p)
    with runtime:
        assert runtime.native_resource_capabilities("optix_program") == {"cuda_ordered"}
        runtime.configure_provider("optix", adapter_path=adapter)
        assert runtime.probe_native_resource("optix_program")["available"]
        start = time.perf_counter()
        context = runtime.native_resource("optix", "context")
        context_ms = (time.perf_counter() - start) * 1000

        def create(params=None, data=code):
            with runtime.host_blob(data) as blob:
                ref = weakref.ref(blob)
                result = runtime.native_resource(
                    "optix",
                    "program",
                    parameters=p if params is None else params,
                    bindings=dict(context=context, module_0=blob),
                )
            del blob
            gc.collect()
            assert ref() is None  # PTX is consumed, not retained by the pipeline.
            return result

        for key, value, message in (
            ("expected_program", "0" * 64, "contract mismatch"),
            ("module_sha256", ["0" * 64], "PTX digest mismatch"),
            ("module_count", 2, "Binding group count"),
            ("payload_count", 33, "register counts"),
            ("parameter_size", 1 << 32, "uint32"),
        ):
            bad = copy.deepcopy(p)
            bad.pop("expected_program")
            bad[key] = value
            raises(lambda: create(bad), message)
        for change, message in (
            (dict(module=1, name="__raygen__render"), "entry module"),
            (dict(module=0, name="__miss__value"), "stage prefix"),
            (dict(module=0, name="__raygen__missing"), "create program"),
        ):
            bad = copy.deepcopy(p)
            bad.pop("expected_program")
            bad["groups"][0]["entry"] = change
            raises(lambda: create(bad), message)
        raises(lambda: create(data=code + b"\0"), "without NUL")
        raises(lambda: create(data=code + b"\n"), "PTX digest mismatch")
        # A real OptiX group-creation failure above must not poison its context.
        start = time.perf_counter()
        first = create()
        compile_ms = (time.perf_counter() - start) * 1000
        # Contract changes must be detected even when both layouts fit the ABI.
        bad = copy.deepcopy(p)
        bad["parameter_size"] = 32
        raises(lambda: create(bad), "contract mismatch")
        second = create()
        first.close()
        first.close()
        context.close()
        raises(create, "closed")
        second.close()
        context = runtime.native_resource("optix", "context")
        third = create()
        if mode == "borrowed":
            ti.reset()
        else:
            runtime.close()
        third.close()
        context.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules
    Path(output).write_text(json.dumps(dict(context_ms=context_ms, compile_ms=compile_ms, ptx_bytes=len(code))))


if __name__ == "__main__":
    main(*sys.argv[1:])
