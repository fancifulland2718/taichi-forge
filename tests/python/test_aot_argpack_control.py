"""Immutable pack owners survive conditional and nested native recordings."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import taichi_forge as ti
from taichi_forge._lib import utils
from taichi_forge_runtime.aot import Scalar
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_argpack_control_and_suffix_adjoint(tmp_path):
    child = ti.types.argpack(step=ti.f32)
    pack = ti.types.argpack(gain=ti.f32, child=child,
                           values=ti.types.ndarray(ti.f32, 1, needs_grad=True))

    @ti.kernel
    def condition(p: pack, count: ti.types.ndarray(ti.i32, 1),
                  predicate: ti.types.ndarray(ti.i32, 1),
                  limits: ti.types.ndarray(ti.i32, 1), slot: ti.i32):
        predicate[0] = int(count[0] < limits[slot] and p.gain > 0)

    @ti.kernel
    def advance(p: pack, count: ti.types.ndarray(ti.i32, 1)):
        count[0] += 1
        for i in p.values:
            p.values[i] += p.child.step

    @ti.kernel
    def zero(count: ti.types.ndarray(ti.i32, 1)):
        count[0] = 0

    @ti.kernel
    def increment(count: ti.types.ndarray(ti.i32, 1)):
        count[0] += 1

    @ti.kernel
    def energy(p: pack, loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.values:
            loss[0] += p.gain * p.values[i] ** 2

    started = time.perf_counter()
    module = ti.aot.Module()
    for kernel in (condition, advance, zero, increment, energy):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    initial = module.add_data(struct.pack("3f", 1, -2, 3))
    seed = module.add_data(struct.pack("f", 1))
    resources = [dict(kind="ndarray", name=n, dtype="i32", shape=[1])
                 for n in ("count", "predicate", "outer", "outer_predicate")]
    resources += [dict(kind="ndarray", name=n, dtype="f32", shape=[3]) for n in ("values", "dx")]
    resources += [dict(kind="ndarray", name="initial", dtype="f32", shape=[3], initial_data=initial),
                  dict(kind="ndarray", name="seed", dtype="f32", shape=[1], initial_data=seed)]
    resources += [Scalar("i32", slot).resource(f"slot{slot}") for slot in (0, 1)]
    prefix = [dict(kind="fill32", destination=n, value=0) for n in ("count", "outer", "loss", "dx")]
    prefix += [dict(kind="copy", source="initial", destination="values")]
    suffix = [dict(kind="kernel", entry="energy", arguments=["pack", "loss", "values"]),
              dict(kind="kernel", entry="backward", arguments=["pack", "loss", "values"]),
              dict(kind="copy", source="dx", destination="gradient"),
              dict(kind="copy", source="count", destination="counts", size_bytes=4),
              dict(kind="copy", source="outer", destination="counts", destination_offset=4, size_bytes=4)]

    def test_condition(count, predicate, slot):
        return dict(kind="kernel", entry="condition",
                    arguments=["pack", count, predicate, "limits", f"slot{slot}", "values"])

    inner_test = test_condition("count", "predicate", 1)
    outer_test = test_condition("outer", "outer_predicate", 0)
    body = dict(kind="kernel", entry="advance", arguments=["pack", "count", "values"])
    reset = dict(kind="kernel", entry="zero", arguments=["count"])
    outer_advance = dict(kind="kernel", entry="increment", arguments=["outer"])
    loop = dict(kind="while", predicate="predicate", max_iterations=3, condition=[inner_test], body=[body])
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for name, budget, strategy, nested, gain, step in (
        ("flat", 3, "compact", False, 2, 1),
        ("zero", 0, "compact", False, 2, 1),
        ("rebound", 3, "compact", False, 3, 2),
        ("nested", 2, "compact", True, 2, 1),
        *(([("chained", 3, "chained", False, 2, 1),
            ("conditional", 3, "conditional", False, 2, 1)]) if arch == "vulkan" else []),
    ):
        selected = dict(loop, max_iterations=budget)
        if nested:
            selected = dict(kind="while", predicate="outer_predicate", max_iterations=budget,
                            condition=[outer_test], body=[reset, loop, outer_advance])
        if arch == "vulkan":
            if nested:
                selected["body"] = [reset, inner_test, body, inner_test, outer_advance, outer_test]
                selected["lowering"] = dict(kind="vulkan_nested", version=1, counter="outer", inners=[dict(
                    predicate="predicate", counter="count", condition_begin=2, body_begin=3, end=5,
                    max_iterations=3, chunk_size=3)])
            else:
                selected["lowering"] = dict(kind="vulkan_structured", version=1, counter="count",
                                            chunks=[dict(iterations=budget, strategy=strategy)] if budget else [])
        module.add_execution_plan(name, ti.aot.ExecutionPlan(
            prefix + [selected] + suffix,
            resources=resources + module.argpack_resources("pack", "energy", 0,
                                                            dict(gain=gain, child=dict(step=step))),
            gradients=dict(values="dx", loss="seed")))
    artifact = tmp_path / "pack-control.tcm"
    module.archive(str(artifact))
    print(json.dumps(dict(control_export_seconds=time.perf_counter() - started,
                          artifact_bytes=artifact.stat().st_size)))
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path),
                       TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
                       TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir())
    ti.reset()
    for mode in ("standalone", "borrowed"):
        result = subprocess.run([sys.executable, *(["-S"] if mode == "standalone" else []),
                                 str(Path(__file__).with_name("aot_argpack_control_probe.py")),
                                 mode, arch, str(artifact)], env=environment, capture_output=True, text=True, timeout=90)
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0, output
        assert "ARGPACK_CONTROL_PASSED" in output
        assert not any(token in output for token in ("VUID-", "Validation Error", "SYNC-HAZARD"))
