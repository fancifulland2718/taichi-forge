"""Persistent control, selected Vulkan recipes, suffix AD and ownership."""

import json
import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_persistent_while(tmp_path):
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"

    @ti.kernel
    def condition(
        counter: ti.types.ndarray(ti.i32, ndim=1),
        predicate: ti.types.ndarray(ti.i32, ndim=1),
        checks: ti.types.ndarray(ti.i32, ndim=1),
        target: ti.i32,
        inverted: ti.i32,
    ):
        predicate[0] = ti.cast(counter[0] < target, ti.i32) ^ inverted
        checks[0] += 1

    @ti.kernel
    def advance(
        counter: ti.types.ndarray(ti.i32, ndim=1),
        x: ti.types.ndarray(ti.f32, ndim=1),
        temporary: ti.types.ndarray(ti.f32, ndim=1),
    ):
        counter[0] += 1
        for i in x:
            x[i] += temporary[i]

    @ti.kernel
    def energy(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True), y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)
    ):
        for i in x:
            y[i] = x[i] * x[i]

    module = ti.aot.Module()
    for kernel in (condition, advance, energy):
        module.add_kernel(kernel)
    module.add_kernel(energy.grad, name="backward")
    resources = [dict(kind="ndarray", name=n, dtype="i32", shape=[1]) for n in ("counter", "predicate", "checks")]
    resources += [dict(kind="ndarray", name=n, dtype="f32", shape=[17]) for n in ("x", "temporary", "dx", "dy")]
    loop = dict(
        kind="while",
        predicate="predicate",
        max_iterations=4,
        condition=[
            dict(kind="kernel", entry="condition", arguments=["counter", "predicate", "checks", "target", "inverted"])
        ],
        body=[
            dict(kind="fill32", destination="temporary", value=1065353216),
            dict(kind="kernel", entry="advance", arguments=["counter", "x", "temporary"]),
        ],
    )
    prefix = [dict(kind="fill32", destination=n, value=0) for n in ("counter", "checks", "dx")]
    prefix += [
        dict(kind="fill32", destination="dy", value=1065353216),
        dict(kind="copy", source="input", destination="x"),
    ]
    suffix = [
        dict(kind="kernel", entry="energy", arguments=["x", "y"]),
        dict(kind="kernel", entry="backward", arguments=["x", "y"]),
        dict(kind="copy", source="dx", destination="gradient"),
    ]
    suffix += [
        dict(kind="copy", source=n, destination="state", destination_offset=i * 4, size_bytes=4)
        for i, n in enumerate(("counter", "predicate", "checks"))
    ]
    for name, budget, inverted in (
        ("normal", 4, False),
        ("zero", 0, False),
        ("wide", 1 << 33, False),
        ("inverted", 4, True),
    ):
        plan = ti.aot.ExecutionPlan(
            prefix + [dict(loop, max_iterations=budget, continue_while_nonzero=not inverted)] + suffix,
            resources=resources,
            gradients={"x": "dx", "y": "dy"},
        )
        module.add_execution_plan(name, plan)
        # The body is serialized once regardless of budget.
        assert len(json.loads(plan.to_json())["operations"]) == len(prefix) + len(suffix) + 1
    # A nested loop in the outer body. Its counter resets on each entry; the
    # outer loop terminates using a separate private predicate and counter.
    inner = dict(loop, max_iterations=2)
    outer = dict(
        kind="while",
        predicate="outer_predicate",
        max_iterations=2,
        condition=[
            dict(
                kind="kernel",
                entry="condition",
                arguments=["outer_counter", "outer_predicate", "outer_checks", "outer_target", "inverted"],
            )
        ],
        body=[
            dict(kind="fill32", destination="counter", value=0),
            inner,
            dict(kind="kernel", entry="advance", arguments=["outer_counter", "x", "temporary"]),
        ],
    )
    nested_resources = resources + [
        dict(kind="ndarray", name=n, dtype="i32", shape=[1])
        for n in ("outer_counter", "outer_predicate", "outer_checks")
    ]
    nested_prefix = prefix + [dict(kind="fill32", destination=n, value=0) for n in ("outer_counter", "outer_checks")]
    module.add_execution_plan(
        "nested",
        ti.aot.ExecutionPlan(
            nested_prefix + [outer] + suffix, resources=nested_resources, gradients={"x": "dx", "y": "dy"}
        ),
    )
    if arch == "vulkan":
        for name, chunks in (("vulkan", [2, 2]), ("vulkan_zero", []), ("vulkan_many", [1] * 9)):
            selected = dict(
                loop,
                max_iterations=sum(chunks),
                body=loop["body"][1:],
                lowering=dict(
                    kind="vulkan_structured",
                    version=1,
                    counter="counter",
                    chunks=[dict(iterations=count, strategy="compact") for count in chunks],
                ),
            )
            module.add_execution_plan(
                name,
                ti.aot.ExecutionPlan(
                    prefix + [loop["body"][0], selected] + suffix,
                    resources=resources,
                    gradients={"x": "dx", "y": "dy"},
                ),
            )
    artifact = tmp_path / "while.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_while_probe.py")), mode, arch, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
