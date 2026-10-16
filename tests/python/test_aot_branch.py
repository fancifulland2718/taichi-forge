"""Persistent branch consumption with actual compiled forward/reverse kernels."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Scalar
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_aot_persistent_branch_forward_reverse_and_ownership(tmp_path):
    _persistent_branch(tmp_path, host=False)


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_host_branch_forward_reverse_and_ownership(tmp_path):
    _persistent_branch(tmp_path, host=True)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_aot_masked_branch_forward_reverse_and_ownership(tmp_path):
    _persistent_branch(tmp_path, host=False, masked=True)


def _persistent_branch(tmp_path, host, masked=False):
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch == "x64" or arch == "arm64":
        arch = "cpu"
    route = "cuda_masked" if masked else "cpu" if arch == "cpu" else f"{arch}_host" if host else "cuda_conditional"

    @ti.kernel
    def condition(
        choice: ti.types.ndarray(ti.i32, 1),
        selector: ti.types.ndarray(ti.i32, 1),
        checks: ti.types.ndarray(ti.i32, 1),
    ):
        selector[0] = choice[0]
        checks[0] += 1

    @ti.kernel
    def forward(
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        gain: ti.f32,
    ):
        for i in x:
            y[i] = gain * x[i] * x[i]

    @ti.kernel
    def seed_reverse(dy: ti.types.ndarray(ti.f32, 1), selector: ti.types.ndarray(ti.i32, 1)):
        for i in dy:
            dy[i] = 1
        # The original selector may change inside the accepted body. Later
        # gated tasks must keep the selected branch, not re-read this value.
        selector[0] = -77

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(condition)
    module.add_kernel(forward)
    module.add_kernel(forward.grad, name="reverse")
    if masked:
        module.add_kernel(seed_reverse)
    compiled = time.perf_counter()
    if masked:
        for name in ("forward", "reverse", "seed_reverse"):
            kernel = module._exported_kernels[name]
            entry = "gated_" + name
            module._aot_builder._add_with_launch_abi(entry, kernel, "cuda_graph_gate_v1")
            module._exported_kernels[entry] = kernel
            module._kernel_launch_abis[entry] = "cuda_graph_gate_v1"
            module._content.append("kernel:" + entry)
    variants_ready = time.perf_counter()
    resources = [dict(kind="ndarray", name=n, dtype="i32", shape=[1]) for n in ("selector", "checks", "p")]
    resources += [dict(kind="ndarray", name=n, dtype="f32", shape=[17]) for n in ("dx", "dy")]
    resources += [Scalar("f32", i + 1).resource(f"gain{i}") for i in range(9)]

    def branch(selection, fallback):
        cases = 1 if selection == "nonzero" else 8
        return dict(
            kind="branch",
            selector="selector",
            selection=selection,
            has_default=fallback,
            lowering=dict(kind=route, version=1),
            condition=[dict(kind="kernel", entry="condition", arguments=["choice", "selector", "checks"])],
            bodies=[
                (
                    [
                        dict(kind="kernel", entry="gated_forward", arguments=["x", "y", f"gain{i}"]),
                        dict(kind="kernel", entry="gated_seed_reverse", arguments=["dy", "selector"]),
                        dict(kind="kernel", entry="gated_reverse", arguments=["x", "y", f"gain{i}"]),
                    ]
                    if masked
                    else [
                        dict(kind="kernel", entry="forward", arguments=["x", "y", f"gain{i}"]),
                        # Reverse consumes y.grad; seed each explicit reverse call,
                        # including another invocation within the nested loop.
                        dict(kind="fill32", destination="dy", value=0x3F800000),
                        dict(kind="kernel", entry="reverse", arguments=["x", "y", f"gain{i}"]),
                        dict(kind="fill32", destination="selector", value=0xFFFFFFB3),  # -77
                    ]
                )
                for i in range(cases + int(fallback))
            ],
        )

    prefix = [dict(kind="fill32", destination=n, value=0) for n in ("checks", "dx")]
    prefix += [
        dict(kind="fill32", destination="dy", value=0x3F800000),
        dict(kind="fill32", destination="y", value=0xC2B60000),
    ]  # -91
    suffix = [dict(kind="copy", source="dx", destination="gradient")]
    suffix += [
        dict(kind="copy", source=n, destination="state", destination_offset=i * 4, size_bytes=4)
        for i, n in enumerate(("checks", "selector"))
    ]
    descriptions = {}
    for name, selection, fallback in (
        ("if", "nonzero", False),
        ("if_else", "nonzero", True),
        ("switch", "indexed", False),
        ("switch_default", "indexed", True),
    ):
        descriptions[name] = branch(selection, fallback)
    nested = branch("nonzero", True)
    nested["bodies"] = [
        [
            dict(
                kind="while",
                predicate="p",
                max_iterations=2,
                condition=[dict(kind="fill32", destination="p", value=1)],
                body=[branch("indexed", False)],
            )
        ],
        [],
    ]
    if masked:
        empty_case = branch("indexed", False)
        empty_case["bodies"][7] = []
        descriptions["switch_empty"] = empty_case
    else:
        descriptions["nested"] = nested
    if host:
        # Two nested host selections, each with its own condition, share the
        # persistent staging. GPU-native WHILE cannot contain a host branch.
        nested["bodies"][0] = [branch("indexed", False), branch("indexed", False)]
    for name, region in descriptions.items():
        plan = ti.aot.ExecutionPlan(prefix + [region] + suffix, resources=resources, gradients={"x": "dx", "y": "dy"})
        assert json.loads(plan.to_json())["version"] == (16 if masked else 12 if host else 11)
        module.add_execution_plan(name, plan)
    artifact = tmp_path / "branch.tcm"
    module.archive(str(artifact))
    exported = time.perf_counter()
    if masked:
        assert len(module._exported_kernels) == 7  # Shared by all five plans.
        with zipfile.ZipFile(artifact) as archive:
            assert sum(name.endswith(".bc") for name in archive.namelist()) == 7
    # Consumers use saved code and descriptions after producer teardown.
    ti.reset()
    evidence = dict(
        arch=arch,
        compile_ms=(compiled - start) * 1000,
        variant_generation_ms=(variants_ready - compiled) * 1000,
        export_ms=(exported - variants_ready) * 1000,
        artifact_bytes=artifact.stat().st_size,
    )
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_branch_probe.py")),
                mode,
                arch,
                str(artifact),
                *(["masked"] if masked else ["host"] if host else []),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        evidence[mode] = json.loads(next(line[7:] for line in result.stdout.splitlines() if line.startswith("RESULT ")))
    (tmp_path / "branch-costs.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


@test_utils.test(arch=[ti.cpu, ti.vulkan], offline_cache=False)
def test_aot_branch_rejects_unmatched_backend_before_work(tmp_path):
    from taichi_forge_runtime import aot

    @ti.kernel
    def write(out: ti.types.ndarray(ti.i32, 1)):
        out[0] = 99

    module = ti.aot.Module()
    module.add_kernel(write)
    artifact = tmp_path / "branch-rejection.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime:
        loaded = runtime.load_module(artifact)
        out, selector = runtime.ndarray("i32", 1), runtime.ndarray("i32", 1)
        import struct

        out.write(struct.pack("<i", 71))
        selector.write(struct.pack("<i", 1))
        body = [dict(kind="kernel", entry="write", arguments=["out"])]
        arch = ti.lang.impl.current_cfg().arch
        for route in (("vulkan_host", "cuda_masked") if arch == ti.cpu else ("cpu", "cuda_conditional", "cuda_masked")):
            branch = dict(
                kind="branch",
                selector="selector",
                selection="nonzero",
                has_default=False,
                lowering=dict(kind=route, version=1),
                condition=[],
                bodies=[body],
            )
            with pytest.raises(aot.AotError, match="declared backend lowering"):
                loaded.prepare_execution(aot.ExecutionPlan([branch]), dict(out=out, selector=selector))
            assert out.read() == struct.pack("<i", 71)
        recovered = loaded.prepare_execution(aot.ExecutionPlan(body), dict(out=out))
        recovered()
        assert out.read() == struct.pack("<i", 99)
        recovered.close()
        loaded.close()
        out.close()
        selector.close()
