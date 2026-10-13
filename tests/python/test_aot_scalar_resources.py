"""Persistent scalar parameters use the ordinary typed kernel argument ABI."""

import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from taichi_forge_runtime.aot import Scalar
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_scalar_resources_forward_reverse_and_exact_bits(tmp_path):
    @ti.kernel
    def forward(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        n: ti.i32,
        gain: ti.f32,
    ):
        for i in range(n):
            y[i] = gain * x[i] * x[i]

    @ti.kernel
    def bits(encoded: ti.types.ndarray(ti.u64, ndim=1), wide: ti.u64, signed: ti.i64, negative_zero: ti.f32):
        encoded[0] = wide
        encoded[1] = ti.bit_cast(signed, ti.u64)
        encoded[2] = ti.cast(ti.bit_cast(negative_zero, ti.u32), ti.u64)

    @ti.kernel
    def narrow(
        encoded: ti.types.ndarray(ti.i64, ndim=1), a: ti.i8, b: ti.i16, c: ti.u8, d: ti.u16, h: ti.f16, z: ti.f64
    ):
        encoded[0] = a
        encoded[1] = b
        encoded[2] = c
        encoded[3] = d
        encoded[4] = ti.cast(h * 8, ti.i64)
        encoded[5] = ti.bit_cast(z, ti.i64)

    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    module = ti.aot.Module(caps=["spirv_has_int64"] if arch == "vulkan" else [])
    module.add_kernel(forward)
    module.add_kernel(forward.grad, name="backward")
    module.add_kernel(bits)
    encoded = dict(kind="kernel", entry="bits", arguments=["encoded", "wide", "signed", "negative_zero"])
    resources = [
        Scalar("i32", 17).resource("n"),
        Scalar("u64", (1 << 64) - 1).resource("wide"),
        Scalar("i64", -(1 << 63)).resource("signed"),
        Scalar("f32", -0.0).resource("negative_zero"),
        dict(kind="ndarray", name="dy", dtype="f32", shape=[17]),
    ]
    for name, gain in (("negative", -1.75), ("positive", 0.5)):
        plan = ti.aot.ExecutionPlan(
            [
                dict(kind="fill32", destination="dx", value=0),
                dict(kind="fill32", destination="dy", value=0x3F800000),
                dict(kind="kernel", entry="forward", arguments=["x", "y", "n", "gain"]),
                dict(kind="kernel", entry="backward", arguments=["x", "y", "n", "gain"]),
                encoded,
            ],
            resources=[*resources, Scalar("f32", gain).resource("gain")],
            gradients=dict(x="dx", y="dy"),
        )
        assert plan.argument_names == ("dx", "encoded", "x", "y")
        module.add_execution_plan(name, plan)
    # Unused ndarray/scalar declarations must not allocate storage or leak arguments.
    module.add_execution_plan("bits_only", ti.aot.ExecutionPlan([encoded], resources=resources))
    if arch == "cpu":
        module.add_kernel(narrow)
        values = dict(
            a=Scalar("i8", -128),
            b=Scalar("i16", -32768),
            c=Scalar("u8", 255),
            d=Scalar("u16", 65535),
            h=Scalar("f16", 1.5),
            z=Scalar("f64", -0.0),
        )
        module.add_execution_plan(
            "narrow",
            ti.aot.ExecutionPlan(
                [dict(kind="kernel", entry="narrow", arguments=["encoded", *values])],
                resources=[value.resource(name) for name, value in values.items()],
            ),
        )
    artifact = tmp_path / "scalar-resources.tcm"
    module.archive(str(artifact))
    for mode in ("standalone", "borrowed"):
        command = [sys.executable] + (["-S"] if mode == "standalone" else [])
        result = subprocess.run(
            [*command, str(Path(__file__).with_name("aot_scalar_resources_probe.py")), mode, arch, str(artifact)],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
