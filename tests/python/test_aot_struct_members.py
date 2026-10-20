"""Typed private parents and affine members share existing storage and executors."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import taichi_forge as ti
from tests import test_utils


def transform(src, dst, size=16, strategy="strided_scalar"):
    return dict(
        kind="native", provider="builtin", operation="transform", version=1,
        parameters=dict(size=size, dtype="f32", scale_bits="4008000000000000",
                        bias_bits="401c000000000000", strategy=strategy),
        bindings=dict(src=src, dst=dst),
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, cpu_max_num_threads=4)
def test_structure_member_resources_and_native_transform(tmp_path):
    record = ti.types.struct(tag=ti.i8, value=ti.f32, other=ti.f32, tail=ti.i8)
    caps = ([ti.DeviceCapability.spirv_has_int8, "spirv_has_int8_storage"]
            if ti.cfg.arch == ti.vulkan else [])
    module = ti.aot.Module(caps=caps)
    module.add_ndarray_layout("records", record)

    @ti.kernel
    def read_records(records: ti.types.ndarray(record, ndim=1), output: ti.types.ndarray(ti.f32, ndim=1)):
        for i in output:
            output[i] = records[i].other

    @ti.kernel
    def energy(x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
               y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)):
        for i in x:
            y[i] = 0.5 * x[i] * x[i]

    @ti.kernel
    def count(n: ti.i32, result: ti.types.ndarray(ti.i32, ndim=1)):
        result[0] = n

    module.add_kernel(read_records)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="reverse")
    module.add_kernel(count)
    # This fixture intentionally checks the exported ABI against these offsets.
    # Poison padding to detect accidental packed copies or member overwrites.
    payload = bytearray([0xA5] * (17 * 16))
    for i in range(17):
        struct.pack_into("=b", payload, i * 16, -3)
        struct.pack_into("=f", payload, i * 16 + 4, i * 0.5)
        struct.pack_into("=f", payload, i * 16 + 8, -9)
        struct.pack_into("=b", payload, i * 16 + 12, 11)
    key = module.add_data(payload)
    private = [
        dict(kind="ndarray", name="state", layout="records", shape=[17], initial_data=key),
        dict(kind="ndarray_member", name="value", array="state", path="value"),
        dict(kind="ndarray_member", name="other", array="state", path="other"),
        # An unrelated declaration must not force allocation or layout loading.
        dict(kind="ndarray", name="unused", layout="not_exported", shape=[2**31 - 1]),
    ]
    member_ops = [
        transform("value", "other"),
        dict(kind="copy", source="state", destination="records_out"),
        dict(kind="kernel", entry="read_records", arguments=["state", "output"]),
        dict(kind="fill32", destination="dx", value=0),
        dict(kind="fill32", destination="dy", value=0x3F800000),
        dict(kind="kernel", entry="energy", arguments=["output", "energy"]),
        dict(kind="kernel", entry="reverse", arguments=["output", "energy"]),
    ]
    plan = ti.aot.ExecutionPlan(member_ops, resources=private, gradients=dict(output="dx", energy="dy"))
    assert json.loads(plan.to_json())["version"] == 23
    assert set(plan.argument_names) == {"records_out", "output", "energy", "dx", "dy"}
    module.add_execution_plan("private", plan)
    module.add_execution_plan(
        "inplace", ti.aot.ExecutionPlan(
            [transform("other", "other"), dict(kind="copy", source="state", destination="records_out")],
            resources=private,
        ),
    )
    external = [
        dict(kind="ndarray_view", name="slice", array="records", offset=1, shape=[16]),
        dict(kind="ndarray_member", name="value", array="slice", path="value"),
        dict(kind="ndarray_member", name="other", array="slice", path="other"),
    ]
    module.add_execution_plan("external", ti.aot.ExecutionPlan(
        [transform("value", "other")], resources=external))
    module.add_execution_plan("packed_output", ti.aot.ExecutionPlan(
        [transform("value", "output")], resources=external))
    for name, operation, resources in (
        ("oversize", transform("value", "other", 17), external),
        ("wrong_path", transform("value", "other"),
         [dict(r, path="absent") if r["name"] == "value" else r for r in external]),
        ("packed_rejects_member", transform("value", "other", strategy="packed_array"), external),
        ("member_copy", dict(kind="copy", source="value", destination="output"), external),
    ):
        module.add_execution_plan(name, ti.aot.ExecutionPlan([operation], resources=resources))
    module.add_execution_plan("empty", ti.aot.ExecutionPlan(
        [dict(kind="kernel", entry="count", arguments=["n", "result"])],
        resources=[
            dict(kind="ndarray", name="empty", layout="records", shape=[0]),
            dict(kind="ndarray_member", name="member", array="empty", path="value"),
            dict(kind="extent", name="n", dtype="i32", array="member", axis=0),
        ],
    ))
    artifact = tmp_path / "members.tcm"
    module.archive(str(artifact))
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).with_name("aot_struct_members_probe.py")),
             mode, arch, str(artifact)],
            env=os.environ.copy(), capture_output=True, text=True, timeout=90,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert not any(word in output for word in ("Validation Error", "SYNC-HAZARD", "cleanup failed")), output


@pytest.mark.parametrize("resource", [
    dict(kind="ndarray", name="a", layout="records", dtype="f32", shape=[3]),
    dict(kind="ndarray", name="a", layout="records", element_shape=[], shape=[3]),
    dict(kind="ndarray_member", name="a", array="a", path="value"),
    dict(kind="ndarray_member", name="a", array="parent", path=""),
])
def test_structure_resource_format_rejects_ambiguity(resource):
    with pytest.raises(ValueError):
        ti.aot.ExecutionPlan([], resources=[resource])
