"""Immutable artifact bytes initialize independent, mutable prepared resources."""

import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_data_resources_forward_reverse_and_lazy_integrity(tmp_path):
    @ti.kernel
    def advance(state: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(17):
            state[i] += 1

    @ti.kernel
    def forward(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        weights: ti.types.ndarray(ti.f32, ndim=1),
        state: ti.types.ndarray(ti.f32, ndim=1),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in range(17):
            y[i] = x[i] * weights[i] + state[i]

    @ti.kernel
    def bytes_to_words(data: ti.types.ndarray(ti.u8, ndim=1), out: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(3):
            out[i] = ti.cast(data[i], ti.i32)

    caps = (
        [ti.DeviceCapability.spirv_has_int8, "spirv_has_int8_storage"]
        if ti.lang.impl.current_cfg().arch == ti.vulkan
        else []
    )
    module = ti.aot.Module(caps=caps)
    for kernel in (advance, forward, bytes_to_words):
        module.add_kernel(kernel)
    module.add_kernel(forward.grad, name="reverse")
    original = struct.pack("<17f", *[(i % 7 + 1) / 8 for i in range(17)])
    mutable = bytearray(original)
    key = module.add_data(mutable)
    assert key == hashlib.sha256(original).hexdigest()
    mutable[:] = b"\0" * len(mutable)
    assert module.add_data(original) == key
    unused_key = module.add_data(b"not part of any selected resource closure")
    byte_key = module.add_data(bytes((1, 128, 255)))
    empty_key = module.add_data(b"")
    resources = [
        dict(kind="ndarray", name=name, dtype="f32", shape=[dict(array="shape_only", axis=0)], initial_data=key)
        for name in ("weights", "state")
    ] + [
        dict(kind="ndarray", name="dy", dtype="f32", shape=[17]),
        dict(kind="ndarray", name="shape_only", dtype="f32", shape=[17], initial_data="a" * 64),
        dict(kind="ndarray", name="unused", dtype="f32", shape=[17], initial_data="b" * 64),
    ]
    plan = ti.aot.ExecutionPlan(
        [
            dict(kind="fill32", destination="dx", value=0),
            dict(kind="fill32", destination="dy", value=0x3F800000),
            dict(kind="kernel", entry="advance", arguments=["state"]),
            dict(kind="kernel", entry="forward", arguments=["x", "weights", "state", "y"]),
            dict(kind="kernel", entry="reverse", arguments=["x", "weights", "state", "y"]),
        ],
        resources=resources,
        gradients=dict(x="dx", y="dy"),
    )
    module.add_execution_plan("step", plan)
    assert plan.data_entries == (key,)
    module.add_execution_plan(
        "initialized",
        ti.aot.ExecutionPlan(
            [dict(kind="copy", source="state", destination="out")],
            initialization=[dict(kind="copy", source="seed", destination="state")],
            resources=[
                dict(kind="ndarray", name="seed", dtype="f32", shape=[17], initial_data=key),
                dict(kind="ndarray", name="state", dtype="f32", shape=[17]),
            ],
        ),
    )
    module.add_execution_plan(
        "bytes",
        ti.aot.ExecutionPlan(
            [
                dict(kind="kernel", entry="bytes_to_words", arguments=["data", "out"]),
            ],
            resources=[dict(kind="ndarray", name="data", dtype="u8", shape=[3], initial_data=byte_key)],
        ),
    )
    module.add_execution_plan(
        "empty",
        ti.aot.ExecutionPlan(
            [
                dict(kind="copy", source="empty", destination="out"),
            ],
            resources=[dict(kind="ndarray", name="empty", dtype="f32", shape=[0], initial_data=empty_key)],
        ),
    )
    with pytest.raises(ValueError, match="unexported data"):
        module.add_execution_plan(
            "missing",
            ti.aot.ExecutionPlan(
                [dict(kind="copy", source="missing", destination="out")],
                resources=[dict(kind="ndarray", name="missing", dtype="f32", shape=[17], initial_data="c" * 64)],
            ),
        )
    assert "missing" not in module._execution_plans
    directory = tmp_path / "data-module"
    artifact = tmp_path / "data.tcm"
    module.save(directory)
    module.archive(str(artifact))
    expected = {f"data/{k}.bin" for k in (key, byte_key, empty_key)}
    with zipfile.ZipFile(artifact) as archive:
        assert {name for name in archive.namelist() if name.startswith("data/")} == expected
        assert archive.read(f"data/{key}.bin") == original
        assert f"data/{unused_key}.bin" not in archive.namelist()
    arch = "cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_data_resources_probe.py")),
                mode,
                arch,
                str(artifact),
                str(directory),
                key,
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "cleanup failed" not in result.stdout + result.stderr
