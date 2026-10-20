"""Module Field roots and native array operations share storage without copies."""

import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_module_root_storage_forward_backward_and_retirement(tmp_path):
    # A producer-only tree makes source IDs differ from exported module IDs.
    unused = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.dense(ti.i, 7).place(unused)
    unused_tree = builder.finalize()
    n = 17
    keys = ti.field(ti.i32, shape=n)
    order = ti.field(ti.i32, shape=n)
    x = ti.field(ti.f32, shape=n, needs_grad=True)
    pair = ti.Vector.field(2, ti.f32, shape=n)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def seed(source: ti.types.ndarray(ti.f32, ndim=1), input_keys: ti.types.ndarray(ti.i32, ndim=1)):
        for i in range(n):
            keys[i], order[i], x[i] = input_keys[i], i, source[i]
            x.grad[i] = 0
            pair[i] = ti.Vector([2 * source[i], 3 * source[i]])
        loss[None] = 0
        loss.grad[None] = 1

    @ti.kernel
    def energy(values: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)):
        for i in range(n):
            loss[None] += (i + 1) * values[order[i]] ** 2

    @ti.kernel
    def report(
        out_keys: ti.types.ndarray(ti.i32, ndim=1),
        out_order: ti.types.ndarray(ti.i32, ndim=1),
        out_loss: ti.types.ndarray(ti.f32, ndim=1),
    ):
        for i in range(n):
            out_keys[i], out_order[i] = keys[i], order[i]
        out_loss[0] = loss[None]

    module = ti.aot.Module()
    for name, field in (("keys", keys), ("order", order), ("values", x), ("gradient", x.grad), ("pair", pair)):
        module.add_field(name, field, as_array=True)
    for as_array in (True, False):
        with pytest.raises(ValueError, match="cannot be rebound"):
            module.add_field("keys", order, as_array=as_array)
    module.add_kernel(seed)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    module.add_kernel(report)
    resources = [
        dict(kind="module_array", name=name, view=name) for name in ("keys", "order", "values", "gradient", "pair")
    ]
    module.add_execution_plan(
        "step",
        ti.aot.ExecutionPlan(
            [
                dict(kind="kernel", entry="seed", arguments=["source", "input_keys"]),
                dict(kind="sort", keys="keys", values="order"),
                dict(kind="kernel", entry="energy", arguments=["values"]),
                dict(kind="kernel", entry="backward", arguments=["values"]),
                dict(kind="kernel", entry="report", arguments=["out_keys", "out_order", "out_loss"]),
                dict(kind="copy", source="gradient", destination="dx"),
                dict(kind="copy", source="pair", destination="out_pair"),
            ],
            resources=resources,
            gradients={"values": "gradient"},
        ),
    )
    # A root alias can parent an ordinary subview or a shape-only dependency.
    module.add_execution_plan(
        "tail",
        ti.aot.ExecutionPlan(
            [
                dict(kind="copy", source="tail", destination="tail_output"),
            ],
            resources=[resources[0], dict(kind="ndarray_view", name="tail", array="keys", offset=2, shape=[n - 2])],
        ),
    )
    artifact = tmp_path / "root-storage.tcm"
    module.archive(str(artifact))
    invalid_artifact = tmp_path / "invalid-root-storage.tcm"
    with zipfile.ZipFile(artifact) as bundle, zipfile.ZipFile(invalid_artifact, "w") as invalid_bundle:
        name = next(name for name in bundle.namelist() if name.endswith("root_storage.json"))
        metadata = json.loads(bundle.read(name))
        views = metadata["views"]
        for item in bundle.infolist():
            data = bundle.read(item.filename)
            if item.filename == name:
                invalid = json.loads(data)
                invalid["views"]["keys"]["byte_offset"] = 1 << 50
                data = json.dumps(invalid).encode()
            invalid_bundle.writestr(item, data)
    assert set(views) == {"keys", "order", "values", "gradient", "pair"}
    assert views["order"]["byte_offset"] > views["keys"]["byte_offset"]
    assert views["pair"]["element_shape"] == [2]
    assert views["keys"]["tree_id"] == 0
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_root_storage_probe.py")),
                mode,
                arch,
                str(artifact),
                str(invalid_artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Unable to release" not in result.stdout + result.stderr
    unused_tree.destroy()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False, demote_dense_struct_fors=False)
def test_aot_module_array_opt_in_preserves_legacy_field_export(tmp_path):
    value = ti.field(ti.i32, shape=4)
    module = ti.aot.Module()
    module.add_field("value", value)
    module.save(tmp_path)
    assert not (tmp_path / "root_storage.json").exists()
    module.add_field("value", value, as_array=True)
    with pytest.raises(RuntimeError, match="relocatable dense trees"):
        module.save(tmp_path)


@test_utils.test(arch=[ti.cpu], offline_cache=False)
def test_aot_module_array_exports_strided_field_layout(tmp_path):
    values = ti.field(ti.i32)
    other = ti.field(ti.i32)
    ti.root.dense(ti.i, 4).place(values, other)
    module = ti.aot.Module()
    module.add_field("strided", values, as_array=True)
    module.save(tmp_path)
    metadata = json.loads((tmp_path / "root_storage.json").read_text())
    assert metadata["version"] == 2
    assert metadata["views"]["strided"]["index_strides_bytes"] == [8]
    assert metadata["views"]["strided"]["element_strides_bytes"] == []
