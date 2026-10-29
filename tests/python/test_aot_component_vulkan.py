"""Vulkan components use existing GFX argument and submission contracts."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import AotError, Runtime
from tests import test_utils


@ti.data_oriented
class _ArrayComponent:
    def __init__(self, capacity=7, active=5):
        self.capacity, self.active, self.seed, self.bump = capacity, active, 13, 2
        scope = ti.experimental.ResourceScope(self)
        scope.dim("active")
        scope.value("seed", ti.i32)
        scope.value("bump", ti.i32)
        scope.records = scope.ndarray(
            ti.types.struct(value=ti.i32, tag=ti.i32, pair=ti.types.vector(2, ti.i32)), shape=scope.dim("capacity")
        )
        scope.alias = self.records
        scope.output = scope.ndarray(ti.i32, shape=scope.dim("capacity"))
        scope.require_le("active_capacity", scope.dim("active"), scope.dim("capacity"), when="records")

    @ti.kernel
    def initialize(self, records: ti.types.ndarray()):
        for i in range(records.shape[0]):
            records[i].value = self.seed + i
            records[i].tag = i % 3
            records[i].pair = ti.Vector([i, 10])

    @ti.kernel
    def step(self, records: ti.types.ndarray(), alias: ti.types.ndarray(), output: ti.types.ndarray()):
        for i in range(self.active):
            records[i].value += self.bump
            alias[i].pair[1] += records[i].tag
            output[i] = records[i].value + alias[i].pair[1]

    def record_initialize(self, builder):
        builder.dispatch(self.initialize, self.records)

    def record(self, builder):
        builder.dispatch(self.step, self.records, self.alias, self.output)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_component_array_instances_and_source_free_replay(tmp_path):
    model = _ArrayComponent()
    model.output.fill(-1)
    model.initialize(model.records)
    model.step(model.records, model.alias, model.output)
    np.testing.assert_array_equal(model.output.to_numpy(), [25 + i + i % 3 for i in range(5)] + [-1, -1])
    component = Component.from_owner(model)
    root = component.compile_graph(model.record, initializer=model.record_initialize)
    assert not root.contract["groups"] and not root.contract["fixed_groups"]
    assert set(root.contract["resources"]["records"]["aliases"]) == {"records", "alias"}
    tasks = [task for entry in root.contract["compilation"]["entries"] for task in entry["tasks"]]
    assert tasks and all(task["backend"] == "vulkan" for task in tasks)
    module = ti.aot.Module()
    root.export(module, "step")
    artifact = tmp_path / "arrays.tcm"
    module.archive(str(artifact))
    frozen = hashlib.sha256(artifact.read_bytes()).hexdigest()
    import taichi_forge_runtime

    package = tmp_path / "consumer_package"
    shutil.copytree(
        Path(taichi_forge_runtime.__file__).parent,
        package / "taichi_forge_runtime",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    consumer = tmp_path / "consume.py"
    shutil.copy2(Path(__file__).with_name("aot_vulkan_component_probe.py"), consumer)
    environment = dict(
        os.environ, FORGE_AOT_TEST_PACKAGE=str(package), TI_LIB_DIR=str(package / "taichi_forge_runtime/_lib/runtime")
    )
    ti.reset()
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(consumer), str(artifact), str(tmp_path / "consumer.json")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VULKAN_COMPONENT_CONSUMER_OK" in result.stdout
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == frozen


@ti.data_oriented
class _DenseComponent:
    def __init__(self):
        self.capacity = 7
        self.active = 5
        self.seed = 13
        self.bump = 2
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.dim("active")
        scope.value("seed", ti.i32)
        scope.value("bump", ti.i32)
        scope.state_record = scope.struct_field(dict(x=ti.i32, y=ti.i32), shape=scope.dim("capacity"))
        scope.x = self.state_record.x
        scope.y = self.state_record.y
        scope.v = scope.vector_field(2, ti.i32, shape=scope.dim("capacity"))
        scope.tally = scope.field(ti.i32, shape=())
        scope.alias = self.x
        scope.require_le("active_capacity", scope.dim("active"), scope.dim("capacity"), when="x")
        self.tree = builder.finalize()

    @ti.kernel
    def initialize(self):
        for i in range(self.x.shape[0]):
            self.x[i] = self.seed + i
            self.y[i] = -99
            self.v[i] = ti.Vector([-3, -4])
        self.tally[None] = 0

    @ti.kernel
    def step(self):
        for i in range(self.active):
            self.x[i] += self.bump
            self.y[i] = 2 * self.alias[i]
        for i in self.v:
            self.v[i] += ti.Vector([self.bump, 1])
        self.tally[None] += self.active

    def record_initialize(self, builder):
        builder.dispatch(self.initialize)

    def record(self, builder):
        builder.dispatch(self.step)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_vulkan_component_dense_instances_and_source_free_replay(tmp_path):
    model = _DenseComponent()
    model.initialize()
    model.step()
    np.testing.assert_array_equal(model.y.to_numpy(), [2 * (15 + i) for i in range(5)] + [-99, -99])
    root = Component.from_owner(model).compile_graph(model.record, initializer=model.record_initialize)
    assert len(root.contract["groups"]) == 1 and not root.contract["fixed_groups"]
    module = ti.aot.Module()
    root.export(module, "dense")
    artifact = tmp_path / "dense.tcm"
    module.archive(str(artifact))
    frozen = hashlib.sha256(artifact.read_bytes()).hexdigest()
    import taichi_forge_runtime

    package = tmp_path / "consumer_package"
    shutil.copytree(
        Path(taichi_forge_runtime.__file__).parent,
        package / "taichi_forge_runtime",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    consumer = tmp_path / "consume.py"
    shutil.copy2(Path(__file__).with_name("aot_vulkan_dense_component_probe.py"), consumer)
    environment = dict(
        os.environ, FORGE_AOT_TEST_PACKAGE=str(package), TI_LIB_DIR=str(package / "taichi_forge_runtime/_lib/runtime")
    )
    ti.reset()
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(consumer), str(artifact), str(tmp_path / "consumer.json")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VULKAN_DENSE_CONSUMER_OK" in result.stdout
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == frozen
    # Metadata alone cannot grant/remove the shader's compiler declaration.
    with zipfile.ZipFile(artifact) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    metadata = json.loads(contents["metadata.json"])
    entry = next(kernel for kernel in metadata["kernels"] if kernel["runtime_affine_arrays"])
    entry_name = entry["name"]
    entry["runtime_affine_arrays"] = []
    contents["metadata.json"] = json.dumps(metadata).encode()
    tampered = tmp_path / "changed_metadata.tcm"
    with zipfile.ZipFile(tampered, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with Runtime("vulkan") as runtime, runtime.load_module(tampered) as loaded:
        with pytest.raises(AotError, match="affine array metadata differs"):
            loaded.kernel(entry_name)
