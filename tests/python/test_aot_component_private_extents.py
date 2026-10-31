"""Private scratch follows declared resource extents in JIT and frozen AOT."""
import json
import struct
from zipfile import ZipFile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class _Workspace:
    def __init__(self, count):
        self.count = count
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.source = scope.field(ti.types.vector(3, ti.f32), shape=scope.dim("count"))
        scope.result = scope.field(ti.i32, shape=())
        scope.array = scope.ndarray(ti.f32, shape=scope.dim("count"))
        self.tree = builder.finalize()

    @ti.kernel
    def fill(self, scratch: ti.types.ndarray(ti.i32, 2)):
        for i, j in scratch:
            scratch[i, j] = i * 100 + j

    @ti.kernel
    def read_last(self, scratch: ti.types.ndarray(ti.i32, 2)):
        self.result[None] += scratch[scratch.shape[0] - 1, 5]

    def record_field(self, builder):
        self._record(builder, self.source)

    def record_array(self, builder):
        self._record(builder, self.array)

    def _record(self, builder, source):
        for _ in range(2):
            region = builder.create_sequential(private_scope=True)
            scratch = region.private_ndarray("scratch", ti.i32, (ti.graph.ArrayExtent(source), 6))
            region.dispatch(self.fill, scratch, template_args={"self": self})
            region.dispatch(self.read_last, scratch, template_args={"self": self})
            builder.append(region)


@pytest.mark.parametrize("record_name", ["record_field", "record_array"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_private_scratch_extents_follow_frozen_consumer_resources(tmp_path, record_name):
    producer = _Workspace(7)
    normal = ti.graph.GraphBuilder()
    getattr(producer, record_name)(normal)
    normal.compile().run({})
    assert producer.result[None] == 2 * 605
    root = Component.from_owner(producer).compile_graph(getattr(producer, record_name))
    different = _Workspace(13)
    other = Component.from_owner(different).compile_graph(getattr(different, record_name))
    assert other.contract["semantic_key"] == root.contract["semantic_key"]
    assert len(root.contract["private_shapes"]) == 2
    assert root.contract["resources"]["source" if record_name == "record_field" else "array"]["access"] == "none"
    shape_sources = {extent["array"] for shape in root.contract["private_shapes"].values()
                     for extent in shape if isinstance(extent, dict)}
    assert shape_sources.isdisjoint(symbol for entry in root.contract["entries"] for symbol in entry["arguments"])
    module = ti.aot.Module()
    root.export(module, "workspace")
    artifact = tmp_path / "workspace.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("workspace")
        with component.instantiate(dict(count=11)) as first, component.instantiate(dict(count=37)) as second:
            for instance in (first, second):
                instance.initialize(zero=component.contract["resources"])
            with first.prepare() as one, second.prepare() as two:
                one.run()
                two.run()
                one.run()
                runtime.wait()
            assert struct.unpack("<i", first.resource("result").read())[0] == 4 * 1005
            assert struct.unpack("<i", second.resource("result").read())[0] == 2 * 3605
            source = "source" if record_name == "record_field" else "array"
            assert not np.frombuffer(second.resource(source).read(), dtype=np.float32).any()
    if backend == "cpu" and record_name == "record_field":
        # A valid semantic hash alone cannot replace the authoritative Plan shape.
        from taichi_forge_runtime.aot._component import _control_identity

        with ZipFile(artifact) as archive:
            payloads = {name: archive.read(name) for name in archive.namelist()}
        contract = json.loads(payloads["components/workspace.json"])
        next(iter(contract["private_shapes"].values()))[1] = 5
        contract["semantic_key"] = _control_identity(contract)
        changed = tmp_path / "bad.tcm"
        with ZipFile(changed, "w") as archive:
            for name, payload in payloads.items():
                archive.writestr(name, json.dumps(contract) if name == "components/workspace.json" else payload)
        with Runtime(backend) as runtime, runtime.load_module(changed) as loaded:
            with pytest.raises(ValueError, match="private array shape differs"):
                loaded.component("workspace")


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_private_extent_rejects_undeclared_and_invalid_sources():
    model = _Workspace(3)
    with pytest.raises(ValueError, match="logical index shape"):
        ti.graph.ArrayExtent(model.source, 1)  # Vector lanes are not index axes.
    with pytest.raises(TypeError, match="Field or whole Ndarray"):
        ti.graph.ArrayExtent(np.zeros(3))
    model.source = ti.field(ti.f32, shape=3)
    with pytest.raises(ti.TaichiCompilationError, match="declared|declaration|layout"):
        Component.from_owner(model).compile_graph(model.record_field)
