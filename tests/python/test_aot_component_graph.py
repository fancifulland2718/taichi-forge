import json
import struct
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component, RuntimeValue
from taichi_forge_runtime.aot import Runtime
from tests import test_utils
from tests.python.test_aot_component_dense import _DenseModel, _environment


@ti.data_oriented
class _RootModel(_DenseModel):
    def record(self, builder):
        builder.dispatch(self.active_pass, template_args={"self": self})
        builder.dispatch(self.capacity_pass, template_args={"self": self})

    @ti.kernel
    def combine(self, a: ti.f32, b: ti.f32):
        for i in range(self.active):
            self.x[i] = a + 2 * b + self.weight

    @ti.kernel
    def randomize(self, scale: ti.f32):
        for i in range(self.active):
            self.x[i] = scale * ti.random(ti.f32)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False, saturating_grid_dim=1, default_gpu_block_dim=64)
def test_normal_graph_root_exports_its_actual_compiled_closure(tmp_path):
    model = _RootModel(7, 7)
    root = _environment().compile_graph(model.record)
    assert len(root.contract["entries"]) == 2
    assert len(root.contract["groups"]) == 2
    facts = root.contract["compilation"]
    assert root.contract["version"] == 2 and facts["task_scope"] == "compiled_not_observed_replay"
    assert all(not entry["opaque"] and entry["tasks"] for entry in facts["entries"])
    assert {name: resource["access"] for name, resource in root.contract["resources"].items()} == {
        "alias_x": "read_write",
        "y": "read_write",
        "v": "write",
        "m": "write",
    }
    assert all(resource["effects_precision"] == "compiler_proven" for resource in root.contract["resources"].values())
    if ti.lang.impl.current_cfg().arch == ti.cuda:
        tasks = [task for entry in facts["entries"] for task in entry["tasks"]]
        assert all(task["range_mapping"] == "grid_stride" for task in tasks)
        assert all(task["selected_grid_size"] == 1 and task["selected_block_size"] == 64 for task in tasks)
        assert all(task["actual_grid_size"] is None for task in tasks)
    module = ti.aot.Module(ti.lang.impl.current_cfg().arch)
    plan = root.export(module, "normal_root")
    assert len(plan.kernel_entries) == 2
    assert not plan.data_entries
    target = tmp_path / "root.tcm"
    module.archive(str(target))
    with ZipFile(target) as archive:
        contract = json.loads(archive.read("components/normal_root.json"))
        assert contract["semantic_key"] == root.contract["semantic_key"]
        assert all(not name.startswith("data/") for name in archive.namelist())
    with Runtime("cpu" if ti.lang.impl.current_cfg().arch == ti.cpu else "cuda") as runtime:
        loaded = runtime.load_module(target)
        component = loaded.component("normal_root")
        # Larger than both the producer and the selected 1 * 64 CUDA grid.
        with component.instantiate(dict(capacity=257, active=251)) as first:
            with pytest.raises(ValueError, match="explicit consumer initialization"):
                first.prepare(values=dict(gravity=[5, 7, 9]))
            first.initialize(
                {
                    name: struct.pack("<f", -11) * (first.resource(name).nbytes // 4)
                    for name in component.contract["resources"]
                }
            )
            with first.prepare(values=dict(gravity=[5, 7, 9])) as prepared:
                prepared.run()
                with pytest.raises(RuntimeError, match="cold boundary"):
                    first.initialize(zero=("x",))
                x = struct.unpack("<257f", first.resource("x").read())
                assert x == tuple([-6] * 251 + [-11] * 6)
                assert struct.unpack("<257f", first.resource("y").read()) == tuple(range(257))
            # A new immutable binding may change active count without resizing.
            with first.prepare(values=dict(gravity=[100, 7, 9]), dimensions=dict(active=0)) as empty:
                empty.run()
            assert struct.unpack("<257f", first.resource("x").read()) == x
            with pytest.raises(ValueError, match="layout dimensions"):
                first.prepare(values=dict(gravity=[5, 7, 9]), dimensions=dict(capacity=41))
        with pytest.raises(ValueError, match="exceeds capacity"):
            component.instantiate(dict(capacity=7, active=8))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_component_root_identity_includes_actual_argument_connections(tmp_path):
    model = _RootModel(7, 7)
    model.weight = 3.0
    a = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "a", ti.f32)
    b = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "b", ti.f32)
    environment = Component(**_environment().declarations, weight=RuntimeValue(ti.f64))
    direct = environment.compile_graph(lambda builder: builder.dispatch(model.combine, a, b))
    swapped = environment.compile_graph(lambda builder: builder.dispatch(model.combine, b, a))
    assert direct.contract["entries"][0]["semantic_key"] == swapped.contract["entries"][0]["semantic_key"]
    assert direct.contract["semantic_key"] != swapped.contract["semantic_key"]
    assert direct.contract["resources"]["x"]["access"] == "write"
    module = ti.aot.Module()
    direct.export(module, "connected")
    good, bad, legacy = (tmp_path / name for name in ("good.tcm", "bad.tcm", "legacy.tcm"))
    module.archive(str(good))
    # Same argument set and native code, but inconsistent connection order.
    with ZipFile(good) as source, ZipFile(bad, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "components/connected.json":
                contract = json.loads(data)
                args = contract["entries"][0]["arguments"]
                args[0], args[1] = args[1], args[0]
                data = json.dumps(contract)
            target.writestr(name, data)
    # Version 1 remains conservative and has no compiled-task projection.
    with ZipFile(good) as source, ZipFile(legacy, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "components/connected.json":
                contract = json.loads(data)
                contract["version"] = 1
                contract.pop("compilation")
                contract["required_capabilities"].remove("component_compiler_facts_v1")
                for resource in contract["resources"].values():
                    resource.update(access="read_write", effects_precision="conservative")
                data = json.dumps(contract)
            target.writestr(name, data)
    with Runtime("cpu") as runtime:
        component = runtime.load_module(good).component("connected")
        with component.instantiate(dict(capacity=11, active=9)) as instance:
            instance.initialize(zero=component.contract["resources"])
            with instance.prepare(arguments=dict(a=1.5, b=2), values=dict(weight=2.5)) as call:
                call.run()
            assert struct.unpack("<11f", instance.resource("x").read()) == tuple([8] * 9 + [0] * 2)
        with pytest.raises(ValueError, match="actual execution plan"):
            runtime.load_module(bad).component("connected")
        assert runtime.load_module(legacy).component("connected").contract["version"] == 1


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_unknown_effects_keep_conservative_resource_contract():
    model = _RootModel(7, 7)
    scale = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "scale", ti.f32)
    root = _environment().compile_graph(lambda builder: builder.dispatch(model.randomize, scale))
    assert root.contract["compilation"]["entries"][0]["opaque"]
    assert root.contract["resources"]["x"]["access"] == "read_write"
    assert root.contract["resources"]["x"]["effects_precision"] == "conservative"
